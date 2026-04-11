"""
Data Packager — 탭별 LLM 전송용 풍부한 데이터 패키징
======================================================
각 탭의 원시 DataFrame을 받아, LLM이 유의미한 인사이트를 생성할 수 있는
구조화된 텍스트 블록으로 변환한다.

기존 ai_analysis.py 대비 개선:
  - 단순 평균 → 분포(평균/중앙값/std/p25/p75) + 일별 트렌드
  - GAP/활성레벨 데이터 통합 (gap_analyzer 결과)
  - 업체 그룹: k-Anonymity 통과한 항목만 포함
  - 장비: 5주 추이 + 비가동 패턴
  - 혼잡도: 시간대별 피크 + 공간 유형별 분포

사용:
    from src.intelligence.data_packager import DataPackager

    text = DataPackager.transit(worker_transit_df, dates=["20260310","20260311"])
    text = DataPackager.equipment(master_df, weekly_bp_df, weekly_overall_df)
    text = DataPackager.worker(worker_df, company_df, date_str)
    text = DataPackager.congestion(summary_dict, ranking_df)
    text = DataPackager.overview(daily_kpi_dict, prev_kpi_dict)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.intelligence.anonymization_pipeline import (
    AnonymizationPipeline as AP,
    K_ANON_MIN,
)

logger = logging.getLogger(__name__)


# ─── 내부 헬퍼 ──────────────────────────────────────────────────────────────

def _dist(series: pd.Series, label: str, unit: str = "분") -> str:
    """분포 통계 1줄 요약 (평균/중앙값/std/max)."""
    s = series.dropna()
    if s.empty:
        return f"- {label}: 데이터 없음"
    return (
        f"- {label}: 평균 {s.mean():.1f}{unit}, "
        f"중앙값 {s.median():.1f}{unit}, "
        f"표준편차 {s.std():.1f}{unit}, "
        f"최대 {s.max():.1f}{unit}, "
        f"상위25% {s.quantile(0.75):.1f}{unit}"
    )


def _pct(value: float | None, total: float | None) -> str:
    """비율 문자열 (분모 0 방어)."""
    if value is None or total is None or total == 0:
        return "N/A"
    return f"{value / total * 100:.1f}%"


def _activity_block(worker_df: pd.DataFrame) -> list[str]:
    """GAP/활성레벨 통계 블록 (컬럼 없으면 생략)."""
    lines: list[str] = []

    # 활성 레벨 분포
    if "ga_dominant_activity" in worker_df.columns:
        dist = worker_df["ga_dominant_activity"].value_counts(normalize=True) * 100
        parts = [f"{k}: {v:.1f}%" for k, v in dist.items() if pd.notna(k)]
        if parts:
            lines.append(f"- 지배적 활성 레벨 분포: {', '.join(parts)}")

    # 평균 GAP 비율
    if "ga_gap_ratio_pct" in worker_df.columns:
        avg_gap = worker_df["ga_gap_ratio_pct"].dropna().mean()
        if not np.isnan(avg_gap):
            lines.append(f"- 평균 음영(GAP) 비율: {avg_gap:.1f}% (건설현장 BLE 특성, 정상 범위)")

    # 활성 레벨 분당 분포
    cols_map = {
        "ga_high_active_min": "HIGH_ACTIVE",
        "ga_active_min": "ACTIVE",
        "ga_inactive_min": "INACTIVE",
        "ga_deep_inactive_min": "DEEP_INACTIVE",
        "ga_estimated_min": "음영추정",
    }
    activity_parts = []
    for col, lbl in cols_map.items():
        if col in worker_df.columns:
            avg = worker_df[col].dropna().mean()
            if not np.isnan(avg) and avg > 0:
                activity_parts.append(f"{lbl} {avg:.0f}분")
    if activity_parts:
        lines.append(f"- 평균 활성 레벨별 시간: {', '.join(activity_parts)}")

    return lines


def _company_rows(
    df: pd.DataFrame,
    company_col: str = "company_name",
    count_col: str = "worker_count",
    metric_cols: list[str] | None = None,
    top_n: int = 10,
    sort_col: str | None = None,
) -> list[str]:
    """
    업체별 통계 행 생성. k-Anonymity 미달 항목 억제.
    업체명은 세션 코드로 치환.
    """
    if df is None or df.empty or company_col not in df.columns:
        return []

    sort_col = sort_col or count_col
    if sort_col in df.columns:
        df = df.nlargest(top_n, sort_col)

    lines: list[str] = []
    for _, r in df.iterrows():
        cnt = r.get(count_col, 0)
        if not isinstance(cnt, (int, float)) or cnt < K_ANON_MIN:
            continue  # k-Anonymity 억제

        company_name = str(r.get(company_col, ""))
        code = AP.get_company_code(company_name) if company_name else "?"

        parts = [f"작업자 {cnt:.0f}명"]
        for col in (metric_cols or []):
            val = r.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                col_label = col.replace("_", " ").upper()
                parts.append(f"{col_label} {val:.2f}" if isinstance(val, float) else f"{col_label} {val}")

        lines.append(f"  - {code}: {', '.join(parts)}")

    return lines


# ─── 공개 패키저 ─────────────────────────────────────────────────────────────

class DataPackager:
    """
    탭별 LLM 전송용 데이터 텍스트 생성기.

    모든 메서드는 익명화 전 원본 데이터를 받아
    익명화 파이프라인이 처리하기 쉬운 구조화 텍스트를 반환.
    (업체명은 여기서 코드로 변환, 날짜는 pipeline에서 상대화)
    """

    # ─── 1. 대기/이동 시간 (Transit) ─────────────────────────────────

    @staticmethod
    def transit(
        worker_df: pd.DataFrame,
        dates: list[str] | None = None,
    ) -> str:
        """
        대기/이동 시간 탭용 풍부한 데이터 텍스트.

        포함:
          - MAT/LBT/EOD 분포 (평균/중앙값/std/max/p75)
          - 일별 트렌드
          - GAP/활성레벨 통계 (있을 경우)
          - 업체 그룹별 MAT 편차 (k-Anonymity 통과)
        """
        if worker_df is None or worker_df.empty:
            return "대기시간 데이터가 없습니다."

        lines: list[str] = ["## 대기/이동 시간 분석 데이터"]

        n_dates = worker_df["date"].nunique() if "date" in worker_df.columns else 1
        n_workers = len(worker_df)
        lines.append(f"- 분석 기간: {n_dates}일, 총 {n_workers}건 (T-Ward 작업자)")

        # 지표별 분포
        lines.append("\n### 지표별 분포")
        for col, label in [
            ("mat_minutes", "출근 대기(MAT)"),
            ("lbt_minutes", "중식 이동(LBT)"),
            ("eod_minutes", "퇴근 이동(EOD)"),
        ]:
            if col in worker_df.columns:
                lines.append(_dist(worker_df[col], label))

        # 전체 대기시간 합계 (TWT)
        twt_cols = [c for c in ["mat_minutes", "lbt_minutes", "eod_minutes"]
                    if c in worker_df.columns]
        if twt_cols:
            twt = worker_df[twt_cols].fillna(0).sum(axis=1)
            lines.append(_dist(twt, "총 대기시간(TWT)"))

        # 일별 트렌드
        if "date" in worker_df.columns and n_dates > 1:
            lines.append("\n### 일별 트렌드")
            lines.append("| 일차 | MAT평균 | LBT평균 | EOD평균 | 작업자수 |")
            lines.append("|------|---------|---------|---------|---------|")
            date_sorted = sorted(worker_df["date"].unique())
            for idx, d in enumerate(date_sorted, 1):
                grp = worker_df[worker_df["date"] == d]
                mat = grp["mat_minutes"].dropna().mean() if "mat_minutes" in grp else float("nan")
                lbt = grp["lbt_minutes"].dropna().mean() if "lbt_minutes" in grp else float("nan")
                eod = grp["eod_minutes"].dropna().mean() if "eod_minutes" in grp else float("nan")
                n = len(grp)
                mat_s = f"{mat:.1f}분" if not np.isnan(mat) else "-"
                lbt_s = f"{lbt:.1f}분" if not np.isnan(lbt) else "-"
                eod_s = f"{eod:.1f}분" if not np.isnan(eod) else "-"
                lines.append(f"| {d} | {mat_s} | {lbt_s} | {eod_s} | {n}명 |")

        # GAP/활성레벨
        act_lines = _activity_block(worker_df)
        if act_lines:
            lines.append("\n### 작업 활성도 현황")
            lines.extend(act_lines)

        # 업체 그룹별 MAT (k-Anonymity 적용)
        if "company_name" in worker_df.columns and "mat_minutes" in worker_df.columns:
            bp_agg = worker_df.groupby("company_name").agg(
                n_workers=("user_no" if "user_no" in worker_df.columns else "mat_minutes", "count"),
                mat_avg=("mat_minutes", lambda x: x.dropna().mean()),
                lbt_avg=("lbt_minutes", lambda x: x.dropna().mean()) if "lbt_minutes" in worker_df.columns else ("mat_minutes", lambda x: float("nan")),
                eod_avg=("eod_minutes", lambda x: x.dropna().mean()) if "eod_minutes" in worker_df.columns else ("mat_minutes", lambda x: float("nan")),
            ).reset_index()

            overall_mat = worker_df["mat_minutes"].dropna().mean()
            eligible = bp_agg[bp_agg["n_workers"] >= K_ANON_MIN]

            if not eligible.empty:
                lines.append(f"\n### 업체별 대기시간 (10명 이상 업체, 전체 MAT 평균 {overall_mat:.1f}분 기준)")
                top_n = eligible.nlargest(10, "mat_avg")
                for _, r in top_n.iterrows():
                    code = AP.get_company_code(str(r["company_name"]))
                    mat = r["mat_avg"]
                    diff = mat - overall_mat
                    sign = "+" if diff >= 0 else ""
                    lines.append(
                        f"  - {code}: MAT {mat:.1f}분 (평균 대비 {sign}{diff:.1f}분), "
                        f"작업자 {r['n_workers']:.0f}명"
                    )

        return "\n".join(lines)

    # ─── 2. 장비 가동률 (Equipment) ──────────────────────────────────

    @staticmethod
    def equipment(
        master_df: pd.DataFrame | None,
        weekly_bp_df: pd.DataFrame | None,
        weekly_overall_df: pd.DataFrame | None,
    ) -> str:
        """
        테이블 리프트 가동률 탭용 풍부한 데이터 텍스트.

        포함:
          - 장비 기본 현황 (대수, 층별 분포)
          - 5주 전체 가동률 추이
          - 업체별 가동률 상위/하위 (k-Anonymity 적용)
          - 비가동 패턴 분석
        """
        lines: list[str] = ["## 테이블 리프트 가동률 데이터"]

        # 장비 기본 현황
        if master_df is not None and not master_df.empty:
            n_equip = len(master_df)
            lines.append(f"- 등록 장비 수: {n_equip}대")
            if "level" in master_df.columns:
                dist = master_df["level"].value_counts()
                dist_str = ", ".join(f"{k}: {v}대" for k, v in dist.items())
                lines.append(f"- 층별 분포: {dist_str}")
            if "work_status" in master_df.columns:
                active = (master_df["work_status"] == 1).sum()
                lines.append(f"- 현재 가동 중: {active}대 ({_pct(active, n_equip)})")

        # 5주 추이
        if weekly_overall_df is not None and not weekly_overall_df.empty:
            lines.append("\n### 주간 전체 가동률 추이")
            opr_col = next((c for c in ["opr_mean", "opr", "avg_opr"] if c in weekly_overall_df.columns), None)
            week_col = next((c for c in ["week_label", "week", "period"] if c in weekly_overall_df.columns), None)
            if opr_col and week_col:
                oprs = []
                for _, r in weekly_overall_df.iterrows():
                    opr = r[opr_col]
                    if isinstance(opr, float) and opr <= 1:
                        opr_str = f"{opr:.1%}"
                        oprs.append(opr)
                    else:
                        opr_str = str(opr)
                    lines.append(f"  - {r[week_col]}: {opr_str}")

                # 추세 방향
                if len(oprs) >= 2:
                    trend = oprs[-1] - oprs[0]
                    direction = "상승" if trend > 0.02 else ("하락" if trend < -0.02 else "유지")
                    lines.append(f"- 전체 추세: {direction} ({oprs[0]:.1%} → {oprs[-1]:.1%})")

        # 업체별 가동률 (k-Anonymity 적용)
        if weekly_bp_df is not None and not weekly_bp_df.empty:
            bp_col = next((c for c in ["bp_name", "company_name", "bp"] if c in weekly_bp_df.columns), None)
            opr_col = next((c for c in ["opr_mean", "opr", "avg_opr"] if c in weekly_bp_df.columns), None)

            if bp_col and opr_col:
                # 장비 수 컬럼 찾기 (k-Anonymity 기준)
                cnt_col = next((c for c in ["n_equip", "equip_count", "count"] if c in weekly_bp_df.columns), None)

                bp_avg = weekly_bp_df.groupby(bp_col)[opr_col].mean().reset_index()
                bp_avg.columns = [bp_col, "avg_opr"]

                if cnt_col:
                    cnt_agg = weekly_bp_df.groupby(bp_col)[cnt_col].mean().reset_index()
                    bp_avg = bp_avg.merge(cnt_agg, on=bp_col, how="left")
                    eligible = bp_avg[bp_avg[cnt_col] >= 1]  # 장비는 대수 기준
                else:
                    eligible = bp_avg

                if not eligible.empty:
                    top3 = eligible.nlargest(3, "avg_opr")
                    bot3 = eligible.nsmallest(3, "avg_opr")

                    lines.append("\n### 가동률 상위 3개 업체")
                    for _, r in top3.iterrows():
                        code = AP.get_company_code(str(r[bp_col]))
                        lines.append(f"  - {code}: 평균 {r['avg_opr']:.1%}")

                    lines.append("\n### 가동률 하위 3개 업체")
                    for _, r in bot3.iterrows():
                        code = AP.get_company_code(str(r[bp_col]))
                        lines.append(f"  - {code}: 평균 {r['avg_opr']:.1%}")

        return "\n".join(lines)

    # ─── 3. 작업자 분석 (Worker/Daily) ───────────────────────────────

    @staticmethod
    def worker(
        worker_df: pd.DataFrame,
        company_df: pd.DataFrame | None,
        date_str: str,
    ) -> str:
        """
        작업자 분석 탭용 풍부한 데이터 텍스트.

        포함:
          - EWI/CRE 분포
          - 고위험 비율
          - 활성레벨 분포 (GAP 통계)
          - 업체 그룹별 위험도 (k-Anonymity)
        """
        if worker_df is None or worker_df.empty:
            return "작업자 데이터가 없습니다."

        lines: list[str] = [f"## 작업자 분석 데이터 ({date_str})"]

        n_workers = len(worker_df)
        lines.append(f"- 총 작업자: {n_workers}명")

        # EWI/CRE 분포
        lines.append("\n### 작업 지표 분포")
        if "ewi" in worker_df.columns:
            lines.append(_dist(worker_df["ewi"], "작업집중도(EWI)", ""))
            # 등급 분포
            ewi = worker_df["ewi"].dropna()
            if not ewi.empty:
                high = (ewi >= 0.7).sum()
                mid = ((ewi >= 0.4) & (ewi < 0.7)).sum()
                low = (ewi < 0.4).sum()
                lines.append(
                    f"  → 상위(≥0.7): {high}명({_pct(high, n_workers)}), "
                    f"중간(0.4~0.7): {mid}명({_pct(mid, n_workers)}), "
                    f"하위(<0.4): {low}명({_pct(low, n_workers)})"
                )

        if "cre" in worker_df.columns:
            lines.append(_dist(worker_df["cre"], "위험노출도(CRE)", ""))
            cre = worker_df["cre"].dropna()
            if not cre.empty:
                high_cre = (cre >= 0.5).sum()
                lines.append(
                    f"  → 고위험(CRE≥0.5): {high_cre}명({_pct(high_cre, n_workers)}) — "
                    f"안전 관리 주의 필요"
                )

        # 근무 시간
        if "work_minutes" in worker_df.columns:
            wm = worker_df["work_minutes"].dropna()
            if not wm.empty:
                lines.append(
                    f"- 평균 근무시간: {wm.mean():.0f}분({wm.mean()/60:.1f}h), "
                    f"최대 {wm.max():.0f}분"
                )

        # 활성레벨 (GAP 통계)
        act_lines = _activity_block(worker_df)
        if act_lines:
            lines.append("\n### 작업 활성도")
            lines.extend(act_lines)

        # 업체별 위험도 (k-Anonymity 적용)
        if company_df is not None and not company_df.empty:
            company_col = next((c for c in ["company_name", "bp_name"] if c in company_df.columns), None)
            count_col = next((c for c in ["worker_count", "n_workers", "count"] if c in company_df.columns), None)

            if company_col and count_col:
                eligible = company_df[company_df[count_col] >= K_ANON_MIN]
                if not eligible.empty:
                    lines.append(f"\n### 업체별 현황 ({K_ANON_MIN}명 이상 업체)")
                    metric_cols = [c for c in ["avg_cre", "avg_ewi"] if c in eligible.columns]
                    rows = _company_rows(
                        eligible,
                        company_col=company_col,
                        count_col=count_col,
                        metric_cols=metric_cols,
                        top_n=8,
                        sort_col=count_col,
                    )
                    lines.extend(rows)

        return "\n".join(lines)

    # ─── 4. 혼잡도 (Congestion) ──────────────────────────────────────

    @staticmethod
    def congestion(
        summary: dict[str, Any],
        ranking_df: pd.DataFrame | None = None,
        hourly_df: pd.DataFrame | None = None,
    ) -> str:
        """
        공간 혼잡도 탭용 데이터 텍스트.

        포함:
          - 피크 공간/시간/인원
          - 공간 유형별 집계
          - 시간대별 패턴 (있을 경우)
          - 혼잡도 상위 공간
        """
        if not summary:
            return "혼잡도 데이터가 없습니다."

        lines: list[str] = ["## 공간 혼잡도 데이터"]

        lines.append(f"- 분석 공간 수: {summary.get('total_spaces', 0)}개")
        lines.append(f"- 전체 평균 점유: {summary.get('avg_occupancy', 0):.1f}명")
        lines.append(f"- 최대 동시 인원: {summary.get('peak_count', 0)}명")
        lines.append(f"- 최대 혼잡 공간: {summary.get('peak_space', '-')}")
        lines.append(f"- 최대 혼잡 시간: {summary.get('peak_time', '-')}")
        lines.append(f"- 가장 붐비는 시간대: {summary.get('busiest_hour', 0)}시")
        lines.append(f"- 가장 한산한 시간대: {summary.get('quietest_hour', 0)}시")

        # 혼잡도 상위 공간 (공간명은 일반 유형으로)
        if ranking_df is not None and not ranking_df.empty:
            lines.append("\n### 혼잡도 상위 공간 (공간 유형 기준)")
            name_col = next((c for c in ["locus_name", "locus_token", "space_name"] if c in ranking_df.columns), None)
            for i, (_, r) in enumerate(ranking_df.head(5).iterrows(), 1):
                space_label = f"공간 {i}"  # 구체적 명칭 대신 순위로
                if name_col:
                    raw_name = str(r.get(name_col, ""))
                    # 토큰 기반 유형 매핑
                    if "work_zone" in raw_name or "work" in raw_name.lower():
                        space_label = f"작업구역 {i}"
                    elif "rest" in raw_name.lower() or "break" in raw_name.lower():
                        space_label = f"휴게구역 {i}"
                    elif "transit" in raw_name.lower() or "hoist" in raw_name.lower():
                        space_label = f"이동구역 {i}"
                    elif "gate" in raw_name.lower() or "timeclock" in raw_name.lower():
                        space_label = f"게이트구역 {i}"
                    else:
                        space_label = f"구역 {i}"

                max_w = r.get("max_workers", r.get("peak_count", 0))
                avg_w = r.get("avg_workers", r.get("avg_occupancy", 0))
                peak_h = r.get("peak_hour", "-")
                lines.append(
                    f"  - {space_label}: 최대 {max_w:.0f}명, "
                    f"평균 {avg_w:.1f}명, 피크 {peak_h}시"
                )

        # 시간대별 패턴
        if hourly_df is not None and not hourly_df.empty:
            hour_col = next((c for c in ["hour", "time_hour"] if c in hourly_df.columns), None)
            count_col = next((c for c in ["avg_workers", "count", "occupancy"] if c in hourly_df.columns), None)
            if hour_col and count_col:
                peak_rows = hourly_df.nlargest(3, count_col)
                lines.append("\n### 피크 시간대 Top 3")
                for _, r in peak_rows.iterrows():
                    lines.append(f"  - {r[hour_col]}시: 평균 {r[count_col]:.1f}명")

        return "\n".join(lines)

    # ─── 5. 현장 개요 (Overview) ─────────────────────────────────────

    @staticmethod
    def overview(
        kpi: dict[str, Any],
        prev_kpi: dict[str, Any] | None = None,
        date_str: str = "",
    ) -> str:
        """
        현장 개요 탭용 데이터 텍스트.

        포함:
          - 당일 KPI (출입/T-Ward/EWI/CRE)
          - 전일 대비 변화
          - 고위험/밀폐공간 현황
        """
        if not kpi:
            return "현장 개요 데이터가 없습니다."

        lines: list[str] = [f"## 현장 개요 ({date_str})"]

        total_access = kpi.get("total_access", 0)
        total_tward = kpi.get("total_tward", 0)
        tward_rate = kpi.get("tward_rate", 0)
        avg_ewi = kpi.get("avg_ewi", None)
        avg_cre = kpi.get("avg_cre", None)
        high_cre = kpi.get("high_cre_count", 0)

        lines.append(f"- 총 출입 인원: {total_access}명")
        lines.append(f"- T-Ward 작업자: {total_tward}명 (전체 대비 {tward_rate:.1%})")

        if avg_ewi is not None:
            ewi_level = "높음" if avg_ewi >= 0.7 else ("보통" if avg_ewi >= 0.4 else "낮음")
            lines.append(f"- 평균 작업집중도(EWI): {avg_ewi:.3f} ({ewi_level} 수준)")

        if avg_cre is not None:
            cre_level = "높음" if avg_cre >= 0.5 else ("보통" if avg_cre >= 0.3 else "낮음")
            lines.append(f"- 평균 위험노출도(CRE): {avg_cre:.3f} ({cre_level} 수준)")
            lines.append(f"- 고위험 작업자(CRE≥0.5): {high_cre}명 ({_pct(high_cre, total_tward)})")

        confined = kpi.get("confined_workers", 0)
        hv = kpi.get("hv_workers", 0)
        if confined:
            lines.append(f"- 밀폐공간 진입자: {confined}명 (2인 1조 원칙 적용 구역)")
        if hv:
            lines.append(f"- 고압전 구역 진입자: {hv}명")

        # 전일 대비 변화
        if prev_kpi:
            lines.append("\n### 전일 대비 변화")
            for key, label, unit in [
                ("total_access", "출입 인원", "명"),
                ("avg_ewi", "EWI", ""),
                ("avg_cre", "CRE", ""),
                ("high_cre_count", "고위험자", "명"),
            ]:
                cur = kpi.get(key)
                prv = prev_kpi.get(key)
                if cur is None or prv is None or prv == 0:
                    continue
                pct = (cur - prv) / abs(prv) * 100
                sign = "▲" if pct > 0 else ("▼" if pct < 0 else "→")
                lines.append(f"  - {label}: {sign} {abs(pct):.1f}% ({prv:.2f} → {cur:.2f})")

        return "\n".join(lines)
