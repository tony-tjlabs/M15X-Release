"""
Report Tab -- 리포트 생성 메뉴
================================
사이드바 메뉴에 추가되어 종합 리포트를 Streamlit 내에서 렌더링하고
다운로드 가능한 형태로 제공한다.

리포트 유형:
  - 일별 리포트: 특정 날짜 종합 분석
  - 기간별 리포트: 시작/종료 날짜 구간 종합 분석

리포트 내용:
  - 작업자 대기/이동 시간 (MAT/LBT/EOD)
  - 테이블 리프트 가동률 (OPR, BP별)
  - 작업자 분석 (EWI/CRE, 업체별)
  - 공간 혼잡도 (Locus별)
  - AI-powered commentary for each section (PDF)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src.dashboard.styles import (
    COLORS,
    PLOTLY_DARK,
    PLOTLY_LEGEND,
    metric_card,
    metric_card_sm,
    section_header,
    sub_header,
    badge,
)
from src.dashboard.date_utils import format_date_label
from src.pipeline.cache_manager import detect_processed_dates

logger = logging.getLogger(__name__)


# =========================================================================
# PDF Report Commentary Generator (LLM-powered)
# =========================================================================

_REPORT_SYSTEM_PROMPT = """
당신은 반도체 FAB 건설현장 데이터 해석 보조입니다.
M15X 건설현장의 분석 리포트에 삽입할 간결한 코멘터리를 작성합니다.

[역할]
- 차트/데이터를 비전문가도 이해할 수 있도록 한국어로 설명
- 핵심 관측점과 실행 가능한 인사이트 제공
- 수치를 직접 인용하되 원인 단정 금지
- 특정 작업자/업체 직접 언급 금지

[형식]
- 3~5문장의 간결한 단락
- "~로 보임", "~가 관측됨" 형식 사용
- 알고리즘/계산 방법 언급 금지
한국어, 간결체 사용.
""".strip()


def _generate_report_commentaries(
    worker_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    equip: dict,
    space_df: pd.DataFrame,
    selected_dates: list[str],
    n_days: int,
) -> dict[str, str]:
    """Generate AI commentaries for each PDF report section.

    Returns a dict with keys: exec_summary, transit, equipment, worker, congestion, conclusion.
    If LLM is unavailable, returns empty dict (PDF generator falls back to statistical summaries).
    """
    try:
        from src.intelligence.llm_gateway import LLMGateway
    except ImportError:
        return {}

    if not LLMGateway.is_available():
        return {}

    # Check session cache
    cache_key = f"_pdf_commentaries_{'_'.join(selected_dates[:3])}"
    if cache_key in st.session_state and st.session_state[cache_key]:
        return st.session_state[cache_key]

    commentaries = {}

    # Prepare data summaries for LLM
    date_range = f"{selected_dates[0]}~{selected_dates[-1]}" if n_days > 1 else selected_dates[0]

    # -- Executive Summary --
    exec_data = _pack_exec_summary_data(worker_df, transit_df, equip, space_df, n_days, date_range)
    if exec_data:
        commentaries["exec_summary"] = _call_report_llm(
            f"다음은 M15X 건설현장 {date_range} 기간의 종합 KPI 데이터입니다.\n"
            f"경영진/현장소장 대상 Executive Summary를 5~7문장으로 작성하세요.\n"
            f"핵심 수치를 인용하고, 주의가 필요한 영역을 식별하세요.\n\n{exec_data}",
            max_tokens=600,
        )

    # -- Transit Commentary --
    transit_data = _pack_transit_data(transit_df, n_days)
    if transit_data:
        commentaries["transit"] = _call_report_llm(
            f"다음은 M15X 건설현장의 작업자 대기/이동 시간(MAT/LBT/EOD) 데이터입니다.\n"
            f"이 차트에 대한 코멘터리를 작성하세요. 병목 구간이나 이상치를 식별하세요.\n\n{transit_data}",
            max_tokens=400,
        )

    # -- Equipment Commentary --
    equip_data = _pack_equipment_data(equip)
    if equip_data:
        commentaries["equipment"] = _call_report_llm(
            f"다음은 M15X 건설현장 테이블 리프트 장비 가동률(OPR) 데이터입니다.\n"
            f"가동률 추세와 개선 여지를 식별하는 코멘터리를 작성하세요.\n\n{equip_data}",
            max_tokens=400,
        )

    # -- Worker Commentary --
    worker_data = _pack_worker_data(worker_df)
    if worker_data:
        commentaries["worker"] = _call_report_llm(
            f"다음은 M15X 건설현장 작업자 EWI(작업강도)/CRE(위험노출) 데이터입니다.\n"
            f"작업 강도 분포와 안전 위험을 분석하는 코멘터리를 작성하세요.\n\n{worker_data}",
            max_tokens=400,
        )

    # -- Congestion Commentary --
    congestion_data = _pack_congestion_data(space_df)
    if congestion_data:
        commentaries["congestion"] = _call_report_llm(
            f"다음은 M15X 건설현장 공간별 혼잡도 데이터입니다.\n"
            f"혼잡 패턴과 주의가 필요한 구역을 식별하는 코멘터리를 작성하세요.\n\n{congestion_data}",
            max_tokens=400,
        )

    # -- Conclusion --
    if commentaries:
        commentaries["conclusion"] = _call_report_llm(
            f"다음은 M15X 건설현장 {date_range} 분석 리포트의 각 섹션 AI 코멘터리입니다.\n"
            f"이를 종합하여 리포트 결론을 3~4문장으로 작성하세요.\n"
            f"데이터 한계(BLE 음영 50%, MAC 랜덤화)를 간략히 언급하세요.\n\n"
            + "\n---\n".join(f"[{k}] {v}" for k, v in commentaries.items() if v),
            max_tokens=400,
        )

    # Cache results
    st.session_state[cache_key] = commentaries
    return commentaries


def _call_report_llm(prompt: str, max_tokens: int = 400) -> str:
    """Call LLM with report-specific system prompt. Returns empty string on failure."""
    try:
        from src.intelligence.llm_gateway import _call_claude_with_system
        return _call_claude_with_system(
            prompt=prompt,
            system=_REPORT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning("Report LLM call failed: %s", e)
        return ""


def _pack_exec_summary_data(
    worker_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    equip: dict,
    space_df: pd.DataFrame,
    n_days: int,
    date_range: str,
) -> str:
    """Pack all KPI data into text for executive summary LLM call."""
    parts = [f"기간: {date_range} ({n_days}일)"]

    if not worker_df.empty and "user_no" in worker_df.columns:
        n_unique = worker_df["user_no"].nunique()
        parts.append(f"고유 작업자: {n_unique}명")
        if "date" in worker_df.columns:
            daily_avg = worker_df.groupby("date")["user_no"].nunique().mean()
            parts.append(f"일 평균 작업자: {daily_avg:.0f}명")
    if not worker_df.empty and "ewi" in worker_df.columns:
        parts.append(f"평균 EWI: {worker_df['ewi'].mean():.3f}")
    if not worker_df.empty and "cre" in worker_df.columns:
        avg_cre = worker_df["cre"].mean()
        high_cre = (worker_df["cre"] >= 0.5).sum()
        pct = high_cre / len(worker_df) * 100
        parts.append(f"평균 CRE: {avg_cre:.3f}, 고위험(>=0.5): {high_cre}명({pct:.1f}%)")
    if not transit_df.empty:
        for col, label in [("mat_minutes", "MAT"), ("lbt_minutes", "LBT"), ("eod_minutes", "EOD")]:
            if col in transit_df.columns:
                vals = transit_df[col].dropna()
                if not vals.empty:
                    parts.append(f"평균 {label}: {vals.mean():.1f}분 (최대 {vals.max():.1f}분)")
    weekly_overall = equip.get("weekly_overall_opr", pd.DataFrame())
    if not weekly_overall.empty:
        opr_col = "opr_mean" if "opr_mean" in weekly_overall.columns else "avg_opr"
        if opr_col in weekly_overall.columns:
            latest = weekly_overall[opr_col].iloc[-1]
            avg = weekly_overall[opr_col].mean()
            parts.append(f"최근 주 장비 가동률: {latest:.1%} (평균: {avg:.1%})")
    if not parts:
        return ""
    return "\n".join(parts)


def _pack_transit_data(transit_df: pd.DataFrame, n_days: int) -> str:
    """Pack transit data for LLM."""
    if transit_df.empty:
        return ""
    parts = [f"분석 일수: {n_days}"]
    for col, label, threshold in [
        ("mat_minutes", "MAT", 15),
        ("lbt_minutes", "LBT", 20),
        ("eod_minutes", "EOD", 18),
    ]:
        if col in transit_df.columns:
            vals = transit_df[col].dropna()
            if not vals.empty:
                over = (vals > threshold).sum()
                pct = over / len(vals) * 100
                parts.append(
                    f"{label}: 평균 {vals.mean():.1f}분, 중앙값 {vals.median():.1f}분, "
                    f"최대 {vals.max():.1f}분, 표준편차 {vals.std():.1f}분, "
                    f"{threshold}분 초과: {over}명({pct:.0f}%)"
                )
    if "total_wait_minutes" in transit_df.columns:
        twt = transit_df["total_wait_minutes"].dropna()
        if not twt.empty:
            parts.append(f"총 대기시간(TWT): 평균 {twt.mean():.1f}분, 최대 {twt.max():.1f}분")
    return "\n".join(parts) if len(parts) > 1 else ""


def _pack_equipment_data(equip: dict) -> str:
    """Pack equipment data for LLM."""
    weekly_overall = equip.get("weekly_overall_opr", pd.DataFrame())
    if weekly_overall.empty:
        return ""
    opr_col = "opr_mean" if "opr_mean" in weekly_overall.columns else "avg_opr"
    week_col = "week_label" if "week_label" in weekly_overall.columns else "week"
    if opr_col not in weekly_overall.columns:
        return ""
    parts = ["주간 전체 가동률:"]
    for _, row in weekly_overall.iterrows():
        w = row.get(week_col, "?")
        v = row.get(opr_col, 0)
        parts.append(f"  {w}: {v:.1%}")
    master = equip.get("master", pd.DataFrame())
    if not master.empty:
        parts.append(f"등록 장비: {len(master)}대")
    return "\n".join(parts)


def _pack_worker_data(worker_df: pd.DataFrame) -> str:
    """Pack worker data for LLM."""
    if worker_df.empty:
        return ""
    n = len(worker_df)
    parts = [f"작업자 수: {n}명"]
    if "ewi" in worker_df.columns:
        ewi = worker_df["ewi"].dropna()
        high = (ewi >= 0.7).sum()
        low = (ewi < 0.4).sum()
        parts.append(f"EWI: 평균 {ewi.mean():.3f}, 고집중(>=0.7) {high}명, 저집중(<0.4) {low}명")
    if "cre" in worker_df.columns:
        cre = worker_df["cre"].dropna()
        high = (cre >= 0.5).sum()
        pct = high / n * 100
        parts.append(f"CRE: 평균 {cre.mean():.3f}, 고위험(>=0.5) {high}명({pct:.1f}%)")
    if "ga_dominant_activity" in worker_df.columns:
        dist = worker_df["ga_dominant_activity"].value_counts(normalize=True) * 100
        activity_parts = [f"{k}: {v:.1f}%" for k, v in dist.head(5).items() if pd.notna(k)]
        if activity_parts:
            parts.append(f"활성 레벨: {', '.join(activity_parts)}")
    return "\n".join(parts) if len(parts) > 1 else ""


def _pack_congestion_data(space_df: pd.DataFrame) -> str:
    """Pack congestion data for LLM."""
    if space_df.empty:
        return ""
    count_cols = [c for c in [
        "avg_workers", "max_workers", "avg_occupancy",
        "unique_workers", "total_person_minutes", "total_visits",
    ] if c in space_df.columns]
    if not count_cols:
        return ""
    main_col = count_cols[0]

    _token_map = {
        "work_zone": "Work Zone (FAB)",
        "breakroom": "Break Room",
        "smoking_area": "Smoking Area",
        "restroom": "Restroom",
        "transit": "Hoist / Transit",
        "timeclock": "Time Clock (Gate)",
        "outdoor_work": "Outdoor Work",
        "unknown": "Unmapped",
    }

    if "locus_token" in space_df.columns:
        agg = space_df.groupby("locus_token")[main_col].sum().sort_values(ascending=False)
        parts = [f"공간별 {main_col}:"]
        for k, v in agg.items():
            label = _token_map.get(str(k), str(k))
            parts.append(f"  {label}: {v:.0f}")
    else:
        grp = next((c for c in ["locus_name", "locus_id"] if c in space_df.columns), None)
        if not grp:
            return ""
        agg = space_df.groupby(grp)[main_col].sum().sort_values(ascending=False)
        parts = [f"공간별 {main_col}:"]
        for k, v in agg.items():
            parts.append(f"  {k}: {v:.0f}")
    return "\n".join(parts)


# =========================================================================
# 데이터 로더 (캐시 적용)
# =========================================================================

@st.cache_data(show_spinner=False, ttl=300)
def _load_report_data(
    sector_id: str, dates: tuple[str, ...]
) -> dict:
    """리포트용 데이터 로드 — 선택 기간의 모든 캐시 병합."""
    paths = cfg.get_sector_paths(sector_id)
    all_workers = []
    all_transit = []
    all_company = []
    all_space = []
    metas = []

    for d in dates:
        date_dir = paths["processed_dir"] / d

        for name, target in [
            ("worker", all_workers),
            ("transit", all_transit),
            ("company", all_company),
            ("space", all_space),
        ]:
            p = date_dir / f"{name}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    df["date"] = d
                    target.append(df)
                except Exception:
                    pass

        meta_path = date_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    m = json.load(f)
                    m["date"] = d
                    metas.append(m)
            except Exception:
                pass

    return {
        "worker": pd.concat(all_workers, ignore_index=True) if all_workers else pd.DataFrame(),
        "transit": pd.concat(all_transit, ignore_index=True) if all_transit else pd.DataFrame(),
        "company": pd.concat(all_company, ignore_index=True) if all_company else pd.DataFrame(),
        "space": pd.concat(all_space, ignore_index=True) if all_space else pd.DataFrame(),
        "metas": metas,
    }


# =========================================================================
# 장비 데이터 로더
# =========================================================================

@st.cache_data(show_spinner=False, ttl=600)
def _load_equipment_data(sector_id: str) -> dict:
    """주간 장비 가동률 데이터 로드."""
    paths = cfg.get_sector_paths(sector_id)
    result = {}
    for name in ["weekly_bp_opr", "weekly_overall_opr"]:
        p = paths["equipment_weekly"] / f"{name}.parquet"
        result[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    master_path = paths["equipment_master"]
    result["master"] = pd.read_parquet(master_path) if master_path.exists() else pd.DataFrame()
    return result


# =========================================================================
# 메인 렌더러
# =========================================================================

def render_report_tab(sector_id: str | None = None):
    """리포트 생성 탭 진입점."""
    sid = sector_id or cfg.SECTOR_ID
    processed = detect_processed_dates(sid)

    if not processed:
        st.info("처리된 데이터가 없습니다. 데이터 관리 탭에서 전처리를 먼저 실행하세요.")
        return

    st.markdown(section_header("리포트 생성"), unsafe_allow_html=True)
    st.caption("데이터 기반 종합 분석 리포트를 생성합니다.")

    # ── 리포트 유형 선택 ──
    report_mode = st.radio(
        "리포트 유형",
        ["일별 리포트", "기간별 리포트"],
        horizontal=True,
        key="report_mode",
    )

    # ── 날짜 선택 ──
    date_list_dt = [datetime.strptime(d, "%Y%m%d") for d in processed]

    if report_mode == "일별 리포트":
        date_options = {format_date_label(d): d for d in reversed(processed)}
        selected_label = st.selectbox("분석 날짜", list(date_options.keys()), key="report_date")
        selected_dates = [date_options[selected_label]]
        period_label = selected_label
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            start_dt = st.date_input(
                "시작일",
                value=date_list_dt[0].date(),
                min_value=date_list_dt[0].date(),
                max_value=date_list_dt[-1].date(),
                key="report_start",
            )
        with col_b:
            end_dt = st.date_input(
                "종료일",
                value=date_list_dt[-1].date(),
                min_value=date_list_dt[0].date(),
                max_value=date_list_dt[-1].date(),
                key="report_end",
            )

        if start_dt > end_dt:
            st.error("시작일이 종료일보다 늦습니다.")
            return

        selected_dates = [
            d for d in processed
            if start_dt <= datetime.strptime(d, "%Y%m%d").date() <= end_dt
        ]
        if not selected_dates:
            st.warning("선택 기간에 데이터가 없습니다.")
            return
        period_label = f"{format_date_label(selected_dates[0])} ~ {format_date_label(selected_dates[-1])}"

    # ── 리포트 생성 버튼 ──
    # session_state 기반: 버튼 클릭 시 데이터를 저장하고,
    # 이후 rerun에서도 리포트가 유지되도록 함 (AI 요약 버튼 등)
    if st.button("리포트 생성", type="primary", key="gen_report"):
        with st.spinner("리포트 데이터 로드 중..."):
            data = _load_report_data(sid, tuple(selected_dates))
            equip = _load_equipment_data(sid)

        st.session_state["_report_data"] = data
        st.session_state["_report_equip"] = equip
        st.session_state["_report_dates"] = selected_dates
        st.session_state["_report_period"] = period_label
        st.session_state["_report_mode"] = report_mode
        st.session_state["_report_sid"] = sid

    # 리포트 데이터가 session_state에 있으면 렌더링
    if "_report_data" in st.session_state:
        _render_report(
            data=st.session_state["_report_data"],
            equip=st.session_state["_report_equip"],
            selected_dates=st.session_state["_report_dates"],
            period_label=st.session_state["_report_period"],
            report_mode=st.session_state["_report_mode"],
            sid=st.session_state["_report_sid"],
        )


# =========================================================================
# 리포트 렌더링
# =========================================================================

def _render_report(
    data: dict,
    equip: dict,
    selected_dates: list[str],
    period_label: str,
    report_mode: str,
    sid: str,
):
    """리포트 내용 렌더링 + 다운로드 제공."""
    worker_df = data["worker"]
    transit_df = data["transit"]
    company_df = data["company"]
    space_df = data["space"]
    n_days = len(selected_dates)

    st.divider()

    # ── 리포트 헤더 ──
    st.markdown(
        f"""
        <div style='background:{cfg.THEME_CARD_BG}; border-radius:12px; padding:24px; margin-bottom:20px;'>
            <div style='font-size:1.5rem; font-weight:700; color:{cfg.THEME_TEXT};'>
                Deep Con at M15X -- {report_mode}
            </div>
            <div style='font-size:0.92rem; color:#7A8FA6; margin-top:8px;'>
                기간: {period_label} ({n_days}일)
            </div>
            <div style='font-size:0.78rem; color:#5A6A7A; margin-top:4px;'>
                생성: {datetime.now().strftime("%Y-%m-%d %H:%M")} | TJLABS Research
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 1: KPI 요약 ──
    _render_kpi_section(worker_df, transit_df, n_days)

    # ── Section 2: 대기시간 분석 ──
    _render_transit_section(transit_df, n_days)

    # ── Section 3: 테이블 리프트 가동률 ──
    _render_equipment_section(equip)

    # ── Section 4: 작업자 분석 ──
    _render_worker_section(worker_df, company_df)

    # ── Section 5: 공간 혼잡도 ──
    _render_congestion_section(space_df)

    # ── AI 요약 (옵션) ──
    _render_ai_summary(worker_df, transit_df, equip, n_days, selected_dates)

    # ── 다운로드 ──
    st.divider()
    _render_download_section(
        worker_df, transit_df, company_df, space_df,
        period_label, selected_dates,
    )


# =========================================================================
# Section Renderers
# =========================================================================

def _render_kpi_section(worker_df: pd.DataFrame, transit_df: pd.DataFrame, n_days: int):
    """KPI 요약 카드."""
    st.markdown(section_header("핵심 지표 요약"), unsafe_allow_html=True)

    if worker_df.empty:
        st.info("작업자 데이터 없음")
        return

    daily_workers = worker_df.groupby("date")["user_no"].nunique() if "user_no" in worker_df.columns else pd.Series()
    avg_workers = daily_workers.mean() if not daily_workers.empty else 0
    total_unique = worker_df["user_no"].nunique() if "user_no" in worker_df.columns else 0
    avg_ewi = worker_df["ewi"].mean() if "ewi" in worker_df.columns else 0
    avg_cre = worker_df["cre"].mean() if "cre" in worker_df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("일 평균 작업자", f"{avg_workers:.0f}명"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("고유 작업자", f"{total_unique:,}명"), unsafe_allow_html=True)
    with col3:
        ewi_color = COLORS["success"] if avg_ewi >= 0.6 else COLORS["warning"] if avg_ewi >= 0.2 else COLORS["danger"]
        st.markdown(metric_card("평균 EWI", f"{avg_ewi:.3f}", color=ewi_color), unsafe_allow_html=True)
    with col4:
        cre_color = COLORS["danger"] if avg_cre >= 0.7 else COLORS["warning"] if avg_cre >= 0.4 else COLORS["success"]
        st.markdown(metric_card("평균 CRE", f"{avg_cre:.3f}", color=cre_color), unsafe_allow_html=True)

    # 대기시간 KPI
    if not transit_df.empty:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        for col, metric_col, label in [
            (col1, "mat_minutes", "평균 MAT (출근 대기)"),
            (col2, "lbt_minutes", "평균 LBT (중식 이동)"),
            (col3, "eod_minutes", "평균 EOD (퇴근 이동)"),
        ]:
            if metric_col in transit_df.columns:
                val = transit_df[metric_col].dropna().mean()
                with col:
                    st.markdown(metric_card_sm(label, f"{val:.1f}분"), unsafe_allow_html=True)


def _render_transit_section(transit_df: pd.DataFrame, n_days: int = 1):
    """대기시간 분석 섹션."""
    st.markdown(section_header("대기/이동 시간 분석"), unsafe_allow_html=True)

    if transit_df.empty or "date" not in transit_df.columns:
        st.info("대기시간 데이터 없음")
        return

    metrics = [
        ("mat_minutes", "#00AEEF", "MAT (출근)"),
        ("lbt_minutes", "#FFB300", "LBT (중식)"),
        ("eod_minutes", "#FF8C42", "EOD (퇴근)"),
    ]

    if n_days == 1:
        # ── 단일 날짜: KPI 바 차트 + 개인별 분포 ──────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            # MAT / LBT / EOD 평균을 가로 바 차트로 표시
            bar_labels, bar_vals, bar_colors = [], [], []
            for col, color, name in metrics:
                if col in transit_df.columns:
                    val = transit_df[col].dropna().mean()
                    if not pd.isna(val):
                        bar_labels.append(name)
                        bar_vals.append(round(val, 1))
                        bar_colors.append(color)

            fig_bar = go.Figure(go.Bar(
                x=bar_vals,
                y=bar_labels,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}분" for v in bar_vals],
                textposition="outside",
                textfont=dict(size=13),
            ))
            fig_bar.update_layout(
                title="평균 대기/이동 시간",
                height=220,
                template="plotly_dark",
                paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
                plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                font_color=PLOTLY_DARK["font_color"],
                margin=dict(l=20, r=60, t=40, b=20),
                xaxis_title="소요시간 (분)",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            # 작업자별 TWT 분포 (히스토그램)
            if "total_wait_minutes" in transit_df.columns:
                twt = transit_df["total_wait_minutes"].dropna()
                fig_hist = go.Figure(go.Histogram(
                    x=twt,
                    nbinsx=20,
                    marker_color="#7C5CBF",
                    opacity=0.85,
                ))
                fig_hist.update_layout(
                    title=f"작업자별 총 대기시간 분포 ({len(twt)}명)",
                    height=220,
                    template="plotly_dark",
                    paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
                    plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                    font_color=PLOTLY_DARK["font_color"],
                    margin=dict(l=20, r=20, t=40, b=40),
                    xaxis_title="총 대기시간 (분)",
                    yaxis_title="작업자 수",
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    else:
        # ── 복수 날짜: 일별 추이 라인 차트 ──────────────────────────────
        daily_transit = transit_df.groupby("date").agg({
            col: lambda x: x.dropna().mean()
            for col, _, _ in metrics
            if col in transit_df.columns
        }).reset_index()
        daily_transit = daily_transit.sort_values("date")
        daily_transit["date_label"] = daily_transit["date"].apply(format_date_label)

        fig = go.Figure()
        for col, color, name in metrics:
            if col in daily_transit.columns:
                fig.add_trace(go.Scatter(
                    x=daily_transit["date_label"],
                    y=daily_transit[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                ))
        fig.update_layout(
            title="일별 평균 대기시간 추이",
            height=300,
            template="plotly_dark",
            paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
            plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
            font_color=PLOTLY_DARK["font_color"],
            margin=dict(l=40, r=20, t=40, b=50),
            xaxis_title="날짜", yaxis_title="대기시간 (분)",
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 통계 테이블 (공통)
    with st.expander("대기시간 상세 통계"):
        stats_data = []
        for col, _, label in metrics:
            if col in transit_df.columns:
                vals = transit_df[col].dropna()
                if not vals.empty:
                    stats_data.append({
                        "지표": label,
                        "평균 (분)": round(vals.mean(), 1),
                        "중앙값 (분)": round(vals.median(), 1),
                        "최소 (분)": round(vals.min(), 1),
                        "최대 (분)": round(vals.max(), 1),
                        "표준편차": round(vals.std(), 1),
                        "측정 인원": len(vals),
                    })
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)


def _render_equipment_section(equip: dict):
    """테이블 리프트 가동률 섹션."""
    st.markdown(section_header("테이블 리프트 가동률"), unsafe_allow_html=True)

    weekly_overall = equip.get("weekly_overall_opr", pd.DataFrame())
    weekly_bp = equip.get("weekly_bp_opr", pd.DataFrame())
    master = equip.get("master", pd.DataFrame())

    if weekly_overall.empty and weekly_bp.empty:
        st.info("장비 가동률 데이터 없음")
        return

    # 전체 가동률 추이
    if not weekly_overall.empty:
        opr_col = "opr_mean" if "opr_mean" in weekly_overall.columns else "opr"
        week_col = "week_label" if "week_label" in weekly_overall.columns else "week"

        if opr_col in weekly_overall.columns and week_col in weekly_overall.columns:
            fig = go.Figure(go.Bar(
                x=weekly_overall[week_col],
                y=weekly_overall[opr_col],
                marker_color=[
                    COLORS["success"] if v >= 0.7 else COLORS["warning"] if v >= 0.5 else COLORS["danger"]
                    for v in weekly_overall[opr_col]
                ],
                text=weekly_overall[opr_col].apply(lambda v: f"{v:.1%}"),
                textposition="outside",
                textfont=dict(color="#D5E5FF"),
            ))
            fig.update_layout(
                title="주간 전체 가동률",
                yaxis_title="가동률",
                yaxis_tickformat=".0%",
                **PLOTLY_DARK, height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

    # BP별 가동률 히트맵
    if not weekly_bp.empty:
        _render_bp_opr_heatmap(weekly_bp)

    # 장비 수 표시
    if not master.empty:
        st.markdown(
            f"<div style='color:#7A8FA6; font-size:0.82rem; margin-top:8px;'>"
            f"등록 장비 수: {len(master)}대</div>",
            unsafe_allow_html=True,
        )


def _render_bp_opr_heatmap(weekly_bp: pd.DataFrame):
    """BP별 주간 가동률 히트맵."""
    bp_col = "bp_name" if "bp_name" in weekly_bp.columns else None
    week_col = "week_label" if "week_label" in weekly_bp.columns else "week"
    opr_col = "opr_mean" if "opr_mean" in weekly_bp.columns else "opr"

    if not bp_col or opr_col not in weekly_bp.columns:
        return

    pivot = weekly_bp.pivot_table(
        index=bp_col, columns=week_col, values=opr_col,
        aggfunc="mean", fill_value=0,
    )

    with st.expander("BP별 주간 가동률 히트맵", expanded=False):
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0, COLORS["danger"]], [0.5, COLORS["warning"]], [1.0, COLORS["success"]],
            ],
            zmin=0, zmax=1,
            hovertemplate="BP: %{y}<br>주차: %{x}<br>가동률: %{z:.1%}<extra></extra>",
            colorbar=dict(
                title=dict(text="가동률", font=dict(color="#D5E5FF")),
                tickformat=".0%",
                tickfont=dict(color="#D5E5FF"),
            ),
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            height=max(300, len(pivot) * 22 + 100),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_worker_section(worker_df: pd.DataFrame, company_df: pd.DataFrame):
    """작업자 분석 섹션."""
    st.markdown(section_header("작업자 분석"), unsafe_allow_html=True)

    if worker_df.empty:
        st.info("작업자 데이터 없음")
        return

    col1, col2 = st.columns(2)

    # EWI 분포
    with col1:
        if "ewi" in worker_df.columns:
            st.markdown(sub_header("EWI 분포"), unsafe_allow_html=True)
            fig = px.histogram(
                worker_df, x="ewi", nbins=20,
                color_discrete_sequence=[cfg.THEME_ACCENT],
            )
            fig.update_layout(**PLOTLY_DARK, height=250, showlegend=False,
                              xaxis_title="EWI", yaxis_title="작업자 수")
            st.plotly_chart(fig, use_container_width=True)

    # CRE 분포
    with col2:
        if "cre" in worker_df.columns:
            st.markdown(sub_header("CRE 분포"), unsafe_allow_html=True)
            fig = px.histogram(
                worker_df, x="cre", nbins=20,
                color_discrete_sequence=["#FF6B35"],
            )
            fig.update_layout(**PLOTLY_DARK, height=250, showlegend=False,
                              xaxis_title="CRE", yaxis_title="작업자 수")
            st.plotly_chart(fig, use_container_width=True)

    # 업체별 작업자 수
    if not company_df.empty and "company_name" in company_df.columns:
        st.markdown(sub_header("업체별 작업자 현황"), unsafe_allow_html=True)
        count_col = "worker_count" if "worker_count" in company_df.columns else "n_workers"
        if count_col in company_df.columns:
            company_agg = company_df.groupby("company_name")[count_col].sum().reset_index()
            company_agg.columns = ["업체명", "작업자 수 (연인원)"]
            top10 = company_agg.nlargest(10, "작업자 수 (연인원)")
            fig = px.bar(
                top10, y="업체명", x="작업자 수 (연인원)",
                orientation="h", color_discrete_sequence=[cfg.THEME_ACCENT],
            )
            fig.update_layout(
                **PLOTLY_DARK, height=300, showlegend=False,
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # 고위험 작업자 통계
    if "cre" in worker_df.columns:
        high_cre = worker_df[worker_df["cre"] >= 0.5]
        if not high_cre.empty:
            st.markdown(
                f"<div style='background:#2A1A1A; border:1px solid #FF4C4C44; "
                f"border-radius:8px; padding:12px; margin-top:12px;'>"
                f"<div style='color:#FF4C4C; font-weight:600;'>"
                f"고위험 작업자 (CRE >= 0.5): {len(high_cre)}명</div>"
                f"<div style='color:#9AB5D4; font-size:0.82rem; margin-top:4px;'>"
                f"전체 {len(worker_df)}명 중 {len(high_cre)/len(worker_df)*100:.1f}%"
                f"</div></div>",
                unsafe_allow_html=True,
            )


def _render_congestion_section(space_df: pd.DataFrame):
    """공간 혼잡도 섹션."""
    st.markdown(section_header("공간 혼잡도"), unsafe_allow_html=True)

    if space_df.empty:
        st.info("공간 혼잡도 데이터 없음")
        return

    # 공간별 집계 컬럼 우선순위 (실제 parquet 컬럼 포함)
    count_cols = [c for c in [
        "avg_workers", "max_workers", "avg_occupancy",
        "unique_workers", "total_person_minutes", "total_visits",
    ] if c in space_df.columns]
    if not count_cols:
        st.info("혼잡도 수치 데이터 없음")
        return

    main_col = count_cols[0]

    # locus_token 기반 집계 (있으면 우선, 없으면 locus_id)
    _token_display = {
        "work_zone": "Work Zone (FAB)",
        "breakroom": "Break Room",
        "smoking_area": "Smoking Area",
        "restroom": "Restroom",
        "transit": "Hoist / Transit",
        "timeclock": "Time Clock (Gate)",
        "outdoor_work": "Outdoor Work",
        "unknown": "Unmapped",
    }
    if "locus_token" in space_df.columns:
        space_agg = space_df.groupby("locus_token")[main_col].sum().reset_index()
        space_agg["공간"] = space_agg["locus_token"].map(lambda t: _token_display.get(t, t))
    else:
        grp_col = next((c for c in ["locus_name", "locus_id"] if c in space_df.columns), None)
        if grp_col is None:
            st.info("공간 데이터 구조 확인 필요")
            return
        space_agg = space_df.groupby(grp_col)[main_col].sum().reset_index()
        space_agg["공간"] = space_agg[grp_col]

    space_agg = space_agg.rename(columns={main_col: "집계값"}).sort_values("집계값", ascending=True)

    x_label = main_col.replace("_", " ").title()
    fig = px.bar(
        space_agg, y="공간", x="집계값",
        orientation="h", color_discrete_sequence=[cfg.THEME_ACCENT],
        labels={"집계값": x_label},
    )
    fig.update_layout(**PLOTLY_DARK, height=max(250, len(space_agg) * 30 + 80), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_ai_summary(
    worker_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    equip: dict,
    n_days: int,
    selected_dates: list[str],
):
    """AI 기반 종합 요약 (옵션) — LLMGateway + DataPackager 사용."""
    try:
        from src.intelligence.llm_gateway import LLMGateway
        from src.intelligence.data_packager import DataPackager
        from src.dashboard.ai_analysis import render_data_comment as _render_ai_result
    except ImportError:
        return

    if not LLMGateway.is_available():
        return

    st.markdown(section_header("AI 종합 요약"), unsafe_allow_html=True)

    cache_key = f"ai_report_{'_'.join(selected_dates[:3])}"
    if cache_key in st.session_state and st.session_state[cache_key]:
        from src.dashboard.ai_analysis import render_data_comment
        render_data_comment("종합 리포트 AI 요약", st.session_state[cache_key])
        return

    if st.button("AI 종합 요약 생성", key="btn_report_ai"):
        with st.spinner("AI 종합 요약 생성 중..."):
            packed = DataPackager.overview(
                kpi={
                    "n_workers": worker_df["user_no"].nunique() if "user_no" in worker_df.columns and not worker_df.empty else 0,
                    "avg_ewi": float(worker_df["ewi"].mean()) if "ewi" in worker_df.columns and not worker_df.empty else None,
                    "avg_cre": float(worker_df["cre"].mean()) if "cre" in worker_df.columns and not worker_df.empty else None,
                    "avg_mat": float(transit_df["mat_minutes"].dropna().mean()) if "mat_minutes" in transit_df.columns and not transit_df.empty else None,
                    "avg_lbt": float(transit_df["lbt_minutes"].dropna().mean()) if "lbt_minutes" in transit_df.columns and not transit_df.empty else None,
                    "avg_eod": float(transit_df["eod_minutes"].dropna().mean()) if "eod_minutes" in transit_df.columns and not transit_df.empty else None,
                    "avg_opr": float(equip.get("weekly_overall_opr", pd.DataFrame()).get("opr_mean", pd.Series()).mean()) if not equip.get("weekly_overall_opr", pd.DataFrame()).empty else None,
                },
                prev_kpi=None,
                date_str=f"{selected_dates[0]}~{selected_dates[-1]}" if n_days > 1 else selected_dates[0],
            )
            company_names = worker_df["company_name"].dropna().unique().tolist() if "company_name" in worker_df.columns else []
            result = LLMGateway.analyze(
                "overview", packed,
                company_names=company_names,
                worker_names=[],
                zone_names=[],
                date_list=selected_dates,
                max_tokens=1000,
            )
            if result:
                st.session_state[cache_key] = result
                _render_ai_result("종합 리포트 AI 요약", result)


# =========================================================================
# 다운로드
# =========================================================================

def _render_download_section(
    worker_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    company_df: pd.DataFrame,
    space_df: pd.DataFrame,
    period_label: str,
    selected_dates: list[str],
):
    """PDF 다운로드 제공 (AI commentary 포함)."""
    st.markdown(sub_header("리포트 다운로드"), unsafe_allow_html=True)

    try:
        from src.report.pdf_generator import generate_report_pdf

        # AI 요약 가져오기 (있으면)
        cache_key = f"ai_report_{'_'.join(selected_dates[:3])}"
        ai_summary = st.session_state.get(cache_key, "")

        # 장비 데이터 가져오기
        equip = st.session_state.get("_report_equip", {})

        # 리포트 모드
        report_mode = st.session_state.get("_report_mode", "")
        n_days = len(selected_dates)

        # AI commentary 생성 (캐시됨 — 같은 날짜 조합이면 재생성 안 함)
        commentary_cache_key = f"_pdf_commentaries_{'_'.join(selected_dates[:3])}"
        if commentary_cache_key in st.session_state:
            commentaries = st.session_state[commentary_cache_key]
        else:
            with st.spinner("AI 코멘터리 생성 중..."):
                commentaries = _generate_report_commentaries(
                    worker_df=worker_df,
                    transit_df=transit_df,
                    equip=equip,
                    space_df=space_df,
                    selected_dates=selected_dates,
                    n_days=n_days,
                )

        n_ai = sum(1 for v in commentaries.values() if v)
        if n_ai > 0:
            st.caption(f"AI 코멘터리 {n_ai}개 섹션 생성 완료")
        else:
            st.caption("AI 코멘터리 미사용 (통계 요약으로 대체)")

        pdf_bytes = generate_report_pdf(
            worker_df=worker_df,
            transit_df=transit_df,
            company_df=company_df,
            space_df=space_df,
            equip=equip,
            selected_dates=selected_dates,
            period_label=period_label,
            report_mode=report_mode,
            ai_summary=ai_summary,
            commentaries=commentaries,
        )

        filename = f"DeepCon_M15X_Report_{'_'.join(selected_dates[:2])}.pdf"
        st.download_button(
            label="PDF 리포트 다운로드",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key="download_report_pdf",
        )
    except ImportError as e:
        st.warning(f"PDF 생성에 필요한 패키지가 없습니다: {e}")
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        st.error(f"PDF 생성 실패: {e}")
