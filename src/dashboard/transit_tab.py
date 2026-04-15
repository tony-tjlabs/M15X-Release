"""
Transit Tab - 작업자 대기/이동 시간 분석
=======================================
고객 요구사항: 출근/중식/퇴근 구간별 대기시간 일별 x BP별 분석

4개 서브탭:
- 일별 분석: 특정 날짜 상세 분석 (날짜 선택기 포함)
- 기간별 분석: 전체 기간 추이 차트 + Box Plot + 요일별
- BP별 분석: Horizontal Bar + 히트맵
- 원본 데이터: 전체 테이블 + 다운로드
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src.dashboard.styles import PLOTLY_DARK, PLOTLY_LEGEND, section_header, sub_header
from src.dashboard.date_utils import (
    get_date_selector,
    fetch_weather_info,
    format_date_label,
    format_date_full,
    parse_date_str,
    get_weekday_korean,
    DAY_NAMES_KR,
)
from src.pipeline.cache_manager import detect_processed_dates

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# --- 색상 상수 ---
COLOR_MAT = "#00AEEF"   # 출근 대기 (파랑)
COLOR_LBT = "#FFB300"   # 중식 이동 (노랑)
COLOR_LMT = "#9B59B6"   # 점심식사 전체 (보라)
COLOR_EOD = "#FF8C42"   # 퇴근 이동 (주황)
COLOR_TWT = "#00C897"   # 총 대기 (민트)

TRANSIT_COLORS = {
    "MAT": COLOR_MAT,
    "LBT": COLOR_LBT,
    "LMT": COLOR_LMT,
    "EOD": COLOR_EOD,
    "TWT": COLOR_TWT,
}


# ============================================================================
# 집계 캐시 함수 (렌더링 함수와 분리 — 재렌더 시 재계산 없음)
# ============================================================================

@st.cache_data(show_spinner=False, ttl=600)
def _compute_daily_agg(df_json: str) -> pd.DataFrame:
    """일별 집계 (MAT/LBT/EOD/TWT/worker_count). JSON 직렬화로 캐시 키 생성."""
    df = pd.read_json(io.StringIO(df_json), orient="split")
    daily_agg = df.groupby("date").agg({
        "mat_minutes": lambda x: x.dropna().mean(),
        "lbt_minutes": lambda x: x.dropna().mean(),
        "eod_minutes": lambda x: x.dropna().mean(),
        "user_no": "count",
    }).reset_index()
    daily_agg.columns = ["date", "MAT", "LBT", "EOD", "worker_count"]
    # TWT = MAT + LBT + EOD (각 항목별 dropna 평균의 합산으로 통일)
    daily_agg["TWT"] = daily_agg[["MAT", "LBT", "EOD"]].fillna(0).sum(axis=1)
    daily_agg = daily_agg.sort_values("date")
    return daily_agg


@st.cache_data(show_spinner=False, ttl=600)
def _compute_dow_agg(df_json: str) -> pd.DataFrame:
    """요일별 집계 (MAT/LBT/EOD 평균). JSON 직렬화로 캐시 키 생성."""
    df = pd.read_json(io.StringIO(df_json), orient="split")
    df["date_parsed"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["요일"] = df["date_parsed"].dt.dayofweek  # 0=월, 6=일
    dow_agg = df.groupby("요일").agg({
        "mat_minutes": lambda x: x.dropna().mean(),
        "lbt_minutes": lambda x: x.dropna().mean(),
        "eod_minutes": lambda x: x.dropna().mean(),
    }).reindex(range(7)).reset_index()
    return dow_agg


@st.cache_data(show_spinner=False, ttl=600)
def _compute_box_data(df_json: str) -> pd.DataFrame:
    """Box plot용 long-form 데이터. iterrows 제거, pd.melt() 사용. LMT 포함."""
    df = pd.read_json(io.StringIO(df_json), orient="split")
    value_cols = [c for c in ["mat_minutes", "lbt_minutes", "lmt_minutes", "eod_minutes"] if c in df.columns]
    label_map = {
        "mat_minutes": "출근 대기(MAT)",
        "lbt_minutes": "중식 이동(LBT)",
        "lmt_minutes": "점심 체류(LMT)",
        "eod_minutes": "퇴근 이동(EOD)",
    }
    box_df = df[value_cols].rename(columns=label_map)
    melted = box_df.melt(var_name="구간", value_name="시간(분)").dropna()
    return melted


@st.cache_data(show_spinner=False, ttl=600)
def _compute_summary_kpis_data(df_json: str) -> dict:
    """KPI 카드 수치 계산. Coverage 필터 적용 후 평균 반환."""
    df = pd.read_json(io.StringIO(df_json), orient="split")

    if "is_coverage_excluded" in df.columns:
        excl_mask = df["is_coverage_excluded"].fillna(False).astype(bool)
        df_kpi = df[~excl_mask]
    else:
        df_kpi = df

    result = {
        "n_kpi": len(df_kpi),
        "n_total": len(df),
        "avg_mat": df_kpi["mat_minutes"].dropna().mean() if "mat_minutes" in df_kpi.columns else 0,
        "avg_lbt": df_kpi["lbt_minutes"].dropna().mean() if "lbt_minutes" in df_kpi.columns else 0,
        "avg_eod": df_kpi["eod_minutes"].dropna().mean() if "eod_minutes" in df_kpi.columns else 0,
        "avg_twt": (
            (df_kpi["mat_minutes"].dropna().mean() if "mat_minutes" in df_kpi.columns else 0)
            + (df_kpi["lbt_minutes"].dropna().mean() if "lbt_minutes" in df_kpi.columns else 0)
            + (df_kpi["eod_minutes"].dropna().mean() if "eod_minutes" in df_kpi.columns else 0)
        ),
        "has_lmt": "lmt_minutes" in df_kpi.columns,
        "avg_lmt": df_kpi["lmt_minutes"].dropna().mean() if "lmt_minutes" in df_kpi.columns else 0,
    }
    return result


# ============================================================================
# 데이터 로더
# ============================================================================

@st.cache_data(show_spinner=False, ttl=300)
def _load_transit_data(date_str: str, sector_id: str) -> pd.DataFrame | None:
    """단일 날짜 transit.parquet 로드."""
    paths = cfg.get_sector_paths(sector_id)
    proc_dir = paths["processed_dir"] / date_str
    transit_path = proc_dir / "transit.parquet"

    if not transit_path.exists():
        return None

    try:
        df = pd.read_parquet(transit_path)
        df["date"] = date_str
        return df
    except Exception as e:
        logger.warning("transit.parquet 로드 실패 [%s]: %s", date_str, e)
        return None


@st.cache_data(show_spinner=False, ttl=300)
def _load_bp_transit_data(date_str: str, sector_id: str) -> pd.DataFrame | None:
    """단일 날짜 bp_transit.parquet 로드."""
    paths = cfg.get_sector_paths(sector_id)
    proc_dir = paths["processed_dir"] / date_str
    bp_path = proc_dir / "bp_transit.parquet"

    if not bp_path.exists():
        return None

    try:
        df = pd.read_parquet(bp_path)
        df["date"] = date_str
        return df
    except Exception as e:
        logger.warning("bp_transit.parquet 로드 실패 [%s]: %s", date_str, e)
        return None


@st.cache_data(show_spinner=False, ttl=300)
def _load_all_transit_data(sector_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """모든 처리된 날짜의 transit 데이터 로드."""
    processed = detect_processed_dates(sector_id)
    if not processed:
        return pd.DataFrame(), pd.DataFrame()

    worker_dfs = []
    bp_dfs = []

    for date_str in processed:
        wdf = _load_transit_data(date_str, sector_id)
        if wdf is not None and not wdf.empty:
            worker_dfs.append(wdf)

        bdf = _load_bp_transit_data(date_str, sector_id)
        if bdf is not None and not bdf.empty:
            bp_dfs.append(bdf)

    worker_all = pd.concat(worker_dfs, ignore_index=True) if worker_dfs else pd.DataFrame()
    bp_all = pd.concat(bp_dfs, ignore_index=True) if bp_dfs else pd.DataFrame()

    return worker_all, bp_all


# ============================================================================
# KPI 카드 컴포넌트
# ============================================================================

def _render_kpi_card(
    label: str,
    value: str,
    unit: str = "분",
    color: str = COLOR_MAT,
) -> None:
    """대기시간 KPI 카드 렌더링."""
    st.markdown(f'''
    <div style="background:{cfg.THEME_CARD_BG}; border-radius:12px; padding:20px;
         text-align:center; border-left:4px solid {color};">
        <div style="color:{cfg.THEME_TEXT}; font-size:0.85rem; margin-bottom:8px;">{label}</div>
        <div style="color:{color}; font-size:2rem; font-weight:700;">{value}</div>
        <div style="color:{cfg.THEME_TEXT}; font-size:0.75rem; opacity:0.7;">{unit}</div>
    </div>
    ''', unsafe_allow_html=True)


def _render_route_segment_chart(
    df: pd.DataFrame,
    seg_cols: list[str],
    seg_labels: list[str],
    seg_colors: list[str],
    title: str,
    caption: str,
    y_label: str,
    gap_cols: list[str] | None = None,
    total_col: str | None = None,
    metric_col: str | None = None,
    metric_label: str | None = None,
) -> None:
    """단일 경로 구간별 소요시간 차트 + 테이블 렌더링.

    각 구간을 [BLE 감지(solid)] + [미감지(hatched)] 두 파트로 분리해 표시한다.
    미감지 구간은 해당 구간 바 바로 뒤에 위치하여 실제 경로 위치를 반영한다.

    Args:
        gap_cols: seg_cols 와 1:1 대응하는 미감지 시간 컬럼명 리스트.
                  None이면 미감지 분리 없이 기존 방식으로 렌더링.
        metric_label: KPI 행 표시명 (None이면 metric_col에서 자동 추출).
    """
    if not all(c in df.columns for c in seg_cols):
        return

    any_data = any(df[c].notna().sum() > 0 for c in seg_cols)
    if not any_data:
        return

    st.markdown(sub_header(title), unsafe_allow_html=True)
    st.caption(caption)

    # KPI와 동일한 모집단: metric_col 이 있는 작업자만
    base = df.copy()
    if metric_col and metric_col in df.columns:
        base = base[base[metric_col].notna()].copy()
    for c in seg_cols:
        base[c] = base[c].fillna(0)

    n_workers = len(base)
    seg_means = [base[c].mean() for c in seg_cols]
    seg_medians = [base[c].median() for c in seg_cols]
    seg_counts = [int((base[c] > 0).sum()) for c in seg_cols]

    # 구간별 미감지 평균 (gap_cols 있을 때만)
    gap_means: list[float] = []
    if gap_cols and all(c in base.columns for c in gap_cols):
        for gc in gap_cols:
            base[gc] = base[gc].fillna(0)
        gap_means = [float(base[gc].mean()) for gc in gap_cols]
    else:
        gap_means = [0.0] * len(seg_cols)

    # ── seg_sum: 각 구간의 전체 시간 합 (gap은 이미 seg_mean 안에 포함되어 있음)
    # seg_mean = 해당 구간 전체 시간 (감지 + 미감지 포함)
    # gap_mean = 그 중 미감지 시간 → seg_mean 안에 이미 포함, 별도 더하면 이중계산!
    seg_sum = sum(seg_means)

    # KPI 전체 평균
    if metric_col and metric_col in base.columns:
        metric_total = base[metric_col].mean()
    else:
        metric_total = seg_sum

    # 경로 미추적: 경로 감지 자체가 안 된 작업자의 KPI 기여분 (position 미확인)
    untracked = max(0.0, metric_total - seg_sum)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig_seg = go.Figure()
        gap_legend_shown = False

        for i, (label, seg_mean, gap_mean, color) in enumerate(
            zip(seg_labels, seg_means, gap_means, seg_colors)
        ):
            # ① 감지 구간 (solid): 구간 내 BLE 감지된 시간
            detected_val = max(0.0, seg_mean - gap_mean)
            fig_seg.add_trace(go.Bar(
                name=label,
                x=[detected_val if detected_val > 0 else seg_mean],
                y=[y_label],
                orientation="h",
                marker_color=color,
                text=[f"{seg_mean:.1f}분" if seg_mean > 0 else ""],
                textposition="inside",
                textfont=dict(size=12, color="white"),
                legendgroup=label,
            ))

            # ② 미감지 구간 (hatched) — 해당 구간 내 음영, 앞뒤 경유지로 귀속됨
            if gap_mean > 0.3:
                fig_seg.add_trace(go.Bar(
                    name="구간 내 음영",
                    x=[gap_mean],
                    y=[y_label],
                    orientation="h",
                    marker=dict(
                        color="rgba(140,140,140,0.30)",
                        pattern=dict(shape="/", fgcolor="rgba(190,190,190,0.55)", size=5),
                        line=dict(color="rgba(180,180,180,0.4)", width=1),
                    ),
                    text=[f"~{gap_mean:.1f}분"],
                    textposition="inside",
                    textfont=dict(size=11, color="rgba(210,210,210,0.8)"),
                    legendgroup="구간 내 음영",
                    showlegend=not gap_legend_shown,
                ))
                gap_legend_shown = True

        # ③ 경로 미추적 구간 (점선 패턴) — 경로 자체가 감지 안 된 작업자의 기여분
        if untracked > 0.3:
            fig_seg.add_trace(go.Bar(
                name="경로 미추적",
                x=[untracked],
                y=[y_label],
                orientation="h",
                marker=dict(
                    color="rgba(100,100,100,0.20)",
                    pattern=dict(shape="x", fgcolor="rgba(160,160,160,0.45)", size=5),
                    line=dict(color="rgba(150,150,150,0.35)", width=1),
                ),
                text=[f"~{untracked:.1f}분"],
                textposition="inside",
                textfont=dict(size=11, color="rgba(190,190,190,0.75)"),
                legendgroup="경로 미추적",
            ))

        fig_seg.update_layout(
            barmode="stack",
            height=120,
            template="plotly_dark",
            paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
            plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
            font_color=PLOTLY_DARK["font_color"],
            margin=dict(l=80, r=20, t=10, b=10),
            xaxis_title="소요시간 (분)",
            legend=dict(orientation="h", y=1.25, traceorder="normal"),
            showlegend=True,
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_b:
        # 테이블:
        # "평균 (분)" = 구간 전체 시간 (감지 + 음영 포함) — 음영도 해당 구간으로 귀속됨
        # "음영 (분)" = 그 중 BLE 미감지 시간
        rows_label, rows_mean, rows_gap, rows_med, rows_count = [], [], [], [], []
        for label, seg_m, gap_m, seg_med, cnt in zip(
            seg_labels, seg_means, gap_means, seg_medians, seg_counts
        ):
            rows_label.append(label)
            rows_mean.append(f"{seg_m:.1f}" if seg_m > 0 else "-")
            rows_gap.append(f"~{gap_m:.1f}" if gap_m > 0.3 else "-")
            rows_med.append(f"{seg_med:.1f}" if seg_med > 0 else "-")
            rows_count.append(cnt)

        # 구간 합계 행
        rows_label.append("**구간 합계**")
        rows_mean.append(f"**{seg_sum:.1f}**")
        rows_gap.append(f"~{sum(gap_means):.1f}" if sum(gap_means) > 0.3 else "-")
        rows_med.append("-")
        rows_count.append(n_workers)

        # 경로 미추적 행 (있을 때만)
        if untracked > 0.3:
            rows_label.append("경로 미추적 (위치 미확인)")
            rows_mean.append(f"~{untracked:.1f}")
            rows_gap.append("-")
            rows_med.append("-")
            rows_count.append(n_workers)

        # KPI 합계 행 — metric_label 파라미터 우선, 없으면 metric_col에서 추출
        if metric_label:
            metric_label_str = metric_label
        elif metric_col:
            _col_label_map = {
                "mat_minutes": "MAT", "lbt_minutes": "LBT",
                "eod_minutes": "EOD", "total_wait_minutes": "TWT",
                "seg_lbt_out_total": "LBT-Out", "seg_lbt_in_total": "LBT-In",
                "seg_eod_total": "EOD", "seg_total": "MAT",
            }
            metric_label_str = _col_label_map.get(metric_col, metric_col.replace("_minutes", "").upper())
        else:
            metric_label_str = ""
        rows_label.append(f"**{metric_label_str} 평균 (KPI)**")
        rows_mean.append(f"**{metric_total:.1f}**")
        rows_gap.append("-")
        rows_med.append("-")
        rows_count.append(n_workers)

        seg_summary = pd.DataFrame({
            "구간": rows_label,
            "합산 (분)": rows_mean,
            "음영 (분)": rows_gap,
            "중앙값 (분)": rows_med,
            "경유 인원": rows_count,
        })
        tbl_height = 155 + len(rows_label) * 35
        st.dataframe(seg_summary, use_container_width=True, hide_index=True, height=tbl_height)

    # 수치 해석 코멘트 (metric_label_str은 위에서 이미 계산됨)
    total_gap = sum(gap_means)
    gap_note_parts = []
    if total_gap > 0.3:
        gap_note_parts.append(
            f"구간 내 음영(///) {total_gap:.1f}분은 앞뒤 경유지를 기준으로 해당 구간 이동 시간으로 귀속되어 "
            f"**합산(분)에 포함**됩니다."
        )
    if untracked > 0.3:
        gap_note_parts.append(
            f"경로 미추적(×) {untracked:.1f}분은 BLE 경로 자체가 감지되지 않은 작업자의 "
            f"{metric_label_str} 기여분이며 위치를 특정할 수 없습니다."
        )
    gap_note_parts.append(
        f"구간 합계({seg_sum:.1f}분) + 경로 미추적({untracked:.1f}분) = "
        f"**{metric_label_str} KPI({metric_total:.1f}분)**"
    )
    st.caption(
        f"※ {n_workers}명 기준 ({metric_label_str} 감지된 작업자) | "
        + " | ".join(gap_note_parts)
    )


def _render_all_route_segments(df: pd.DataFrame) -> None:
    """4가지 경로 구간별 소요시간 차트를 모두 렌더링."""

    # 1. 출근 (MAT): 타각기 -> 야외 -> 호이스트 -> FAB
    # 경로 미추적 = 타각기 통과 후 FAB 도달이 확인되었으나 3개 경유지 BLE 미감지 작업자의 MAT 기여분
    _render_route_segment_chart(
        df,
        seg_cols=["seg_gate_to_outdoor", "seg_outdoor_to_hoist", "seg_hoist_to_fab"],
        gap_cols=["seg_gap_gate_outdoor", "seg_gap_outdoor_hoist", "seg_gap_hoist_fab"],
        seg_labels=["타각기 → 야외", "야외 → 호이스트", "호이스트 → FAB"],
        seg_colors=["#00AEEF", "#FFB300", "#00C897"],
        title="출근 경로 구간별 소요시간 (MAT)",
        caption=(
            "경로: 타각기(출입) → 야외(공사현장) → 호이스트 → FAB 작업층 | "
            "회색(///) = 해당 구간 내 BLE 미감지 시간 | "
            "경로 미추적 = FAB 도착은 확인되었으나 경유지 BLE 미감지 작업자의 평균 기여분"
        ),
        y_label="출근 동선",
        total_col="seg_total",
        metric_col="mat_minutes",
    )

    # 2. 점심 외출 (LBT-Out): FAB -> 호이스트 -> 야외 -> 타각기
    # metric_col = seg_lbt_out_total (편도만): 타각기 이후 현장 외부는 BLE 추적 불가
    # 경로 미추적 = LBT-Out KPI 중 3개 구간으로 설명되지 않는 부분 (경유지 미감지)
    _render_route_segment_chart(
        df,
        seg_cols=["seg_lbt_out_fab_to_hoist", "seg_lbt_out_hoist_to_outdoor", "seg_lbt_out_outdoor_to_gate"],
        gap_cols=["seg_gap_lbt_out_fab_hoist", "seg_gap_lbt_out_hoist_outdoor", "seg_gap_lbt_out_outdoor_gate"],
        seg_labels=["FAB → 호이스트", "호이스트 → 야외", "야외 → 타각기"],
        seg_colors=["#FFB300", "#FF8C42", "#E74C3C"],
        title="점심 외출 경로 구간별 소요시간 (LBT-Out)",
        caption=(
            "경로: FAB 작업층 → 호이스트 → 야외(공사현장) → 타각기(출입) | "
            "회색(///) = 해당 구간 내 BLE 미감지 시간 | "
            "경로 미추적 = 타각기 통과 이후 현장 외부 진출 (BLE 추적 범위 외) — "
            "타각기 이후의 외부 체류 시간은 LMT 지표에서 별도 측정됨"
        ),
        y_label="점심 외출",
        total_col="seg_lbt_out_total",
        metric_col="lbt_out_minutes",   # lbt_minutes와 동일 집단 — out + in = lbt_minutes 보장
    )

    # 3. 점심 복귀 (LBT-In): 타각기 -> 야외 -> 호이스트 -> FAB
    # metric_col = lbt_in_minutes: lbt_minutes와 동일 집단·동일 방식으로 계산된 In 편도
    # → avg(lbt_out_minutes) + avg(lbt_in_minutes) = avg(lbt_minutes) (동일 집단)
    _render_route_segment_chart(
        df,
        seg_cols=["seg_lbt_in_gate_to_outdoor", "seg_lbt_in_outdoor_to_hoist", "seg_lbt_in_hoist_to_fab"],
        gap_cols=["seg_gap_lbt_in_gate_outdoor", "seg_gap_lbt_in_outdoor_hoist", "seg_gap_lbt_in_hoist_fab"],
        seg_labels=["타각기 → 야외", "야외 → 호이스트", "호이스트 → FAB"],
        seg_colors=["#3498DB", "#9B59B6", "#1ABC9C"],
        title="점심 복귀 경로 구간별 소요시간 (LBT-In)",
        caption=(
            "경로: 타각기(출입) → 야외(공사현장) → 호이스트 → FAB 작업층 | "
            "회색(///) = 해당 구간 내 BLE 미감지 시간 | "
            "경로 미추적 = 타각기 진입 이후 FAB 도달까지 경유지 BLE 미감지 기여분"
        ),
        y_label="점심 복귀",
        total_col="seg_lbt_in_total",
        metric_col="lbt_in_minutes",    # lbt_minutes와 동일 집단 — out + in = lbt_minutes 보장
    )

    # 4. 퇴근 (EOD): FAB -> 호이스트 -> 야외 -> 타각기
    # 경로 미추적 = 타각기 통과 후 ~ AccessLog 퇴장 기록까지의 간극 (현장 외부 이동)
    _render_route_segment_chart(
        df,
        seg_cols=["seg_eod_fab_to_hoist", "seg_eod_hoist_to_outdoor", "seg_eod_outdoor_to_gate"],
        gap_cols=["seg_gap_eod_fab_hoist", "seg_gap_eod_hoist_outdoor", "seg_gap_eod_outdoor_gate"],
        seg_labels=["FAB → 호이스트", "호이스트 → 야외", "야외 → 타각기"],
        seg_colors=["#FF8C42", "#E67E22", "#C0392B"],
        title="퇴근 경로 구간별 소요시간 (EOD)",
        caption=(
            "경로: FAB 작업층 → 호이스트 → 야외(공사현장) → 타각기(출입) | "
            "회색(///) = 해당 구간 내 BLE 미감지 시간 | "
            "경로 미추적 = 타각기 마지막 BLE 감지 → AccessLog 공식 퇴장 기록까지의 간극"
        ),
        y_label="퇴근 동선",
        total_col="seg_eod_total",
        metric_col="eod_minutes",
    )

    st.markdown("")


def _render_coverage_transparency(df: pd.DataFrame) -> None:
    """
    투명성 패널: 이 수치는 어떻게 계산되었나?
    Coverage ratio, threshold, 보정 방법, 집계 기준 명시.
    """
    from src.pipeline.journey_reconstructor import DEFAULT_COVERAGE_THRESHOLD

    has_coverage = "coverage_pct" in df.columns and df["coverage_pct"].notna().any()
    has_excluded = "is_coverage_excluded" in df.columns

    n_total_transit = len(df)
    n_included = int((~df["is_coverage_excluded"].fillna(False).astype(bool)).sum()) if has_excluded else n_total_transit
    n_excluded = n_total_transit - n_included

    threshold_pct = int(DEFAULT_COVERAGE_THRESHOLD * 100)

    with st.expander("이 수치는 어떻게 계산되었나?", expanded=False):
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown(
                f"""
**집계 기준 (모집단)**
- T-Ward 착용자: {n_total_transit}명 분석
- BLE coverage >= {threshold_pct}% 통과: **{n_included}명** (KPI 집계 대상)
- BLE coverage < {threshold_pct}% 제외: {n_excluded}명 (신뢰도 미달)
- 모든 KPI 카드, 차트, 테이블이 **동일한 {n_included}명 기준**
                """
            )

        with col_b:
            st.markdown(
                f"""
**지표 계산 방법**
- **MAT** = AccessLog 입장시각 → T-Ward 첫 work_zone 감지 (분)
- **EOD** = T-Ward 마지막 work_zone → AccessLog 퇴장시각 (분)
- **LBT** = 중식 왕복 이동 시간 합계 (LBT-Out + LBT-In, 최대 90분)
  - LBT-Out: 마지막 work_zone → 타각기 첫 감지
  - LBT-In: 타각기 마지막 감지 → 첫 work_zone 복귀 (45분 내)
- **LMT** = 타각기 첫 감지 → 타각기 마지막 감지 (순수 건물 밖 체류)
- **TWT** = MAT + LBT + EOD 합계
- **Coverage** = T-Ward 감지분 / AccessLog 체류총분 × 100%
                """
            )

        st.markdown("---")
        st.markdown(
            f"""
**Coverage 분류 기준**

| 등급 | Coverage | 의미 |
|------|----------|------|
| 불충분 | < 30% | BLE 거의 없음, 이동 경로 재현 불가 → 집계 제외 |
| 부분 | 30~50% | 이동 경로 부분 재현, 시간 추정 오차 있음 |
| 양호 | 50~70% | 주요 경로 포착, 실용적 정확도 |
| 충분 | > 70% | BLE 충분, 높은 신뢰도 |

> M15X FAB 현장 특성: 철골 구조물·금속 자재로 BLE 신호 감쇠가 심함.
> 30% threshold는 데이터 분포를 고려한 균형점입니다.
            """
        )

        if has_coverage:
            st.markdown("**현재 날짜 Coverage 분포**")
            dist_data = df["coverage_pct"].dropna()
            import plotly.graph_objects as go
            fig_cov = go.Figure(go.Histogram(
                x=dist_data,
                nbinsx=20,
                marker_color="#00AEEF",
                opacity=0.8,
            ))
            fig_cov.add_vline(
                x=threshold_pct,
                line_dash="dash",
                line_color="#FF4C4C",
                annotation_text=f"기준({threshold_pct}%)",
                annotation_font_color="#FF4C4C",
            )
            fig_cov.update_layout(
                height=200,
                margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#C8D6E8",
                xaxis_title="BLE Coverage (%)",
                yaxis_title="작업자 수",
                showlegend=False,
            )
            st.plotly_chart(fig_cov, use_container_width=True)


def _render_summary_kpis(worker_df: pd.DataFrame) -> None:
    """4~5열 KPI 요약 카드 렌더링 (LMT 컬럼 존재 시 5열).

    집계 기준: coverage_threshold를 통과한 작업자만 (is_coverage_excluded=False).
    집계 계산은 _compute_summary_kpis_data()에 위임하여 재렌더 시 캐시 적중.
    """
    kpi = _compute_summary_kpis_data(worker_df.to_json(orient="split"))

    avg_mat = kpi["avg_mat"]
    avg_lbt = kpi["avg_lbt"]
    avg_eod = kpi["avg_eod"]
    avg_twt = kpi["avg_twt"]
    n_kpi   = kpi["n_kpi"]
    n_total = kpi["n_total"]

    if kpi["has_lmt"]:
        avg_lmt = kpi["avg_lmt"]
        cols = st.columns(5)
        with cols[0]:
            _render_kpi_card("평균 출근 대기 (MAT)", f"{avg_mat:.1f}", "분", COLOR_MAT)
        with cols[1]:
            _render_kpi_card("평균 중식 이동 (LBT, 왕복)", f"{avg_lbt:.1f}", "분", COLOR_LBT)
        with cols[2]:
            _render_kpi_card("평균 퇴근 이동 (EOD)", f"{avg_eod:.1f}", "분", COLOR_EOD)
        with cols[3]:
            _render_kpi_card("평균 총 대기 (TWT)", f"{avg_twt:.1f}", "분", COLOR_TWT)
        with cols[4]:
            _render_kpi_card("평균 점심 외부 체류 (LMT)", f"{avg_lmt:.1f}", "분", COLOR_LMT)
        st.caption(
            "LMT (점심 외부 체류): 타각기 첫 감지 → 타각기 마지막 감지. 순수 건물 밖 체류 시간 (이동 제외)."
            " | LBT (왕복 이동): LBT-Out + LBT-In 합계."
        )
    else:
        cols = st.columns(4)
        with cols[0]:
            _render_kpi_card("평균 출근 대기 (MAT)", f"{avg_mat:.1f}", "분", COLOR_MAT)
        with cols[1]:
            _render_kpi_card("평균 중식 이동 (LBT, 왕복)", f"{avg_lbt:.1f}", "분", COLOR_LBT)
        with cols[2]:
            _render_kpi_card("평균 퇴근 이동 (EOD)", f"{avg_eod:.1f}", "분", COLOR_EOD)
        with cols[3]:
            _render_kpi_card("평균 총 대기 (TWT)", f"{avg_twt:.1f}", "분", COLOR_TWT)

    # 집계 기준 안내
    st.caption(
        f"※ 집계 기준: Coverage >= 30% 작업자 {n_kpi}명 / 전체 T-Ward {n_total}명 | "
        f"'이 수치는 어떻게 계산되었나?' 항목 참조"
    )


# ============================================================================
# 서브탭 A: 일별 분석 (특정 날짜 상세)
# ============================================================================

def _render_daily_single_subtab(
    worker_df: pd.DataFrame,
    sector_id: str,
    bp_filter: list[str] | None = None,
) -> None:
    """일별 분석 서브탭 - 특정 날짜 선택 후 상세 분석."""
    if worker_df.empty:
        st.info("대기시간 데이터가 없습니다. 파이프라인 탭에서 먼저 전처리를 실행하세요.")
        return

    # 날짜 선택기
    available_dates = sorted(worker_df["date"].unique())
    st.markdown(
        "<div style='font-size:0.8rem; color:#7A8FA6; margin-bottom:6px;'>분석 날짜</div>",
        unsafe_allow_html=True,
    )
    selected_date = get_date_selector(available_dates, key="transit_daily_date_selector")

    if not selected_date:
        st.info("날짜를 선택하세요.")
        return

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # 선택된 날짜 데이터 필터링
    df = worker_df[worker_df["date"] == selected_date].copy()
    if bp_filter:
        df = df[df["company_name"].isin(bp_filter)]

    if df.empty:
        st.warning(f"{selected_date} 데이터가 없습니다.")
        return

    # 날짜 정보 표시
    parsed_date = parse_date_str(selected_date)
    if parsed_date:
        weekday = get_weekday_korean(parsed_date)
        st.markdown(
            f"<div style='font-size:1.1rem; font-weight:600; color:{cfg.THEME_TEXT}; margin-bottom:16px;'>"
            f"{format_date_full(selected_date)} ({weekday})</div>",
            unsafe_allow_html=True,
        )

    # 0) 투명성 패널: 수치 계산 방법 공개
    _render_coverage_transparency(df)
    st.markdown("")

    # 1) KPI 요약 카드
    _render_summary_kpis(df)
    st.markdown("")

    # 1.5) 4가지 경로 구간별 소요시간 분석
    # Coverage 통과 작업자만 (KPI와 동일 모집단)
    if "is_coverage_excluded" in df.columns:
        excl_mask = df["is_coverage_excluded"].fillna(False).astype(bool)
        df_seg = df[~excl_mask].copy()
    else:
        df_seg = df.copy()
    _render_all_route_segments(df_seg)

    # 2) 시간대별 분포 히스토그램 (2열) — coverage 통과 작업자만
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(sub_header("출근 대기시간 분포 (MAT)"), unsafe_allow_html=True)
        mat_data = df_seg["mat_minutes"].dropna()
        if not mat_data.empty:
            fig_mat = px.histogram(
                x=mat_data,
                nbins=20,
                color_discrete_sequence=[COLOR_MAT],
            )
            fig_mat.update_layout(
                height=250,
                template="plotly_dark",
                paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
                plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                font_color=PLOTLY_DARK["font_color"],
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title="대기시간 (분)",
                yaxis_title="작업자 수",
                showlegend=False,
            )
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.info("MAT 데이터 없음")

    with col2:
        st.markdown(sub_header("퇴근 이동시간 분포 (EOD)"), unsafe_allow_html=True)
        eod_data = df_seg["eod_minutes"].dropna()
        if not eod_data.empty:
            fig_eod = px.histogram(
                x=eod_data,
                nbins=20,
                color_discrete_sequence=[COLOR_EOD],
            )
            fig_eod.update_layout(
                height=250,
                template="plotly_dark",
                paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
                plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                font_color=PLOTLY_DARK["font_color"],
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title="이동시간 (분)",
                yaxis_title="작업자 수",
                showlegend=False,
            )
            st.plotly_chart(fig_eod, use_container_width=True)
        else:
            st.info("EOD 데이터 없음")

    # 3) 업체별 대기시간 TOP 10 (해당 날짜) — coverage 통과 작업자만
    st.markdown(sub_header("업체별 평균 대기시간 TOP 10"), unsafe_allow_html=True)

    if "company_name" in df_seg.columns:
        bp_agg = df_seg.groupby("company_name").agg(
            MAT=("mat_minutes", lambda x: x.dropna().mean()),
            LBT=("lbt_minutes", lambda x: x.dropna().mean()),
            EOD=("eod_minutes", lambda x: x.dropna().mean()),
            작업자수=("user_no", "count"),
        ).reset_index()
        bp_agg.columns = ["업체", "MAT", "LBT", "EOD", "작업자수"]
        # TWT = 각 지표 평균의 합 (NaN은 0으로 처리, 단 전부 NaN이면 제외)
        bp_agg["TWT"] = bp_agg[["MAT", "LBT", "EOD"]].fillna(0).sum(axis=1)
        has_any = bp_agg[["MAT", "LBT", "EOD"]].notna().any(axis=1)
        bp_agg = bp_agg[has_any]
        # 최소 3명 이상 업체만 (1~2명은 신뢰도 낮음)
        bp_agg = bp_agg[bp_agg["작업자수"] >= 3]
        bp_agg = bp_agg.sort_values("TWT", ascending=True).tail(10)

        fig_bp = go.Figure()
        for metric, color, name in [
            ("MAT", COLOR_MAT, "MAT"),
            ("LBT", COLOR_LBT, "LBT"),
            ("EOD", COLOR_EOD, "EOD"),
        ]:
            fig_bp.add_trace(go.Bar(
                y=bp_agg["업체"],
                x=bp_agg[metric],
                orientation="h",
                name=name,
                marker_color=color,
            ))

        fig_bp.update_layout(
            height=max(300, len(bp_agg) * 30),
            template="plotly_dark",
            paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
            plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
            font_color=PLOTLY_DARK["font_color"],
            margin=dict(l=150, r=20, t=30, b=40),
            barmode="stack",
            xaxis_title="대기시간 (분)",
            yaxis_title="",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
        )
        st.plotly_chart(fig_bp, use_container_width=True)
    else:
        st.info("업체 정보 없음")

    # 4) 작업자별 상세 테이블 — coverage 정보 포함, 전체 표시 (coverage 컬럼 포함)
    with st.expander("작업자별 상세 데이터 (전체)", expanded=False):
        display_cols = [
            "user_no", "user_name", "company_name",
            "mat_minutes", "lbt_minutes", "eod_minutes",
            "total_wait_minutes", "coverage_pct", "coverage_label", "is_coverage_excluded",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        display_df = df[available_cols].copy()

        col_rename = {
            "user_no": "작업자 ID",
            "user_name": "작업자명",
            "company_name": "업체명",
            "mat_minutes": "MAT (분)",
            "lbt_minutes": "LBT (분)",
            "eod_minutes": "EOD (분)",
            "total_wait_minutes": "TWT (분)",
            "coverage_pct": "Coverage (%)",
            "coverage_label": "Coverage 등급",
            "is_coverage_excluded": "집계 제외",
        }
        display_df = display_df.rename(columns=col_rename)

        # 집계 제외 작업자 표시
        if "집계 제외" in display_df.columns:
            st.caption(
                f"전체 {len(display_df)}명 표시 (집계 포함 {(~display_df['집계 제외']).sum()}명, "
                f"Coverage 미달 제외 {display_df['집계 제외'].sum()}명)"
            )

        # 작업자명 마스킹
        if "작업자명" in display_df.columns:
            try:
                from src.utils.anonymizer import mask_name
                display_df["작업자명"] = display_df["작업자명"].apply(mask_name)
            except ImportError:
                pass

        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# 서브탭 B: 기간별 분석 (전체 기간 추이)
# ============================================================================

def _render_period_subtab(
    worker_df: pd.DataFrame,
    bp_filter: list[str] | None = None,
) -> None:
    """기간별 분석 서브탭 - 전체 기간 추이 분석."""
    if worker_df.empty:
        st.info("대기시간 데이터가 없습니다. 파이프라인 탭에서 먼저 전처리를 실행하세요.")
        return

    # BP 필터 적용
    df = worker_df.copy()
    if bp_filter:
        df = df[df["company_name"].isin(bp_filter)]

    if df.empty:
        st.warning("선택한 업체에 해당하는 데이터가 없습니다.")
        return

    # 1) KPI 요약 카드 (전체 기간)
    _render_summary_kpis(df)
    st.markdown("")

    # 2) 일별 대기시간 추이 차트
    st.markdown(sub_header("일별 대기시간 추이"), unsafe_allow_html=True)

    _df_json = df.to_json(orient="split")
    daily_agg = _compute_daily_agg(_df_json)

    # 날짜 포맷팅
    daily_agg["date_label"] = daily_agg["date"].astype(str).apply(format_date_label)

    # Multi-line chart
    fig_trend = go.Figure()

    for metric, color, name in [
        ("MAT", COLOR_MAT, "출근 대기 (MAT)"),
        ("LBT", COLOR_LBT, "중식 이동 (LBT)"),
        ("EOD", COLOR_EOD, "퇴근 이동 (EOD)"),
    ]:
        fig_trend.add_trace(go.Scatter(
            x=daily_agg["date_label"],
            y=daily_agg[metric],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=8),
            hovertemplate=f"{name}: %{{y:.1f}}분<extra></extra>",
        ))

    fig_trend.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
        plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
        font_color=PLOTLY_DARK["font_color"],
        margin=dict(l=40, r=20, t=30, b=50),
        xaxis_title="날짜",
        yaxis_title="대기시간 (분)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            **PLOTLY_LEGEND,
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # 3) 구간별 분포 + 요일별 평균
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(sub_header("구간별 분포 (Box Plot)"), unsafe_allow_html=True)

        # Box plot 데이터 준비 — pd.melt() 기반 캐시 함수 사용 (LMT 포함)
        melted = _compute_box_data(_df_json)

        if not melted.empty:
            color_discrete_map = {
                "출근 대기(MAT)": COLOR_MAT,
                "중식 이동(LBT)": COLOR_LBT,
                "점심 체류(LMT)": COLOR_LMT,
                "퇴근 이동(EOD)": COLOR_EOD,
            }
            fig_box = px.box(
                melted,
                x="구간",
                y="시간(분)",
                color="구간",
                color_discrete_map=color_discrete_map,
            )
            fig_box.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
                plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                font_color=PLOTLY_DARK["font_color"],
                margin=dict(l=40, r=20, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Box Plot 데이터 없음")

    with col2:
        st.markdown(sub_header("요일별 평균 대기시간"), unsafe_allow_html=True)

        # 요일별 집계 — 캐시 함수 사용
        dow_agg = _compute_dow_agg(_df_json)
        dow_agg["요일명"] = dow_agg["요일"].map(lambda x: DAY_NAMES_KR[x] if pd.notna(x) else "")

        fig_dow = go.Figure()
        for metric, color, name in [
            ("mat_minutes", COLOR_MAT, "MAT"),
            ("lbt_minutes", COLOR_LBT, "LBT"),
            ("eod_minutes", COLOR_EOD, "EOD"),
        ]:
            fig_dow.add_trace(go.Bar(
                x=dow_agg["요일명"],
                y=dow_agg[metric],
                name=name,
                marker_color=color,
            ))

        fig_dow.update_layout(
            height=350,
            template="plotly_dark",
            paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
            plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
            font_color=PLOTLY_DARK["font_color"],
            margin=dict(l=40, r=20, t=20, b=40),
            barmode="group",
            xaxis_title="요일",
            yaxis_title="대기시간 (분)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
            ),
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    # 4) 일별 상세 데이터 테이블
    st.markdown(sub_header("일별 상세 데이터"), unsafe_allow_html=True)

    daily_table = daily_agg[["date", "worker_count", "MAT", "LBT", "EOD", "TWT"]].copy()
    daily_table.columns = ["날짜", "작업자 수", "평균 MAT", "평균 LBT", "평균 EOD", "평균 TWT"]
    daily_table["날짜"] = daily_table["날짜"].astype(str).apply(format_date_full)
    daily_table = daily_table.sort_values("날짜", ascending=False)

    # 수치 포맷팅
    for col in ["평균 MAT", "평균 LBT", "평균 EOD", "평균 TWT"]:
        daily_table[col] = daily_table[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    st.dataframe(
        daily_table,
        use_container_width=True,
        hide_index=True,
    )

    # 5) 데이터 해석 (접힌 영역)
    with st.expander("데이터 해석", expanded=False):
        avg_mat = df["mat_minutes"].dropna().mean()
        avg_lbt = df["lbt_minutes"].dropna().mean()
        avg_eod = df["eod_minutes"].dropna().mean()

        interpretation = f"""
        **대기시간 분석 요약**

        - **출근 대기 (MAT)**: 평균 {avg_mat:.1f}분 - 타각기에서 FAB 작업층까지 소요시간
        - **중식 이동 (LBT)**: 평균 {avg_lbt:.1f}분 - 점심시간 FAB에서 타각기까지 이동시간
        - **퇴근 이동 (EOD)**: 평균 {avg_eod:.1f}분 - 퇴근 시 FAB에서 타각기까지 이동시간

        **해석 가이드**:
        - MAT가 길면 출근 시 호이스트 대기열 확인 필요
        - LBT/EOD가 높으면 점심/퇴근 시 병목 발생 가능성
        """
        st.markdown(interpretation)


# ============================================================================
# 서브탭 C: BP별 분석
# ============================================================================

def _render_bp_subtab(
    worker_df: pd.DataFrame,
    bp_df: pd.DataFrame,
    bp_filter: list[str] | None = None,
) -> None:
    """BP별 분석 서브탭 렌더링."""
    if worker_df.empty:
        st.info("대기시간 데이터가 없습니다.")
        return

    # 컨트롤 패널
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        all_dates = sorted(worker_df["date"].unique())
        if len(all_dates) >= 2:
            date_range = st.select_slider(
                "날짜 범위",
                options=all_dates,
                value=(all_dates[0], all_dates[-1]),
                format_func=format_date_full,
            )
            start_date, end_date = date_range[0], date_range[1]
        else:
            start_date = all_dates[0] if all_dates else ""
            end_date = all_dates[-1] if all_dates else ""

    with col2:
        all_companies = sorted(worker_df["company_name"].dropna().unique())
        selected_companies = st.multiselect(
            "업체 선택",
            options=all_companies,
            default=bp_filter if bp_filter else all_companies[:20],
        )

    with col3:
        segment_options = ["출근 (MAT)", "중식 (LBT)", "퇴근 (EOD)", "전체 (TWT)"]
        selected_segment = st.selectbox("구간", segment_options, index=3)

    # 데이터 필터링
    df_filtered = worker_df.copy()
    if start_date and end_date:
        df_filtered = df_filtered[(df_filtered["date"] >= start_date) & (df_filtered["date"] <= end_date)]
    if selected_companies:
        df_filtered = df_filtered[df_filtered["company_name"].isin(selected_companies)]

    if df_filtered.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    # 세그먼트 컬럼 매핑
    segment_map = {
        "출근 (MAT)": "mat_minutes",
        "중식 (LBT)": "lbt_minutes",
        "퇴근 (EOD)": "eod_minutes",
        "전체 (TWT)": "total_wait_minutes",
    }
    metric_col = segment_map[selected_segment]

    # 1) BP별 평균 대기시간 Horizontal Bar
    st.markdown(sub_header(f"BP별 평균 대기시간 ({selected_segment})"), unsafe_allow_html=True)

    bp_agg = df_filtered.groupby("company_name").agg({
        metric_col: lambda x: x.dropna().mean(),
        "user_no": "count",
    }).reset_index()
    bp_agg.columns = ["업체", "평균대기", "작업자수"]
    bp_agg = bp_agg.dropna(subset=["평균대기"])
    bp_agg = bp_agg.sort_values("평균대기", ascending=True).tail(30)

    # 평균 대비 색상 코딩
    overall_avg = bp_agg["평균대기"].mean()
    bp_agg["color"] = bp_agg["평균대기"].apply(
        lambda x: cfg.THEME_DANGER if x > overall_avg * 1.3 else (
            cfg.THEME_WARNING if x > overall_avg else cfg.THEME_SUCCESS
        )
    )

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bp_agg["평균대기"],
        y=bp_agg["업체"],
        orientation="h",
        marker_color=bp_agg["color"],
        text=bp_agg.apply(lambda r: f"{r['평균대기']:.1f}분 ({int(r['작업자수'])}명)", axis=1),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>평균: %{x:.1f}분<extra></extra>",
    ))

    # 평균선 추가
    fig_bar.add_vline(
        x=overall_avg,
        line_dash="dash",
        line_color="#7A8FA6",
        annotation_text=f"평균 {overall_avg:.1f}분",
        annotation_position="top",
    )

    fig_bar.update_layout(
        height=max(400, len(bp_agg) * 25),
        template="plotly_dark",
        paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
        plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
        font_color=PLOTLY_DARK["font_color"],
        margin=dict(l=150, r=80, t=30, b=40),
        xaxis_title="대기시간 (분)",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2) BP x 날짜 히트맵
    st.markdown(sub_header("BP x 날짜 대기시간 히트맵"), unsafe_allow_html=True)

    pivot_df = df_filtered.pivot_table(
        index="company_name",
        columns="date",
        values=metric_col,
        aggfunc=lambda x: x.dropna().mean(),
    )

    if not pivot_df.empty:
        # 날짜 라벨 포맷팅
        pivot_df.columns = [format_date_label(str(c)) for c in pivot_df.columns]

        # TWT 기준 정렬 (내림차순)
        pivot_df["_avg"] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values("_avg", ascending=True).drop(columns=["_avg"]).tail(25)

        fig_heat = px.imshow(
            pivot_df,
            color_continuous_scale="RdYlGn_r",  # 빨강 = 높음, 초록 = 낮음
            aspect="auto",
            labels={"color": "대기시간 (분)"},
        )
        fig_heat.update_layout(
            height=max(350, len(pivot_df) * 20),
            template="plotly_dark",
            paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
            plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
            font_color=PLOTLY_DARK["font_color"],
            margin=dict(l=150, r=20, t=30, b=50),
            xaxis_title="날짜",
            yaxis_title="",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("히트맵 데이터 없음")

    # 3) BP 상세 테이블
    st.markdown(sub_header("BP별 상세 통계"), unsafe_allow_html=True)

    bp_detail = df_filtered.groupby("company_name").agg({
        "user_no": "count",
        "mat_minutes": lambda x: x.dropna().mean(),
        "lbt_minutes": lambda x: x.dropna().mean(),
        "eod_minutes": lambda x: x.dropna().mean(),
        "total_wait_minutes": "max",
    }).reset_index()
    bp_detail.columns = ["업체명", "작업자 수", "평균 MAT", "평균 LBT", "평균 EOD", "최대 TWT"]
    # 평균 TWT = MAT + LBT + EOD (dropna 평균의 합산으로 통일)
    bp_detail["평균 TWT"] = bp_detail[["평균 MAT", "평균 LBT", "평균 EOD"]].fillna(0).sum(axis=1)
    bp_detail = bp_detail.sort_values("평균 TWT", ascending=False)

    # 수치 포맷팅
    for col in ["평균 MAT", "평균 LBT", "평균 EOD", "평균 TWT", "최대 TWT"]:
        bp_detail[col] = bp_detail[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    st.dataframe(
        bp_detail,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================================
# 서브탭 D: 원본 데이터
# ============================================================================

def _render_raw_subtab(
    worker_df: pd.DataFrame,
    bp_filter: list[str] | None = None,
) -> None:
    """원본 데이터 서브탭 렌더링."""
    if worker_df.empty:
        st.info("대기시간 데이터가 없습니다.")
        return

    # 필터 컨트롤
    col1, col2, col3 = st.columns(3)

    with col1:
        all_dates = sorted(worker_df["date"].unique())
        date_options = ["전체"] + [format_date_full(str(d)) for d in all_dates]
        selected_date = st.selectbox("날짜 선택", date_options)

    with col2:
        all_companies = sorted(worker_df["company_name"].dropna().unique())
        company_options = ["전체"] + list(all_companies)
        selected_company = st.selectbox("업체 선택", company_options)

    with col3:
        segment_filter = st.selectbox(
            "구간 필터",
            ["전체", "MAT 있음", "LBT 있음", "EOD 있음"],
        )

    # 필터 적용
    df = worker_df.copy()

    if selected_date != "전체":
        date_raw = selected_date.replace("-", "")
        df = df[df["date"] == date_raw]

    if selected_company != "전체":
        df = df[df["company_name"] == selected_company]

    if bp_filter:
        df = df[df["company_name"].isin(bp_filter)]

    if segment_filter == "MAT 있음":
        df = df[df["mat_minutes"].notna()]
    elif segment_filter == "LBT 있음":
        df = df[df["lbt_minutes"].notna()]
    elif segment_filter == "EOD 있음":
        df = df[df["eod_minutes"].notna()]

    if df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    # 테이블 표시
    st.markdown(f"**{len(df):,}건 데이터**")

    display_cols = [
        "date", "user_no", "user_name", "company_name",
        "mat_minutes", "lbt_minutes", "eod_minutes",
        "total_wait_minutes", "work_minutes", "transit_ratio",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()

    # 컬럼명 한글화
    col_rename = {
        "date": "날짜",
        "user_no": "작업자 ID",
        "user_name": "작업자명",
        "company_name": "업체명",
        "mat_minutes": "MAT (분)",
        "lbt_minutes": "LBT (분)",
        "eod_minutes": "EOD (분)",
        "total_wait_minutes": "TWT (분)",
        "work_minutes": "근무시간 (분)",
        "transit_ratio": "대기비율 (%)",
    }
    display_df = display_df.rename(columns=col_rename)

    # 날짜 포맷팅
    if "날짜" in display_df.columns:
        display_df["날짜"] = display_df["날짜"].astype(str).apply(format_date_full)

    # 작업자명 마스킹
    if "작업자명" in display_df.columns:
        try:
            from src.utils.anonymizer import mask_name
            display_df["작업자명"] = display_df["작업자명"].apply(mask_name)
        except ImportError:
            pass

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # CSV 다운로드
    csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSV 다운로드",
        data=csv_data,
        file_name="transit_data.csv",
        mime="text/csv",
    )

    # 계산 기준 안내
    with st.expander("지표 정의 및 계산 방법", expanded=False):
        st.markdown("""
### 지표 정의

| 지표 | 정의 | 계산 방법 |
|------|------|----------|
| **MAT** | Morning Arrival Time (출근 대기) | 타각기 출근 타각 -> FAB 작업층 최초 도착까지 전체 소요 시간 (분). 경로: 타각기 -> 야외(공사현장) -> 호이스트 -> FAB |
| **LBT** | Lunch Break Transit (중식 이동) | 중식(11~14시) FAB 작업층 -> 타각기 출문까지 소요 시간 (분) |
| **EOD** | End-of-Day Transit (퇴근 이동) | 퇴근(17시 이후) FAB 작업층 -> 타각기 출문까지 소요 시간 (분) |
| **TWT** | Total Wait Time (총 대기시간) | MAT + LBT + EOD 합계 |
| **TRR** | Transit Ratio (이동 비율) | TWT / 근무시간 (%) |

---

### 대기시간 추출 방법

1. **데이터 소스**: T-Ward BLE 태그 (1분 단위 위치 기록)
2. **공간 분류**: 11개 Locus (타각기, 호이스트, FAB 5F/6F/7F/RF, 화장실, 휴게실, 흡연실, 공사현장)
3. **MAT 계산**: 작업자별 해당 날짜 첫 번째 '타각기' 위치 시각 -> 첫 번째 'FAB 작업층' 위치 시각의 차이. 중간 경유지(야외 공사현장, 호이스트) 구간별 소요시간도 별도 추출하여 병목 구간 식별 가능
4. **LBT 계산**: 중식 시간대(11~14시)에 마지막 'FAB 작업층' -> 첫 번째 '타각기' 위치 시각의 차이
5. **EOD 계산**: 퇴근 시간대(17시 이후) 마지막 'FAB 작업층' -> 마지막 '타각기' 위치 시각의 차이
6. **BP별 집계**: 업체(company_name)별 평균/중앙값/최대값 산출

---

### 참고사항

- 각 지표가 `NaN`인 경우 해당 구간 이동 패턴이 감지되지 않았음을 의미
- MAT: 작업층 도착 전 타각기 기록이 없으면 계산 불가
- LBT/EOD: 해당 시간대에 FAB->타각기 이동이 없으면 계산 불가
        """)


# ============================================================================
# 메인 렌더 함수
# ============================================================================

def render_transit_tab(
    sector_id: str,
    date_str: str | None = None,
    bp_filter: list[str] | None = None,
) -> None:
    """
    작업자 대기/이동 시간 탭 진입점.

    Args:
        sector_id: Sector ID (e.g., "M15X_SKHynix")
        date_str: 선택된 날짜 (YYYYMMDD) - None이면 전체 기간
        bp_filter: BP(업체) 필터 리스트 - None이면 전체
    """
    st.markdown(section_header("작업자 대기/이동 시간 분석"), unsafe_allow_html=True)

    # 데이터 로드
    with st.spinner("대기시간 데이터 로드 중..."):
        worker_df, bp_df = _load_all_transit_data(sector_id)

    if worker_df.empty:
        st.info(
            "대기시간 데이터가 없습니다. "
            "[데이터 관리] 탭에서 전처리를 실행하면 transit 데이터가 생성됩니다."
        )

        # 처리된 날짜 확인
        processed = detect_processed_dates(sector_id)
        if processed:
            st.caption(f"처리된 날짜: {len(processed)}일")

            # transit 파일 존재 확인
            paths = cfg.get_sector_paths(sector_id)
            missing_transit = []
            for d in processed[:5]:  # 처음 5개만 체크
                t_path = paths["processed_dir"] / d / "transit.parquet"
                if not t_path.exists():
                    missing_transit.append(d)

            if missing_transit:
                st.warning(
                    f"transit.parquet 파일이 없는 날짜가 있습니다: {missing_transit[:3]}... "
                    "전처리 파이프라인에서 transit 분석이 포함되어 있는지 확인하세요."
                )
        return

    # 데이터 요약 표시
    n_dates = worker_df["date"].nunique()
    n_workers = len(worker_df)
    n_companies = worker_df["company_name"].nunique()

    st.caption(
        f"데이터 범위: {n_dates}일 | 총 {n_workers:,}건 | {n_companies}개 업체"
    )

    # 5개 서브탭 (AI Analysis 추가)
    tabs = st.tabs(["일별 분석", "기간별 분석", "BP별 분석", "원본 데이터", "AI Analysis"])

    with tabs[0]:
        _render_daily_single_subtab(worker_df, sector_id, bp_filter)

    with tabs[1]:
        _render_period_subtab(worker_df, bp_filter)

    with tabs[2]:
        _render_bp_subtab(worker_df, bp_df, bp_filter)

    with tabs[3]:
        _render_raw_subtab(worker_df, bp_filter)

    with tabs[4]:
        from src.dashboard.ai_analysis import render_transit_ai
        render_transit_ai(
            worker_transit=worker_df,
            cache_key=f"ai_transit_{sector_id}",
            sid=sector_id,
            anchor_date=date_str,
        )
