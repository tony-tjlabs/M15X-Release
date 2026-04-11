"""
Equipment Tab - Table Lift (OPR) Analysis
==========================================
M15X FAB 건설현장 테이블 리프트(고소작업대) 가동률 분석 대시보드.

고객 핵심 요구:
> "주 단위 BP별 가동률 평균" Data 요청 드립니다.
> PH4 와드 분출 시점부터 3월 2주차까지 자료가 필요합니다.

서브탭 구조:
- 일별 분석: 일일 가동 현황, 시간대별 동시 가동, 업체별/층별 분포
- 주간 현황: BP별 가동률 히트맵, 전체 추이
- 장비 상세: 마스터 테이블, 업체별 분포
- 층별 분포: 층별 장비 현황

Author: developer (agent)
Created: 2026-04-07
Updated: 2026-04-08 - 일별 분석 서브탭 추가
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

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
    badge,
)
from src.dashboard.date_utils import get_date_selector


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
OPR_GOOD_THRESHOLD = 0.70      # >= 70% = 정상 (초록)
OPR_WARN_THRESHOLD = 0.50      # >= 50% = 주의 (노랑)
# < 50% = 저조 (빨강)

# OPR Colorscale: 빨강 -> 노랑 -> 초록
OPR_COLORSCALE = [
    [0.0, COLORS["danger"]],    # 0% - 빨강
    [0.5, COLORS["warning"]],   # 50% - 노랑
    [1.0, COLORS["success"]],   # 100% - 초록
]


# ============================================================
# Data Loading (Cached)
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def _load_weekly_bp_opr(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """주간 BP별 가동률 데이터 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_weekly"] / "weekly_bp_opr.parquet"
    if not path.exists():
        logger.warning(f"Weekly BP OPR cache not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def _load_weekly_overall_opr(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """주간 전체 가동률 데이터 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_weekly"] / "weekly_overall_opr.parquet"
    if not path.exists():
        logger.warning(f"Weekly overall OPR cache not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def _load_weekly_floor_distribution(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """주간 층별 분포 데이터 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_weekly"] / "weekly_floor_distribution.parquet"
    if not path.exists():
        logger.warning(f"Weekly floor distribution cache not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def _load_equipment_master(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """장비 마스터 데이터 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_master"]
    if not path.exists():
        logger.warning(f"Equipment master cache not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="장비 일별 데이터 로드 중...", ttl=600)
def _load_daily_equipment_cache(date_str: str) -> pd.DataFrame:
    """사전 캐시된 장비 일별 시간대별 집계 로드 (매우 빠름)."""
    cache_path = cfg.EQUIPMENT_DIR / "M15X_SKHynix" / "daily" / f"{date_str}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    # 캐시 없으면 빈 DataFrame
    return pd.DataFrame()


@st.cache_data(show_spinner="동시 가동 데이터 로드 중...", ttl=600)
def _load_daily_concurrent_cache(date_str: str) -> pd.DataFrame:
    """사전 캐시된 동시 가동 장비 수 로드."""
    cache_path = cfg.EQUIPMENT_DIR / "M15X_SKHynix" / "daily" / f"{date_str}_concurrent.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame()


def _detect_equipment_dates(sector_id: str = "M15X_SKHynix") -> list[str]:
    """캐시된 장비 일별 데이터 날짜 목록."""
    cache_dir = cfg.EQUIPMENT_DIR / sector_id / "daily"
    if not cache_dir.exists():
        return []
    dates = sorted([
        f.stem for f in cache_dir.glob("*.parquet")
        if not f.stem.endswith("_concurrent") and len(f.stem) == 8
    ])
    return dates


# ============================================================
# Helpers
# ============================================================
def _get_week_date_range(year: int, week: int) -> tuple[str, str]:
    """
    ISO 주차의 날짜 범위 반환.

    Returns:
        (start_date_str, end_date_str) - "MM/DD" 형식
    """
    # ISO 주차 1의 첫 번째 목요일 기준
    from datetime import date
    jan4 = date(year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday())
    start_of_target_week = start_of_week1 + timedelta(weeks=week - 1)
    end_of_target_week = start_of_target_week + timedelta(days=6)

    return (
        start_of_target_week.strftime("%m/%d"),
        end_of_target_week.strftime("%m/%d")
    )


def _format_week_label(week_label: str) -> str:
    """
    주차 라벨을 읽기 쉬운 형식으로 변환.

    "2026-W07" -> "W07 (02/09~02/15)"
    """
    try:
        year, week_str = week_label.split("-W")
        week = int(week_str)
        start, end = _get_week_date_range(int(year), week)
        return f"W{week:02d} ({start}~{end})"
    except Exception:
        return week_label


def _get_opr_color(opr: float) -> str:
    """OPR 값에 따른 색상 반환."""
    if opr >= OPR_GOOD_THRESHOLD:
        return COLORS["success"]
    elif opr >= OPR_WARN_THRESHOLD:
        return COLORS["warning"]
    return COLORS["danger"]


def _get_opr_badge(opr: float) -> str:
    """OPR 값에 따른 뱃지 반환."""
    if opr >= OPR_GOOD_THRESHOLD:
        return badge(f"{opr*100:.1f}%", "success")
    elif opr >= OPR_WARN_THRESHOLD:
        return badge(f"{opr*100:.1f}%", "warning")
    return badge(f"{opr*100:.1f}%", "danger")


# ============================================================
# Main Entry Point
# ============================================================
def render_equipment_tab(sector_id: str = "M15X_SKHynix") -> None:
    """테이블 리프트 가동률 탭 메인 렌더링."""
    st.markdown(section_header("Table Lift Operating Rate Analysis"), unsafe_allow_html=True)

    # 데이터 로드
    weekly_bp_df = _load_weekly_bp_opr(sector_id)
    weekly_overall_df = _load_weekly_overall_opr(sector_id)
    weekly_floor_df = _load_weekly_floor_distribution(sector_id)
    master_df = _load_equipment_master(sector_id)

    # 데이터 없으면 안내 메시지
    if weekly_bp_df.empty:
        st.warning(
            "주간 가동률 데이터가 없습니다. "
            "**데이터 관리** 탭에서 장비 데이터 전처리를 실행해 주세요."
        )
        return

    # 서브탭 구성 (AI Analysis 추가)
    subtab_daily, subtab1, subtab2, subtab3, subtab_ai = st.tabs([
        "일별 분석",
        "주간 현황",
        "장비 상세",
        "층별 분포",
        "AI Analysis",
    ])

    with subtab_daily:
        _render_daily_subtab(master_df, sector_id)

    with subtab1:
        _render_weekly_subtab(weekly_bp_df, weekly_overall_df, master_df)

    with subtab2:
        _render_equipment_subtab(master_df, weekly_bp_df)

    with subtab3:
        _render_floor_subtab(weekly_floor_df, master_df)

    with subtab_ai:
        from src.dashboard.ai_analysis import render_equipment_ai
        render_equipment_ai(
            master_df=master_df,
            weekly_bp_df=weekly_bp_df,
            weekly_overall_df=weekly_overall_df,
            cache_key=f"ai_equipment_{sector_id}",
        )

    # -- 지표 정의 (접힌 상태) --
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.expander("지표 정의 및 계산 방법", expanded=False):
        st.markdown("""
### 가동률 지표 정의

| 지표 | 정의 | 계산 방법 |
|------|------|----------|
| **OPR** | Operating Rate (가동률) | 가동 분 / 전체 기록 분 x 100% |
| **가동시간** | 장비가 실제 작동한 시간 | ActiveSignal_count > 0인 분 합계 |
| **대기시간** | 장비가 대기 상태인 시간 | ActiveSignal_count = 0인 분 합계 |
| **주간 BP OPR** | 업체별 주간 평균 가동률 | 해당 주의 모든 장비-일별 OPR을 업체별로 평균 |

---

### 가동률 추출 방법

1. **데이터 소스**: T-Ward BLE 태그 (1분 단위 장비 위치 기록) + RSSI 센서 (초단위 감지)
2. **가동 판정**: ActiveSignal_count > 0인 분 = 가동 중 (장비가 움직이거나 작업 중)
3. **OPR (Operational Rate)**: 가동 분 / 전체 기록 분 x 100%
4. **근무시간**: 07:00~19:00 기준 (야간 데이터 제외)
5. **주간 BP별 OPR**: 해당 주의 모든 장비-일별 OPR을 업체별로 평균

---

### OPR 해석 기준

| OPR 범위 | 상태 | 의미 |
|---------|------|------|
| >= 70% | 정상 (초록) | 장비 활용도 양호 |
| 50~70% | 주의 (노랑) | 개선 여지 있음 |
| < 50% | 저조 (빨강) | 장비 유휴 또는 비효율 의심 |
        """)


# ============================================================
# Subtab 0: Daily Analysis (NEW)
# ============================================================
def _render_daily_subtab(
    master_df: pd.DataFrame,
    sector_id: str = "M15X_SKHynix",
) -> None:
    """일별 분석 서브탭."""

    # ─── 날짜 선택기 ───
    available_dates = st.session_state.get("equipment_available_dates")
    if available_dates is None:
        with st.spinner("날짜 범위 감지 중..."):
            available_dates = _detect_equipment_dates(sector_id)
            st.session_state["equipment_available_dates"] = available_dates

    if not available_dates:
        st.info("장비 위치 데이터가 없습니다.")
        return

    date_str = get_date_selector(available_dates, key="equip_daily_date")
    if not date_str:
        st.warning("날짜를 선택해 주세요.")
        return

    # ─── 데이터 로드 (사전 캐시 → 즉시 로드) ───
    hourly_cache = _load_daily_equipment_cache(date_str)
    concurrent_cache = _load_daily_concurrent_cache(date_str)

    if hourly_cache.empty:
        st.warning(
            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} 캐시가 없습니다. "
            "데이터 관리 탭에서 장비 데이터 캐시를 생성하세요."
        )
        return

    # ─── 캐시 기반 장비별 가동 통계 ───
    equipment_stats = hourly_cache.groupby("equipment_no").agg(
        total_minutes=("total_min", "sum"),
        active_minutes=("active_min", "sum"),
        equipment_name=("equipment_name", "first"),
        floor=("floor", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    ).reset_index()

    equipment_stats["opr"] = (
        equipment_stats["active_minutes"] / equipment_stats["total_minutes"]
    ).fillna(0)
    equipment_stats["active_hours"] = equipment_stats["active_minutes"] / 60

    # 마스터 테이블과 조인하여 업체 정보 추가
    if not master_df.empty and "equipment_no" in master_df.columns:
        equipment_stats = equipment_stats.merge(
            master_df[["equipment_no", "company_name"]],
            on="equipment_no",
            how="left"
        )

        equipment_stats["company_name"] = equipment_stats["company_name"].fillna("Unknown")
    else:
        equipment_stats["company_name"] = "Unknown"

    # work_hours 대체: 히트맵 등에서 사용할 시간대별 캐시
    work_hours = hourly_cache  # 이미 시간대별 집계된 데이터

    # ─── 시간대별 동시 가동 장비 수 (캐시 기반) ───
    if not concurrent_cache.empty:
        concurrent_by_minute = concurrent_cache.set_index("minute")["concurrent_count"]
        concurrent_by_hour = concurrent_cache.groupby("hour")["concurrent_count"].mean()
    else:
        concurrent_by_minute = pd.Series(dtype=int)
        concurrent_by_hour = pd.Series(dtype=float)

    # ── OPR >= 99% 센서 이상 장비 필터링 (통계 왜곡 방지) ──
    opr_anomaly_threshold = getattr(cfg, "OPR_SENSOR_ANOMALY_THRESHOLD", 0.99)
    anomaly_mask = equipment_stats["opr"] >= opr_anomaly_threshold
    n_anomalous = int(anomaly_mask.sum())
    if n_anomalous > 0:
        equipment_stats_filtered = equipment_stats[~anomaly_mask]
        st.info(
            f"통계에서 **{n_anomalous}대** 제외됨 "
            f"(OPR >= {opr_anomaly_threshold*100:.0f}%, 센서 이상 의심). "
            f"상세 내용은 아래 '센서 이상 의심 장비' 참조."
        )
    else:
        equipment_stats_filtered = equipment_stats

    # ─── KPI 계산 (필터링된 통계 사용) ───
    total_equipment = len(equipment_stats_filtered)
    avg_active_hours = equipment_stats_filtered["active_hours"].mean() if total_equipment > 0 else 0
    avg_opr = equipment_stats_filtered["opr"].mean() if total_equipment > 0 else 0
    peak_concurrent = concurrent_by_minute.max() if not concurrent_by_minute.empty else 0

    # ─── Row 1: KPI 카드 4열 ───
    date_display = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    st.markdown(f"##### {date_display} 일일 현황")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            metric_card("가동 장비 수", f"{total_equipment:,}대"),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            metric_card(
                "평균 일일 가동 시간",
                f"{avg_active_hours:.1f}시간",
                color=COLORS["accent"]
            ),
            unsafe_allow_html=True,
        )

    with col3:
        opr_color = _get_opr_color(avg_opr)
        st.markdown(
            metric_card(
                "평균 OPR",
                f"{avg_opr*100:.1f}%",
                color=opr_color
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            metric_card(
                "피크 동시 가동",
                f"{int(peak_concurrent):,}대",
                color=COLORS["success"]
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Always-Active Equipment Warning ───
    # 전 시간대 100% 가동은 센서 이상 가능성이 높음
    always_active = equipment_stats[equipment_stats["opr"] >= 0.99]
    if not always_active.empty:
        n_suspect = len(always_active)
        suspect_names = always_active["equipment_name"].tolist()
        names_display = ", ".join(suspect_names[:5])
        if len(suspect_names) > 5:
            names_display += f" 외 {len(suspect_names) - 5}대"
        st.warning(
            f"**센서 이상 의심 장비 {n_suspect}대 감지** -- "
            f"가동률 99% 이상으로 비정상적 수치입니다. "
            f"T-Ward 진동 센서 오작동 또는 충전기 위 진동 환경이 원인일 수 있습니다.\n\n"
            f"해당 장비: {names_display}"
        )
        with st.expander(f"센서 이상 의심 장비 상세 ({n_suspect}대)", expanded=False):
            suspect_display = always_active[[
                "equipment_no", "equipment_name", "company_name", "floor",
                "active_minutes", "total_minutes",
            ]].copy()
            suspect_display["opr_pct"] = (always_active["opr"] * 100).round(1)
            suspect_display.columns = [
                "장비번호", "장비명", "업체명", "층",
                "가동분", "전체분", "OPR(%)"
            ]
            st.dataframe(suspect_display, use_container_width=True, hide_index=True)

    # ─── Row 2: 시간대별 동시 가동 장비 수 Area Chart ───
    st.markdown("##### 시간대별 동시 가동 장비 수")

    if not concurrent_by_minute.empty:
        # 분 단위 데이터를 시간대별로 평균하여 표시
        hourly_concurrent = concurrent_by_minute.reset_index()
        hourly_concurrent.columns = ["time", "concurrent_count"]
        hourly_concurrent["hour"] = hourly_concurrent["time"].dt.hour + hourly_concurrent["time"].dt.minute / 60

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_concurrent["time"],
            y=hourly_concurrent["concurrent_count"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0, 174, 239, 0.3)",
            line=dict(color=COLORS["accent"], width=2),
            hovertemplate="<b>%{x|%H:%M}</b><br>동시 가동: %{y}대<extra></extra>",
        ))

        fig.update_layout(
            height=280,
            xaxis_title="시간",
            yaxis_title="동시 가동 장비 수",
            xaxis=dict(
                tickformat="%H:%M",
                dtick=3600000 * 2,  # 2시간 간격
            ),
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor="#111820",
            font_color=COLORS["text"],
            margin=dict(l=50, r=20, t=20, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("가동 데이터가 없습니다.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 3: 2열 (업체별 평균 가동 시간 + 층별 분포) ───
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 업체별 평균 가동 시간")

        company_stats = equipment_stats.groupby("company_name").agg(
            equipment_count=("equipment_no", "count"),
            avg_active_hours=("active_hours", "mean"),
            avg_opr=("opr", "mean"),
        ).reset_index()
        company_stats = company_stats.sort_values("avg_active_hours", ascending=True)

        if len(company_stats) > 0:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=company_stats["avg_active_hours"],
                y=company_stats["company_name"],
                orientation="h",
                marker_color=company_stats["avg_opr"].apply(_get_opr_color),
                text=company_stats.apply(
                    lambda r: f"{r['avg_active_hours']:.1f}h ({r['equipment_count']}대)",
                    axis=1
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "평균 가동: %{x:.1f}시간<extra></extra>"
                ),
            ))

            fig.update_layout(
                height=max(300, len(company_stats) * 28),
                xaxis_title="평균 가동 시간",
                yaxis_title="",
                xaxis_range=[0, max(company_stats["avg_active_hours"].max() * 1.3, 2)],
                paper_bgcolor=COLORS["card_bg"],
                plot_bgcolor="#111820",
                font_color=COLORS["text"],
                margin=dict(l=120, r=80, t=20, b=50),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("업체별 데이터가 없습니다.")

    with col_right:
        st.markdown("##### 층별 가동 장비 분포")

        floor_order = ["5F", "6F", "7F", "RF"]
        floor_stats = equipment_stats.groupby("floor").agg(
            equipment_count=("equipment_no", "count"),
            avg_opr=("opr", "mean"),
            avg_active_hours=("active_hours", "mean"),
        ).reset_index()

        # 층 순서 정렬
        floor_stats["floor_order"] = floor_stats["floor"].apply(
            lambda x: floor_order.index(x) if x in floor_order else 99
        )
        floor_stats = floor_stats.sort_values("floor_order")

        if len(floor_stats) > 0:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=floor_stats["floor"],
                y=floor_stats["equipment_count"],
                marker_color=floor_stats["avg_opr"].apply(_get_opr_color),
                text=floor_stats.apply(
                    lambda r: f"{r['equipment_count']}대<br>OPR {r['avg_opr']*100:.0f}%",
                    axis=1
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "장비 수: %{y}대<extra></extra>"
                ),
            ))

            fig.update_layout(
                height=300,
                xaxis_title="층",
                yaxis_title="장비 수",
                yaxis_range=[0, max(floor_stats["equipment_count"].max() * 1.3, 5)],
                paper_bgcolor=COLORS["card_bg"],
                plot_bgcolor="#111820",
                font_color=COLORS["text"],
                margin=dict(l=50, r=20, t=20, b=50),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("층별 데이터가 없습니다.")

    # ─── Row 4: 상세 데이터 테이블 ───
    with st.expander("장비별 상세 데이터", expanded=False):
        display_df = equipment_stats[[
            "equipment_no", "equipment_name", "company_name", "floor",
            "active_hours", "opr", "total_minutes", "active_minutes"
        ]].copy()

        display_df = display_df.sort_values("active_hours", ascending=False)
        display_df["opr_pct"] = (display_df["opr"] * 100).round(1)
        display_df = display_df.drop(columns=["opr"])

        display_df.columns = [
            "장비번호", "장비명", "업체명", "층",
            "가동시간(h)", "총기록(분)", "가동기록(분)", "OPR(%)"
        ]

        st.dataframe(
            display_df.head(100),
            use_container_width=True,
            height=400,
        )

        st.caption(f"총 {len(equipment_stats):,}대, 상위 100개 표시")

        # CSV 다운로드
        csv = equipment_stats.to_csv(index=False)
        st.download_button(
            label="CSV 다운로드",
            data=csv,
            file_name=f"equipment_daily_{date_str}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ─── Row 5: 업체별/층별 시간대별 가동 히트맵 ───
    st.markdown("##### 상세 시간대별 가동 현황")

    col_bp, col_floor = st.columns(2)

    with col_bp:
        st.markdown("###### 업체별 장비 시간대별 가동")

        # 업체 선택
        bp_list = sorted(equipment_stats["company_name"].unique())
        if bp_list:
            selected_bp = st.selectbox(
                "업체 선택",
                bp_list,
                key="equip_bp_select",
            )

            # 해당 업체 장비 필터
            bp_equip_list = equipment_stats[
                equipment_stats["company_name"] == selected_bp
            ]["equipment_no"].tolist()

            bp_hourly_cache = work_hours[
                work_hours["equipment_no"].isin(bp_equip_list)
            ].copy()

            if not bp_hourly_cache.empty:
                # 캐시는 이미 시간대별 집계: equipment_no, hour, total_min, active_min, opr
                bp_hourly = bp_hourly_cache.rename(columns={
                    "total_min": "total_minutes",
                    "active_min": "active_minutes",
                })
                bp_hourly["active_pct"] = (
                    bp_hourly["active_minutes"] / bp_hourly["total_minutes"] * 100
                ).fillna(0)

                # 피벗 테이블
                bp_pivot = bp_hourly.pivot_table(
                    index="equipment_name",
                    columns="hour",
                    values="active_pct",
                    fill_value=0
                )

                # 시간 범위 (07~18)
                hour_cols = [h for h in range(7, 19) if h in bp_pivot.columns]
                if not hour_cols:
                    hour_cols = bp_pivot.columns.tolist()
                bp_pivot = bp_pivot.reindex(columns=sorted(hour_cols), fill_value=0)

                # 장비명 정렬 (평균 가동률 내림차순)
                bp_pivot["avg"] = bp_pivot.mean(axis=1)
                bp_pivot = bp_pivot.sort_values("avg", ascending=True)
                bp_pivot = bp_pivot.drop(columns=["avg"])

                # 히트맵
                hour_labels = [f"{h:02d}:00" for h in bp_pivot.columns]
                z_values = bp_pivot.values.round(0)

                # 장비명 축약 (긴 이름 처리)
                equip_labels = [
                    n[-20:] if len(n) > 20 else n
                    for n in bp_pivot.index.tolist()
                ]

                fig_bp = go.Figure(data=go.Heatmap(
                    z=z_values,
                    x=hour_labels,
                    y=equip_labels,
                    colorscale=[
                        [0.0, COLORS["card_bg"]],
                        [0.5, COLORS["warning"]],
                        [1.0, COLORS["success"]],
                    ],
                    zmin=0,
                    zmax=100,
                    text=z_values,
                    texttemplate="%{text:.0f}",
                    textfont={"size": 9},
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "%{x}<br>"
                        "가동률: %{z:.0f}%<extra></extra>"
                    ),
                    colorbar=dict(
                        title=dict(text="%", side="right"),
                        ticksuffix="%",
                        len=0.5,
                    ),
                    showscale=False,
                ))

                fig_bp.update_layout(
                    height=max(200, len(bp_pivot) * 22 + 80),
                    xaxis_title="시간",
                    yaxis_title="",
                    paper_bgcolor=COLORS["card_bg"],
                    plot_bgcolor="#111820",
                    font_color=COLORS["text"],
                    margin=dict(l=140, r=10, t=10, b=40),
                )

                st.plotly_chart(fig_bp, use_container_width=True)
                st.caption(f"{selected_bp}: {len(bp_pivot)}대 장비")
            else:
                st.info(f"{selected_bp} 장비 데이터가 없습니다.")
        else:
            st.info("업체 데이터가 없습니다.")

    with col_floor:
        st.markdown("###### 층별 장비 시간대별 가동")

        # 층 선택
        floor_list = ["5F", "6F", "7F", "RF"]
        available_floors = [
            f for f in floor_list
            if f in work_hours["floor"].unique()
        ]

        if available_floors:
            selected_floor = st.selectbox(
                "층 선택",
                available_floors,
                key="equip_floor_select",
            )

            # 해당 층 장비 필터
            floor_equip_list = equipment_stats[
                equipment_stats["floor"] == selected_floor
            ]["equipment_no"].tolist()

            floor_work_hours = work_hours[
                work_hours["equipment_no"].isin(floor_equip_list)
            ].copy()

            if not floor_work_hours.empty:
                # 캐시는 이미 시간대별 집계: equipment_no, hour, total_min, active_min, opr
                floor_hourly = floor_work_hours.rename(columns={
                    "total_min": "total_minutes",
                    "active_min": "active_minutes",
                })
                floor_hourly["active_pct"] = (
                    floor_hourly["active_minutes"] / floor_hourly["total_minutes"] * 100
                ).fillna(0)

                # 피벗 테이블
                floor_pivot = floor_hourly.pivot_table(
                    index="equipment_name",
                    columns="hour",
                    values="active_pct",
                    fill_value=0
                )

                # 시간 범위 (07~18)
                hour_cols = [h for h in range(7, 19) if h in floor_pivot.columns]
                if not hour_cols:
                    hour_cols = floor_pivot.columns.tolist()
                floor_pivot = floor_pivot.reindex(columns=sorted(hour_cols), fill_value=0)

                # 장비명 정렬 (평균 가동률 내림차순)
                floor_pivot["avg"] = floor_pivot.mean(axis=1)
                floor_pivot = floor_pivot.sort_values("avg", ascending=True)
                floor_pivot = floor_pivot.drop(columns=["avg"])

                # 장비가 많으면 상위 20개만 표시
                if len(floor_pivot) > 20:
                    floor_pivot = floor_pivot.tail(20)
                    show_limit_msg = True
                else:
                    show_limit_msg = False

                # 히트맵
                hour_labels = [f"{h:02d}:00" for h in floor_pivot.columns]
                z_values = floor_pivot.values.round(0)

                # 장비명 축약
                equip_labels = [
                    n[-20:] if len(n) > 20 else n
                    for n in floor_pivot.index.tolist()
                ]

                fig_floor = go.Figure(data=go.Heatmap(
                    z=z_values,
                    x=hour_labels,
                    y=equip_labels,
                    colorscale=[
                        [0.0, COLORS["card_bg"]],
                        [0.5, COLORS["warning"]],
                        [1.0, COLORS["success"]],
                    ],
                    zmin=0,
                    zmax=100,
                    text=z_values,
                    texttemplate="%{text:.0f}",
                    textfont={"size": 9},
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "%{x}<br>"
                        "가동률: %{z:.0f}%<extra></extra>"
                    ),
                    showscale=False,
                ))

                fig_floor.update_layout(
                    height=max(200, len(floor_pivot) * 22 + 80),
                    xaxis_title="시간",
                    yaxis_title="",
                    paper_bgcolor=COLORS["card_bg"],
                    plot_bgcolor="#111820",
                    font_color=COLORS["text"],
                    margin=dict(l=140, r=10, t=10, b=40),
                )

                st.plotly_chart(fig_floor, use_container_width=True)
                caption = f"{selected_floor}: {len(floor_pivot)}대 장비"
                if show_limit_msg:
                    caption += " (상위 20개 표시)"
                st.caption(caption)
            else:
                st.info(f"{selected_floor} 장비 데이터가 없습니다.")
        else:
            st.info("층 데이터가 없습니다.")

    st.markdown("---")

    # ─── Row 6: 요일별 전체 가동률 추이 ───
    st.markdown("##### 요일별 가동률 패턴")

    # weekly_bp_opr 데이터에서 요일별 패턴 추출
    weekly_bp_df_dow = _load_weekly_bp_opr(sector_id)

    if not weekly_bp_df_dow.empty and "week_label" in weekly_bp_df_dow.columns:
        # 각 날짜별 OPR을 계산하기 위해 weekly_overall 사용
        weekly_overall_df_dow = _load_weekly_overall_opr(sector_id)

        if not weekly_overall_df_dow.empty:
            # 현재 선택 날짜의 요일 강조 + 일반 패턴 표시
            from datetime import datetime as dt

            # 현재 날짜의 요일
            current_date = dt.strptime(date_str, "%Y%m%d")
            current_dow = current_date.strftime("%a")
            dow_map = {
                "Mon": "월", "Tue": "화", "Wed": "수",
                "Thu": "목", "Fri": "금", "Sat": "토", "Sun": "일"
            }
            current_dow_kr = dow_map.get(current_dow, current_dow)

            # 선택된 날짜의 평균 OPR
            selected_date_opr = avg_opr * 100

            # 주간 평균 OPR (비교용)
            overall_avg_opr = weekly_overall_df_dow["avg_opr"].mean() * 100 if not weekly_overall_df_dow.empty else 0

            col_dow1, col_dow2 = st.columns([1, 2])

            with col_dow1:
                delta_val = selected_date_opr - overall_avg_opr
                delta_str = f"{'+' if delta_val >= 0 else ''}{delta_val:.1f}%p vs 주간 평균"
                delta_up = delta_val >= 0

                st.markdown(
                    metric_card(
                        f"{current_dow_kr}요일 ({date_display})",
                        f"{selected_date_opr:.1f}%",
                        delta=delta_str,
                        delta_up=delta_up,
                        color=_get_opr_color(avg_opr)
                    ),
                    unsafe_allow_html=True,
                )

            with col_dow2:
                # 요일별 패턴 정보 (사용 가능한 날짜 기반)
                dow_pattern = []
                for d_str in available_dates:
                    try:
                        d_dt = dt.strptime(d_str, "%Y%m%d")
                        dow_pattern.append({
                            "date": d_str,
                            "dow": d_dt.weekday(),
                            "dow_name": dow_map.get(d_dt.strftime("%a"), d_dt.strftime("%a"))
                        })
                    except Exception:
                        continue

                if dow_pattern:
                    dow_df = pd.DataFrame(dow_pattern)
                    dow_counts = dow_df.groupby("dow_name").size().reset_index(name="count")
                    dow_order = ["월", "화", "수", "목", "금", "토", "일"]
                    dow_counts["order"] = dow_counts["dow_name"].apply(
                        lambda x: dow_order.index(x) if x in dow_order else 99
                    )
                    dow_counts = dow_counts.sort_values("order")

                    # 데이터 분포 표시
                    st.markdown(f"**데이터 분포** (총 {len(available_dates)}일)")
                    dow_str = " | ".join([
                        f"{r['dow_name']}: {r['count']}일"
                        for _, r in dow_counts.iterrows()
                    ])
                    st.caption(dow_str)

                    # 주간 평균 기준선 정보
                    st.caption(
                        f"전체 주간 평균 OPR: {overall_avg_opr:.1f}% "
                        f"(최근 {len(weekly_overall_df_dow)}주)"
                    )
    else:
        st.info("주간 가동률 데이터가 없어 요일별 패턴을 표시할 수 없습니다.")

    # ─── 지표 정의 토글 ───
    with st.expander("지표 정의", expanded=False):
        st.markdown("""
        | 지표 | 정의 |
        |------|------|
        | **OPR (Operating Rate)** | 가동률 = 가동 시간 / 근무 시간 (07:00~19:00) |
        | **가동 상태** | active_count > 0 이면 가동 중으로 판정 |
        | **피크 동시 가동** | 분 단위 최대 동시 가동 장비 수 |
        | **시간대별 가동률** | 해당 시간대 내 가동 분 / 총 기록 분 x 100% |
        """)


# ============================================================
# Subtab 1: Weekly Overview
# ============================================================
def _render_weekly_subtab(
    weekly_bp_df: pd.DataFrame,
    weekly_overall_df: pd.DataFrame,
    master_df: pd.DataFrame,
) -> None:
    """주간 현황 서브탭."""

    if weekly_overall_df.empty:
        st.info("주간 가동률 데이터가 없습니다.")
        return

    # 주차 선택기
    all_weeks = weekly_overall_df["week_label"].unique().tolist()
    all_weeks_sorted = sorted(all_weeks)
    week_options = {_format_week_label(w): w for w in reversed(all_weeks_sorted)}
    selected_week_display = st.selectbox(
        "주차 선택",
        list(week_options.keys()),
        key="equip_weekly_selector",
    )
    selected_week = week_options[selected_week_display]

    latest_week = selected_week
    latest_week_display = _format_week_label(latest_week)

    # 선택 주차 데이터
    _filtered = weekly_overall_df[weekly_overall_df["week_label"] == latest_week]
    latest_overall = _filtered.iloc[0].to_dict() if not _filtered.empty else {}
    latest_bp = weekly_bp_df[weekly_bp_df["week_label"] == latest_week] if not weekly_bp_df.empty else pd.DataFrame()

    # ─── Row 1: KPI 카드 4열 ───
    st.markdown(f"##### {latest_week_display} 현황")

    # 전체 등록 장비 수 vs 이번 주 가동 장비 수 표시
    total_registered = len(master_df)
    active_this_week = int(latest_overall.get("total_equipment", 0))
    active_pct = active_this_week / total_registered * 100 if total_registered > 0 else 0

    st.markdown(
        f"<div style='background:#1A2332; border:1px solid #2A3A50; border-radius:8px; "
        f"padding:10px 16px; margin-bottom:12px; font-size:0.88rem; color:#9AB5D4;'>"
        f"전체 등록 장비: <b style='color:#D5E5FF;'>{total_registered:,}대</b>"
        f" &nbsp;|&nbsp; 이번 주 가동 장비: <b style='color:#00AEEF;'>{active_this_week:,}대</b>"
        f" ({active_pct:.1f}%)"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            metric_card("이번 주 가동 장비", f"{active_this_week:,}대",
                        delta=f"/{total_registered:,}대"),
            unsafe_allow_html=True,
        )

    with col2:
        avg_opr = latest_overall.get("avg_opr", 0)
        opr_color = _get_opr_color(avg_opr)
        st.markdown(
            metric_card("이번 주 평균 OPR", f"{avg_opr*100:.1f}%", color=opr_color),
            unsafe_allow_html=True,
        )

    with col3:
        if not latest_bp.empty:
            top_bp = latest_bp.loc[latest_bp["avg_opr"].idxmax()]
            top_bp_name = top_bp["company_name"][:12] + "..." if len(top_bp["company_name"]) > 12 else top_bp["company_name"]
            top_bp_opr = top_bp["avg_opr"]
        else:
            top_bp_name = "N/A"
            top_bp_opr = 0
        st.markdown(
            metric_card(
                f"최고 OPR BP",
                f"{top_bp_opr*100:.1f}%",
                delta=top_bp_name,
                delta_up=True,
                color=COLORS["success"]
            ),
            unsafe_allow_html=True,
        )

    with col4:
        if not latest_bp.empty:
            bottom_bp = latest_bp.loc[latest_bp["avg_opr"].idxmin()]
            bottom_bp_name = bottom_bp["company_name"][:12] + "..." if len(bottom_bp["company_name"]) > 12 else bottom_bp["company_name"]
            bottom_bp_opr = bottom_bp["avg_opr"]
        else:
            bottom_bp_name = "N/A"
            bottom_bp_opr = 0
        st.markdown(
            metric_card(
                f"최저 OPR BP",
                f"{bottom_bp_opr*100:.1f}%",
                delta=bottom_bp_name,
                delta_up=False,
                color=COLORS["danger"]
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 2: 주간 전체 가동률 추이 Bar Chart ───
    st.markdown("##### 주간 전체 가동률 추이")

    if not weekly_overall_df.empty:
        overall_chart_df = weekly_overall_df.copy()
        overall_chart_df["week_display"] = overall_chart_df["week_label"].apply(_format_week_label)
        overall_chart_df["opr_pct"] = overall_chart_df["avg_opr"] * 100
        overall_chart_df["color"] = overall_chart_df["avg_opr"].apply(_get_opr_color)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=overall_chart_df["week_display"],
            y=overall_chart_df["opr_pct"],
            marker_color=overall_chart_df["color"],
            text=overall_chart_df["opr_pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>평균 OPR: %{y:.1f}%<extra></extra>",
        ))

        fig.update_layout(
            height=280,
            xaxis_title="주차",
            yaxis_title="평균 가동률 (%)",
            yaxis_range=[0, max(overall_chart_df["opr_pct"].max() * 1.2, 30)],
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor="#111820",
            font_color=COLORS["text"],
            margin=dict(l=50, r=20, t=20, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 3: 주간 BP별 가동률 히트맵 (핵심) ───
    st.markdown("##### 주간 BP별 가동률 히트맵")
    st.caption("X축: 주차, Y축: BP(업체), 색상: 가동률 (%)")

    if not weekly_bp_df.empty:
        # 피벗 테이블 생성
        pivot_df = weekly_bp_df.pivot_table(
            index="company_name",
            columns="week_label",
            values="avg_opr",
            aggfunc="mean"
        )

        # 주차 순서 정렬
        week_order = sorted(pivot_df.columns.tolist())
        pivot_df = pivot_df[week_order]

        # 전체 기간 평균으로 BP 정렬 (내림차순)
        pivot_df["avg_all"] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values("avg_all", ascending=True)
        pivot_df = pivot_df.drop(columns=["avg_all"])

        # 주차 라벨 포맷팅
        week_labels = [_format_week_label(w) for w in week_order]

        # 히트맵 생성
        z_values = (pivot_df.values * 100).round(1)  # 퍼센트

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=week_labels,
            y=pivot_df.index.tolist(),
            colorscale=OPR_COLORSCALE,
            zmin=0,
            zmax=100,
            text=z_values,
            texttemplate="%{text:.0f}%",
            textfont={"size": 10},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}<br>"
                "가동률: %{z:.1f}%<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text="OPR (%)", side="right"),
                ticksuffix="%",
            ),
        ))

        fig.update_layout(
            height=max(400, len(pivot_df) * 25),
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor="#111820",
            font_color=COLORS["text"],
            margin=dict(l=150, r=20, t=20, b=50),
            xaxis_title="주차",
            yaxis_title="",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 4: BP별 평균 가동률 Horizontal Bar (전체 기간) ───
    st.markdown("##### BP별 평균 가동률 (전체 기간)")

    if not weekly_bp_df.empty:
        bp_avg = weekly_bp_df.groupby("company_name").agg({
            "avg_opr": "mean",
            "equipment_count": "max",
        }).reset_index()
        bp_avg = bp_avg.sort_values("avg_opr", ascending=True)
        bp_avg["opr_pct"] = bp_avg["avg_opr"] * 100
        bp_avg["color"] = bp_avg["avg_opr"].apply(_get_opr_color)

        # ⚠️ 100% 또는 99% 이상 업체 경고
        suspicious_bp = bp_avg[bp_avg["opr_pct"] >= 99.0]
        if not suspicious_bp.empty:
            names = ", ".join(suspicious_bp["company_name"].tolist())
            st.warning(
                f"⚠️ **데이터 이상 의심**: {names} 업체의 평균 가동률이 99% 이상입니다. "
                f"장비가 24시간 내내 가동 기록되거나 센서 오류 가능성이 있습니다. "
                f"해당 업체의 원본 데이터를 검토하세요. "
                f"(100% 가동률 데이터는 통계에서 제외되었으나 주간 평균에 영향을 줄 수 있습니다.)"
            )

        # 전체 평균 기준선
        overall_avg = bp_avg["avg_opr"].mean()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=bp_avg["opr_pct"],
            y=bp_avg["company_name"],
            orientation="h",
            marker_color=bp_avg["color"],
            text=bp_avg.apply(
                lambda r: f"{r['opr_pct']:.1f}% ({r['equipment_count']:.0f}대)",
                axis=1
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "평균 OPR: %{x:.1f}%<extra></extra>"
            ),
        ))

        # 전체 평균 기준선
        fig.add_vline(
            x=overall_avg * 100,
            line_dash="dash",
            line_color=COLORS["accent"],
            annotation_text=f"전체 평균 {overall_avg*100:.1f}%",
            annotation_position="top",
            annotation_font_color=COLORS["accent"],
        )

        # 99% 이상 이상값 경고선
        if bp_avg["opr_pct"].max() >= 99:
            fig.add_vline(
                x=99,
                line_dash="dot",
                line_color="rgba(255,80,80,0.6)",
                annotation_text="⚠️ 99% 이상 = 데이터 이상 의심",
                annotation_position="bottom right",
                annotation_font_color="rgba(255,120,120,0.9)",
                annotation_font_size=11,
            )

        fig.update_layout(
            height=max(400, len(bp_avg) * 30),
            xaxis_title="평균 가동률 (%)",
            yaxis_title="",
            xaxis_range=[0, max(bp_avg["opr_pct"].max() * 1.2, 50)],
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor="#111820",
            font_color=COLORS["text"],
            margin=dict(l=150, r=80, t=20, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)

    # ─── Row 5: 데이터 테이블 ───
    with st.expander("상세 데이터 테이블", expanded=False):
        if not weekly_bp_df.empty:
            # 피벗 형태로 표시
            display_pivot = weekly_bp_df.pivot_table(
                index="company_name",
                columns="week_label",
                values="avg_opr",
                aggfunc="mean"
            )

            # 전체 평균 추가
            display_pivot["전체 평균"] = display_pivot.mean(axis=1)
            display_pivot = display_pivot.sort_values("전체 평균", ascending=False)

            # 퍼센트로 포맷팅
            display_formatted = display_pivot.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
            display_formatted.index.name = "업체명"

            st.dataframe(display_formatted, use_container_width=True)

            # CSV 다운로드
            csv = display_pivot.to_csv()
            st.download_button(
                label="CSV 다운로드",
                data=csv,
                file_name="weekly_bp_opr.csv",
                mime="text/csv",
            )


# ============================================================
# Subtab 2: Equipment Detail
# ============================================================
def _render_equipment_subtab(
    master_df: pd.DataFrame,
    weekly_bp_df: pd.DataFrame,
) -> None:
    """장비 상세 서브탭."""

    if master_df.empty:
        st.info("장비 마스터 데이터가 없습니다.")
        return

    # ─── Row 1: KPI 카드 4열 ───
    st.markdown("##### 장비 현황 요약")

    total_count = len(master_df)
    working_count = len(master_df[master_df["status"] == "working"])
    out_count = total_count - working_count
    company_count = master_df["company_name"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            metric_card("전체 장비", f"{total_count:,}대"),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            metric_card("현장 가동", f"{working_count:,}대", color=COLORS["success"]),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            metric_card("반출 완료", f"{out_count:,}대", color=COLORS["text_muted"]),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            metric_card("활성 업체", f"{company_count:,}개"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 2: 2열 (도넛 + 업체별 Bar) ───
    col1, col2 = st.columns([4, 8])

    with col1:
        st.markdown("##### 장비 상태 분포")

        # 도넛 차트
        status_counts = master_df["status"].value_counts()
        labels = ["가동", "반출"]
        values = [
            status_counts.get("working", 0),
            status_counts.get("out", 0),
        ]
        colors = [COLORS["success"], COLORS["text_muted"]]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker_colors=colors,
            textinfo="percent+value",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value}대 (%{percent})<extra></extra>",
        )])

        fig.update_layout(
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor=COLORS["card_bg"],
            font_color=COLORS["text"],
            margin=dict(l=20, r=20, t=20, b=60),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### 업체별 장비 현황")

        # 업체별 장비 수 (가동/반출 스택)
        company_status = master_df.groupby(["company_name", "status"]).size().unstack(fill_value=0)
        company_status["total"] = company_status.sum(axis=1)
        company_status = company_status.sort_values("total", ascending=True)

        fig = go.Figure()

        if "working" in company_status.columns:
            fig.add_trace(go.Bar(
                name="가동",
                x=company_status.get("working", 0),
                y=company_status.index,
                orientation="h",
                marker_color=COLORS["success"],
            ))

        if "out" in company_status.columns:
            fig.add_trace(go.Bar(
                name="반출",
                x=company_status.get("out", 0),
                y=company_status.index,
                orientation="h",
                marker_color=COLORS["text_muted"],
            ))

        fig.update_layout(
            barmode="stack",
            height=max(300, len(company_status) * 25),
            xaxis_title="장비 수",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor="#111820",
            font_color=COLORS["text"],
            margin=dict(l=150, r=20, t=40, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 3: 장비 마스터 테이블 ───
    st.markdown("##### 장비 마스터 테이블")

    # 필터 컨트롤
    filter_col1, filter_col2, filter_col3 = st.columns([2, 4, 2])

    with filter_col1:
        status_filter = st.selectbox(
            "상태 필터",
            options=["전체", "가동", "반출"],
            index=0,
            key="equipment_status_filter",
        )

    with filter_col2:
        company_options = ["전체"] + sorted(master_df["company_name"].unique().tolist())
        company_filter = st.multiselect(
            "업체 필터",
            options=company_options,
            default=["전체"],
            key="equipment_company_filter",
        )

    with filter_col3:
        spec_options = ["전체"] + sorted(master_df["specification"].dropna().unique().tolist())
        spec_filter = st.selectbox(
            "사양 필터",
            options=spec_options,
            index=0,
            key="equipment_spec_filter",
        )

    # 필터 적용
    filtered_df = master_df.copy()

    if status_filter == "가동":
        filtered_df = filtered_df[filtered_df["status"] == "working"]
    elif status_filter == "반출":
        filtered_df = filtered_df[filtered_df["status"] == "out"]

    if "전체" not in company_filter and company_filter:
        filtered_df = filtered_df[filtered_df["company_name"].isin(company_filter)]

    if spec_filter != "전체":
        filtered_df = filtered_df[filtered_df["specification"] == spec_filter]

    # 표시 컬럼 선택
    display_cols = [
        "equipment_no", "equipment_name", "company_name",
        "status", "specification", "manufacturer",
        "inbound_date",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    display_df = filtered_df[display_cols].copy()
    display_df.columns = [
        "장비번호", "장비명", "업체명",
        "상태", "사양", "제조사",
        "반입일"
    ][:len(display_cols)]

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        height=400,
    )

    st.caption(f"총 {len(filtered_df):,}대 (필터 적용), 상위 100개 표시")


# ============================================================
# Subtab 3: Floor Distribution
# ============================================================
def _render_floor_subtab(
    weekly_floor_df: pd.DataFrame,
    master_df: pd.DataFrame,
) -> None:
    """층별 분포 서브탭."""

    if weekly_floor_df.empty:
        st.info("층별 분포 데이터가 없습니다.")
        return

    # Unknown 제외
    floor_df = weekly_floor_df[weekly_floor_df["floor_name"] != "Unknown"].copy()

    # 층 순서 정의
    floor_order = ["5F", "6F", "7F", "RF"]
    floor_df["floor_order"] = floor_df["floor_name"].apply(
        lambda x: floor_order.index(x) if x in floor_order else 99
    )
    floor_df = floor_df.sort_values(["week_label", "floor_order"])

    # ─── Row 1: 층별 KPI 카드 (최신 주차) ───
    st.markdown("##### 층별 장비 현황")

    latest_week = floor_df["week_label"].max()
    latest_floor = floor_df[floor_df["week_label"] == latest_week]

    cols = st.columns(4)

    for i, floor in enumerate(floor_order):
        with cols[i]:
            floor_data = latest_floor[latest_floor["floor_name"] == floor]
            if not floor_data.empty:
                count = floor_data["equipment_count"].iloc[0]
                opr = floor_data["avg_opr"].iloc[0]
                opr_color = _get_opr_color(opr)
            else:
                count = 0
                opr = 0
                opr_color = COLORS["text_muted"]

            st.markdown(
                metric_card_sm(
                    f"{floor} ({count}대)",
                    f"{opr*100:.1f}%",
                    color=opr_color,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 2: 주간 층별 장비 분포 Stacked Bar ───
    st.markdown("##### 주간 층별 장비 분포")

    # 주차별 라벨 포맷팅
    floor_df["week_display"] = floor_df["week_label"].apply(_format_week_label)

    fig = px.bar(
        floor_df,
        x="week_display",
        y="equipment_count",
        color="floor_name",
        color_discrete_map={
            "5F": "#3B82F6",
            "6F": "#10B981",
            "7F": "#F59E0B",
            "RF": "#8B5CF6",
        },
        category_orders={"floor_name": floor_order},
        labels={
            "week_display": "주차",
            "equipment_count": "장비 수",
            "floor_name": "층",
        },
    )

    fig.update_layout(
        height=350,
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        margin=dict(l=50, r=20, t=40, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 3: 층별 평균 가동률 비교 ───
    st.markdown("##### 층별 평균 가동률 비교")

    # 전체 기간 층별 평균
    floor_avg = floor_df.groupby("floor_name").agg({
        "equipment_count": "mean",
        "avg_opr": "mean",
    }).reset_index()
    floor_avg["floor_order"] = floor_avg["floor_name"].apply(
        lambda x: floor_order.index(x) if x in floor_order else 99
    )
    floor_avg = floor_avg.sort_values("floor_order")
    floor_avg["opr_pct"] = floor_avg["avg_opr"] * 100
    floor_avg["color"] = floor_avg["avg_opr"].apply(_get_opr_color)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=floor_avg["floor_name"],
        y=floor_avg["opr_pct"],
        marker_color=floor_avg["color"],
        text=floor_avg["opr_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "평균 OPR: %{y:.1f}%<extra></extra>"
        ),
    ))

    fig.update_layout(
        height=300,
        xaxis_title="층",
        yaxis_title="평균 가동률 (%)",
        yaxis_range=[0, max(floor_avg["opr_pct"].max() * 1.3, 30)],
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        margin=dict(l=50, r=20, t=20, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─── Row 4: 층별 주간 OPR 히트맵 ───
    st.markdown("##### 층별 주간 가동률 히트맵")

    pivot_floor = floor_df.pivot_table(
        index="floor_name",
        columns="week_label",
        values="avg_opr",
        aggfunc="mean"
    )

    # 층 순서 정렬
    pivot_floor = pivot_floor.reindex([f for f in floor_order if f in pivot_floor.index])

    # 주차 순서 정렬
    week_order = sorted(pivot_floor.columns.tolist())
    pivot_floor = pivot_floor[week_order]

    week_labels = [_format_week_label(w) for w in week_order]
    z_values = (pivot_floor.values * 100).round(1)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=week_labels,
        y=pivot_floor.index.tolist(),
        colorscale=OPR_COLORSCALE,
        zmin=0,
        zmax=100,
        text=z_values,
        texttemplate="%{text:.0f}%",
        textfont={"size": 12},
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x}<br>"
            "가동률: %{z:.1f}%<extra></extra>"
        ),
        colorbar=dict(
            title="OPR (%)",
            ticksuffix="%",
        ),
    ))

    fig.update_layout(
        height=250,
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis_title="주차",
        yaxis_title="층",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Debug / Test
# ============================================================
if __name__ == "__main__":
    # Standalone test
    import streamlit as st
    st.set_page_config(page_title="Equipment Tab Test", layout="wide")
    render_equipment_tab("M15X_SKHynix")
