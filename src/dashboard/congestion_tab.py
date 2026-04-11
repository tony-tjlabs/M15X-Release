"""
Congestion Tab — 공간 혼잡도 분석 탭
======================================
건설현장 공간별 시간대별 혼잡도를 시각화.

구성:
  📊 개요     : 혼잡도 KPI + 피크 공간/시간
  🗺️ 히트맵  : 공간 × 시간대 혼잡도 히트맵
  📈 추이     : 시간대별 혼잡도 곡선 (공간별)
  📅 기간분석 : 날짜별/요일별 혼잡도 패턴
"""
from __future__ import annotations

import logging
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

import config as cfg
from src.dashboard.styles import (
    metric_card, section_header, PLOTLY_DARK, PLOTLY_LEGEND,
)
from src.pipeline.cache_manager import detect_processed_dates, _date_dir, load_journey
from src.spatial.loader import load_locus_dict
from src.dashboard.components import DWELL_CATEGORY_STYLES
from src.utils.weather import date_label

logger = logging.getLogger(__name__)

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND

# 혼잡도 등급 색상
_CONGESTION_COLORS = {
    "여유": "#00C897",
    "보통": "#FFB300",
    "혼잡": "#FF6B35",
    "과밀": "#FF4C4C",
}

# dwell_category 스타일 (components.py에서 import)
_DWELL_CATEGORY_STYLES = DWELL_CATEGORY_STYLES


def _congestion_grade(count: int) -> str:
    """혼잡도 등급 판정 (작업자 수 기준 — 하위 호환)."""
    if count >= 100:
        return "과밀"
    if count >= 50:
        return "혼잡"
    if count >= 20:
        return "보통"
    return "여유"


def _congestion_grade_by_max(current: int, max_occ: int) -> tuple[str, str]:
    """
    max_concurrent_occupancy 대비 혼잡률 기반 등급.

    Args:
        current: 현재 인원
        max_occ: 최대 수용 인원 (locus.csv의 max_concurrent_occupancy)

    Returns:
        (등급 문자열, 색상 코드)
    """
    if max_occ <= 0:
        return "여유", "#00C897"

    pct = current / max_occ * 100
    if pct >= 90:
        return "과밀", "#FF4C4C"
    elif pct >= 70:
        return "혼잡", "#FF6B35"
    elif pct >= 40:
        return "보통", "#FFB300"
    else:
        return "여유", "#00C897"


def render_congestion_tab(sector_id: str | None = None):
    """혼잡도 분석 탭 진입점."""
    sid = sector_id or cfg.SECTOR_ID
    processed = detect_processed_dates(sid)

    if not processed:
        st.info("처리된 데이터가 없습니다. [파이프라인] 탭에서 먼저 전처리를 실행하세요.")
        return

    locus_dict = load_locus_dict(sid)

    st.markdown(section_header("🏢 공간 혼잡도 분석"), unsafe_allow_html=True)
    st.caption(
        "건설현장 구역별 동시 체류 인원 분석 — "
        "시간대별 혼잡도 변화, 피크 공간/시간 탐지, 요일별 패턴"
    )

    # ── 분석 모드 선택 ───────────────────────────────────────────
    mode = st.radio(
        "분석 범위",
        ["📊 일별 분석", "📅 기간 분석"],
        horizontal=True,
        key="congestion_mode",
    )

    if mode == "📊 일별 분석":
        _render_single_day(sid, processed, locus_dict)
    else:
        _render_multi_day(sid, processed, locus_dict)


# ══════════════════════════════════════════════════════════════════
# 일별 혼잡도 분석
# ══════════════════════════════════════════════════════════════════

def _render_single_day(sid: str, processed: list[str], locus_dict: dict):
    """단일 날짜 혼잡도 분석."""
    date_options = {date_label(d): d for d in reversed(processed)}
    selected_label = st.selectbox("분석 날짜", list(date_options.keys()), key="cong_date")
    date_str = date_options[selected_label]

    # journey 로드 (★ 캐시 활용 — 2회차부터 즉시 로드)
    journey_full = load_journey(date_str, sid)
    if journey_full.empty:
        st.warning("journey.parquet가 없습니다. 로컬 환경에서 파이프라인을 실행하세요.")
        return

    with st.spinner("혼잡도 데이터 분석 중..."):
        cols = [c for c in ["timestamp", "user_no", "locus_id", "locus_token"] if c in journey_full.columns]
        journey_df = journey_full[cols]
        from src.pipeline.congestion import (
            compute_congestion, compute_hourly_profile,
            compute_congestion_summary, compute_space_ranking,
        )
        congestion_df = compute_congestion(journey_df, time_bin_minutes=30, locus_dict=locus_dict)
        hourly_df = compute_hourly_profile(journey_df, locus_dict)
        summary = compute_congestion_summary(congestion_df)
        ranking_df = compute_space_ranking(congestion_df, top_n=15)

    if congestion_df.empty:
        st.warning("혼잡도 데이터가 없습니다.")
        return

    # ── KPI 카드 ──────────────────────────────────────────────────
    st.divider()
    cols = st.columns(6)
    kpis = [
        ("피크 공간", summary["peak_space"], None),
        ("최대 동시 인원 (30분 단위)", f"{summary['peak_count']}명", "#FF4C4C" if summary["peak_count"] >= 100 else "#FFB300"),
        ("피크 시간", summary["peak_time"], None),
        ("최붐비는 시간", f"{summary['busiest_hour']}시", "#FF6B35"),
        ("가장 한산한 시간", f"{summary['quietest_hour']}시", "#00C897"),
        ("분석 공간 수", f"{summary['total_spaces']}개", None),
    ]
    for col, (label, val, color) in zip(cols, kpis):
        with col:
            if color:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{color}'>"
                    f"{val}</div><div class='metric-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(metric_card(label, str(val)), unsafe_allow_html=True)

    st.caption(
        "※ 최대 동시 인원은 30분 단위 피크 기준. "
        "히트맵은 1시간 단위 평균/최대 기준으로 수치가 다를 수 있습니다."
    )
    st.divider()

    # ── 탭 구성 ──────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["🗺️ 히트맵", "📈 시간대 추이", "🏆 공간 랭킹", "AI Analysis"])

    with t1:
        _render_heatmap(hourly_df)

    with t2:
        _render_time_series(congestion_df)

    with t3:
        _render_space_ranking(ranking_df, locus_dict)

    with t4:
        from src.dashboard.ai_analysis import render_congestion_ai
        render_congestion_ai(
            summary_dict=summary,
            ranking_df=ranking_df,
            cache_key=f"ai_congestion_{sid}_{date_str}",
        )


def _render_heatmap(hourly_df: pd.DataFrame):
    """공간 × 시간대 혼잡도 히트맵 (3종: 평균/최대/밀도)."""
    if hourly_df.empty:
        st.info("히트맵 데이터 없음")
        return

    # ── 히트맵 유형 선택 ──
    heatmap_type = st.radio(
        "히트맵 유형",
        ["평균 인원수", "최대 동시 인원", "단위 면적당 인원수"],
        horizontal=True,
        key="heatmap_type_radio",
    )

    # 유형별 설명
    _HEATMAP_DESCRIPTIONS = {
        "평균 인원수": {
            "title": "공간 x 시간대 평균 체류 인원",
            "desc": (
                "각 공간에 해당 시간대에 평균적으로 몇 명이 체류하고 있었는지를 보여줍니다. "
                "색이 진할수록(빨간색에 가까울수록) 해당 시간대에 해당 공간의 평균 체류 인원이 많습니다. "
                "일과 중 어떤 공간이 지속적으로 사람이 많은지 파악하는 데 유용합니다."
            ),
        },
        "최대 동시 인원": {
            "title": "공간 x 시간대 최대 동시 체류 인원 (피크 기준)",
            "desc": (
                "각 공간에 해당 시간대에 동시에 가장 많이 체류했던 순간의 인원수입니다. "
                "순간적으로 밀집되는 병목 구간이나 안전 위험 시간대를 탐지하는 데 유용합니다. "
                "평균 대비 최대 인원이 크게 높으면 특정 시점에 인원 쏠림이 발생한 것입니다."
            ),
        },
        "단위 면적당 인원수": {
            "title": "공간 x 시간대 단위 면적당 인원 밀도 (명/100m2)",
            "desc": (
                "공간의 면적을 고려한 실질적 혼잡 밀도입니다. "
                "면적이 넓은 작업층(FAB 5F~7F 등)은 인원이 많아도 밀도가 낮을 수 있고, "
                "면적이 좁은 편의시설(화장실, 흡연실 등)은 소수 인원으로도 높은 밀도를 보일 수 있습니다. "
                "안전 관리 관점에서 실질적 과밀 공간을 식별하는 핵심 지표입니다."
            ),
        },
    }
    info = _HEATMAP_DESCRIPTIONS[heatmap_type]
    st.markdown(f"#### {info['title']}")
    st.caption(info["desc"])

    # 작업 구간만 필터 (5~23시) + 음영지역 제외
    df = hourly_df[(hourly_df["hour"] >= 5) & (hourly_df["hour"] <= 23)].copy()
    if "locus_id" in df.columns:
        df = df[df["locus_id"] != "shadow_zone"]
    if "locus_token" in df.columns:
        df = df[df["locus_token"] != "shadow_zone"]

    if df.empty:
        st.info("업무 시간(5~23시) 데이터 없음")
        return

    # ── Locus별 면적 정의 (m2) ──
    # 실측/도면 기반 추정 면적. 정확한 면적 데이터 확보 시 교체.
    _LOCUS_AREA_M2: dict[str, float] = {
        "FAB 5F":          3200.0,
        "FAB 6F":          3200.0,
        "FAB 7F":          3200.0,
        "FAB RF":          2400.0,
        "FAB 건너 화장실":   80.0,
        "FAB 건너 휴게실":  150.0,
        "FAB 건너 흡연실":   40.0,
        "공사현장":        5000.0,
        "타각기":           200.0,
        "호이스트2":        100.0,
        "호이스트3":        100.0,
    }

    if heatmap_type == "평균 인원수":
        value_col = "avg_workers"
        colorbar_title = "평균 인원"
        hover_template = "공간: %{y}<br>시간: %{x}<br>평균 인원: %{z:.1f}명<extra></extra>"
        colorscale = [
            [0, "#0D1B2A"], [0.2, "#1A3A5C"], [0.4, "#00AEEF"],
            [0.6, "#FFB300"], [0.8, "#FF6B35"], [1.0, "#FF4C4C"],
        ]
        aggfunc = "mean"

    elif heatmap_type == "최대 동시 인원":
        value_col = "max_workers"
        colorbar_title = "최대 인원"
        hover_template = "공간: %{y}<br>시간: %{x}<br>최대 인원: %{z:.0f}명<extra></extra>"
        colorscale = [[0, "#0D1B2A"], [0.5, "#FFB300"], [1.0, "#FF4C4C"]]
        aggfunc = "max"

    else:
        # 단위 면적당 인원수: avg_workers / (area / 100)
        value_col = "avg_workers"
        colorbar_title = "밀도 (명/100m2)"
        hover_template = "공간: %{y}<br>시간: %{x}<br>밀도: %{z:.2f}명/100m2<extra></extra>"
        colorscale = [
            [0, "#0D1B2A"], [0.25, "#1A3A5C"], [0.5, "#9B59B6"],
            [0.75, "#E74C3C"], [1.0, "#FF4C4C"],
        ]
        aggfunc = "mean"

    if value_col not in df.columns:
        st.warning(f"{value_col} 컬럼이 없습니다.")
        return

    pivot = df.pivot_table(
        index="locus_name",
        columns="hour",
        values=value_col,
        aggfunc=aggfunc,
        fill_value=0,
    )

    # 밀도 계산: 면적 기반 변환
    if heatmap_type == "단위 면적당 인원수":
        for locus_name in pivot.index:
            area = _LOCUS_AREA_M2.get(locus_name, 500.0)
            pivot.loc[locus_name] = (pivot.loc[locus_name] / (area / 100.0)).round(2)

    # 총합 기준 정렬
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=True)
    pivot = pivot.drop(columns=["_total"])

    # 열 라벨 정리
    pivot.columns = [f"{h}시" for h in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
        hovertemplate=hover_template,
        colorbar=dict(
            title=dict(text=colorbar_title, font=dict(color="#D5E5FF")),
            tickfont=dict(color="#D5E5FF"),
        ),
    ))
    fig.update_layout(
        **_DARK,
        height=max(350, len(pivot) * 28 + 100),
        xaxis_title="시간대",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 밀도 히트맵에 면적 참조 테이블 표시
    if heatmap_type == "단위 면적당 인원수":
        with st.expander("공간별 면적 참조 (추정값)", expanded=False):
            area_data = []
            for name in pivot.index:
                area = _LOCUS_AREA_M2.get(name, 500.0)
                area_data.append({"공간": name, "면적 (m2)": area, "비고": "추정값"})
            st.dataframe(pd.DataFrame(area_data), use_container_width=True, hide_index=True)
            st.caption(
                "면적은 도면 기반 추정값입니다. 정확한 면적 데이터 확보 시 업데이트됩니다."
            )


def _render_time_series(congestion_df: pd.DataFrame):
    """시간대별 혼잡도 추이 (상위 공간)."""
    if congestion_df.empty:
        st.info("추이 데이터 없음")
        return

    st.markdown("#### 📈 시간대별 혼잡도 추이")

    # 상위 10개 공간 필터
    top_spaces = (
        congestion_df.groupby("locus_name")["worker_count"]
        .max()
        .nlargest(10)
        .index.tolist()
    )

    selected_spaces = st.multiselect(
        "공간 선택 (최대 10개)",
        options=top_spaces,
        default=top_spaces[:5],
        key="cong_space_select",
    )

    if not selected_spaces:
        st.info("공간을 선택하세요.")
        return

    df = congestion_df[congestion_df["locus_name"].isin(selected_spaces)].copy()
    # 작업 시간만
    df = df[(df["hour"] >= 5) & (df["hour"] <= 23)]
    df["time_label"] = df["time_bin"].dt.strftime("%H:%M")

    fig = px.line(
        df,
        x="time_bin",
        y="worker_count",
        color="locus_name",
        title="공간별 동시 체류 인원 추이 (30분 단위)",
        labels={"worker_count": "동시 인원", "time_bin": "시간", "locus_name": "공간"},
        markers=True,
    )
    fig.update_layout(**_DARK, height=420, legend=_LEG)
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)

    # 전체 현장 총 인원 추이
    total_by_time = congestion_df.groupby("time_bin")["worker_count"].sum().reset_index()
    total_by_time = total_by_time[
        (total_by_time["time_bin"].dt.hour >= 5) & (total_by_time["time_bin"].dt.hour <= 23)
    ]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=total_by_time["time_bin"],
        y=total_by_time["worker_count"],
        fill="tozeroy",
        fillcolor="rgba(0, 174, 239, 0.15)",
        line=dict(color="#00AEEF", width=2),
        name="현장 전체",
        hovertemplate="시간: %{x|%H:%M}<br>전체 인원: %{y}명<extra></extra>",
    ))
    fig2.update_layout(
        title="현장 전체 동시 체류 인원 추이",
        yaxis_title="전체 인원",
        **_DARK, height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_space_ranking(ranking_df: pd.DataFrame, locus_dict: dict | None = None):
    """공간별 혼잡도 랭킹 — max_concurrent_occupancy 대비 혼잡률 포함."""
    if ranking_df.empty:
        st.info("랭킹 데이터 없음")
        return

    st.markdown("#### 🏆 공간별 혼잡도 랭킹")
    st.caption("최대 동시 인원 기준 상위 공간 — 피크 시간대 + 혼잡률 포함")

    # ★ max_concurrent_occupancy 대비 혼잡률 계산 (enriched locus 활용)
    ranking_df = ranking_df.copy()
    if locus_dict:
        ranking_df["_max_capacity"] = ranking_df["locus_id"].apply(
            lambda lid: locus_dict.get(lid, {}).get("max_concurrent_occupancy") or 50
        )
        ranking_df["_dwell_cat"] = ranking_df["locus_id"].apply(
            lambda lid: locus_dict.get(lid, {}).get("dwell_category", "UNKNOWN")
        )
    else:
        ranking_df["_max_capacity"] = 50
        ranking_df["_dwell_cat"] = "UNKNOWN"

    ranking_df["occupancy_pct"] = (
        ranking_df["max_workers"] / ranking_df["_max_capacity"] * 100
    ).clip(0, 150).round(1)

    # 수평 바 차트
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ranking_df["locus_name"],
        x=ranking_df["max_workers"],
        orientation="h",
        name="최대 인원",
        marker_color="#FF6B35",
        text=ranking_df.apply(
            lambda r: f"{int(r['max_workers'])}명 ({r['occupancy_pct']:.0f}%)", axis=1
        ),
        textposition="outside",
        textfont=dict(color="#D5E5FF", size=11),
    ))
    fig.add_trace(go.Bar(
        y=ranking_df["locus_name"],
        x=ranking_df["avg_workers"],
        orientation="h",
        name="평균 인원",
        marker_color="#00AEEF",
        text=ranking_df["avg_workers"].apply(lambda v: f"{v:.1f}명"),
        textposition="outside",
        textfont=dict(color="#D5E5FF", size=10),
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(ranking_df) * 40 + 100),
        legend=_LEG,
        **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ★ 유형별 혼잡도 비교 차트 추가
    if locus_dict and "_dwell_cat" in ranking_df.columns:
        _render_category_congestion_chart(ranking_df)

    # 상세 테이블
    with st.expander("📋 상세 데이터"):
        display = ranking_df[[
            "locus_name", "_dwell_cat", "max_workers", "_max_capacity",
            "occupancy_pct", "avg_workers", "peak_hour"
        ]].copy()
        display.columns = ["공간", "유형", "최대 인원", "수용 기준", "혼잡률(%)", "평균 인원", "피크 시간"]
        display["피크 시간"] = display["피크 시간"].astype(int).astype(str) + "시"
        # 유형 한글화
        display["유형"] = display["유형"].apply(
            lambda x: _DWELL_CATEGORY_STYLES.get(x, {}).get("label", x)
        )
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_category_congestion_chart(ranking_df: pd.DataFrame):
    """dwell_category별 평균 혼잡률 비교 차트."""
    if "_dwell_cat" not in ranking_df.columns or ranking_df.empty:
        return

    cat_agg = ranking_df.groupby("_dwell_cat").agg(
        avg_occupancy=("occupancy_pct", "mean"),
        count=("locus_id", "count"),
    ).reset_index()

    if cat_agg.empty:
        return

    with st.expander("📊 공간 유형별 평균 혼잡률", expanded=False):
        cat_agg["label"] = cat_agg["_dwell_cat"].apply(
            lambda x: _DWELL_CATEGORY_STYLES.get(x, {}).get("label", x)
        )
        cat_agg["color"] = cat_agg["_dwell_cat"].apply(
            lambda x: _DWELL_CATEGORY_STYLES.get(x, {}).get("color", "#6B7280")
        )

        fig = go.Figure(go.Bar(
            x=cat_agg["label"],
            y=cat_agg["avg_occupancy"],
            marker_color=cat_agg["color"].tolist(),
            text=cat_agg.apply(lambda r: f"{r['avg_occupancy']:.1f}% ({r['count']}개)", axis=1),
            textposition="outside",
            textfont=dict(color="#D5E5FF"),
        ))
        fig.update_layout(
            yaxis_title="평균 혼잡률 (%)",
            **_DARK,
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 기간 혼잡도 분석
# ══════════════════════════════════════════════════════════════════

def _render_multi_day(sid: str, processed: list[str], locus_dict: dict):
    """복수 날짜 혼잡도 분석 — 날짜별/요일별 패턴."""
    date_list_dt = [datetime.strptime(d, "%Y%m%d") for d in processed]

    col_a, col_b = st.columns(2)
    with col_a:
        start_dt = st.date_input(
            "시작일",
            value=date_list_dt[0].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
            key="cong_start",
        )
    with col_b:
        end_dt = st.date_input(
            "종료일",
            value=date_list_dt[-1].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
            key="cong_end",
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

    st.caption(
        f"분석 기간: **{date_label(selected_dates[0])}** ~ "
        f"**{date_label(selected_dates[-1])}** ({len(selected_dates)}일)"
    )

    with st.spinner(f"기간 혼잡도 분석 중... ({len(selected_dates)}일)"):
        from src.pipeline.congestion import (
            compute_multi_day_congestion, compute_day_of_week_pattern,
        )
        multi_df = compute_multi_day_congestion(selected_dates, sid, locus_dict)

    if multi_df.empty:
        st.warning("혼잡도 데이터 없음 (journey.parquet 필요)")
        return

    st.divider()

    # ── 탭 ─────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📊 날짜별 추이", "📅 요일별 패턴", "🏢 공간별 비교"])

    with t1:
        _render_daily_congestion_trend(multi_df)

    with t2:
        dow_df = compute_day_of_week_pattern(multi_df)
        _render_dow_pattern(dow_df)

    with t3:
        _render_space_comparison(multi_df)


def _render_daily_congestion_trend(multi_df: pd.DataFrame):
    """날짜별 전체 혼잡도 추이."""
    st.markdown("#### 📊 날짜별 혼잡도 추이")

    # 날짜별 시간대별 총 인원
    daily_agg = (
        multi_df.groupby(["date", "hour"])["avg_workers"]
        .sum()
        .reset_index()
    )
    from src.utils.weather import date_label_short
    daily_agg["date_fmt"] = daily_agg["date"].apply(date_label_short)

    # 작업 시간만
    daily_agg = daily_agg[(daily_agg["hour"] >= 5) & (daily_agg["hour"] <= 23)]

    fig = px.line(
        daily_agg,
        x="hour",
        y="avg_workers",
        color="date_fmt",
        title="날짜별 시간대 혼잡도 (전체 공간 합산)",
        labels={"avg_workers": "평균 인원 합계", "hour": "시간", "date_fmt": "날짜"},
        markers=True,
    )
    fig.update_layout(**_DARK, height=400, legend=_LEG)
    st.plotly_chart(fig, use_container_width=True)

    # 날짜별 피크 인원 카드
    daily_peak = multi_df.groupby("date").agg(
        total_avg=("avg_workers", "sum"),
        max_single=("max_workers", "max"),
    ).reset_index()
    daily_peak["date_fmt"] = daily_peak["date"].apply(date_label_short)

    cols = st.columns(min(len(daily_peak), 5))
    for i, (_, row) in enumerate(daily_peak.iterrows()):
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:#1A2A3A; border:1px solid #2A3A4A; "
                f"border-radius:8px; padding:10px; text-align:center; margin:4px 0;'>"
                f"<div style='color:#9AB5D4; font-size:0.78rem;'>{row['date_fmt']}</div>"
                f"<div style='color:#FF6B35; font-size:1.3rem; font-weight:700;'>"
                f"{int(row['max_single'])}명</div>"
                f"<div style='color:#6A7A95; font-size:0.72rem;'>단일 공간 최대</div></div>",
                unsafe_allow_html=True,
            )


def _render_dow_pattern(dow_df: pd.DataFrame):
    """요일별 혼잡도 패턴."""
    if dow_df.empty:
        st.info("요일별 패턴 데이터 부족 (최소 2일 이상)")
        return

    st.markdown("#### 📅 요일별 시간대 혼잡도 패턴")
    st.caption("같은 요일의 시간대별 평균 혼잡도 — 요일 간 차이를 비교")

    # 작업 시간만
    df = dow_df[(dow_df["hour"] >= 5) & (dow_df["hour"] <= 23)].copy()

    fig = px.line(
        df,
        x="hour",
        y="avg_workers",
        color="day_name",
        title="요일별 시간대 평균 혼잡도 (전체 공간 합산)",
        labels={"avg_workers": "평균 인원", "hour": "시간", "day_name": "요일"},
        markers=True,
        category_orders={"day_name": ["월", "화", "수", "목", "금", "토", "일"]},
    )
    fig.update_layout(**_DARK, height=380, legend=_LEG)
    st.plotly_chart(fig, use_container_width=True)

    # 요일별 총 평균 비교 (바 차트)
    dow_total = df.groupby(["day_of_week", "day_name"])["avg_workers"].mean().reset_index()
    dow_total = dow_total.sort_values("day_of_week")

    fig2 = go.Figure(go.Bar(
        x=dow_total["day_name"],
        y=dow_total["avg_workers"],
        marker_color=["#00AEEF" if d < 5 else "#FF6B35" for d in dow_total["day_of_week"]],
        text=dow_total["avg_workers"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
        textfont=dict(color="#D5E5FF"),
    ))
    fig2.update_layout(
        title="요일별 평균 혼잡도 비교",
        xaxis_title="요일",
        yaxis_title="평균 인원",
        **_DARK, height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_space_comparison(multi_df: pd.DataFrame):
    """공간별 날짜 간 혼잡도 비교."""
    if multi_df.empty:
        return

    st.markdown("#### 🏢 공간별 날짜 간 혼잡도 비교")

    # 공간별 날짜별 평균
    space_daily = (
        multi_df.groupby(["date", "locus_name"])["avg_workers"]
        .mean()
        .reset_index()
    )
    from src.utils.weather import date_label_short
    space_daily["date_fmt"] = space_daily["date"].apply(date_label_short)

    # 상위 10개 공간
    top_spaces = (
        space_daily.groupby("locus_name")["avg_workers"]
        .mean()
        .nlargest(10)
        .index.tolist()
    )
    df = space_daily[space_daily["locus_name"].isin(top_spaces)]

    fig = px.bar(
        df,
        x="locus_name",
        y="avg_workers",
        color="date_fmt",
        barmode="group",
        title="상위 10개 공간 — 날짜별 평균 혼잡도",
        labels={"avg_workers": "평균 인원", "locus_name": "공간", "date_fmt": "날짜"},
    )
    fig.update_layout(**_DARK, height=420, legend=_LEG)
    st.plotly_chart(fig, use_container_width=True)
