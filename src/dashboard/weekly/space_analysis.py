"""
Weekly Space Analysis 모듈
==========================
공간별 집계 + 업체별 현황 + 안전 분석.
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import PLOTLY_DARK
from src.utils.anonymizer import mask_names_in_df
from src.spatial.loader import load_locus_dict

logger = logging.getLogger(__name__)

_DARK = PLOTLY_DARK


def render_weekly_space(space_df: pd.DataFrame, sector_id: str | None = None):
    """주간 공간별 누적 체류 현황."""
    if space_df is None or space_df.empty:
        st.info("공간 데이터 없음")
        return

    locus_dict = load_locus_dict(sector_id)
    df = space_df.copy()
    df["locus_name"] = df["locus_id"].map(
        lambda lid: locus_dict.get(lid, {}).get("locus_name", lid)
    )

    # 날짜별 집계 -> locus_id 기준 합산
    agg = df.groupby(["locus_id", "locus_name", "locus_token"]).agg(
        total_person_minutes=("total_person_minutes", "sum"),
        max_unique_workers=("unique_workers", "max"),
    ).reset_index()

    top20 = agg.nlargest(20, "total_person_minutes")

    fig = px.bar(
        top20, x="total_person_minutes", y="locus_name",
        orientation="h",
        color="locus_token",
        title="주간 공간별 누적 체류 (상위 20개)",
        labels={"total_person_minutes": "총 체류 인분", "locus_name": "공간", "locus_token": "공간 유형"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500, **_DARK,
                      legend=dict(font=dict(color="#C8D6E8")))
    st.plotly_chart(fig, use_container_width=True)


def render_weekly_company(company_df: pd.DataFrame):
    """주간 업체별 현황."""
    if company_df is None or company_df.empty:
        st.info("업체 데이터 없음")
        return

    # 날짜 수 계산
    num_days = company_df["date"].nunique() if "date" in company_df.columns else 1

    agg = company_df.groupby("company_name").agg(
        total_workers=("worker_count", "sum"),
        avg_work_zone=("avg_work_zone_minutes", "mean"),
        total_confined=("total_confined_minutes", "sum"),
        active_days=("date", "nunique") if "date" in company_df.columns else ("worker_count", "count"),
    ).reset_index().sort_values("total_workers", ascending=False)

    # 일평균 = 연인원 / 실제 활동 날짜 수
    agg["avg_workers_per_day"] = (agg["total_workers"] / agg["active_days"]).round(1)

    top20 = agg.head(20)

    # 차트 선택 토글
    view_mode = st.radio(
        "표시 기준",
        ["연인원 (기간 합산)", "일평균 인원"],
        horizontal=True,
        key="company_view_mode",
    )

    if view_mode == "연인원 (기간 합산)":
        x_col = "total_workers"
        x_label = f"연인원 ({num_days}일 합산)"
        title = f"주간 업체별 총 작업자 수 - 연인원 ({num_days}일 합산)"
        fmt_fn = lambda v: f"{v:,.0f}명"
    else:
        x_col = "avg_workers_per_day"
        x_label = "일평균 작업자 수"
        title = f"주간 업체별 일평균 작업자 수 ({num_days}일 기준)"
        fmt_fn = lambda v: f"{v:.1f}명/일"

    fig = px.bar(
        top20, x=x_col, y="company_name",
        orientation="h",
        color="avg_work_zone",
        color_continuous_scale="Blues",
        title=title,
        labels={x_col: x_label, "company_name": "업체", "avg_work_zone": "평균 작업시간(분)"},
        text=top20[x_col].apply(fmt_fn),
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#C8D6E8", size=10))
    fig.update_layout(yaxis=dict(autorange="reversed"), height=520, **_DARK,
                      coloraxis_colorbar=dict(tickfont=dict(color="#C8D6E8")))
    st.plotly_chart(fig, use_container_width=True)

    # 전체 테이블
    with st.expander(f"전체 업체 목록 ({len(agg)}개)"):
        display_df = agg[["company_name", "total_workers", "avg_workers_per_day",
                           "avg_work_zone", "total_confined"]].copy()
        display_df.columns = ["업체", f"연인원({num_days}일)", "일평균(명/일)",
                               "평균작업시간(분)", "밀폐공간(분)"]
        display_df["일평균(명/일)"] = display_df["일평균(명/일)"].round(1)
        display_df["평균작업시간(분)"] = display_df["평균작업시간(분)"].round(0).astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 밀폐공간 업체 top
    confined_co = agg[agg["total_confined"] > 0].sort_values("total_confined", ascending=False)
    if not confined_co.empty:
        st.markdown("**밀폐공간 체류 업체 (주간 합산)**")
        st.dataframe(
            confined_co[["company_name", "total_workers", "avg_workers_per_day", "total_confined"]]
            .rename(columns={"company_name": "업체", "total_workers": "연인원",
                             "avg_workers_per_day": "일평균(명/일)", "total_confined": "밀폐총시간(분)"}),
            use_container_width=True, hide_index=True
        )


def render_weekly_time_breakdown(worker_df: pd.DataFrame, date_list: list[str]) -> None:
    """
    주간 업체별 시간 분류 (Time Breakdown).

    - 업체별 평균 작업비율 막대 차트
    - 업체별 시간 분류 스택 바 (주간 평균)
    """
    import numpy as np

    if worker_df is None or worker_df.empty:
        st.info("시간 분류 데이터 없음")
        return

    df = worker_df.copy()
    if date_list and "date" in df.columns:
        df = df[df["date"].isin(set(date_list))]

    if df.empty:
        st.info("선택 기간 데이터가 없습니다.")
        return

    # 업체 컬럼 탐색
    company_col = next(
        (c for c in ["hcon_company_name", "company_name", "bp_name"] if c in df.columns),
        None,
    )
    if company_col is None:
        st.info("업체 컬럼을 찾을 수 없습니다.")
        return

    # time breakdown fallback 컬럼 생성
    if "work_dwell_min" not in df.columns:
        work_cols = [c for c in ["intense_work_min", "light_work_min", "idle_at_work_min"] if c in df.columns]
        df["work_dwell_min"] = df[work_cols].sum(axis=1) if work_cols else 0

    if "transit_dwell_min" not in df.columns:
        t_cols = [c for c in ["transit_min", "in_transit_min"] if c in df.columns]
        df["transit_dwell_min"] = df[t_cols].sum(axis=1) if t_cols else 0

    if "rest_dwell_min" not in df.columns:
        df["rest_dwell_min"] = df["rest_min"] if "rest_min" in df.columns else 0

    # 업체별 평균 집계
    agg_dict: dict = {
        "work_dwell_min":   ("work_dwell_min",   "mean"),
        "transit_dwell_min":("transit_dwell_min", "mean"),
        "rest_dwell_min":   ("rest_dwell_min",    "mean"),
    }
    if "work_minutes" in df.columns:
        agg_dict["work_minutes"] = ("work_minutes", "mean")

    agg = df.groupby(company_col).agg(**agg_dict).reset_index()
    agg = agg.sort_values("work_dwell_min", ascending=False).head(10)

    if agg.empty:
        st.info("집계 데이터가 없습니다.")
        return

    # 작업 비율 계산
    classified = agg["work_dwell_min"] + agg["transit_dwell_min"] + agg["rest_dwell_min"]
    if "work_minutes" in agg.columns:
        denom = agg["work_minutes"].where(agg["work_minutes"] > 0, classified.replace(0, np.nan))
    else:
        denom = classified.replace(0, np.nan)
    agg["work_pct"] = (agg["work_dwell_min"] / denom * 100).fillna(0).round(1)

    # ── 1: 업체별 평균 작업비율 막대 차트 ────────────────────────────
    st.markdown("**업체별 평균 작업 집중도 (TOP 10)**")
    top10_sorted = agg.sort_values("work_pct", ascending=True)

    fig_bar = go.Figure(go.Bar(
        y=top10_sorted[company_col].tolist(),
        x=top10_sorted["work_pct"].tolist(),
        orientation="h",
        marker=dict(
            color=top10_sorted["work_pct"].tolist(),
            colorscale=[[0, "#1A3A5C"], [0.5, "#00AEEF"], [1, "#00C897"]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in top10_sorted["work_pct"].tolist()],
        textposition="outside",
        textfont=dict(color="#D5E5FF", size=11),
        hovertemplate="작업 비율: %{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        **_DARK,
        margin=dict(l=15, r=60, t=30, b=15),
        height=max(260, len(top10_sorted) * 32 + 60),
        xaxis=dict(range=[0, 110], title="작업 집중도 (%)"),
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 2: 업체별 시간 분류 스택 바 ──────────────────────────────────
    st.markdown("**업체별 시간 분류 (주간 평균, 분)**")
    stack_sorted = agg.sort_values("work_dwell_min", ascending=True)
    company_names = stack_sorted[company_col].tolist()

    fig_stack = go.Figure()
    for col_key, label, color in [
        ("work_dwell_min",    "작업공간 체류", "#00AEEF"),
        ("transit_dwell_min", "이동",         "#FFB300"),
        ("rest_dwell_min",    "휴식",         "#4CAF50"),
    ]:
        fig_stack.add_trace(go.Bar(
            y=company_names,
            x=stack_sorted[col_key].tolist(),
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"{label}: %{{x:.1f}}분<extra></extra>",
        ))

    # work_pct 주석
    max_x = (stack_sorted["work_dwell_min"] + stack_sorted["transit_dwell_min"] + stack_sorted["rest_dwell_min"]).max()
    annotations = [
        dict(
            x=float(stack_sorted["work_dwell_min"].iloc[i]
                    + stack_sorted["transit_dwell_min"].iloc[i]
                    + stack_sorted["rest_dwell_min"].iloc[i]) + max_x * 0.02,
            y=company_names[i],
            text=f"<b>{float(stack_sorted['work_pct'].iloc[i]):.1f}%</b>",
            showarrow=False,
            font=dict(color="#00AEEF", size=11),
            xanchor="left",
            yanchor="middle",
        )
        for i in range(len(company_names))
    ]

    fig_stack.update_layout(
        **_DARK,
        barmode="stack",
        margin=dict(l=15, r=80, t=30, b=15),
        height=max(300, len(company_names) * 36 + 80),
        xaxis_title="시간 (분)",
        yaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#D5E5FF"),
        ),
        annotations=annotations,
        hovermode="y unified",
    )
    st.plotly_chart(fig_stack, use_container_width=True)


def render_weekly_safety(worker_df: pd.DataFrame, date_list: list[str]):
    """주간 안전 현황."""
    # 선택 기간 필터링
    if date_list and "date" in worker_df.columns:
        worker_df = worker_df[worker_df["date"].isin(set(date_list))]

    if "confined_minutes" not in worker_df.columns:
        st.info("안전 데이터 없음")
        return

    # 밀폐공간 장시간 체류 (주간 합산)
    st.markdown("**밀폐공간 장시간 체류 작업자 (주간 누적 30분 이상)**")
    confined = worker_df[worker_df["confined_minutes"] >= 30].sort_values(
        "confined_minutes", ascending=False
    )
    if confined.empty:
        st.success("해당 없음")
    else:
        st.dataframe(
            mask_names_in_df(confined[["user_name", "company_name", "confined_minutes", "date"]], "user_name")
            .rename(columns={"user_name": "작업자", "company_name": "업체",
                             "confined_minutes": "밀폐체류(분)", "date": "날짜"}),
            use_container_width=True, hide_index=True
        )

    # 고압전 구역
    if "high_voltage_minutes" in worker_df.columns:
        st.markdown("**고압전 구역 체류 작업자 (주간)**")
        hv = worker_df[worker_df["high_voltage_minutes"] > 0].sort_values(
            "high_voltage_minutes", ascending=False
        )
        if hv.empty:
            st.success("해당 없음")
        else:
            st.dataframe(
                mask_names_in_df(hv[["user_name", "company_name", "high_voltage_minutes", "date"]], "user_name")
                .rename(columns={"user_name": "작업자", "company_name": "업체",
                                 "high_voltage_minutes": "고압전체류(분)", "date": "날짜"}),
                use_container_width=True, hide_index=True
            )
