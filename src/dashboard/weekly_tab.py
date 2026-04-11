"""
Weekly Tab - 주간 리포트 탭 (라우터)
=====================================
weekly/ 패키지의 서브모듈들을 조합하여 주간 리포트 탭을 렌더링합니다.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

import config as cfg
from src.dashboard.styles import section_header
from src.dashboard.llm_apollo import cached_weekly_trend_analysis, render_data_comment, is_llm_available
from src.pipeline.cache_manager import detect_processed_dates, load_multi_day_results
from src.utils.weather import date_label

# weekly 패키지에서 서브모듈 import
from src.dashboard.weekly import (
    render_weekly_kpi,
    render_weekly_site_status,
    render_daily_trend,
    render_ewi_cre_trend,
    render_weekly_space,
    render_weekly_company,
    render_weekly_safety,
    render_weekly_time_breakdown,
    render_day_of_week_analysis,
    render_shift_trend,
)
from src.dashboard.weekly.report import render_report_download

logger = logging.getLogger(__name__)


def render_weekly_tab(sector_id: str | None = None):
    """주간 리포트 탭 (메인 진입점)."""
    sid = sector_id or cfg.SECTOR_ID
    processed = detect_processed_dates(sid)
    if len(processed) < 1:
        st.info("처리된 데이터가 없습니다. [파이프라인] 탭에서 먼저 전처리를 실행하세요.")
        return

    # -- 날짜 범위 선택 --
    st.markdown(section_header("분석 기간 설정"), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    date_list_dt = [datetime.strptime(d, "%Y%m%d") for d in processed]

    with col_a:
        start_dt = st.date_input(
            "시작일",
            value=date_list_dt[0].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
        )
    with col_b:
        end_dt = st.date_input(
            "종료일",
            value=date_list_dt[-1].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
        )

    if start_dt > end_dt:
        st.error("시작일이 종료일보다 늦습니다.")
        return

    # 선택 범위 내 처리된 날짜
    selected_dates = [
        d for d in processed
        if start_dt <= datetime.strptime(d, "%Y%m%d").date() <= end_dt
    ]

    if not selected_dates:
        st.warning("선택 기간에 처리된 데이터가 없습니다.")
        return

    st.caption(
        f"분석 기간: **{date_label(selected_dates[0])}** ~ "
        f"**{date_label(selected_dates[-1])}** ({len(selected_dates)}일)"
    )
    st.divider()

    # -- 데이터 로드 --
    with st.spinner("주간 데이터 로딩 중..."):
        multi = load_multi_day_results(tuple(selected_dates), sid, skip_journey=True)

    worker_df = multi.get("worker")
    space_df = multi.get("space")
    company_df = multi.get("company")
    metas = multi.get("metas", [])

    if worker_df is None or worker_df.empty:
        st.warning("데이터 없음")
        return

    # -- 주간 KPI --
    has_ewi, has_cre = render_weekly_kpi(worker_df, metas, selected_dates)

    st.divider()

    # -- 건설현장 주간 핵심 현황 --
    render_weekly_site_status(worker_df, metas, selected_dates, has_ewi, has_cre)

    st.divider()

    # -- 탭 구성 (5탭) --
    t1, t2, t3, t4, t5 = st.tabs([
        "트렌드", "요일/패턴", "안전/업체", "상세", "리포트",
    ])

    with t1:
        render_daily_trend(worker_df, selected_dates, metas)
        st.divider()
        render_ewi_cre_trend(worker_df, selected_dates, has_ewi, has_cre)

    with t2:
        render_day_of_week_analysis(worker_df, metas, selected_dates, has_ewi, has_cre)

    with t3:
        render_weekly_safety(worker_df, selected_dates)
        st.divider()
        render_weekly_company(company_df)
        st.divider()
        render_weekly_time_breakdown(worker_df, selected_dates)

    with t4:
        with st.expander("주간/야간 비교", expanded=True):
            render_shift_trend(worker_df, selected_dates, metas, has_ewi, has_cre)
        with st.expander("공간별 현황", expanded=False):
            render_weekly_space(space_df, sid)

    with t5:
        render_report_download(
            selected_dates, sid, metas, worker_df, company_df,
            has_ewi, has_cre,
        )

    # -- 데이터 해석 (접이식) --
    if is_llm_available() and metas:
        st.divider()
        with st.expander("주간 데이터 해석 보기", expanded=False):
            from src.utils.weather import date_label_short
            date_range = (
                f"{date_label(selected_dates[0])} ~ "
                f"{date_label(selected_dates[-1])}"
            )
            trend_lines = []
            for m in metas:
                d = m.get("date_str", "")
                trend_lines.append(
                    f"  - {date_label_short(d)}: 출입 {m.get('total_workers_access', 0):,}명 / "
                    f"T-Ward {m.get('total_workers_move', 0):,}명"
                )
            if "date" in worker_df.columns and "ewi" in worker_df.columns:
                daily_ewi = worker_df.groupby("date")["ewi"].mean()
                for d, v in daily_ewi.items():
                    trend_lines.append(f"  - {date_label_short(d)} EWI: {v:.3f}")
            trend_summary = "\n".join(trend_lines)
            result = cached_weekly_trend_analysis(
                sector_id=sid,
                date_range=date_range,
                trend_summary=trend_summary,
            )
            render_data_comment(f"{date_range} 추이 요약", result)
