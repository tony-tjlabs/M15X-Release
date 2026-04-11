"""
Daily Tab Package
=================
일별 분석 탭 패키지 라우터.

5개 탭 + 패턴 탭을 서브모듈로 분리하여 구성.
render_daily_tab()을 re-export하여 기존 import 경로 유지.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as cfg
from src.pipeline.cache_manager import detect_processed_dates, load_daily_results, load_journey
from src.pipeline.summary_index import load_summary_index
from src.spatial.loader import load_locus_dict
from src.dashboard.date_utils import get_date_selector, fetch_weather_info

# 서브모듈 임포트
from .summary import render_summary
from .productivity import render_productivity, render_space_analysis_section
from .safety import render_safety
from .individual import render_individual
from .company import render_company
from .patterns import render_journey_patterns

__all__ = ["render_daily_tab"]


def render_daily_tab(sector_id: str | None = None):
    """일별 분석 탭 진입점."""
    sid = sector_id or cfg.SECTOR_ID
    processed = detect_processed_dates(sid)
    if not processed:
        st.info("처리된 데이터가 없습니다. [파이프라인] 탭에서 먼저 전처리를 실행하세요.")
        return

    # ── 날짜 선택 + KPI 프리뷰 ────────────────────────────────────
    # get_date_selector 사용 (요일 + 날씨 포함)
    summary_idx = load_summary_index(sid)
    summary_dates = summary_idx.get("dates", {})

    # available_dates를 session_state에 저장 (다른 탭에서 재사용)
    st.session_state["available_dates"] = processed

    col_sel, col_prev = st.columns([1, 3])
    with col_sel:
        date_str = get_date_selector(processed, key="daily_tab_date", show_label=True, label="분석 날짜")

    if not date_str:
        st.warning("날짜를 선택해 주세요.")
        return

    # 날짜 변경 시 journey 캐시 무효화
    _prev_date_key = f"daily_prev_date_{sid}"
    if st.session_state.get(_prev_date_key) != date_str:
        _old = st.session_state.get(_prev_date_key)
        if _old:
            st.session_state.pop(f"journey_{sid}_{_old}", None)
        st.session_state[_prev_date_key] = date_str

    # Summary Index 기반 즉시 KPI 프리뷰 (Parquet 로드 전)
    with col_prev:
        _render_kpi_preview(summary_dates, date_str)

    try:
        data = load_daily_results(date_str, sid)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    meta = data.get("meta", {})
    worker_df = data.get("worker", pd.DataFrame())
    space_df = data.get("space", pd.DataFrame())
    company_df = data.get("company", pd.DataFrame())
    transit_df = data.get("transit", pd.DataFrame())
    locus_dict = load_locus_dict(sid)

    # transit_df → worker_df에 대기시간(MAT/LBT/LMT/EOD) 병합
    if not transit_df.empty and not worker_df.empty:
        _transit_cols = ["user_no"]
        for _tc in ["mat_minutes", "lbt_minutes", "lmt_minutes", "eod_minutes"]:
            if _tc in transit_df.columns and _tc not in worker_df.columns:
                _transit_cols.append(_tc)
        if len(_transit_cols) > 1:
            worker_df = worker_df.merge(
                transit_df[_transit_cols], on="user_no", how="left",
            )

    if worker_df.empty:
        st.warning("작업자 데이터 없음")
        return

    has_ewi = "ewi" in worker_df.columns
    has_cre = "cre" in worker_df.columns

    # BLE 커버리지 등급 동적 계산 (기존 parquet 호환)
    if "ble_coverage" not in worker_df.columns and "gap_ratio" in worker_df.columns:
        import numpy as np
        gr = worker_df["gap_ratio"].fillna(1.0)
        worker_df["ble_coverage"] = np.where(
            gr <= 0.2, "정상",
            np.where(gr <= 0.5, "부분음영",
                     np.where(gr <= 0.8, "음영", "미측정"))
        )
        worker_df["ble_coverage_pct"] = ((1 - gr) * 100).clip(0, 100).round(1)

    # ── 탭 구성 ────────────────────────────────────────────────────
    tabs = st.tabs(["📊 요약", "⚡ 생산성", "🛡️ 안전", "👷 개인별", "🏗️ 업체별", "🧬 패턴", "AI Analysis"])

    with tabs[0]:
        render_summary(meta, worker_df, space_df, locus_dict, date_str, sid, has_ewi, has_cre)

    with tabs[1]:
        render_productivity(worker_df, space_df, locus_dict, has_ewi)

    with tabs[2]:
        # journey는 session_state에서 재사용 (탭 전환 시 중복 로드 방지)
        _cache_key = f"journey_{sid}_{date_str}"
        journey_safety = st.session_state.get(_cache_key)
        if journey_safety is None:
            with st.spinner("이동 데이터 로드 중..."):
                journey_safety = load_journey(date_str, sid)
            st.session_state[_cache_key] = journey_safety
        render_safety(worker_df, space_df, locus_dict, date_str, sid, has_cre, journey_safety)

    with tabs[3]:
        # journey 로드 (session_state 캐시)
        _cache_key_ind = f"journey_{sid}_{date_str}"
        _journey_ind = st.session_state.get(_cache_key_ind)
        if _journey_ind is None:
            with st.spinner("이동 데이터 로드 중..."):
                _journey_ind = load_journey(date_str, sid)
            st.session_state[_cache_key_ind] = _journey_ind
        st.session_state["current_journey_df"] = _journey_ind
        render_individual(worker_df, locus_dict, has_ewi, has_cre)

    with tabs[4]:
        render_company(company_df, worker_df, has_ewi, has_cre)

    with tabs[5]:
        # session_state 캐시에서 재사용 (tabs[2]에서 이미 로드했을 수 있음)
        _cache_key = f"journey_{sid}_{date_str}"
        journey_df = st.session_state.get(_cache_key)
        if journey_df is None:
            with st.spinner("이동 데이터 로드 중..."):
                journey_df = load_journey(date_str, sid)
            st.session_state[_cache_key] = journey_df
        render_journey_patterns(worker_df, journey_df, sid)

    with tabs[6]:
        from src.dashboard.ai_analysis import render_worker_ai
        render_worker_ai(
            worker_df=worker_df,
            company_df=company_df,
            date_str=date_str,
            cache_key=f"ai_worker_{sid}_{date_str}",
            sid=sid,
        )


def _render_kpi_preview(summary_dates: dict, date_str: str):
    """Summary Index 기반 KPI 프리뷰 칩 렌더링."""
    entry = summary_dates.get(date_str)
    if not entry:
        return

    avg_cre = entry.get("avg_cre")
    avg_ewi = entry.get("avg_ewi")
    high_cre = entry.get("high_cre_count", 0)
    top_title = entry.get("top_insight_title", "")

    chips = []
    chips.append(
        f"<span style='background:#1A2A3A; border:1px solid #2A3A4A; "
        f"border-radius:4px; padding:2px 8px; color:#C8D6E8;'>"
        f"📡 {entry.get('total_workers_access', 0):,}명</span>"
    )
    if avg_ewi is not None:
        chips.append(
            f"<span style='background:#1A2A3A; border:1px solid #2A3A4A; "
            f"border-radius:4px; padding:2px 8px; color:#00AEEF;'>"
            f"EWI {avg_ewi:.3f}</span>"
        )
    if avg_cre is not None:
        cre_col = "#FF4C4C" if avg_cre >= 0.5 else "#FFB300" if avg_cre >= 0.35 else "#00C897"
        chips.append(
            f"<span style='background:#1A2A3A; border:1px solid #2A3A4A; "
            f"border-radius:4px; padding:2px 8px; color:{cre_col};'>"
            f"CRE {avg_cre:.3f}</span>"
        )
    if high_cre:
        chips.append(
            f"<span style='background:#2A1A1A; border:1px solid #FF4C4C44; "
            f"border-radius:4px; padding:2px 8px; color:#FF4C4C;'>"
            f"고위험 {high_cre}명</span>"
        )
    if top_title:
        sev_colors = {4: "#FF4C4C", 3: "#FF8C42", 2: "#FFB300", 1: "#7A8FA6"}
        top_color = sev_colors.get(entry.get("top_insight_severity", 0), "#7A8FA6")
        chips.append(
            f"<span style='background:#1A2A3A; border:1px solid {top_color}44; "
            f"border-radius:4px; padding:2px 8px; color:{top_color};'>"
            f"💡 {top_title}</span>"
        )

    st.markdown(
        "<div style='display:flex; gap:6px; flex-wrap:wrap; padding:6px 0;'>"
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )
