"""
AI Analysis — 탭별 AI 분석 렌더링 + 데이터 패키징
===================================================
모든 LLM 호출은 LLMGateway를 통해 이루어진다.
데이터 패키징은 DataPackager가 담당하며,
익명화는 AnonymizationPipeline이 의무적으로 처리한다.

보안 원칙:
  - 업체명: Company_A/B/... 코드로 자동 치환 (세션 내 일관)
  - 작업자명/ID: 완전 차단
  - 날짜: "분석 N일차(요일)" 상대화
  - 알고리즘 공식: ANONYMIZE_LOGIC=true 시 추상화
  - k-Anonymity: 10명 미만 집단 수치 억제
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

import config as cfg
from src.dashboard.llm_apollo import (
    is_llm_available,
    render_data_comment,
)
from src.intelligence.data_packager import DataPackager
from src.intelligence.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


# ========================================================================
# 공통: AI Analysis 탭 렌더링 래퍼
# ========================================================================

def render_ai_analysis_section(
    title: str,
    summary_text: str,
    cache_key: str,
    max_tokens: int = 900,
    tab_id: str = "generic",
    company_names: list[str] | None = None,
    worker_names: list[str] | None = None,
    date_list: list[str] | None = None,
):
    """
    AI 분석 섹션 렌더링.

    Args:
        title:         분석 제목
        summary_text:  DataPackager가 생성한 풍부한 데이터 텍스트
        cache_key:     세션 캐시 키
        max_tokens:    LLM 최대 응답 토큰
        tab_id:        탭 식별자 (LLMGateway 탭별 시스템 프롬프트 선택)
        company_names: 익명화할 업체명 목록
        worker_names:  익명화할 작업자명 목록
        date_list:     익명화할 날짜 목록
    """
    if not is_llm_available():
        st.info(
            "AI 분석을 사용하려면 `.env` 파일에 `ANTHROPIC_API_KEY`를 설정하거나, "
            "`LLM_BACKEND=bedrock`으로 AWS Bedrock을 구성하세요."
        )
        return

    st.markdown(
        "<div style='font-size:0.88rem; color:#7A8FA6; margin-bottom:12px;'>"
        "Claude AI가 익명화된 핵심 통계를 분석합니다. "
        "데이터 해석 보조 역할만 수행하며, 원인 단정이나 조치 권고는 하지 않습니다."
        "</div>",
        unsafe_allow_html=True,
    )

    # 캐시 확인
    if cache_key in st.session_state and st.session_state[cache_key]:
        render_data_comment(title, st.session_state[cache_key])
        if st.button("다시 분석", key=f"re_{cache_key}"):
            del st.session_state[cache_key]
            st.rerun()
        return

    if st.button("AI 분석 실행", key=f"btn_{cache_key}", type="primary"):
        with st.spinner("Claude AI 분석 중 (익명화 처리 포함)..."):
            result = LLMGateway.analyze(
                tab_id=tab_id,
                packed_text=summary_text,
                company_names=company_names,
                worker_names=worker_names,
                date_list=date_list,
                max_tokens=max_tokens,
            )
            if result:
                st.session_state[cache_key] = result
                render_data_comment(title, result)
            else:
                st.warning("AI 분석 결과를 가져오지 못했습니다. API 키와 네트워크를 확인하세요.")


# ========================================================================
# 1. 대기시간 (Transit) 분석
# ========================================================================

def build_transit_summary(
    worker_transit: pd.DataFrame,
    bp_transit: pd.DataFrame | None = None,  # 하위 호환용, 내부 미사용
) -> str:
    """대기시간 탭용 풍부한 데이터 텍스트 생성 (DataPackager 위임)."""
    dates = sorted(worker_transit["date"].unique().tolist()) \
        if "date" in worker_transit.columns else []
    return DataPackager.transit(worker_transit, dates=dates)


def render_transit_ai(
    worker_transit: pd.DataFrame,
    cache_key: str = "ai_transit",
    sid: str | None = None,
    anchor_date: str | None = None,
) -> None:
    """대기시간 탭 AI 분석 렌더링."""
    if worker_transit is None or worker_transit.empty:
        return

    import config as cfg
    sid = sid or cfg.SECTOR_ID

    dates = sorted(worker_transit["date"].unique().tolist()) \
        if "date" in worker_transit.columns else []
    company_names = worker_transit["company_name"].dropna().unique().tolist() \
        if "company_name" in worker_transit.columns else []

    packed = DataPackager.transit(worker_transit, dates=dates)

    # 비교 컨텍스트 추가 (단일 날짜 분석 시 가장 최신 날짜 기준)
    _anchor = anchor_date or (max(dates) if dates else None)
    if _anchor:
        try:
            from src.intelligence.context_builder import ContextBuilder
            ctx = ContextBuilder.build_transit(_anchor, sid)
            if ctx:
                packed = packed + "\n" + ctx
        except Exception as e:
            logger.debug("transit context build failed: %s", e)

    render_ai_analysis_section(
        title="대기/이동 시간 AI 분석",
        summary_text=packed,
        cache_key=cache_key,
        tab_id="transit",
        company_names=company_names,
        date_list=dates,
    )


# ========================================================================
# 2. 장비 가동률 (Equipment) 분석
# ========================================================================

def build_equipment_summary(
    weekly_bp_opr: pd.DataFrame | None,
    weekly_overall: pd.DataFrame | None,
    master_df: pd.DataFrame | None,
) -> str:
    """장비 탭용 풍부한 데이터 텍스트 생성 (DataPackager 위임)."""
    return DataPackager.equipment(master_df, weekly_bp_opr, weekly_overall)


def render_equipment_ai(
    master_df: pd.DataFrame | None,
    weekly_bp_df: pd.DataFrame | None,
    weekly_overall_df: pd.DataFrame | None,
    cache_key: str = "ai_equipment",
) -> None:
    """장비 탭 AI 분석 렌더링."""
    packed = DataPackager.equipment(master_df, weekly_bp_df, weekly_overall_df)

    # 업체명 수집
    company_names: list[str] = []
    for df in [weekly_bp_df, master_df]:
        if df is not None and not df.empty:
            for col in ["bp_name", "company_name"]:
                if col in df.columns:
                    company_names.extend(df[col].dropna().unique().tolist())
    company_names = list(set(company_names))

    render_ai_analysis_section(
        title="테이블 리프트 가동률 AI 분석",
        summary_text=packed,
        cache_key=cache_key,
        tab_id="equipment",
        company_names=company_names,
    )


# ========================================================================
# 3. 작업자 분석 (Worker/Daily)
# ========================================================================

def build_worker_analysis_summary(
    worker_df: pd.DataFrame,
    company_df: pd.DataFrame | None,
    date_str: str,
) -> str:
    """작업자 탭용 풍부한 데이터 텍스트 생성 (DataPackager 위임)."""
    return DataPackager.worker(worker_df, company_df, date_str)


def render_worker_ai(
    worker_df: pd.DataFrame,
    company_df: pd.DataFrame | None,
    date_str: str,
    cache_key: str = "ai_worker",
    sid: str | None = None,
) -> None:
    """작업자 탭 AI 분석 렌더링."""
    import config as cfg
    sid = sid or cfg.SECTOR_ID

    packed = DataPackager.worker(worker_df, company_df, date_str)

    # 비교 컨텍스트 추가
    if date_str:
        try:
            from src.intelligence.context_builder import ContextBuilder
            ctx = ContextBuilder.build_worker(date_str, sid)
            if ctx:
                packed = packed + "\n" + ctx
        except Exception as e:
            logger.debug("worker context build failed: %s", e)

    company_names: list[str] = []
    if company_df is not None and not company_df.empty:
        for col in ["company_name", "bp_name"]:
            if col in company_df.columns:
                company_names.extend(company_df[col].dropna().unique().tolist())
    if "company_name" in worker_df.columns:
        company_names.extend(worker_df["company_name"].dropna().unique().tolist())
    company_names = list(set(company_names))

    render_ai_analysis_section(
        title="작업자 분석 AI 인사이트",
        summary_text=packed,
        cache_key=cache_key,
        tab_id="worker",
        company_names=company_names,
        date_list=[date_str] if date_str else [],
    )


# ========================================================================
# 4. 혼잡도 (Congestion) 분석
# ========================================================================

def build_congestion_summary(
    congestion_summary: dict,
    ranking_df: pd.DataFrame | None = None,
) -> str:
    """혼잡도 탭용 풍부한 데이터 텍스트 생성 (DataPackager 위임)."""
    return DataPackager.congestion(congestion_summary, ranking_df)


def render_congestion_ai(
    summary_dict: dict,
    ranking_df: pd.DataFrame | None = None,
    hourly_df: pd.DataFrame | None = None,
    cache_key: str = "ai_congestion",
) -> None:
    """혼잡도 탭 AI 분석 렌더링."""
    packed = DataPackager.congestion(summary_dict, ranking_df, hourly_df)

    render_ai_analysis_section(
        title="공간 혼잡도 AI 분석",
        summary_text=packed,
        cache_key=cache_key,
        tab_id="congestion",
    )


# ========================================================================
# 5. 현장 개요 (Overview) 분석
# ========================================================================

def build_overview_summary(
    kpi: dict[str, Any],
    prev_kpi: dict[str, Any] | None = None,
    date_str: str = "",
) -> str:
    """현장 개요 탭용 풍부한 데이터 텍스트 생성 (DataPackager 위임)."""
    return DataPackager.overview(kpi, prev_kpi, date_str)


def render_overview_ai(
    kpi: dict[str, Any],
    prev_kpi: dict[str, Any] | None = None,
    date_str: str = "",
    cache_key: str = "ai_overview",
) -> None:
    """현장 개요 탭 AI 분석 렌더링."""
    packed = DataPackager.overview(kpi, prev_kpi, date_str)

    render_ai_analysis_section(
        title="현장 개요 AI 인사이트",
        summary_text=packed,
        cache_key=cache_key,
        tab_id="overview",
        date_list=[date_str] if date_str else [],
    )
