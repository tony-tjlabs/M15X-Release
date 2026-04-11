"""
DeepCon-M15X — M15X FAB 건설현장 분석 대시보드
===============================================
고객사 제공용 프로페셔널 대시보드.

실행: streamlit run main.py --server.port 8530

탭 구조 (6개):
  1. 현장 개요 — 일별 KPI 요약 + 공간 현황
  2. 작업자 대기/이동 시간 — MAT/LBT/EOD 분석 (고객 핵심 요구 1)
  3. 테이블 리프트 — 주간 BP별 가동률 (고객 핵심 요구 2)
  4. 작업자 분석 — EWI/CRE/SII 심층 분석
  5. 공간 혼잡도 — 층별/구역별 인원 밀집 현황
  6. 데이터 관리 — 파이프라인 실행 + 캐시 상태 (Admin)
"""
import streamlit as st

# -- 페이지 설정 (반드시 첫 번째) ------------------------------------------
st.set_page_config(
    page_title="Deep Con at M15X",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import config as cfg
from src.dashboard.auth import (
    is_logged_in,
    require_login,
    get_current_user,
    get_current_sector,
    set_current_sector,
    get_allowed_sectors,
    is_admin,
    logout,
)
from src.dashboard.styles import (
    inject_css,
    metric_card,
    metric_card_sm,
    section_header,
    sub_header,
    badge,
    COLORS,
    PLOTLY_DARK,
)
from src.dashboard.date_utils import (
    DAY_NAMES_KR,
    WEATHER_ICONS,
    fetch_weather_info,
    get_date_selector,
    get_weekday_korean,
    get_date_badge,
    parse_date_str,
    format_date_label,
)

# -- 로그인 게이트 ---------------------------------------------------------
inject_css()
require_login()


# ==========================================================================
# 헬퍼 함수
# ==========================================================================

@st.cache_data(show_spinner=False, ttl=600)
def _get_available_dates(sector_id: str) -> list[str]:
    """processed 폴더에서 사용 가능한 날짜 목록 (오름차순)."""
    try:
        from src.pipeline.cache_manager import detect_processed_dates
        return detect_processed_dates(sector_id)
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=300)
def _get_available_bps(sector_id: str, date_str: str) -> list[str]:
    """특정 날짜의 BP(업체) 목록."""
    try:
        paths = cfg.get_sector_paths(sector_id)
        company_path = paths["processed_dir"] / date_str / "company.parquet"
        if company_path.exists():
            df = pd.read_parquet(company_path, columns=["company_name"])
            return sorted(df["company_name"].dropna().unique().tolist())
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False, ttl=300)
def _load_daily_cache(sector_id: str, date_str: str) -> dict[str, Any]:
    """
    Parquet 캐시 로드.

    반환 키: worker, space, company, meta, transit, bp_transit
    """
    data: dict[str, Any] = {}
    try:
        paths = cfg.get_sector_paths(sector_id)
        date_dir = paths["processed_dir"] / date_str

        # 핵심 캐시 파일들
        for name in ["worker", "space", "company"]:
            p = date_dir / f"{name}.parquet"
            data[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()

        # 메타데이터
        meta_path = date_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                data["meta"] = json.load(f)
        else:
            data["meta"] = {}

        # 대기시간 캐시 (옵션)
        transit_path = date_dir / "transit.parquet"
        data["transit"] = pd.read_parquet(transit_path) if transit_path.exists() else pd.DataFrame()

        bp_transit_path = date_dir / "bp_transit.parquet"
        data["bp_transit"] = pd.read_parquet(bp_transit_path) if bp_transit_path.exists() else pd.DataFrame()

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"캐시 로드 실패 [{date_str}]: {e}")

    return data


def _get_weekday_korean(d: date) -> str:
    """요일 한글 반환 (date 객체용)."""
    return get_weekday_korean(d)


def _get_date_badge(d: date) -> str:
    """날짜 뱃지 HTML (주중/주말) - date 객체용."""
    return get_date_badge(d)


@st.cache_data(ttl=300, show_spinner="전체 기간 데이터 로딩 중...")
def _load_all_daily_summary(sector_id: str, dates: tuple) -> tuple:
    """
    전체 기간 일별 요약 로드 (캐시).

    Args:
        sector_id: 섹터 ID
        dates: 날짜 목록 (tuple로 받아야 캐시 가능)

    Returns:
        (workers_df, transit_df, company_df, summary_df)
    """
    all_workers = []
    all_transit = []
    all_company = []
    daily_summary = []

    for d in dates:
        cache = _load_daily_cache(sector_id, d)
        worker_df = cache.get("worker")
        transit_df = cache.get("transit")
        company_df = cache.get("company")

        if worker_df is not None and not worker_df.empty:
            w = worker_df.copy()
            w["date"] = d
            all_workers.append(w)

            # 일별 요약
            daily_summary.append({
                "date": d,
                "n_workers": len(worker_df),
                "avg_ewi": worker_df["ewi"].mean() if "ewi" in worker_df.columns else 0,
                "avg_cre": worker_df["cre"].mean() if "cre" in worker_df.columns else 0,
            })

        if transit_df is not None and not transit_df.empty:
            t = transit_df.copy()
            t["date"] = d
            all_transit.append(t)

        if company_df is not None and not company_df.empty:
            c = company_df.copy()
            c["date"] = d
            all_company.append(c)

    workers_df = pd.concat(all_workers, ignore_index=True) if all_workers else pd.DataFrame()
    transit_df_all = pd.concat(all_transit, ignore_index=True) if all_transit else pd.DataFrame()
    company_df_all = pd.concat(all_company, ignore_index=True) if all_company else pd.DataFrame()
    summary_df = pd.DataFrame(daily_summary) if daily_summary else pd.DataFrame()

    return workers_df, transit_df_all, company_df_all, summary_df


# ==========================================================================
# 사이드바 렌더링
# ==========================================================================

def render_sidebar() -> tuple[str, str]:
    """
    사이드바 렌더링.

    날짜/BP 필터는 각 탭 내부에서 처리.
    available_dates는 session_state에 저장.

    반환: (page, sector_id)
    """
    user = get_current_user()
    sector_id = get_current_sector() or cfg.SECTOR_ID

    with st.sidebar:
        # -- 브랜드 섹션 ------------------------------------------------
        st.markdown(
            f"""
            <div style='text-align:center; padding: 16px 0 12px 0;'>
                <div style='font-size:2.2rem;'>🏭</div>
                <div style='font-size:1.3rem; font-weight:700; color:{cfg.THEME_ACCENT};
                            letter-spacing:1px;'>Deep Con at M15X</div>
                <div style='font-size:0.72rem; color:#7A8FA6; margin-top:4px;'>
                    Spatial Data Analysis<br>using Deep Con Prototype
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # -- 메뉴 네비게이션 --------------------------------------------
        st.markdown(
            "<div style='font-size:0.8rem; color:#7A8FA6; margin-bottom:6px;'>"
            "메뉴</div>",
            unsafe_allow_html=True,
        )

        # Admin 전용 탭 포함 여부
        if is_admin() and not cfg.CLOUD_MODE:
            pages = [
                "📊 현장 개요",
                "⏱️ 작업자 대기/이동 시간",
                "🏗️ 테이블 리프트",
                "👷 작업자 분석",
                "🏢 공간 혼잡도",
                "📋 리포트 생성",
                "⚙️ 데이터 관리",
            ]
        else:
            pages = [
                "📊 현장 개요",
                "⏱️ 작업자 대기/이동 시간",
                "🏗️ 테이블 리프트",
                "👷 작업자 분석",
                "🏢 공간 혼잡도",
                "📋 리포트 생성",
            ]

        page = st.radio(
            "페이지",
            pages,
            label_visibility="collapsed",
            key="page_radio",
        )

        st.divider()

        # -- 데이터 기간 안내 -------------------------------------------
        # available_dates 조회 및 session_state 저장
        available_dates = _get_available_dates(sector_id)
        st.session_state["available_dates"] = available_dates

        data_info = _get_data_period_info(sector_id)
        st.markdown(
            f"""
            <div style='background:{cfg.THEME_CARD_BG}; border-radius:8px; padding:12px;'>
                <div style='font-size:0.78rem; color:#7A8FA6;'>데이터 기간</div>
                <div style='font-size:0.92rem; color:{cfg.THEME_ACCENT}; font-weight:600; margin-top:4px;'>
                    {data_info['period']}
                </div>
                <div style='font-size:0.75rem; color:#5A6A7A; margin-top:6px;'>
                    {data_info['summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not available_dates:
            st.warning("처리된 데이터가 없습니다.")
            st.markdown(
                "<div style='font-size:0.82rem; color:#9AB5D4;'>"
                "데이터 관리 탭에서 전처리를 실행하세요.</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # -- 사용자 정보 + 로그아웃 -------------------------------------
        st.markdown(
            f"""
            <div style='font-size:0.82rem; color:#7A8FA6;'>
                {user['icon']} <b style='color:#C8D6E8'>{user['label']}</b><br>
                <span style='font-size:0.72rem; color:#3A4A5A;'>
                    {'관리자' if is_admin() else '클라이언트'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("🚪 로그아웃", use_container_width=True, key="logout_btn"):
            logout()

        st.markdown(
            f"<div style='text-align:center; color:#2A3A4A; font-size:0.72rem;"
            f"margin-top:12px;'>v{cfg.APP_VERSION} | TJLABS Research</div>",
            unsafe_allow_html=True,
        )

    return page, sector_id


def _get_data_period_info(sector_id: str) -> dict[str, str]:
    """데이터 기간 정보 반환."""
    dates = _get_available_dates(sector_id)
    if not dates:
        return {"period": "데이터 없음", "summary": ""}

    start = dates[0]
    end = dates[-1]
    n_days = len(dates)

    return {
        "period": f"{start} ~ {end}",
        "summary": f"{n_days}일 데이터",
    }


# ==========================================================================
# 탭 1: 현장 개요 (전체 기간 요약)
# ==========================================================================

def _render_overview_tab(sector_id: str, available_dates: list[str]) -> None:
    """현장 개요 탭 렌더링 - 전체 기간 요약 대시보드."""
    import plotly.express as px
    import plotly.graph_objects as go

    if not available_dates:
        st.info("처리된 데이터가 없습니다. 데이터 관리 탭에서 전처리를 실행하세요.")
        return

    # 전체 날짜의 데이터 로드 (캐시 적용)
    # tuple로 변환해야 캐시 키로 사용 가능
    workers_df, transit_df, company_df, summary_df = _load_all_daily_summary(
        sector_id, tuple(available_dates)
    )

    if workers_df.empty:
        st.warning("전처리 데이터가 없습니다.")
        return

    # -- 페이지 헤더 ----------------------------------------------------
    start_date = available_dates[0]
    end_date = available_dates[-1]
    n_days = len(available_dates)

    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px;'>
            <div style='font-size:1.5rem; font-weight:700; color:{cfg.THEME_TEXT};'>
                전체 기간 요약
            </div>
            {badge(f"{n_days}일", "info")}
        </div>
        <div style='font-size:0.88rem; color:#7A8FA6; margin-bottom:20px;'>
            {start_date} ~ {end_date}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -- Row 1: 전체 KPI 카드 (3열 x 2행) -----------------------------------
    # 연인원 (일별 작업자 수 단순 합산)
    total_延인원 = int(summary_df["n_workers"].sum()) if not summary_df.empty else 0
    # 고유 작업자 (전체 기간 unique user_no = unique T-Ward 기준)
    total_unique_workers = workers_df["user_no"].nunique() if "user_no" in workers_df.columns else 0
    # 일 평균 / 일 최대
    daily_avg_workers = summary_df["n_workers"].mean() if not summary_df.empty else 0
    daily_max_workers = int(summary_df["n_workers"].max()) if not summary_df.empty else 0
    # 평균 EWI/CRE
    overall_ewi = summary_df["avg_ewi"].mean() if not summary_df.empty else 0
    overall_cre = summary_df["avg_cre"].mean() if not summary_df.empty else 0

    # 상단 3열: 작업자 현황
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("일 평균 작업자", f"{daily_avg_workers:.0f}명"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("일 최대 작업자", f"{daily_max_workers:,}명"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("연인원 (일별 합산)", f"{total_延인원:,}명"), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # 하단 3열: 고유 작업자 + EWI + CRE
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(metric_card("고유 작업자 (T-Ward 기준)", f"{total_unique_workers:,}명"), unsafe_allow_html=True)
    with col5:
        ewi_color = COLORS["success"] if overall_ewi >= 0.6 else COLORS["warning"] if overall_ewi >= 0.2 else COLORS["danger"]
        st.markdown(metric_card("평균 EWI", f"{overall_ewi:.2f}", color=ewi_color), unsafe_allow_html=True)
    with col6:
        cre_color = COLORS["danger"] if overall_cre >= 0.7 else COLORS["warning"] if overall_cre >= 0.4 else COLORS["success"]
        st.markdown(metric_card("평균 CRE", f"{overall_cre:.2f}", color=cre_color), unsafe_allow_html=True)

    # -- 메트릭 정의 (접이식) ------------------------------------------
    with st.expander("📌 지표 정의 — EWI · CRE · MAT · LBT · EOD", expanded=False):
        st.markdown(
            """
<div style='display:grid; grid-template-columns:1fr 1fr; gap:12px 32px; font-size:0.84rem; line-height:1.7;'>

<div>

**생산성 지표**

| 약어 | 전체 명칭 | 의미 |
|------|-----------|------|
| **EWI** | Effective Work Index | 실효 작업 지수. Σ(공간가중치 × 활동가중치) / 총 작업시간. 공간(작업구역=1.0, 이동/휴식=0) × 활동(활성신호 비율)을 결합한 종합 집중도. 범위 0–1, 높을수록 작업 집중도 높음 |
| **CRE** | Congestion Risk Estimator | 혼잡 위험 지수. 특정 공간·시간대의 밀집도를 0–1로 정규화. 높을수록 병목·안전 위험 |

</div>

<div>

**대기/이동 시간 지표**

| 약어 | 전체 명칭 | 의미 |
|------|-----------|------|
| **MAT** | Morning Arrival Transit | 출근 대기시간. AccessLog 입장 기록 → T-Ward 첫 work_zone 감지까지 (분) |
| **LBT** | Lunch Break Transit | 중식 왕복 이동시간. FAB → 타각기(LBT-Out) + 타각기 → FAB(LBT-In) 합계 (분) |
| **LMT** | Lunch Meal Time | 점심 외부 체류시간. 현장 전체 BLE 무신호 구간 (타각기 통과 확인된 경우만) (분) |
| **EOD** | End-of-Day Transit | 퇴근 이동시간. T-Ward 마지막 work_zone 감지 → AccessLog 퇴장 기록까지 (분) |
| **TWT** | Total Wait Time | MAT + LBT + EOD 합계. 하루 중 비작업 이동·대기 총합 (분) |

</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -- Row 2: 대기시간 요약 카드 (5열) --------------------------------
    # Coverage 필터 적용 (transit_tab과 동일 기준)
    if not transit_df.empty:
        if "is_coverage_excluded" in transit_df.columns:
            _kpi_df = transit_df[~transit_df["is_coverage_excluded"]]
        else:
            _kpi_df = transit_df
        avg_mat = _kpi_df["mat_minutes"].dropna().mean() if "mat_minutes" in _kpi_df.columns else 0
        avg_lbt = _kpi_df["lbt_minutes"].dropna().mean() if "lbt_minutes" in _kpi_df.columns else 0
        avg_lmt = _kpi_df["lmt_minutes"].dropna().mean() if "lmt_minutes" in _kpi_df.columns else 0
        avg_eod = _kpi_df["eod_minutes"].dropna().mean() if "eod_minutes" in _kpi_df.columns else 0
    else:
        avg_mat, avg_lbt, avg_lmt, avg_eod = 0, 0, 0, 0

    avg_twt = avg_mat + avg_lbt + avg_eod  # TWT = MAT + LBT + EOD

    st.markdown(section_header("전체 기간 대기시간"), unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(metric_card_sm("평균 총 대기 (TWT)", f"{avg_twt:.1f}분"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_sm("출근 대기 (MAT)", f"{avg_mat:.1f}분"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_sm("중식 이동 (LBT)", f"{avg_lbt:.1f}분"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card_sm("점심 외부 (LMT)", f"{avg_lmt:.1f}분"), unsafe_allow_html=True)
    with col5:
        st.markdown(metric_card_sm("퇴근 이동 (EOD)", f"{avg_eod:.1f}분"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -- Row 3: 작업자 시간 분류 (Time Breakdown) -----------------------
    _render_time_breakdown_section(workers_df, transit_df)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -- Row 4: 일별 추이 차트 (2열) ------------------------------------
    st.markdown(section_header("일별 추이"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(sub_header("일별 작업자 수"), unsafe_allow_html=True)
        summary_df["date_label"] = summary_df["date"].apply(format_date_label)
        fig_workers = px.bar(
            summary_df,
            x="date_label",
            y="n_workers",
            color_discrete_sequence=[cfg.THEME_ACCENT],
        )
        fig_workers.update_layout(
            **PLOTLY_DARK,
            height=280,
            showlegend=False,
            xaxis_title="날짜",
            yaxis_title="작업자 수",
        )
        st.plotly_chart(fig_workers, use_container_width=True)

    with col2:
        st.markdown(sub_header("일별 평균 EWI"), unsafe_allow_html=True)
        fig_ewi = go.Figure()
        fig_ewi.add_trace(go.Scatter(
            x=summary_df["date_label"],
            y=summary_df["avg_ewi"],
            mode="lines+markers",
            line=dict(color=cfg.THEME_ACCENT, width=3),
            marker=dict(size=8),
            hovertemplate="EWI: %{y:.3f}<extra></extra>",
        ))
        fig_ewi.update_layout(
            **PLOTLY_DARK,
            height=280,
            showlegend=False,
            xaxis_title="날짜",
            yaxis_title="평균 EWI",
        )
        st.plotly_chart(fig_ewi, use_container_width=True)

    # -- Row 5: 대기시간 추이 + 업체 TOP 10 (2열) ------------------------
    col1, col2 = st.columns([7, 5])

    with col1:
        st.markdown(sub_header("일별 평균 대기시간 추이"), unsafe_allow_html=True)

        if not transit_df.empty and "date" in transit_df.columns:
            # 일별 대기시간 집계
            agg_cols = {c: (lambda x: x.dropna().mean()) for c in ["mat_minutes", "lbt_minutes", "lmt_minutes", "eod_minutes"] if c in transit_df.columns}
            transit_daily = transit_df.groupby("date").agg(agg_cols).reset_index()
            transit_daily = transit_daily.sort_values("date")
            transit_daily["date_label"] = transit_daily["date"].apply(format_date_label)

            fig_transit = go.Figure()
            for col, color, name in [
                ("mat_minutes", "#00AEEF", "MAT"),
                ("lbt_minutes", "#FFB300", "LBT"),
                ("lmt_minutes", "#A78BFA", "LMT"),
                ("eod_minutes", "#FF8C42", "EOD"),
            ]:
                if col in transit_daily.columns:
                    fig_transit.add_trace(go.Scatter(
                        x=transit_daily["date_label"],
                        y=transit_daily[col],
                        mode="lines+markers",
                        name=name,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                    ))

            fig_transit.update_layout(
                height=300,
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
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_transit, use_container_width=True)
        else:
            st.info("대기시간 데이터가 없습니다.")

    with col2:
        st.markdown(sub_header("업체별 작업자 수 TOP 10"), unsafe_allow_html=True)
        _render_company_top10_period(company_df)

    # -- Row 5: 지표 정의 (접힌 상태) -----------------------------------
    with st.expander("지표 정의 및 계산 방법", expanded=False):
        st.markdown("""
| 지표 | 정의 | 계산 방법 |
|------|------|----------|
| **일 평균 작업자** | 하루 평균 출입 작업자 수 | 일별 AccessLog unique User_no 평균 |
| **일 최대 작업자** | 하루 최대 출입 작업자 수 | 일별 AccessLog unique User_no 최대 |
| **연인원** | 일별 작업자 수를 단순 합산 | 각 날짜의 작업자 수를 모두 더한 값. 동일인이 매일 출근하면 매번 카운트 |
| **고유 작업자 (T-Ward 기준)** | 전체 기간 고유 작업자 수 | 전체 기간 unique User_no. T-Ward 태그 기준으로 실제 몇 명이 현장에 투입되었는지 |
| **EWI** | Effective Work Intensity (유효작업집중도) | (고활성시간 x 1.0 + 저활성시간 x 0.5) / 근무시간. 높을수록 작업 집중도 높음 |
| **CRE** | Construction Risk Exposure (건설위험노출도) | 공간위험도(40%) + 동적밀집(15%) + 개인위험(45%). 높을수록 위험 노출 높음 |
| **MAT** | Morning Arrival Time (출근 대기) | 타각기 출근 타각 -> 작업 공간(FAB 작업층 또는 야외 공사현장) 최초 도착까지 소요 시간 (분) |
| **LBT** | Lunch Break Transit (중식 이동) | 중식(11~14시) FAB 작업층 -> 타각기 출문까지 소요 시간 (분) |
| **EOD** | End-of-Day Transit (퇴근 이동) | 퇴근(17시 이후) FAB 작업층 -> 타각기 출문까지 소요 시간 (분) |
        """)


def _compute_time_breakdown(workers_df: pd.DataFrame, transit_df: pd.DataFrame) -> dict:
    """
    workers_df + transit_df 에서 4개 카테고리 평균 시간(분)과 비율을 계산.

    Returns: dict with keys
        work_min, transit_min, rest_min, lunch_min, unclassified_min,
        work_pct, transit_pct, rest_pct, lunch_pct, unclassified_pct
    """
    import numpy as np

    df = workers_df.copy()

    # work_dwell_min fallback
    if "work_dwell_min" not in df.columns:
        work_cols = [c for c in ["intense_work_min", "light_work_min", "idle_at_work_min"] if c in df.columns]
        df["work_dwell_min"] = df[work_cols].sum(axis=1) if work_cols else 0

    # transit_dwell_min fallback
    if "transit_dwell_min" not in df.columns:
        t_cols = [c for c in ["transit_min", "in_transit_min"] if c in df.columns]
        df["transit_dwell_min"] = df[t_cols].sum(axis=1) if t_cols else 0

    # rest_dwell_min fallback
    if "rest_dwell_min" not in df.columns:
        df["rest_dwell_min"] = df["rest_min"] if "rest_min" in df.columns else 0

    avg_work    = float(df["work_dwell_min"].mean()) if "work_dwell_min" in df.columns else 0.0
    avg_transit = float(df["transit_dwell_min"].mean()) if "transit_dwell_min" in df.columns else 0.0
    avg_rest    = float(df["rest_dwell_min"].mean()) if "rest_dwell_min" in df.columns else 0.0

    # lmt_minutes는 transit.parquet에서 직접 평균 계산 (user_no merge 시 멀티데이 cartesian product 문제)
    avg_lunch = 0.0
    if not transit_df.empty and "lmt_minutes" in transit_df.columns:
        lmt_vals = transit_df["lmt_minutes"].dropna()
        avg_lunch = float(lmt_vals.mean()) if len(lmt_vals) > 0 else 0.0

    # shadow_min: Phase 2 보간 후에도 남아있는 BLE 미감지 (30분 이상 gap)
    avg_shadow = float(df["shadow_min"].mean()) if "shadow_min" in df.columns else 0.0

    # 총 근무시간 기준 분모
    avg_work_minutes = float(df["work_minutes"].mean()) if "work_minutes" in df.columns else 0.0
    # 명시적으로 집계된 항목 합계
    classified_sum = avg_work + avg_transit + avg_rest + avg_lunch + avg_shadow
    # 나머지 residual (shadow_zone 이외의 미집계 — 이론상 매우 작아야 함)
    avg_residual = max(0.0, avg_work_minutes - classified_sum)

    denom = avg_work_minutes if avg_work_minutes > 0 else max(classified_sum, 1.0)

    def pct(v: float) -> float:
        return round(v / denom * 100, 1)

    return {
        "work_min":          round(avg_work, 1),
        "transit_min":       round(avg_transit, 1),
        "rest_min":          round(avg_rest, 1),
        "lunch_min":         round(avg_lunch, 1),
        "unclassified_min":  round(avg_shadow + avg_residual, 1),  # 30분+ 음영 + 기타 잔여
        "work_pct":          pct(avg_work),
        "transit_pct":       pct(avg_transit),
        "rest_pct":          pct(avg_rest),
        "lunch_pct":         pct(avg_lunch),
        "unclassified_pct":  pct(avg_shadow + avg_residual),
    }


def _render_time_breakdown_section(workers_df: pd.DataFrame, transit_df: pd.DataFrame) -> None:
    """작업자 시간 분류(Time Breakdown) 섹션 — 도넛 차트 + 수치 테이블 + 업체별 스택 바."""
    import numpy as np
    import plotly.graph_objects as go

    st.markdown(section_header("작업자 시간 분류 (Time Breakdown)"), unsafe_allow_html=True)

    # ── 전체 평균 계산 ────────────────────────────────────────────────
    tb = _compute_time_breakdown(workers_df, transit_df)

    # ── 2열 레이아웃: 도넛 차트 | 수치 테이블 ────────────────────────
    col1, col2 = st.columns([5, 5])

    with col1:
        st.markdown(sub_header("전체 평균 시간 분류"), unsafe_allow_html=True)

        labels = ["작업공간 체류", "이동", "휴식", "점심 외부", "BLE 미감지"]
        values = [
            tb["work_min"], tb["transit_min"], tb["rest_min"],
            tb["lunch_min"], tb["unclassified_min"],
        ]
        colors = ["#00AEEF", "#FFB300", "#4CAF50", "#A78BFA", "#5A6A7A"]

        # 0 이하 값 제거
        filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
        if not filtered:
            st.info("시간 분류 데이터가 없습니다.")
        else:
            f_labels, f_values, f_colors = zip(*filtered)
            center_text = f"평균<br><b>{tb['work_pct']}%</b><br>작업 집중"

            fig_donut = go.Figure(go.Pie(
                labels=list(f_labels),
                values=list(f_values),
                hole=0.55,
                marker=dict(colors=list(f_colors)),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.1f}분 (%{percent})<extra></extra>",
                textfont=dict(size=11),
            ))
            fig_donut.update_layout(
                paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
                plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                font_color=PLOTLY_DARK["font_color"],
                margin=dict(l=10, r=10, t=30, b=10),
                height=300,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                annotations=[dict(
                    text=center_text,
                    x=0.5, y=0.5,
                    font=dict(size=13, color="#D5E5FF"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown(sub_header("카테고리별 수치"), unsafe_allow_html=True)

        rows = [
            ("작업공간 체류", tb["work_min"],         tb["work_pct"],         "#00AEEF"),
            ("이동",          tb["transit_min"],       tb["transit_pct"],       "#FFB300"),
            ("휴식",          tb["rest_min"],          tb["rest_pct"],          "#4CAF50"),
            ("점심 외부",      tb["lunch_min"],         tb["lunch_pct"],         "#A78BFA"),
            ("BLE 미감지",     tb["unclassified_min"],  tb["unclassified_pct"],  "#5A6A7A"),
        ]

        # ── div 기반 테이블 (Streamlit이 <table>/<tr>/<td> 스트립) ──
        grid = "display:grid; grid-template-columns:1fr auto auto; font-size:0.86rem;"
        header_html = (
            f"<div style='{grid} border-bottom:1px solid #2A3A4A; color:#9AB5D4; padding:6px 0;'>"
            "<div style='font-weight:600; padding:0 4px;'>카테고리</div>"
            "<div style='font-weight:600; padding:0 4px; text-align:right;'>평균(분)</div>"
            "<div style='font-weight:600; padding:0 4px; text-align:right; min-width:110px;'>비율</div>"
            "</div>"
        )
        rows_html = ""
        for label, minutes, pct_val, color in rows:
            bar_pct = min(pct_val, 100)
            rows_html += (
                f"<div style='{grid} border-bottom:1px solid #1A2A3A; padding:7px 0; align-items:center;'>"
                f"<div style='padding:0 4px;'>"
                f"<span style='display:inline-block; width:10px; height:10px; "
                f"background:{color}; border-radius:2px; margin-right:6px; vertical-align:middle;'></span>"
                f"<span style='color:#D5E5FF;'>{label}</span></div>"
                f"<div style='text-align:right; padding:0 4px; color:#D5E5FF; font-weight:600;'>{minutes:.1f}분</div>"
                f"<div style='text-align:right; padding:0 4px;'>"
                f"<div style='display:flex; align-items:center; justify-content:flex-end; gap:6px;'>"
                f"<div style='background:#1A2A3A; border-radius:3px; width:60px; height:8px; overflow:hidden;'>"
                f"<div style='background:{color}; width:{bar_pct}%; height:100%; border-radius:3px;'></div></div>"
                f"<span style='color:{color}; font-weight:600; min-width:36px; text-align:right;'>{pct_val}%</span>"
                f"</div></div></div>"
            )
        st.markdown(header_html + rows_html, unsafe_allow_html=True)

        # BLE 미감지 안내
        if tb.get("unclassified_pct", 0) > 5:
            st.markdown(
                "<div style='margin-top:10px; padding:8px 12px; background:#1A2530; "
                "border-left:3px solid #5A6A7A; border-radius:4px; font-size:0.80rem; color:#8A9BB0;'>"
                "ℹ️ <b>BLE 미감지</b>: 작업자가 FAB 내부에 있으나 BLE 센서 음영 구역에 위치하여 "
                "위치가 감지되지 않은 시간입니다. 실제 작업 중이지만 집계에서 제외됩니다."
                "</div>",
                unsafe_allow_html=True,
            )

    # ── 업체별 시간 분류 스택 바 ──────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _render_company_time_breakdown(workers_df, transit_df)


def _render_company_time_breakdown(workers_df: pd.DataFrame, transit_df: pd.DataFrame) -> None:
    """업체별 시간 분류 가로 스택 바 차트 (TOP 10)."""
    import numpy as np
    import plotly.graph_objects as go

    # 업체 컬럼 탐색
    company_col = next(
        (c for c in ["hcon_company_name", "company_name", "bp_name"] if c in workers_df.columns),
        None,
    )
    if company_col is None:
        return  # 업체 컬럼 없으면 생략

    st.markdown(sub_header("업체별 시간 분류 TOP 10"), unsafe_allow_html=True)

    df = workers_df.copy()

    # time breakdown fallback 컬럼 생성
    if "work_dwell_min" not in df.columns:
        work_cols = [c for c in ["intense_work_min", "light_work_min", "idle_at_work_min"] if c in df.columns]
        df["work_dwell_min"] = df[work_cols].sum(axis=1) if work_cols else 0

    if "transit_dwell_min" not in df.columns:
        t_cols = [c for c in ["transit_min", "in_transit_min"] if c in df.columns]
        df["transit_dwell_min"] = df[t_cols].sum(axis=1) if t_cols else 0

    if "rest_dwell_min" not in df.columns:
        df["rest_dwell_min"] = df["rest_min"] if "rest_min" in df.columns else 0

    # lmt_minutes merge (user_no 기준)
    if not transit_df.empty and "lmt_minutes" in transit_df.columns and "user_no" in transit_df.columns:
        lmt_df = transit_df[["user_no", "lmt_minutes"]].dropna(subset=["lmt_minutes"])
        # 동일 user_no가 여러 날짜에 있을 수 있으므로 평균 사용
        lmt_agg = lmt_df.groupby("user_no")["lmt_minutes"].mean().reset_index()
        df = df.merge(lmt_agg, on="user_no", how="left")
        df["lmt_minutes"] = df["lmt_minutes"].fillna(0)
    else:
        df["lmt_minutes"] = 0

    # 업체별 집계 (평균)
    agg = df.groupby(company_col).agg(
        work_avg      = ("work_dwell_min",   "mean"),
        transit_avg   = ("transit_dwell_min", "mean"),
        rest_avg      = ("rest_dwell_min",    "mean"),
        lunch_avg     = ("lmt_minutes",       "mean"),
        work_minutes  = ("work_minutes",      "mean") if "work_minutes" in df.columns else ("work_dwell_min", "mean"),
        n_workers     = (company_col,         "count"),
    ).reset_index()

    # TOP 10 (작업공간 체류 기준 정렬)
    top10 = agg.nlargest(10, "work_avg").sort_values("work_avg", ascending=True)

    if top10.empty:
        st.info("업체별 시간 분류 데이터가 없습니다.")
        return

    # work 비율 계산 (분모: work_minutes, 없으면 분류합계)
    classified = top10["work_avg"] + top10["transit_avg"] + top10["rest_avg"] + top10["lunch_avg"]
    denom = top10["work_minutes"].where(top10["work_minutes"] > 0, classified.replace(0, np.nan))
    work_pct = (top10["work_avg"] / denom * 100).fillna(0).round(1)

    company_names = top10[company_col].tolist()

    # 스택 바
    fig = go.Figure()
    for col_key, label, color in [
        ("work_avg",    "작업공간 체류", "#00AEEF"),
        ("transit_avg", "이동",         "#FFB300"),
        ("rest_avg",    "휴식",         "#4CAF50"),
        ("lunch_avg",   "점심 외부",    "#A78BFA"),
    ]:
        fig.add_trace(go.Bar(
            y=company_names,
            x=top10[col_key].tolist(),
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"{label}: %{{x:.1f}}분<extra></extra>",
        ))

    # 작업 비율 텍스트 주석
    max_x = classified.max()
    annotations = []
    for i, (company, pct_val) in enumerate(zip(company_names, work_pct.tolist())):
        annotations.append(dict(
            x=classified.iloc[top10.index.get_loc(top10[top10[company_col] == company].index[0])] + max_x * 0.02,
            y=company,
            text=f"<b>{pct_val}%</b>",
            showarrow=False,
            font=dict(color="#00AEEF", size=11),
            xanchor="left",
            yanchor="middle",
        ))

    fig.update_layout(
        paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
        plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
        font_color=PLOTLY_DARK["font_color"],
        margin=dict(l=15, r=80, t=30, b=15),
        barmode="stack",
        height=max(300, len(company_names) * 36 + 80),
        xaxis_title="시간 (분)",
        yaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        annotations=annotations,
        hovermode="y unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_company_top10_period(company_df: pd.DataFrame) -> None:
    """전체 기간 업체별 작업자 TOP 10 Horizontal Bar."""
    try:
        import plotly.express as px

        if company_df.empty or "company_name" not in company_df.columns:
            st.info("업체별 분포 데이터가 없습니다.")
            return

        # 전체 기간 업체별 집계
        count_col = "worker_count" if "worker_count" in company_df.columns else "n_workers"
        if count_col not in company_df.columns:
            st.info("작업자 수 데이터가 없습니다.")
            return

        # 업체별 합산
        company_agg = company_df.groupby("company_name")[count_col].sum().reset_index()
        company_agg.columns = ["업체명", "작업자 수"]
        top10 = company_agg.nlargest(10, "작업자 수")

        fig = px.bar(
            top10,
            y="업체명",
            x="작업자 수",
            orientation="h",
            color_discrete_sequence=[cfg.THEME_ACCENT],
        )
        fig.update_layout(
            **PLOTLY_DARK,
            showlegend=False,
            xaxis_title="작업자 수 (연인원)",
            yaxis_title="",
            height=300,
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"차트 렌더링 실패: {e}")


# ==========================================================================
# 메인 함수
# ==========================================================================

def main():
    page, sector_id = render_sidebar()

    if not sector_id:
        st.warning("Sector를 선택하세요.")
        return

    # available_dates는 session_state에서 가져옴
    available_dates = st.session_state.get("available_dates", [])

    sec_info = cfg.SECTOR_REGISTRY.get(sector_id, {})

    # -- 페이지 헤더 ----------------------------------------------------
    st.markdown(
        f"""
        <h1 style='color:{cfg.THEME_TEXT}; font-size:1.7rem; margin-bottom:2px;'>
            {sec_info.get('icon', '🏭')} Deep Con at M15X
            <span style='color:{cfg.THEME_ACCENT};'>{sec_info.get('label', '')}</span>
        </h1>
        <p style='color:#7A8FA6; font-size:0.88rem; margin-top:0;'>
            Spatial Data Analysis using Deep Con Prototype | {sec_info.get('subtitle', '')}
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # -- 페이지 라우팅 --------------------------------------------------
    if "현장 개요" in page:
        _render_overview_tab(sector_id, available_dates)

    elif "작업자 대기/이동 시간" in page:
        try:
            from src.dashboard.transit_tab import render_transit_tab
            # transit_tab은 자체적으로 전체 데이터 로드 + 내부 필터링
            render_transit_tab(sector_id)
        except ImportError:
            st.info("작업자 대기/이동 시간 분석 탭은 추후 구현 예정입니다.")
            st.markdown(
                """
                **구현 예정 기능:**
                - 일별 MAT/LBT/EOD 추이 분석
                - BP별 대기시간 비교
                - 원본 데이터 조회 및 다운로드
                """
            )

    elif "테이블 리프트" in page:
        try:
            from src.dashboard.equipment_tab import render_equipment_tab
            render_equipment_tab(sector_id)
        except ImportError:
            st.info("테이블 리프트 분석 탭은 추후 구현 예정입니다.")
            st.markdown(
                """
                **구현 예정 기능:**
                - 주간 BP별 가동률 (OPR) 분석
                - 장비 현황 대시보드
                - 층별 장비 분포
                """
            )

    elif "작업자 분석" in page:
        try:
            from src.dashboard.daily_tab import render_daily_tab
            render_daily_tab(sector_id)
        except ImportError:
            st.info("작업자 분석 탭은 추후 구현 예정입니다.")
            st.markdown(
                """
                **구현 예정 기능:**
                - EWI/CRE/SII 심층 분석
                - 업체별 비교
                - 개인 프로파일 조회
                """
            )

    elif "공간 혼잡도" in page:
        try:
            from src.dashboard.congestion_tab import render_congestion_tab
            render_congestion_tab(sector_id)
        except ImportError:
            st.info("공간 혼잡도 탭은 추후 구현 예정입니다.")
            st.markdown(
                """
                **구현 예정 기능:**
                - 일별 혼잡도 히트맵
                - 시간대별 층별 인원 추이
                - 기간별 패턴 분석
                """
            )

    elif "리포트 생성" in page:
        try:
            from src.dashboard.report_tab import render_report_tab
            render_report_tab(sector_id)
        except ImportError as e:
            st.error(f"리포트 생성 탭 로드 실패: {e}")

    elif "데이터 관리" in page:
        if is_admin() and not cfg.CLOUD_MODE:
            try:
                from src.dashboard.pipeline_tab import render_pipeline_tab
                render_pipeline_tab(sector_id)
            except ImportError as e:
                st.error(f"파이프라인 탭 로드 실패: {e}")
        else:
            st.warning("데이터 관리 탭은 관리자 전용입니다.")


if __name__ == "__main__":
    main()
