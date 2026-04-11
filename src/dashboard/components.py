"""
Dashboard Components — 재사용 가능한 대시보드 컴포넌트
======================================================
DeepCon-M15X 대시보드 전반에서 사용되는 공통 UI 컴포넌트.

컴포넌트:
  - alert_center: 사이드바 알림 센터
  - metric_card_enhanced: 향상된 메트릭 카드 (스파크라인 + 컨텍스트)
  - journey_timeline: Journey 타임라인 Plotly Figure
  - drill_panel: 드릴다운 상세 패널
  - validation_badge: 데이터 품질 뱃지
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import COLORS, PLOTLY_DARK
from src.utils.anonymizer import mask_name

if TYPE_CHECKING:
    pass


# ─── 알림 센터 ─────────────────────────────────────────────────────────────


def alert_center(alerts: list[dict]) -> None:
    """
    사이드바 알림 센터.

    Args:
        alerts: [
            {"severity": "critical", "title": "...", "count": 2, "detail": "..."},
            ...
        ]
        severity: critical > high > medium > low
    """
    severity_order = ["critical", "high", "medium", "low"]
    severity_colors = {
        "critical": "#FF4C4C",
        "high": "#FF8C00",
        "medium": "#FFB300",
        "low": "#00AEEF",
    }
    severity_icons = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "⚡",
        "low": "💡",
    }

    if not alerts:
        st.markdown(
            "<div style='color:#4A6A8A; font-size:0.85rem;'>알림 없음</div>",
            unsafe_allow_html=True,
        )
        return

    # 심각도별 카운트 뱃지
    counts = {s: 0 for s in severity_order}
    for a in alerts:
        sev = a.get("severity", "low")
        counts[sev] += a.get("count", 1)

    badges = []
    for sev in severity_order:
        if counts[sev] > 0:
            color = severity_colors[sev]
            badges.append(
                f"<span style='background:{color}22; color:{color}; "
                f"padding:2px 10px; border-radius:12px; font-size:0.82rem; "
                f"font-weight:600;'>{counts[sev]}</span>"
            )

    st.markdown(
        f"<div style='display:flex; gap:8px; margin-bottom:8px;'>{''.join(badges)}</div>",
        unsafe_allow_html=True,
    )

    # 알림 목록 (우선순위 정렬, 상위 5개)
    sorted_alerts = sorted(
        alerts,
        key=lambda x: severity_order.index(x.get("severity", "low")),
    )[:5]

    with st.expander("알림 상세", expanded=False):
        for alert in sorted_alerts:
            sev = alert.get("severity", "low")
            color = severity_colors.get(sev, "#00AEEF")
            icon = severity_icons.get(sev, "💡")
            title = alert.get("title", "")
            detail = alert.get("detail", "")

            st.markdown(
                f"""
                <div style='border-left:3px solid {color}; padding:6px 10px;
                            margin:6px 0; background:#111820; border-radius:4px;'>
                    <div style='font-size:0.85rem; color:{color}; font-weight:600;'>
                        {icon} {title}
                    </div>
                    <div style='font-size:0.78rem; color:#9AB5D4; margin-top:4px;'>
                        {detail}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── 향상된 메트릭 카드 ─────────────────────────────────────────────────────


def metric_card_enhanced(
    label: str,
    value: str,
    delta: str = "",
    delta_up: bool = True,
    sparkline_data: list = None,
    context: str = "",
    color: str = None,
) -> str:
    """
    향상된 메트릭 카드 (스파크라인 + 컨텍스트).

    Args:
        label: 라벨
        value: 주 값
        delta: 변화량 (예: "+5%")
        delta_up: 증가가 긍정적인지 (True면 초록, False면 빨강)
        sparkline_data: 최근 7일 등 추이 데이터 리스트
        context: 추가 컨텍스트 (예: "지난주 평균 대비")
        color: 값 색상 오버라이드

    Returns:
        HTML 문자열
    """
    # 변화량 색상
    delta_color = COLORS["success"] if delta_up else COLORS["danger"]
    delta_html = (
        f'<div style="font-size:0.85rem; color:{delta_color}; margin-top:4px;">{delta}</div>'
        if delta
        else ""
    )

    # 값 색상
    value_color = color or COLORS["accent"]

    # 스파크라인 (미니 SVG)
    sparkline_html = ""
    if sparkline_data and len(sparkline_data) >= 2:
        sparkline_html = _render_sparkline_svg(sparkline_data)

    # 컨텍스트
    context_html = (
        f'<div style="font-size:0.72rem; color:#6A7A95; margin-top:4px;">{context}</div>'
        if context
        else ""
    )

    return f"""
    <div class="metric-card" style="min-height:110px;">
        <div class="metric-value" style="color:{value_color};">{value}</div>
        {delta_html}
        {sparkline_html}
        <div class="metric-label">{label}</div>
        {context_html}
    </div>"""


def _render_sparkline_svg(
    data: list,
    width: int = 80,
    height: int = 20,
    color: str = "#00AEEF",
) -> str:
    """미니 스파크라인 SVG."""
    if not data or len(data) < 2:
        return ""

    # NaN 제거
    clean_data = [v for v in data if v is not None and not (isinstance(v, float) and v != v)]
    if len(clean_data) < 2:
        return ""

    min_v, max_v = min(clean_data), max(clean_data)
    range_v = max_v - min_v if max_v != min_v else 1

    points = []
    for i, v in enumerate(clean_data):
        x = i * width / (len(clean_data) - 1)
        y = height - ((v - min_v) / range_v) * (height - 2) - 1
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    return f"""
    <svg width="{width}" height="{height}" style="margin:6px auto; display:block; opacity:0.7;">
        <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
    </svg>"""


# ─── Journey 타임라인 ──────────────────────────────────────────────────────


def journey_timeline(
    journey_df: pd.DataFrame,
    user_no: str,
    locus_dict: dict,
    height: int = 100,
) -> go.Figure:
    """
    작업자 Journey 타임라인 Plotly Figure.

    Args:
        journey_df: Journey 데이터 (block_type 포함 시 사용)
        user_no: 작업자 ID
        locus_dict: locus_id → 정보 딕셔너리
        height: 차트 높이

    Returns:
        Plotly Figure
    """
    # 해당 작업자 데이터 필터링
    user_journey = journey_df[journey_df["user_no"] == user_no].copy()
    if user_journey.empty:
        fig = go.Figure()
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[
                dict(text="데이터 없음", x=0.5, y=0.5, showarrow=False)
            ],
            **PLOTLY_DARK,
        )
        return fig

    user_journey = user_journey.sort_values("timestamp")

    # 블록 추출
    if "block_id" in user_journey.columns:
        # 토큰화된 데이터 사용
        blocks = _extract_blocks_from_tokenized(user_journey, locus_dict)
    else:
        # 기본 방식: 연속 동일 locus 그룹화
        blocks = _extract_blocks_basic(user_journey, locus_dict)

    if not blocks:
        fig = go.Figure()
        fig.update_layout(height=height, **PLOTLY_DARK)
        return fig

    # Plotly 타임라인 구성
    fig = go.Figure()

    # 블록 타입별 색상
    color_map = {
        "GATE_IN": "#6B7280",
        "GATE_OUT": "#6B7280",
        "WORK": "#3B82F6",
        "REST": "#10B981",
        "TRANSIT": "#9CA3AF",
        "ADMIN": "#8B5CF6",
        "UNKNOWN": "#4B5563",
    }

    # 토큰별 색상 (블록 타입 없을 때)
    # v1 영문 토큰 + v2에서는 block_type 기반 color_map이 우선 적용됨
    token_color_map = {
        "work_zone": "#3B82F6",
        "outdoor_work": "#60A5FA",
        "breakroom": "#10B981",
        "smoking_area": "#34D399",
        "timeclock": "#6B7280",
        "main_gate": "#9CA3AF",
        "confined_space": "#EF4444",
        "high_voltage": "#DC2626",
        # v2: locus_type 기반 (token 컬럼 값)
        "work_area": "#3B82F6",
        "gate": "#6B7280",
        "vertical": "#F59E0B",
    }

    # 시작 시간 기준
    start_time = blocks[0]["start"]

    for block in blocks:
        block_type = block.get("block_type", "UNKNOWN")
        token = block.get("token", "unknown")
        color = color_map.get(block_type, token_color_map.get(token, "#4B5563"))

        # 상대 시간 (분)
        start_min = (block["start"] - start_time).total_seconds() / 60
        duration = block["duration"]
        name = block.get("name", block.get("locus_id", ""))

        fig.add_trace(go.Bar(
            x=[duration],
            y=["Journey"],
            orientation="h",
            base=start_min,
            marker_color=color,
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"유형: {block_type}<br>"
                f"시간: {block['start'].strftime('%H:%M')} - {block['end'].strftime('%H:%M')}<br>"
                f"체류: {duration:.0f}분<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        barmode="stack",
        showlegend=False,
        height=height,
        margin=dict(l=0, r=0, t=10, b=20),
        xaxis=dict(
            title="시간 (분)",
            showgrid=True,
            gridcolor="#2A3A4A",
        ),
        yaxis=dict(showticklabels=False),
        **PLOTLY_DARK,
    )

    return fig


def _extract_blocks_from_tokenized(
    user_journey: pd.DataFrame,
    locus_dict: dict,
) -> list[dict]:
    """토큰화된 데이터에서 블록 추출."""
    blocks = []
    for block_id in user_journey["block_id"].unique():
        block_data = user_journey[user_journey["block_id"] == block_id]
        if block_data.empty:
            continue

        locus_id = block_data["locus_id"].iloc[0]
        locus_info = locus_dict.get(locus_id, {})

        blocks.append({
            "locus_id": locus_id,
            "name": locus_info.get("locus_name", locus_id),
            "token": locus_info.get("token", "unknown"),
            "block_type": block_data["block_type"].iloc[0] if "block_type" in block_data.columns else "UNKNOWN",
            "start": block_data["timestamp"].min(),
            "end": block_data["timestamp"].max(),
            "duration": len(block_data),
        })

    return blocks


def _extract_blocks_basic(
    user_journey: pd.DataFrame,
    locus_dict: dict,
) -> list[dict]:
    """기본 방식으로 블록 추출 (연속 동일 locus 그룹화)."""
    blocks = []
    current_locus = None
    block_start = None
    block_rows = []

    for _, row in user_journey.iterrows():
        if row["locus_id"] != current_locus:
            if current_locus is not None and block_rows:
                locus_info = locus_dict.get(current_locus, {})
                blocks.append({
                    "locus_id": current_locus,
                    "name": locus_info.get("locus_name", current_locus),
                    "token": locus_info.get("token", "unknown"),
                    "block_type": "UNKNOWN",
                    "start": block_start,
                    "end": block_rows[-1]["timestamp"],
                    "duration": len(block_rows),
                })
            current_locus = row["locus_id"]
            block_start = row["timestamp"]
            block_rows = [row]
        else:
            block_rows.append(row)

    # 마지막 블록
    if current_locus is not None and block_rows:
        locus_info = locus_dict.get(current_locus, {})
        blocks.append({
            "locus_id": current_locus,
            "name": locus_info.get("locus_name", current_locus),
            "token": locus_info.get("token", "unknown"),
            "block_type": "UNKNOWN",
            "start": block_start,
            "end": block_rows[-1]["timestamp"] if isinstance(block_rows[-1], dict) else block_rows[-1]["timestamp"],
            "duration": len(block_rows),
        })

    return blocks


# ─── 데이터 품질 뱃지 ──────────────────────────────────────────────────────


def validation_badge(level: str, score: float = None) -> str:
    """
    데이터 품질 뱃지 HTML.

    Args:
        level: "pass" | "warning" | "fail"
        score: 0~1 점수 (선택)

    Returns:
        HTML 문자열
    """
    colors = {
        "pass": COLORS["success"],
        "warning": COLORS["warning"],
        "fail": COLORS["danger"],
    }
    icons = {
        "pass": "✓",
        "warning": "⚠",
        "fail": "✗",
    }
    labels = {
        "pass": "정상",
        "warning": "주의",
        "fail": "문제",
    }

    color = colors.get(level, COLORS["text_muted"])
    icon = icons.get(level, "?")
    label = labels.get(level, level)

    score_text = f" ({score * 100:.0f}%)" if score is not None else ""

    return (
        f'<span style="background:{color}22; color:{color}; '
        f'padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:600;">'
        f'{icon} {label}{score_text}</span>'
    )


# ─── 드릴다운 패널 ─────────────────────────────────────────────────────────


def render_worker_detail_panel(
    user_no: str,
    worker_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    locus_dict: dict,
) -> None:
    """작업자 상세 패널 렌더링."""
    # 작업자 정보 조회
    worker_row = worker_df[worker_df["user_no"] == user_no]
    if worker_row.empty:
        st.info("작업자 정보 없음")
        return

    row = worker_row.iloc[0]

    # 헤더
    st.markdown(
        f"""
        <div style='background:#1A2A3A; border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
            <div style='font-size:1.1rem; font-weight:600; color:#D5E5FF;'>
                {mask_name(row.get('user_name', user_no))}
            </div>
            <div style='font-size:0.82rem; color:#7A8FA6;'>
                {row.get('company_name', '소속 미확인')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI 3개
    col1, col2, col3 = st.columns(3)
    with col1:
        ewi = row.get("ewi", 0)
        ewi_color = "#FF4C4C" if ewi >= 0.8 else "#FFB300" if ewi >= 0.6 else "#00C897"
        st.markdown(
            metric_card_enhanced("EWI", f"{ewi:.3f}", color=ewi_color),
            unsafe_allow_html=True,
        )
    with col2:
        cre = row.get("cre", 0)
        cre_color = "#FF4C4C" if cre >= 0.6 else "#FFB300" if cre >= 0.3 else "#00C897"
        st.markdown(
            metric_card_enhanced("CRE", f"{cre:.3f}", color=cre_color),
            unsafe_allow_html=True,
        )
    with col3:
        work_min = row.get("work_minutes", 0)
        st.markdown(
            metric_card_enhanced("근무 시간", f"{work_min:.0f}분"),
            unsafe_allow_html=True,
        )

    # Journey 타임라인
    st.markdown("### Journey")
    if not journey_df.empty:
        fig = journey_timeline(journey_df, user_no, locus_dict)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Journey 데이터 없음")

    # 상세 지표
    with st.expander("상세 지표", expanded=False):
        detail_cols = [
            ("고활성 시간", "high_active_min", "분"),
            ("저활성 시간", "low_active_min", "분"),
            ("대기 시간", "standby_min", "분"),
            ("휴식 시간", "rest_min", "분"),
            ("이동 시간", "transit_min", "분"),
            ("방문 공간 수", "unique_loci", "개"),
            ("피로도", "fatigue_score", ""),
            ("단독작업 비율", "alone_ratio", ""),
        ]

        detail_data = []
        for label, col, unit in detail_cols:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    if isinstance(val, float):
                        detail_data.append({"지표": label, "값": f"{val:.2f}{unit}"})
                    else:
                        detail_data.append({"지표": label, "값": f"{val}{unit}"})

        if detail_data:
            st.table(pd.DataFrame(detail_data))


def render_space_detail_panel(
    locus_id: str,
    space_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    locus_dict: dict,
) -> None:
    """공간 상세 패널 렌더링."""
    locus_info = locus_dict.get(locus_id, {})

    # 헤더
    st.markdown(
        f"""
        <div style='background:#1A2A3A; border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
            <div style='font-size:1.1rem; font-weight:600; color:#D5E5FF;'>
                {locus_info.get('locus_name', locus_id)}
            </div>
            <div style='font-size:0.82rem; color:#7A8FA6;'>
                {locus_info.get('token', '')} · {locus_info.get('locus_type', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 공간 지표 (space_df에서)
    if not space_df.empty and "locus_id" in space_df.columns:
        space_row = space_df[space_df["locus_id"] == locus_id]
        if not space_row.empty:
            row = space_row.iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("체류 인원", f"{row.get('total_workers', 0):,}명")
            with col2:
                st.metric("평균 체류", f"{row.get('avg_dwell_min', 0):.0f}분")
            with col3:
                st.metric("평균 EWI", f"{row.get('avg_ewi', 0):.3f}")

    # 시간대별 체류 현황
    if not journey_df.empty and "locus_id" in journey_df.columns:
        space_journey = journey_df[journey_df["locus_id"] == locus_id]
        if not space_journey.empty and "timestamp" in space_journey.columns:
            # 시간대별 집계
            space_journey["hour"] = pd.to_datetime(space_journey["timestamp"]).dt.hour
            hourly = space_journey.groupby("hour")["user_no"].nunique().reset_index()
            hourly.columns = ["시간", "인원"]

            st.markdown("### 시간대별 체류 인원")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly["시간"],
                y=hourly["인원"],
                marker_color="#00AEEF",
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=10, b=30),
                xaxis_title="시간",
                yaxis_title="인원",
                **PLOTLY_DARK,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Enriched Locus 기반 공간 알림 (2026-03-28 추가) ─────────────────────

# dwell_category 컬러 및 라벨
DWELL_CATEGORY_STYLES = {
    "TRANSIT": {"color": "#6B7280", "bg": "#6B728022", "label": "통과형", "icon": "🚶"},
    "SHORT_STAY": {"color": "#10B981", "bg": "#10B98122", "label": "단기", "icon": "☕"},
    "LONG_STAY": {"color": "#3B82F6", "bg": "#3B82F622", "label": "장기", "icon": "🔧"},
    "HAZARD_ZONE": {"color": "#EF4444", "bg": "#EF444422", "label": "고위험", "icon": "⚠️"},
    "ADMIN": {"color": "#8B5CF6", "bg": "#8B5CF622", "label": "관리", "icon": "📋"},
    "UNKNOWN": {"color": "#4B5563", "bg": "#4B556322", "label": "미분류", "icon": "❓"},
}


def get_dwell_category_style(category: str) -> dict:
    """dwell_category에 해당하는 스타일 반환."""
    return DWELL_CATEGORY_STYLES.get(category, DWELL_CATEGORY_STYLES["UNKNOWN"])


def build_space_alerts(
    journey_df: pd.DataFrame,
    locus_dict: dict,
    space_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Enriched locus 정보 기반 공간 관련 알림 생성.

    알림 유형:
    1. HAZARD_ZONE 장시간 체류 (avg_dwell의 2배 초과) → critical
    2. TRANSIT 이상 체류 (30분 초과) → medium
    3. 공간 과밀 (max_concurrent_occupancy의 90% 초과) → high
    4. 공간 혼잡 (70% 초과) → medium

    Args:
        journey_df: 현재 journey 데이터
        locus_dict: enriched locus 정보
        space_df: 공간별 집계 데이터 (optional)

    Returns:
        list[dict]: 각 dict는 {severity, title, detail, count} 포함
    """
    alerts = []

    if journey_df.empty or not locus_dict:
        return alerts

    # 최근 30분 데이터 추출
    if "timestamp" in journey_df.columns:
        try:
            latest = pd.to_datetime(journey_df["timestamp"]).max()
            recent = journey_df[pd.to_datetime(journey_df["timestamp"]) >= latest - pd.Timedelta(minutes=30)]
        except Exception:
            recent = journey_df.tail(1000)
    else:
        recent = journey_df.tail(1000)

    if recent.empty or "locus_id" not in recent.columns:
        return alerts

    # Locus별 분석
    seen_alerts = set()

    for locus_id in recent["locus_id"].dropna().unique():
        info = locus_dict.get(locus_id, {})
        if not info:
            continue

        cat = info.get("dwell_category", "UNKNOWN")
        avg_dwell = info.get("avg_dwell_minutes") or 30
        max_occ = info.get("max_concurrent_occupancy") or 50
        locus_name = info.get("locus_name", locus_id)

        locus_journey = recent[recent["locus_id"] == locus_id]
        current_workers = locus_journey["user_no"].nunique()

        # 1. HAZARD_ZONE 장시간 체류 체크
        if cat == "HAZARD_ZONE":
            for user_no in locus_journey["user_no"].unique():
                user_data = locus_journey[locus_journey["user_no"] == user_no]
                dwell_min = len(user_data)  # 분 단위 근사
                if dwell_min > avg_dwell * 2:
                    alert_key = f"hazard_dwell_{locus_id}_{user_no}"
                    if alert_key not in seen_alerts:
                        seen_alerts.add(alert_key)
                        alerts.append({
                            "severity": "critical",
                            "title": f"고위험 구역 장시간 체류: {locus_name}",
                            "detail": f"작업자 {user_no}: {dwell_min}분 체류 (평균 {avg_dwell:.0f}분의 {dwell_min/avg_dwell:.1f}배)",
                            "count": 1,
                        })

        # 2. TRANSIT 이상 체류 체크
        if cat == "TRANSIT":
            for user_no in locus_journey["user_no"].unique():
                user_data = locus_journey[locus_journey["user_no"] == user_no]
                dwell_min = len(user_data)
                if dwell_min > 30:
                    alert_key = f"transit_dwell_{locus_id}_{user_no}"
                    if alert_key not in seen_alerts:
                        seen_alerts.add(alert_key)
                        alerts.append({
                            "severity": "medium",
                            "title": f"통과 구역 이상 체류: {locus_name}",
                            "detail": f"작업자 {user_no}: {dwell_min}분 체류 (통과형 구역)",
                            "count": 1,
                        })

        # 3. 공간 과밀/혼잡 체크
        if max_occ > 0:
            occ_pct = current_workers / max_occ * 100
            if occ_pct >= 90:
                alert_key = f"overcrowd_{locus_id}"
                if alert_key not in seen_alerts:
                    seen_alerts.add(alert_key)
                    alerts.append({
                        "severity": "high",
                        "title": f"공간 과밀 경고: {locus_name}",
                        "detail": f"현재 {current_workers}명 / 최대 {max_occ}명 ({occ_pct:.0f}%)",
                        "count": 1,
                    })
            elif occ_pct >= 70:
                alert_key = f"crowded_{locus_id}"
                if alert_key not in seen_alerts:
                    seen_alerts.add(alert_key)
                    alerts.append({
                        "severity": "medium",
                        "title": f"공간 혼잡 주의: {locus_name}",
                        "detail": f"현재 {current_workers}명 / 최대 {max_occ}명 ({occ_pct:.0f}%)",
                        "count": 1,
                    })

    # 심각도별 정렬
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(alerts, key=lambda x: severity_order.get(x.get("severity", "low"), 3))


def render_space_card(
    locus_id: str,
    locus_dict: dict,
    current_workers: int = 0,
) -> str:
    """
    공간 카드 HTML 생성.

    Args:
        locus_id: Locus ID
        locus_dict: enriched locus 정보
        current_workers: 현재 체류 인원

    Returns:
        HTML 문자열
    """
    info = locus_dict.get(locus_id, {})
    cat = info.get("dwell_category", "UNKNOWN")
    style = get_dwell_category_style(cat)

    locus_name = info.get("locus_name", locus_id)
    avg_dwell = info.get("avg_dwell_minutes") or 0
    peak_hour = info.get("peak_hour") or 0
    max_occ = info.get("max_concurrent_occupancy") or 50
    hazard_level = str(info.get("hazard_level", "")).lower()

    # 혼잡률 계산
    occ_pct = min(100, (current_workers / max_occ * 100)) if max_occ > 0 else 0

    # 혼잡도 색상
    if occ_pct >= 90:
        occ_color = "#FF4C4C"
    elif occ_pct >= 70:
        occ_color = "#FF6B35"
    elif occ_pct >= 40:
        occ_color = "#FFB300"
    else:
        occ_color = "#00C897"

    # 고위험 뱃지
    hazard_badge = ""
    if hazard_level in ("high", "critical"):
        hazard_badge = f"<span style='color:#EF4444;font-size:0.78rem;'>⚡ 고위험</span>"

    return f"""
    <div style='background:#1A2A3A; border:1px solid #2A3A4A; border-radius:8px;
                padding:12px 14px; margin:6px 0;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <span style='background:{style["bg"]}; color:{style["color"]}; padding:2px 6px;
                             border-radius:4px; font-size:0.72rem;'>{style["icon"]} {style["label"]}</span>
                <span style='color:#D5E5FF; font-weight:600; margin-left:6px;'>
                    {locus_name}
                </span>
            </div>
            {hazard_badge}
        </div>
        <div style='color:#9AB5D4; font-size:0.82rem; margin-top:8px;'>
            👥 {current_workers}명 / {max_occ}명 ({occ_pct:.1f}%)
        </div>
        <div style='color:#7A8FA6; font-size:0.78rem; margin-top:4px;'>
            평균 {avg_dwell:.0f}분 · 피크 {peak_hour}시
        </div>
        <div style='background:#2A3A4A; height:5px; border-radius:3px; margin-top:8px; overflow:hidden;'>
            <div style='background:{occ_color}; width:{occ_pct:.1f}%; height:100%;'></div>
        </div>
    </div>
    """


def render_dwell_category_badge(category: str) -> str:
    """dwell_category 뱃지 HTML 생성."""
    style = get_dwell_category_style(category)
    return (
        f"<span style='background:{style['bg']}; color:{style['color']}; "
        f"padding:2px 8px; border-radius:4px; font-size:0.75rem;'>"
        f"{style['icon']} {style['label']}</span>"
    )
