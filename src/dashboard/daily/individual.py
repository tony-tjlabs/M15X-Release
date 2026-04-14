"""
Daily Tab - Individual Section
==============================
개인별 분석 탭 관련 함수들.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import (
    section_header, sub_header,
    PLOTLY_DARK, PLOTLY_LEGEND,
)
from src.dashboard.components import journey_timeline
from src.utils.anonymizer import mask_name, mask_names_in_df

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND


def render_individual(worker_df, locus_dict, has_ewi, has_cre):
    """개인별 분석 탭 전체 렌더링."""
    st.markdown(section_header("개인별 분석"), unsafe_allow_html=True)

    # 검색 + 정렬
    col_search, col_sort = st.columns([3, 2])
    with col_search:
        search = st.text_input("작업자 이름 또는 업체 검색", placeholder="예: 홍길동, A건설")
    with col_sort:
        sort_options = {
            "EWI 높은 순": "ewi",
            "CRE 높은 순": "cre",
            "신호수신 많은 순": "recorded_minutes",
            "근무시간 긴 순": "work_minutes",
            "밀폐공간 긴 순": "confined_minutes",
        }
        sort_valid = {k: v for k, v in sort_options.items() if v in worker_df.columns}
        if sort_valid:
            sort_by_label = st.selectbox("정렬 기준", list(sort_valid.keys()))
            sort_col = sort_valid[sort_by_label]
        else:
            sort_col = "recorded_minutes"

    df = worker_df.copy()
    if search:
        mask = (
            df["user_name"].str.contains(search, case=False, na=False)
            | df["company_name"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    if df.empty:
        st.info("검색 결과 없음")
        return

    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Shift 필터 (shift_type 컬럼 있을 때)
    if "shift_type" in df.columns:
        shift_filter = st.radio(
            "근무 교대", ["전체", "sun 주간", "moon 야간"],
            horizontal=True, key="individual_shift_filter",
        )
        if shift_filter == "sun 주간":
            df = df[df["shift_type"] == "day"]
        elif shift_filter == "moon 야간":
            df = df[df["shift_type"] == "night"]

        # Shift 배지 컬럼 생성
        df = df.copy()
        df["shift"] = df["shift_type"].map({"day": "sun", "night": "moon", "unknown": "?"}).fillna("?")

        # 헬멧 방치 의심 플래그
        if "helmet_abandoned" in df.columns:
            df["헬멧방치?"] = df["helmet_abandoned"].map({True: "!", False: ""})

    if df.empty:
        st.info("해당 교대의 작업자 없음")
        return

    # BLE 커버리지 등급 컬럼 추가
    if "ble_coverage" in df.columns:
        cov_icons = {"정상": "green", "부분음영": "yellow", "음영": "orange", "미측정": "red"}
        df["BLE"] = df["ble_coverage"].map(cov_icons).fillna("?")

    # 표시 컬럼 구성
    base_cols = ["shift", "user_name", "company_name", "work_minutes",
                 "valid_ble_minutes", "recorded_minutes",
                 "unique_loci", "transition_count", "confined_minutes", "high_voltage_minutes"]
    if "ble_coverage" in df.columns:
        base_cols.insert(3, "BLE")
    if "helmet_abandoned" in df.columns:
        base_cols.append("헬멧방치?")
    metric_cols = []
    if has_ewi:
        metric_cols += ["ewi", "high_active_min", "low_active_min", "standby_min"]
    if has_cre:
        metric_cols += ["cre", "fatigue_score", "sii"]

    show_cols = [c for c in base_cols + metric_cols if c in df.columns]
    rename = {
        "shift": "교대", "user_name": "작업자", "company_name": "업체",
        "work_minutes": "근무(분)", "valid_ble_minutes": "유효BLE(분)",
        "recorded_minutes": "전체BLE(분)",
        "unique_loci": "방문구역", "transition_count": "이동횟수",
        "confined_minutes": "밀폐(분)", "high_voltage_minutes": "고압전(분)",
        "ewi": "EWI", "high_active_min": "고활성(분)", "low_active_min": "저활성(분)",
        "standby_min": "대기(분)", "cre": "CRE", "fatigue_score": "피로도",
        "sii": "SII",
    }

    st.caption(f"총 {len(df)}명 / 상위 100명 표시")
    display_top = mask_names_in_df(df[show_cols].head(100), "user_name")
    st.dataframe(
        display_top.rename(columns=rename)
        .style.format({
            "EWI": "{:.3f}", "CRE": "{:.3f}", "SII": "{:.3f}", "피로도": "{:.3f}",
            "근무(분)": "{:.0f}", "유효BLE(분)": "{:.0f}", "전체BLE(분)": "{:.0f}",
        }, na_rep="-"),
        use_container_width=True, hide_index=True,
    )

    # 개별 작업자 상세 보기 (그래픽 포함)
    st.divider()
    st.markdown(section_header("작업자 상세 분석"), unsafe_allow_html=True)
    worker_options = df["user_name"].head(50).tolist()
    if not worker_options:
        return
    masked_options = [mask_name(n) for n in worker_options]
    selected_idx = st.selectbox("작업자 선택", range(len(worker_options)),
                                format_func=lambda i: masked_options[i])
    selected_worker = worker_options[selected_idx]
    wrow = df[df["user_name"] == selected_worker].iloc[0]

    # 프로필 헤더
    in_t = str(wrow.get("in_datetime", ""))[:16] if pd.notna(wrow.get("in_datetime")) else "-"
    out_t = str(wrow.get("out_datetime", ""))[:16] if pd.notna(wrow.get("out_datetime")) else "-"
    wm = wrow.get("work_minutes", 0)
    wm_h = f"{wm/60:.1f}h" if wm > 0 else "-"
    shift_icon = {"day": "sun", "night": "moon"}.get(wrow.get("shift_type", ""), "?")

    st.markdown(
        f"<div style='background:#1A2A3A; border:1px solid #2A3A4A; border-radius:10px; "
        f"padding:16px 20px; margin-bottom:16px;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<div>"
        f"<span style='font-size:1.3rem; font-weight:700; color:#D5E5FF;'>"
        f"{shift_icon} {mask_name(selected_worker)}</span>"
        f"<span style='color:#9AB5D4; font-size:0.88rem; margin-left:12px;'>"
        f"{wrow.get('company_name', '-')}</span>"
        f"</div>"
        f"<div style='text-align:right;'>"
        f"<div style='color:#9AB5D4; font-size:0.82rem;'>출근 {in_t} -> 퇴근 {out_t}</div>"
        f"<div style='color:#00AEEF; font-size:1.1rem; font-weight:700;'>{wm_h}</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # BLE 커버리지 경고
    cov_level = wrow.get("ble_coverage", "")
    cov_pct = wrow.get("ble_coverage_pct", 0)
    if cov_level in ("음영", "미측정"):
        cov_color = "#FF4C4C" if cov_level == "미측정" else "#FF6B35"
        st.markdown(
            f"<div style='background:#1A1015; border:1px solid {cov_color}44; "
            f"border-left:4px solid {cov_color}; border-radius:8px; "
            f"padding:10px 16px; margin-bottom:12px; font-size:0.85rem; color:#D5E5FF;'>"
            f"signal <b>BLE 커버리지 {cov_pct:.0f}%</b> - "
            f"<span style='color:{cov_color}'>{cov_level}</span> | "
            f"근무 {wm:.0f}분 중 BLE 기록 {wrow.get('recorded_minutes', 0)}분 | "
            f"EWI/CRE 지표를 신뢰할 수 없습니다. "
            f"T-Ward 미착용, 센서 음영, 또는 장비 고장 가능성이 있습니다."
            f"</div>",
            unsafe_allow_html=True,
        )

    # T-Ward 방치 의심 경고
    # 조건: 방문 구역 1개 + 이동 0회 + 활성신호 극히 낮음 + 근무 6시간 이상
    _unique_loci = wrow.get("unique_loci", 0)
    _transitions = wrow.get("transition_count", 0)
    _ewi_val = wrow.get("ewi", 1)
    if _unique_loci <= 1 and _transitions <= 1 and _ewi_val < 0.05 and wm > 360:
        st.markdown(
            "<div style='background:#1A1510; border:1px solid #FF8C4244; "
            "border-left:4px solid #FF8C42; border-radius:8px; "
            "padding:10px 16px; margin-bottom:12px; font-size:0.85rem; color:#D5E5FF;'>"
            "⚠️ <b>T-Ward 방치 의심</b> | "
            f"근무 {wm:.0f}분 동안 이동 {_transitions}회, 방문 구역 {_unique_loci}개, "
            f"EWI {_ewi_val:.3f} | "
            "T-Ward(헬멧)가 작업 구역에 고정된 채 작업자가 실제 착용하지 않았을 가능성이 높습니다. "
            "이 작업자의 활동 시간·대기시간 지표는 신뢰할 수 없습니다."
            "</div>",
            unsafe_allow_html=True,
        )

    # 핵심 지표 4칸
    kpi_cols = st.columns(4)
    kpis = [
        ("EWI (생산성)", f"{wrow.get('ewi', 0):.3f}" if has_ewi else "-",
         "#00AEEF" if wrow.get("ewi", 0) < 0.6 else "#FF4C4C"),
        ("CRE (위험노출)", f"{wrow.get('cre', 0):.3f}" if has_cre else "-",
         "#00C897" if wrow.get("cre", 0) < 0.4 else "#FFB300" if wrow.get("cre", 0) < 0.6 else "#FF4C4C"),
        ("피로도", f"{wrow.get('fatigue_score', 0):.2f}" if "fatigue_score" in wrow else "-",
         "#00C897" if wrow.get("fatigue_score", 0) < 0.5 else "#FF4C4C"),
        ("방문 구역", f"{wrow.get('unique_loci', 0)}개", "#D5E5FF"),
    ]
    for col, (label, val, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value-secondary' style='color:{color}'>{val}</div>"
                f"<div class='metric-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    # 대기/이동시간 카드 (transit_df에서 merge된 컬럼이 있을 때만)
    _mat = wrow.get("mat_minutes")
    _lbt = wrow.get("lbt_minutes")
    _lmt = wrow.get("lmt_minutes")
    _eod = wrow.get("eod_minutes")
    _has_transit = any(pd.notna(v) for v in [_mat, _lbt, _lmt, _eod] if v is not None)
    if _has_transit:
        _mat_v = float(_mat) if pd.notna(_mat) else 0
        _lbt_v = float(_lbt) if pd.notna(_lbt) else 0
        _lmt_v = float(_lmt) if pd.notna(_lmt) else 0
        _eod_v = float(_eod) if pd.notna(_eod) else 0
        _twt_v = _mat_v + _lbt_v + _eod_v
        transit_cols = st.columns(5)
        transit_kpis = [
            ("총 대기 (TWT)", f"{_twt_v:.1f}분", "#00C897"),
            ("출근 대기 (MAT)", f"{_mat_v:.1f}분" if pd.notna(_mat) else "-", "#00AEEF"),
            ("중식 이동 (LBT)", f"{_lbt_v:.1f}분" if pd.notna(_lbt) else "-", "#FFB300"),
            ("점심 외부 (LMT)", f"{_lmt_v:.1f}분" if pd.notna(_lmt) else "-", "#A78BFA"),
            ("퇴근 이동 (EOD)", f"{_eod_v:.1f}분" if pd.notna(_eod) else "-", "#FF8C42"),
        ]
        for col, (label, val, color) in zip(transit_cols, transit_kpis):
            with col:
                st.markdown(
                    f"<div style='background:#111820; border-radius:8px; padding:10px; text-align:center;'>"
                    f"<div style='color:{color}; font-size:1.1rem; font-weight:700;'>{val}</div>"
                    f"<div style='color:#9AB5D4; font-size:0.75rem;'>{label}</div></div>",
                    unsafe_allow_html=True,
                )

    # 활동 시간 분배 차트
    col_chart, col_space = st.columns([3, 2])

    with col_chart:
        st.markdown(sub_header("time 활동 시간 분배"), unsafe_allow_html=True)
        # EWI v2 컬럼 우선 사용, 없으면 v1 폴백
        has_ewi_v2 = "intense_work_min" in wrow.index
        if has_ewi_v2:
            activity_data = {
                "집중 작업 (FAB 고활성)": wrow.get("intense_work_min", 0),
                "경작업 (FAB 저활성)": wrow.get("light_work_min", 0),
                "작업구역 대기": wrow.get("idle_at_work_min", 0),
                "휴식 (휴게/흡연)": wrow.get("rest_min", 0),
                "점심 외부 (LMT)": wrow.get("lmt_minutes", 0) if pd.notna(wrow.get("lmt_minutes")) else 0,
                "이동 중 (감지)": wrow.get("in_transit_min", 0),
                "호이스트/출입": wrow.get("transit_min", 0),
                "BLE 미감지": wrow.get("shadow_min", 0),
            }
            act_colors = {
                "집중 작업 (FAB 고활성)": "#00AEEF",
                "경작업 (FAB 저활성)": "#0078AA",
                "작업구역 대기": "#FFB300",
                "휴식 (휴게/흡연)": "#00C897",
                "점심 외부 (LMT)": "#A78BFA",
                "이동 중 (감지)": "#FF8C42",
                "호이스트/출입": "#9AB5D4",
                "BLE 미감지": "#5A6A7A",
            }
        else:
            activity_data = {
                "고활성 작업": wrow.get("high_active_min", 0),
                "저활성 작업": wrow.get("low_active_min", 0),
                "대기": wrow.get("standby_min", 0),
                "휴게": wrow.get("rest_min", 0),
                "점심 외부 (LMT)": wrow.get("lmt_minutes", 0) if pd.notna(wrow.get("lmt_minutes")) else 0,
                "이동": wrow.get("transit_min", 0),
                "BLE 미감지": wrow.get("shadow_min", 0),
            }
            act_colors = {
                "고활성 작업": "#00AEEF", "저활성 작업": "#0078AA",
                "대기": "#FFB300", "휴게": "#00C897",
                "점심 외부 (LMT)": "#A78BFA",
                "이동": "#9AB5D4", "BLE 미감지": "#5A6A7A",
            }
        activity_data = {k: v for k, v in activity_data.items() if v > 0}
        if activity_data:
            fig = go.Figure(go.Bar(
                x=list(activity_data.values()),
                y=list(activity_data.keys()),
                orientation="h",
                marker_color=[act_colors.get(k, "#666") for k in activity_data.keys()],
                text=[f"{v:.0f}분" for v in activity_data.values()],
                textposition="auto",
            ))
            fig.update_layout(
                paper_bgcolor="#1A2A3A",
                plot_bgcolor="#111820",
                font_color="#C8D6E8",
                margin=dict(l=80, r=15, t=10, b=30),
                height=220, showlegend=False, xaxis_title="시간(분)",
            )
            st.plotly_chart(fig, use_container_width=True)
            if has_ewi_v2:
                st.caption(
                    "집중 작업: FAB 내 고활성(ar≥0.90) | 경작업: FAB 내 저활성 | "
                    "이동 중: 5분 내 3회+ 장소 변경 감지 | "
                    "점심 외부: 타각기 통과 확인된 외부 체류 | "
                    "BLE 미감지: 30분+ 센서 음영 ※야간 근무자는 D+1 새벽 기록이 누락될 수 있음"
                )
        else:
            st.caption("활동 데이터 없음")

    with col_space:
        st.markdown(sub_header("pin 공간 체류"), unsafe_allow_html=True)

        # locus_token → 표시 카테고리 매핑
        # outdoor_work = FAB 외부 이동 통로 (작업 공간 아님)
        _TOKEN_TO_SPACE = {
            "work_zone":    "FAB 작업구역",
            "outdoor_work": "이동구간",
            "transit":      "이동구간",
            "timeclock":    "이동구간",
            "breakroom":    "휴게공간",
            "restroom":     "휴게공간",
            "smoking_area": "휴게공간",
            "dining_hall":  "휴게공간",
            "shadow_zone":  "BLE미감지",
        }
        sp_colors = {
            "FAB 작업구역": "#00AEEF", "이동구간": "#9AB5D4",
            "휴게공간": "#00C897", "밀폐공간": "#FF6B35",
            "고압전구역": "#FF4C4C", "BLE미감지": "#5A6A7A",
        }

        # Journey 데이터에서 직접 계산 (분 단위 deduplicate → 정확한 비율)
        space_data: dict = {}
        _user_no_sp = wrow.get("user_no")
        _jdf_sp = st.session_state.get("current_journey_df")
        if _user_no_sp and _jdf_sp is not None and not _jdf_sp.empty:
            uj = _jdf_sp[_jdf_sp["user_no"] == _user_no_sp]
            if not uj.empty and "locus_token" in uj.columns:
                # 같은 분에 복수 S-Ward 감지 시 신호 강도 최대 위치 우선
                sc = "signal_count" if "signal_count" in uj.columns else None
                uj_dedup = (
                    uj.sort_values(sc, ascending=False) if sc else uj
                ).drop_duplicates(subset=["timestamp"])
                for token, cnt in uj_dedup["locus_token"].value_counts().items():
                    label = _TOKEN_TO_SPACE.get(str(token))
                    if label:
                        space_data[label] = space_data.get(label, 0) + int(cnt)
                # 밀폐/고압전은 locus_id 기반이라 pre-computed 값 보완
                for extra_key, extra_col in [("밀폐공간", "confined_minutes"), ("고압전구역", "high_voltage_minutes")]:
                    v = wrow.get(extra_col, 0)
                    if v and v > 0 and extra_key not in space_data:
                        space_data[extra_key] = v

        # Fallback: pre-aggregated columns
        if not space_data:
            space_data = {
                "FAB 작업구역": wrow.get("work_zone_minutes", 0),
                "이동구간":     wrow.get("outdoor_minutes", 0),
                "휴게공간":     wrow.get("rest_minutes", 0),
                "밀폐공간":     wrow.get("confined_minutes", 0),
                "고압전구역":   wrow.get("high_voltage_minutes", 0),
                "BLE미감지":    wrow.get("shadow_minutes", 0),
            }

        space_data = {k: v for k, v in space_data.items() if v > 0}
        if space_data:
            fig = go.Figure(go.Pie(
                labels=list(space_data.keys()),
                values=list(space_data.values()),
                marker_colors=[sp_colors.get(k, "#888") for k in space_data.keys()],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                paper_bgcolor="#1A2A3A",
                plot_bgcolor="#111820",
                font_color="#C8D6E8",
                margin=dict(l=10, r=10, t=10, b=10),
                height=220, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("공간 체류 데이터 없음")

    # 위험 지표 상세
    if has_cre:
        st.markdown(sub_header("warning 위험 구성 요소"), unsafe_allow_html=True)
        risk_cols = st.columns(4)
        risks = [
            ("공간위험", wrow.get("static_norm", 0), 0.5),
            ("밀집도", wrow.get("dynamic_norm", 0), 0.5),
            ("피로도", wrow.get("fatigue_score", 0), 0.6),
            ("단독작업", wrow.get("alone_ratio", 0), 0.5),
        ]
        for col, (label, val, threshold) in zip(risk_cols, risks):
            with col:
                color = "#FF4C4C" if val >= threshold else "#FFB300" if val >= threshold * 0.7 else "#00C897"
                st.markdown(
                    f"<div style='background:#111820; border-radius:8px; padding:10px; text-align:center;'>"
                    f"<div style='color:{color}; font-size:1.2rem; font-weight:700;'>"
                    f"{val:.2f}</div>"
                    f"<div style='color:#9AB5D4; font-size:0.78rem;'>{label}</div></div>",
                    unsafe_allow_html=True,
                )

    # 이동 경로
    if "locus_sequence" in wrow and pd.notna(wrow.get("locus_sequence")):
        st.markdown(sub_header("map 이동 경로"), unsafe_allow_html=True)

        # Journey 타임라인 시각화
        user_no = wrow.get("user_no", "")
        if user_no:
            try:
                if "current_journey_df" in st.session_state and not st.session_state.current_journey_df.empty:
                    journey_df_for_timeline = st.session_state.current_journey_df
                    fig_timeline = journey_timeline(journey_df_for_timeline, user_no, locus_dict)
                    st.plotly_chart(fig_timeline, use_container_width=True)
            except Exception:
                pass  # journey_timeline은 선택적 기능

        tokens = str(wrow["locus_sequence"]).split()
        locus_names = [locus_dict.get(t, {}).get("locus_name", t) for t in tokens]
        # 중복 연속 제거하여 이동 흐름만 표시
        path = []
        for name in locus_names:
            if not path or path[-1] != name:
                path.append(name)
        path_display = " -> ".join(f"`{p}`" for p in path[:25])
        st.markdown(path_display)
        if len(path) > 25:
            st.caption(f"(+{len(path)-25}개 추가 이동)")
        _ble_min  = int(wrow.get("recorded_minutes", 0))
        _work_min = wrow.get("work_minutes", 0)
        _blocks   = len(tokens)
        st.caption(
            f"위치 블록 {_blocks}개 | "
            f"BLE 기록 {_ble_min}분 / 근무 {_work_min:.0f}분 | "
            f"{wrow.get('transition_count', 0)}회 이동"
        )

    # ── Journey 기반 시간대별 분석 + 맥락 분석 ────────────────────────
    _user_journey = None
    _user_no = wrow.get("user_no", "")
    if _user_no and "current_journey_df" in st.session_state:
        _jdf = st.session_state.current_journey_df
        if _jdf is not None and not _jdf.empty:
            _user_journey = _jdf[_jdf["user_no"] == _user_no].copy()
            if _user_journey.empty:
                _user_journey = None

    _render_journey_analysis(_user_journey, wrow, locus_dict)
    _render_context_analysis(wrow, has_ewi, has_cre, _user_journey)


def _render_journey_analysis(journey_df, wrow, locus_dict):
    """시간대별 Journey 타임라인 — 어디서 무엇을 했는지 시각화."""
    if journey_df is None or journey_df.empty:
        return

    st.markdown(sub_header("🕐 시간대별 활동 타임라인"), unsafe_allow_html=True)

    jj = journey_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(jj["timestamp"]):
        jj["timestamp"] = pd.to_datetime(jj["timestamp"])
    jj["_hour"] = jj["timestamp"].dt.hour

    # 시간대별 집계: 토큰별 분 수 + 평균 활성비율
    has_ar = "active_ratio" in jj.columns

    # 토큰 색상 매핑
    token_colors = {
        "work_zone": "#00AEEF", "outdoor_work": "#0078AA",
        "transit": "#FF8C42", "timeclock": "#FFB300",
        "breakroom": "#00C897", "restroom": "#4CAF50", "smoking_area": "#66BB6A",
        "shadow_zone": "#5A6A7A", "unknown": "#3A3A3A",
    }
    token_labels = {
        "work_zone": "작업구역", "outdoor_work": "야외",
        "transit": "호이스트", "timeclock": "타각기",
        "breakroom": "휴게실", "restroom": "화장실", "smoking_area": "흡연실",
        "shadow_zone": "BLE미감지", "unknown": "미분류",
    }

    # 시간대별 토큰 분포 (stacked bar)
    hourly = jj.groupby(["_hour", "locus_token"]).size().unstack(fill_value=0)
    # 근무 시간대만 (기록 있는 시간)
    active_hours = sorted(jj["_hour"].unique())

    fig = go.Figure()
    all_tokens = hourly.columns.tolist()
    # 토큰 순서 정렬 (작업 → 이동 → 휴식 → 미감지)
    token_order = ["work_zone", "outdoor_work", "transit", "timeclock",
                   "breakroom", "restroom", "smoking_area", "shadow_zone", "unknown"]
    sorted_tokens = [t for t in token_order if t in all_tokens]
    sorted_tokens += [t for t in all_tokens if t not in sorted_tokens]

    for token in sorted_tokens:
        if token not in hourly.columns:
            continue
        vals = hourly[token].reindex(active_hours, fill_value=0)
        fig.add_trace(go.Bar(
            x=[f"{h:02d}시" for h in active_hours],
            y=vals.values,
            name=token_labels.get(token, token),
            marker_color=token_colors.get(token, "#666"),
            hovertemplate=f"{token_labels.get(token, token)}: %{{y}}분<extra></extra>",
        ))

    # 활성비율 라인 추가
    if has_ar:
        hourly_ar = jj.groupby("_hour")["active_ratio"].mean().reindex(active_hours, fill_value=0)
        fig.add_trace(go.Scatter(
            x=[f"{h:02d}시" for h in active_hours],
            y=hourly_ar.values * 60,  # 0~1 → 0~60 스케일로 (분 축에 맞춤)
            name="활성비율(×60)",
            line=dict(color="#FF4C4C", width=2, dash="dot"),
            yaxis="y",
            hovertemplate="활성비율: %{customdata:.2f}<extra></extra>",
            customdata=hourly_ar.values,
        ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#1A2A3A",
        plot_bgcolor="#111820",
        font_color="#C8D6E8",
        margin=dict(l=40, r=10, t=10, b=40),
        height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)),
        xaxis_title="시간대",
        yaxis_title="분",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 간단한 시간대 요약
    if has_ar:
        # 주요 이벤트 추출
        events = []
        for h in active_hours:
            h_data = jj[jj["_hour"] == h]
            tokens = h_data["locus_token"].value_counts()
            ar_mean = h_data["active_ratio"].mean() if has_ar else 0
            dominant = tokens.index[0] if len(tokens) > 0 else "?"
            dominant_pct = tokens.iloc[0] / len(h_data) * 100 if len(h_data) > 0 else 0

            # 특이 이벤트만 표시
            if dominant == "shadow_zone" and dominant_pct > 50:
                events.append(f"`{h:02d}시` BLE미감지 {dominant_pct:.0f}%")
            elif dominant == "timeclock":
                events.append(f"`{h:02d}시` 타각기 통과")
            elif dominant in ("breakroom", "restroom", "smoking_area") and dominant_pct > 30:
                events.append(f"`{h:02d}시` {token_labels.get(dominant, dominant)}")

        if events:
            st.caption("주요 이벤트: " + " · ".join(events))


def _render_context_analysis(wrow, has_ewi: bool, has_cre: bool, journey_df=None):
    """규칙 기반 맥락 분석 — 작업자의 하루를 해석하여 핵심 인사이트 제공."""
    st.markdown(sub_header("📋 맥락 분석"), unsafe_allow_html=True)

    wm = wrow.get("work_minutes", 0)
    if wm <= 0:
        st.caption("근무 데이터 부족")
        return

    findings = []  # (icon, title, detail, severity)  severity: info/warning/danger

    # ── 1. 근무 패턴 요약 ──────────────────────────────────────────
    wm_h = wm / 60
    shift = wrow.get("shift_type", "unknown")
    shift_label = {"day": "주간", "night": "야간"}.get(shift, "미분류")

    intense = wrow.get("intense_work_min", 0) or 0
    light = wrow.get("light_work_min", 0) or 0
    idle = wrow.get("idle_at_work_min", 0) or 0
    rest = wrow.get("rest_min", 0) or 0
    transit_m = wrow.get("transit_min", 0) or 0
    in_transit = wrow.get("in_transit_min", 0) or 0
    shadow = wrow.get("shadow_min", 0) or 0
    total_work = intense + light

    if wm > 0:
        work_pct = total_work / wm * 100
        idle_pct = idle / wm * 100
        shadow_pct = shadow / wm * 100
    else:
        work_pct = idle_pct = shadow_pct = 0

    findings.append(("⏱️", "근무 개요",
                      f"{shift_label} 근무 {wm_h:.1f}시간 | "
                      f"실작업 {total_work:.0f}분({work_pct:.0f}%) · "
                      f"대기 {idle:.0f}분({idle_pct:.0f}%) · "
                      f"이동 {transit_m + in_transit:.0f}분 · "
                      f"휴식 {rest:.0f}분", "info"))

    # ── 2. 점심 분석 (journey 기반 강화) ────────────────────────────
    lmt = wrow.get("lmt_minutes")
    lbt = wrow.get("lbt_minutes")
    has_lmt = pd.notna(lmt) and float(lmt) > 0
    has_lbt = pd.notna(lbt) and float(lbt) > 0

    # journey에서 점심 시간대(11~14시) 활동 패턴 분석
    lunch_journey_detail = ""
    if journey_df is not None and not journey_df.empty:
        jj = journey_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(jj["timestamp"]):
            jj["timestamp"] = pd.to_datetime(jj["timestamp"])
        lunch_period = jj[(jj["timestamp"].dt.hour >= 11) & (jj["timestamp"].dt.hour < 14)]
        if not lunch_period.empty:
            lunch_tokens = lunch_period["locus_token"].value_counts()
            lunch_unique = lunch_period["locus_token"].nunique()
            lunch_total = len(lunch_period)
            # 점심 시간대 shadow_zone 비율
            lunch_shadow = (lunch_period["locus_token"] == "shadow_zone").sum()
            lunch_shadow_pct = lunch_shadow / lunch_total * 100 if lunch_total > 0 else 0
            # 점심 시간대 timeclock 통과 여부
            lunch_tc = (lunch_period["locus_token"] == "timeclock").sum()
            # 점심 시간대 활성 비율
            lunch_ar = lunch_period["active_ratio"].mean() if "active_ratio" in lunch_period.columns else 0

            parts = []
            if lunch_tc > 0:
                parts.append(f"타각기 통과 {lunch_tc}회")
            if lunch_shadow_pct > 30:
                parts.append(f"BLE 미감지 {lunch_shadow_pct:.0f}%")
            top_token = lunch_tokens.index[0] if len(lunch_tokens) > 0 else "?"
            top_pct = lunch_tokens.iloc[0] / lunch_total * 100 if lunch_total > 0 else 0
            parts.append(f"주요 위치: {top_token}({top_pct:.0f}%)")
            parts.append(f"활성비율: {lunch_ar:.2f}")
            lunch_journey_detail = " | ".join(parts)

    if has_lmt:
        findings.append(("🍽️", "점심 확인",
                          f"점심 외부 체류 {float(lmt):.0f}분"
                          + (f" (왕복 이동 {float(lbt):.0f}분)" if has_lbt else "")
                          + (f" — 11~14시: {lunch_journey_detail}" if lunch_journey_detail else ""),
                          "info"))
    elif wm_h >= 6:
        # 6시간 이상 근무인데 점심 기록 없음
        detail = f"6시간 이상 근무이나 점심 기록 없음."
        if lunch_journey_detail:
            detail += f" 11~14시 패턴: {lunch_journey_detail}."
        if shadow > 30:
            detail += f" BLE 미감지 {shadow:.0f}분 중 점심시간이 포함되었을 가능성."
        else:
            detail += " 타각기 미통과 또는 현장 내 식사 가능성."
        findings.append(("🍽️", "점심 미확인", detail, "warning"))

    # ── 3. 출퇴근 대기 ────────────────────────────────────────────
    mat = wrow.get("mat_minutes")
    eod = wrow.get("eod_minutes")
    has_mat = pd.notna(mat) and float(mat) > 0
    has_eod = pd.notna(eod) and float(eod) > 0

    if has_mat or has_eod:
        parts = []
        if has_mat:
            parts.append(f"출근 대기 {float(mat):.1f}분")
        if has_eod:
            parts.append(f"퇴근 이동 {float(eod):.1f}분")
        sev = "warning" if (has_mat and float(mat) > 15) or (has_eod and float(eod) > 20) else "info"
        findings.append(("🚶", "출퇴근", " · ".join(parts), sev))

    # ── 4. BLE 미감지 ─────────────────────────────────────────────
    if shadow_pct > 30:
        findings.append(("📡", "BLE 미감지 높음",
                          f"전체 근무의 {shadow_pct:.0f}%({shadow:.0f}분)가 센서 음영. "
                          "실제 활동이 집계에서 누락되어 EWI가 과소평가될 수 있음.",
                          "warning"))
    elif shadow_pct > 10:
        findings.append(("📡", "BLE 부분 미감지",
                          f"근무의 {shadow_pct:.0f}%({shadow:.0f}분) 센서 음영 발생.",
                          "info"))

    # ── 5. T-Ward 이상 패턴 (journey 기반 심층 분석) ──────────────
    _unique_loci = wrow.get("unique_loci", 0)
    _transitions = wrow.get("transition_count", 0)
    ewi_val = wrow.get("ewi", 1) if has_ewi else 1

    # journey 기반 추가 근거
    journey_evidence = []
    if journey_df is not None and not journey_df.empty:
        jj = journey_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(jj["timestamp"]):
            jj["timestamp"] = pd.to_datetime(jj["timestamp"])

        # 시간대별 토큰 다양성
        jj["_hour"] = jj["timestamp"].dt.hour
        hourly_diversity = jj.groupby("_hour")["locus_token"].nunique()
        max_diversity = hourly_diversity.max() if not hourly_diversity.empty else 0
        all_same_token = (jj["locus_token"].nunique() == 1)

        # 활성 비율 시간대별 변화
        if "active_ratio" in jj.columns:
            hourly_ar = jj.groupby("_hour")["active_ratio"].mean()
            ar_std = hourly_ar.std() if len(hourly_ar) > 1 else 0
            ar_max = hourly_ar.max()
            always_low = ar_max < 0.10  # 모든 시간대 활성 비율 < 10%
        else:
            ar_std, ar_max, always_low = 0, 0, False

        # 24시간 연속 기록 여부 (정상 작업자는 출퇴근 사이만 기록)
        hour_range = jj["_hour"].max() - jj["_hour"].min() + 1
        is_24h = hour_range >= 22  # 거의 24시간 연속

        if all_same_token:
            journey_evidence.append("전 시간 동일 위치")
        if always_low:
            journey_evidence.append(f"모든 시간대 활성비율 < 10%(최대 {ar_max:.2f})")
        if is_24h:
            journey_evidence.append(f"{hour_range}시간 연속 BLE 감지")
        if ar_std < 0.01 and len(hourly_ar) > 4:
            journey_evidence.append("시간대별 활성 변화 없음(단조)")

    evidence_str = " · ".join(journey_evidence) if journey_evidence else ""

    if _unique_loci <= 1 and _transitions <= 1 and ewi_val < 0.05 and wm > 360:
        detail = (
            f"단일 지점 고정, 활동 비율 극히 낮음(EWI={ewi_val:.3f}). "
            "헬멧이 작업 구역에 방치되었을 가능성이 높습니다."
        )
        if evidence_str:
            detail += f" 근거: {evidence_str}."
        findings.append(("🪖", "T-Ward 방치 의심", detail, "danger"))
    elif idle_pct > 80 and wm > 360:
        detail = f"근무 시간의 {idle_pct:.0f}%가 대기 상태. T-Ward 저활성 또는 실제 대기 가능성."
        if evidence_str:
            detail += f" 패턴: {evidence_str}."
        findings.append(("⚠️", "대기 비율 과다", detail, "warning"))

    # ── 6. 생산성 판정 ────────────────────────────────────────────
    if has_ewi and ewi_val > 0:
        if ewi_val >= 0.7:
            findings.append(("💪", "높은 집중도",
                              f"EWI {ewi_val:.3f} — 작업 구역에서 고활성 비율이 높은 집중 근무.",
                              "info"))
        elif ewi_val < 0.2 and _unique_loci > 1:
            findings.append(("📉", "낮은 집중도",
                              f"EWI {ewi_val:.3f} — 작업 구역 내 활성 비율이 낮음. "
                              "대기·이동이 많거나 저강도 작업 위주.",
                              "warning"))

    # ── 7. 안전 ─────────────────────────────────────────────────
    fatigue = wrow.get("fatigue_score", 0) or 0
    if fatigue >= 0.6:
        findings.append(("🔴", "피로 위험",
                          f"피로도 {fatigue:.2f} — 장시간 연속 작업 또는 고강도 활동 누적.",
                          "danger"))
    confined = wrow.get("confined_minutes", 0) or 0
    if confined > 60:
        findings.append(("🔒", "밀폐공간 장시간",
                          f"밀폐공간 {confined:.0f}분 체류 — 안전 모니터링 강화 필요.",
                          "warning"))

    # ── 렌더링 ────────────────────────────────────────────────────
    severity_colors = {
        "info": "#1A2A3A",
        "warning": "#2A2510",
        "danger": "#2A1015",
    }
    severity_borders = {
        "info": "#2A4A6A",
        "warning": "#FF8C42",
        "danger": "#FF4C4C",
    }
    html_parts = []
    for icon, title, detail, severity in findings:
        bg = severity_colors.get(severity, "#1A2A3A")
        border = severity_borders.get(severity, "#2A4A6A")
        html_parts.append(
            f"<div style='background:{bg}; border-left:3px solid {border}; "
            f"border-radius:6px; padding:10px 14px; margin-bottom:6px; font-size:0.84rem;'>"
            f"<span style='font-size:1rem;'>{icon}</span> "
            f"<b style='color:#D5E5FF;'>{title}</b>"
            f"<span style='color:#9AB5D4; margin-left:8px;'>{detail}</span>"
            f"</div>"
        )

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)
