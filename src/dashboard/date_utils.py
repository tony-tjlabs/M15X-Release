"""
Date Utilities - 날짜 선택 및 날씨 유틸리티
==========================================
main.py에서 분리된 공통 날짜 관련 함수.
각 탭에서 import하여 사용.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

import config as cfg
from src.pipeline.weather import fetch_weather

# ═══════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════

DAY_NAMES_KR = ["월", "화", "수", "목", "금", "토", "일"]
WEATHER_ICONS = {"Sunny": "☀️", "Rain": "🌧️", "Snow": "❄️", "Unknown": "🌤️"}

# M15X 현장 좌표 (이천시)
M15X_LATITUDE = 37.235
M15X_LONGITUDE = 127.209


# ═══════════════════════════════════════════════════════════════════
# 날씨 정보 조회
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_weather_info(dates: list[str]) -> dict[str, dict]:
    """
    날짜 목록에 대한 요일+날씨 정보 반환.

    Parameters
    ----------
    dates : YYYYMMDD 또는 YYYY-MM-DD 형식 날짜 목록

    Returns
    -------
    dict[date_str, {day_kr, weather, icon, temp_max, temp_min, label}]
    """
    if not dates:
        return {}

    # 날짜 형식 통일 (YYYY-MM-DD)
    dates_fmt = []
    for d in sorted(dates):
        if len(d) == 8 and d.isdigit():
            dates_fmt.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
        else:
            dates_fmt.append(d)

    # 날씨 API 호출
    try:
        weather_df = fetch_weather(
            dates_fmt[0], dates_fmt[-1],
            latitude=M15X_LATITUDE, longitude=M15X_LONGITUDE
        )
    except Exception:
        weather_df = pd.DataFrame()

    info = {}
    for d_orig, d_fmt in zip(sorted(dates), dates_fmt):
        try:
            dt = datetime.strptime(d_fmt, "%Y-%m-%d")
        except ValueError:
            continue

        day_kr = DAY_NAMES_KR[dt.weekday()]
        weather = "Unknown"
        temp_max, temp_min = None, None

        if not weather_df.empty and d_fmt in weather_df["date"].values:
            row = weather_df[weather_df["date"] == d_fmt].iloc[0]
            weather = row.get("weather", "Unknown")
            temp_max = row.get("temp_max")
            temp_min = row.get("temp_min")

        icon = WEATHER_ICONS.get(weather, "🌤️")
        temp_str = f" {temp_min:.0f}~{temp_max:.0f}" if temp_max is not None else ""

        # YYYYMMDD 형식으로 라벨 생성 (MM/DD 형식)
        mm_dd = f"{dt.month:02d}/{dt.day:02d}"

        info[d_orig] = {
            "day_kr": day_kr,
            "weather": weather,
            "icon": icon,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "label": f"{mm_dd} ({day_kr}) {icon}{temp_str}",
        }

    return info


# ═══════════════════════════════════════════════════════════════════
# 날짜 선택기 위젯
# ═══════════════════════════════════════════════════════════════════

def get_date_selector(
    dates: list[str],
    key: str = "date_selector",
    default_index: int | None = None,
    label: str = "날짜 선택",
    show_label: bool = False,
) -> str | None:
    """
    요일+날씨 포함 날짜 선택기.

    각 탭에서 호출하여 사용.

    Parameters
    ----------
    dates : 사용 가능한 날짜 목록 (YYYYMMDD 형식)
    key : Streamlit widget 키
    default_index : 기본 선택 인덱스 (None이면 마지막 날짜)
    label : 라벨 텍스트
    show_label : 라벨 표시 여부

    Returns
    -------
    선택된 날짜 (YYYYMMDD 형식) 또는 None
    """
    if not dates:
        return None

    weather_info = fetch_weather_info(dates)
    date_labels = [weather_info.get(d, {}).get("label", d) for d in dates]

    if default_index is None:
        default_index = len(dates) - 1

    selected_label = st.selectbox(
        label,
        date_labels,
        index=default_index,
        key=key,
        label_visibility="visible" if show_label else "collapsed",
    )

    # 선택된 라벨 -> 날짜 매핑
    try:
        idx = date_labels.index(selected_label)
        return dates[idx]
    except (ValueError, IndexError):
        return dates[-1] if dates else None


def get_weekday_korean(d) -> str:
    """요일 한글 반환 (date 또는 datetime 객체)."""
    return DAY_NAMES_KR[d.weekday()]


def get_date_badge(d, styles_module=None) -> str:
    """날짜 뱃지 HTML (주중/주말)."""
    if styles_module is None:
        from src.dashboard.styles import badge
    else:
        badge = styles_module.badge

    if d.weekday() >= 5:
        return badge("주말", "warning")
    return badge("주중", "info")


def parse_date_str(date_str: str):
    """YYYYMMDD 또는 YYYY-MM-DD 형식 날짜 문자열 파싱."""
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def format_date_label(date_str) -> str:
    """YYYYMMDD -> MM/DD 형식으로 변환. int/float 입력도 처리."""
    date_str = str(date_str).split(".")[0]  # int/float 방어
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[4:6]}/{date_str[6:]}"
    return date_str


def format_date_full(date_str) -> str:
    """YYYYMMDD -> YYYY-MM-DD 형식으로 변환. int/float 입력도 처리."""
    date_str = str(date_str).split(".")[0]  # int/float 방어
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str
