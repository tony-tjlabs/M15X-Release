"""
Transit Analyzer - 이동/대기 시간 분석 모듈
=============================================
고객 요구사항에 따른 출근/중식/퇴근 대기시간 분석.

**고객 원문 요구사항:**
> 출근 시간: 근로 현장까지 도착 시간
> 중식 시간: 호이스트에서 출문까지 소요 시간
> 퇴근 시간: 호이스트에서 출문까지 소요 시간
> 각각의 Data를 일별 / BP별로 분석

**M15X 공간 구조 (11개 Locus):**
- timeclock: 타각기 (출입문) - 출퇴근 기록 지점
- transit: 호이스트2/3 - 수직이동 수단 (층간 이동)
- work_zone: FAB 5F/6F/7F/RF - 실제 작업 장소
- outdoor_work: 야외 공사현장
- breakroom/restroom/smoking_area: 휴게시설

**지표 정의:**
- MAT (Morning Arrival Time): 타각기 출근 → FAB 작업층 최초 도착 (분)
- LBT (Lunch Break Transit): 중식 시간대 FAB → 호이스트 → 타각기 (분)
- EOD (End-of-Day Transit): 퇴근 시간대 FAB → 호이스트 → 타각기 (분)
- TWT (Total Wait Time): MAT + LBT + EOD
- TRR (Transit Ratio): TWT / 근무시간 (%)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─── Locus 토큰 상수 ──────────────────────────────────────────────────
# M15X v1: spot_name 기반 11개 Locus
# FAB 작업층만 최종 목적지 (outdoor_work는 이동 경로 상 중간 지점)
WORK_ZONE_TOKENS = {"work_zone"}         # FAB 5F/6F/7F/RF
GATE_TOKENS = {"timeclock"}              # 타각기 (출입문)
TRANSIT_TOKENS = {"transit"}             # 호이스트2/3 (수직이동)
OUTDOOR_TOKENS = {"outdoor_work"}        # 야외 공사현장 (이동 경로)
REST_TOKENS = {"breakroom", "restroom", "smoking_area"}


def _get_config_values() -> dict:
    """config.py에서 임계값 로드."""
    try:
        import config as cfg
        return {
            "gap_threshold": getattr(cfg, "TRANSIT_GAP_THRESHOLD_MIN", 5),
            "lunch_start": getattr(cfg, "LUNCH_START_HOUR", 11),
            "lunch_end": getattr(cfg, "LUNCH_END_HOUR", 14),
            "work_end": getattr(cfg, "WORK_END_HOUR_THRESHOLD", 17),
        }
    except ImportError:
        # 테스트 환경용 기본값
        return {
            "gap_threshold": 5,
            "lunch_start": 11,
            "lunch_end": 14,
            "work_end": 17,
        }


def _calc_mat_for_worker(df: pd.DataFrame) -> tuple[float | None, bool]:
    """
    MAT (Morning Arrival Time) 계산 - 단일 작업자용.

    정의: AccessLog in_datetime → 첫 번째 work_zone BLE 기록 시간 차이 (분)
    - 출입 시각: AccessLog의 공식 출입 기록 (정확한 기준)
    - 도착 시각: T-Ward BLE의 FAB 작업층 첫 감지 (gap-filled 포함)

    Args:
        df: 단일 작업자의 시간순 정렬된 journey DataFrame
            (in_datetime, is_gap_filled 컬럼 포함 가능)

    Returns:
        (MAT 분, is_estimated) — is_estimated=True이면 gap-filled 레코드 기반 추정값
    """
    in_time = df["in_datetime"].iloc[0] if "in_datetime" in df.columns else None
    if pd.isna(in_time):
        return None, False

    work_records = df[
        (df["locus_token"].isin(WORK_ZONE_TOKENS)) & (df["timestamp"] > in_time)
    ]
    if work_records.empty:
        return None, False

    first_work_time = work_records["timestamp"].min()
    mat_minutes = (first_work_time - in_time).total_seconds() / 60

    if mat_minutes > 60 or mat_minutes < 0:   # 게이트→FAB 현실 상한 60분
        return None, False

    # 첫 work_zone 레코드가 gap-filled인지 확인
    first_rec = work_records[work_records["timestamp"] == first_work_time].iloc[0]
    is_estimated = bool(first_rec.get("is_gap_filled", False))

    return round(mat_minutes, 1), is_estimated


def _count_seg_gap(records: pd.DataFrame, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> float:
    """
    구간 내 BLE 미감지 시간(분) 계산.

    1분 단위 BLE 레코드를 기준으로,
    구간 내 실제 감지된 레코드 수를 제외한 나머지 = 미감지(음영) 구간.

    Args:
        records: 해당 작업자의 전체 journey DataFrame (timestamp 컬럼 필요)
        seg_start: 구간 시작 타임스탬프 (exclusive — 체크포인트 감지 시각)
        seg_end:   구간 종료 타임스탬프 (exclusive — 다음 체크포인트 감지 시각)

    Returns:
        gap_min: 미감지 시간 (분, 소수 1자리)
    """
    if pd.isna(seg_start) or pd.isna(seg_end) or seg_end <= seg_start:
        return 0.0
    total_min = (seg_end - seg_start).total_seconds() / 60
    # 구간 내 BLE 기록 수 (시작·종료 체크포인트 제외, 중간 기록만)
    n_detected = int(
        ((records["timestamp"] > seg_start) & (records["timestamp"] < seg_end)).sum()
    )
    gap_min = max(0.0, total_min - n_detected)
    return round(gap_min, 1)


def _calc_route_segments_for_worker(df: pd.DataFrame) -> dict | None:
    """
    출근 경로 구간별 소요시간 분석 — 단일 작업자용.

    기준: AccessLog in_datetime → BLE 중간 경유지 → FAB 작업층
    - 출발: AccessLog 공식 입장 시각
    - 중간: T-Ward BLE 감지 (공사현장, 호이스트)
    - 도착: FAB 작업층 첫 BLE 감지

    Returns:
        dict with keys:
        - gate_to_outdoor: 입장 → 공사현장 소요시간 (분), 없으면 None
        - gap_gate_outdoor: gate_to_outdoor 구간 내 미감지 시간 (분)
        - outdoor_to_hoist: 공사현장 → 호이스트 소요시간 (분), 없으면 None
        - gap_outdoor_hoist: outdoor_to_hoist 구간 내 미감지 시간 (분)
        - hoist_to_fab: 호이스트 → FAB 작업층 소요시간 (분), 없으면 None
        - gap_hoist_fab: hoist_to_fab 구간 내 미감지 시간 (분)
        - gate_to_fab_total: 입장 → FAB 전체 소요시간 (분)
        - route_sequence: 실제 경유한 locus_token 순서 리스트
        또는 None (데이터 부족 시)
    """
    # AccessLog 입장 시각 (공식 기준)
    in_time = df["in_datetime"].iloc[0] if "in_datetime" in df.columns else None
    if pd.isna(in_time):
        return None

    # 입장 이후 FAB 작업층 첫 BLE 기록
    work_recs = df[(df["locus_token"].isin(WORK_ZONE_TOKENS)) & (df["timestamp"] > in_time)]
    if work_recs.empty:
        return None
    first_fab = work_recs["timestamp"].min()

    # 입장~FAB 사이의 모든 BLE 기록 (경유지 추출)
    between = df[(df["timestamp"] >= in_time) & (df["timestamp"] <= first_fab)]
    if between.empty:
        return None

    # 경유 순서 (중복 제거, 순서 유지)
    route_tokens = between["locus_token"].tolist()
    seen = set()
    route_seq = []
    for t in route_tokens:
        if t not in seen:
            seen.add(t)
            route_seq.append(t)

    total_min = (first_fab - in_time).total_seconds() / 60
    if total_min > 60 or total_min < 0:   # MAT 구간 분석 상한 60분
        return None

    result = {
        "gate_to_outdoor": None,
        "gap_gate_outdoor": 0.0,
        "outdoor_to_hoist": None,
        "gap_outdoor_hoist": 0.0,
        "hoist_to_fab": None,
        "gap_hoist_fab": 0.0,
        "gate_to_fab_total": round(total_min, 1),
        "route_sequence": route_seq,
    }

    # 공사현장(outdoor_work) 첫 BLE 기록
    outdoor_recs = between[between["locus_token"].isin(OUTDOOR_TOKENS)]
    if not outdoor_recs.empty:
        first_outdoor = outdoor_recs["timestamp"].min()
        result["gate_to_outdoor"] = round(
            (first_outdoor - in_time).total_seconds() / 60, 1
        )
        result["gap_gate_outdoor"] = _count_seg_gap(between, in_time, first_outdoor)

        # 호이스트(transit) 첫 BLE 기록 (공사현장 이후)
        hoist_recs = between[
            (between["locus_token"].isin(TRANSIT_TOKENS))
            & (between["timestamp"] > first_outdoor)
        ]
        if not hoist_recs.empty:
            first_hoist = hoist_recs["timestamp"].min()
            result["outdoor_to_hoist"] = round(
                (first_hoist - first_outdoor).total_seconds() / 60, 1
            )
            result["gap_outdoor_hoist"] = _count_seg_gap(between, first_outdoor, first_hoist)
            result["hoist_to_fab"] = round(
                (first_fab - first_hoist).total_seconds() / 60, 1
            )
            result["gap_hoist_fab"] = _count_seg_gap(between, first_hoist, first_fab)
        else:
            # 호이스트 없이 바로 FAB 도착
            result["outdoor_to_hoist"] = None
            result["hoist_to_fab"] = round(
                (first_fab - first_outdoor).total_seconds() / 60, 1
            )
            result["gap_hoist_fab"] = _count_seg_gap(between, first_outdoor, first_fab)
    else:
        # 공사현장 거치지 않고 바로 호이스트 또는 FAB
        hoist_recs = between[
            (between["locus_token"].isin(TRANSIT_TOKENS))
            & (between["timestamp"] > in_time)
        ]
        if not hoist_recs.empty:
            first_hoist = hoist_recs["timestamp"].min()
            result["gate_to_outdoor"] = None
            result["outdoor_to_hoist"] = round(
                (first_hoist - in_time).total_seconds() / 60, 1
            )
            result["gap_outdoor_hoist"] = _count_seg_gap(between, in_time, first_hoist)
            result["hoist_to_fab"] = round(
                (first_fab - first_hoist).total_seconds() / 60, 1
            )
            result["gap_hoist_fab"] = _count_seg_gap(between, first_hoist, first_fab)

    return result


def _calc_lbt_out_segments_for_worker(df: pd.DataFrame, cfg_values: dict) -> dict | None:
    """
    점심 외출 (LBT-Out) 경로 구간별 소요시간 분석 -- 단일 작업자용.

    경로: FAB 작업층 -> 호이스트 -> 야외(공사현장) -> 타각기
    중식 시간대(11~14시)에 work_zone에서 마지막으로 떠나 timeclock에 도착하는 구간.

    Returns:
        dict with keys: fab_to_hoist, hoist_to_outdoor, outdoor_to_gate, fab_to_gate_total
        또는 None
    """
    lunch_start = cfg_values["lunch_start"]
    lunch_end = cfg_values["lunch_end"]

    df_lunch = df[
        (df["timestamp"].dt.hour >= lunch_start) &
        (df["timestamp"].dt.hour < lunch_end)
    ]
    if df_lunch.empty:
        return None

    # work_zone 마지막 기록 (출발)
    work_in_lunch = df_lunch[df_lunch["locus_token"].isin(WORK_ZONE_TOKENS)]
    if work_in_lunch.empty:
        return None
    last_work_time = work_in_lunch["timestamp"].max()

    # timeclock 기록 (도착, work_zone 이후)
    gate_after = df_lunch[
        (df_lunch["locus_token"].isin(GATE_TOKENS)) &
        (df_lunch["timestamp"] > last_work_time)
    ]
    if gate_after.empty:
        return None
    first_gate = gate_after["timestamp"].min()

    total_min = (first_gate - last_work_time).total_seconds() / 60
    if total_min > 120 or total_min < 0:
        return None

    between = df_lunch[
        (df_lunch["timestamp"] >= last_work_time) &
        (df_lunch["timestamp"] <= first_gate)
    ]

    result = {
        "fab_to_hoist": None,
        "gap_fab_hoist": 0.0,
        "hoist_to_outdoor": None,
        "gap_hoist_outdoor": 0.0,
        "outdoor_to_gate": None,
        "gap_outdoor_gate": 0.0,
        "fab_to_gate_total": round(total_min, 1),
    }

    # 호이스트 (transit) 첫 기록 (work_zone 이후)
    hoist_recs = between[
        (between["locus_token"].isin(TRANSIT_TOKENS)) &
        (between["timestamp"] > last_work_time)
    ]
    if not hoist_recs.empty:
        first_hoist = hoist_recs["timestamp"].min()
        result["fab_to_hoist"] = round(
            (first_hoist - last_work_time).total_seconds() / 60, 1
        )
        result["gap_fab_hoist"] = _count_seg_gap(between, last_work_time, first_hoist)

        # 야외 (outdoor_work) 기록 (호이스트 이후)
        outdoor_recs = between[
            (between["locus_token"].isin(OUTDOOR_TOKENS)) &
            (between["timestamp"] > first_hoist)
        ]
        if not outdoor_recs.empty:
            first_outdoor = outdoor_recs["timestamp"].min()
            result["hoist_to_outdoor"] = round(
                (first_outdoor - first_hoist).total_seconds() / 60, 1
            )
            result["gap_hoist_outdoor"] = _count_seg_gap(between, first_hoist, first_outdoor)
            result["outdoor_to_gate"] = round(
                (first_gate - first_outdoor).total_seconds() / 60, 1
            )
            result["gap_outdoor_gate"] = _count_seg_gap(between, first_outdoor, first_gate)
        else:
            result["hoist_to_outdoor"] = None
            result["outdoor_to_gate"] = round(
                (first_gate - first_hoist).total_seconds() / 60, 1
            )
            result["gap_outdoor_gate"] = _count_seg_gap(between, first_hoist, first_gate)

    return result


def _calc_lbt_in_segments_for_worker(df: pd.DataFrame, cfg_values: dict) -> dict | None:
    """
    점심 복귀 (LBT-In) 경로 구간별 소요시간 분석 -- 단일 작업자용.

    경로: 타각기 -> 야외(공사현장) -> 호이스트 -> FAB 작업층
    중식 시간대(11~14시)에 timeclock에서 출발하여 work_zone에 복귀하는 구간.
    (MAT과 동일 패턴이지만 점심 시간대)

    Returns:
        dict with keys: gate_to_outdoor, outdoor_to_hoist, hoist_to_fab, gate_to_fab_total
        또는 None
    """
    lunch_start = cfg_values["lunch_start"]
    lunch_end = cfg_values["lunch_end"]

    df_lunch = df[
        (df["timestamp"].dt.hour >= lunch_start) &
        (df["timestamp"].dt.hour < lunch_end)
    ]
    if df_lunch.empty:
        return None

    # timeclock에서 work_zone 이후 재출현하는 timeclock 기록 (복귀 시작)
    gate_in_lunch = df_lunch[df_lunch["locus_token"].isin(GATE_TOKENS)]
    work_in_lunch = df_lunch[df_lunch["locus_token"].isin(WORK_ZONE_TOKENS)]

    if gate_in_lunch.empty or work_in_lunch.empty:
        return None

    # LBT-Out 이후의 복귀: timeclock 이후 work_zone에 다시 나타나는 패턴
    # 점심 시간대의 마지막 timeclock 기록을 복귀 출발로 사용
    last_gate_time = gate_in_lunch["timestamp"].max()

    # 그 이후 work_zone 첫 기록 (복귀 도착) — 45분 이내만 유효
    _search_limit = last_gate_time + pd.Timedelta(minutes=45)
    work_after_gate = df[
        (df["locus_token"].isin(WORK_ZONE_TOKENS)) &
        (df["timestamp"] > last_gate_time) &
        (df["timestamp"] <= _search_limit)
    ]
    if work_after_gate.empty:
        return None
    first_work = work_after_gate["timestamp"].min()

    total_min = (first_work - last_gate_time).total_seconds() / 60
    if total_min > 45 or total_min < 0:
        return None

    between = df[
        (df["timestamp"] >= last_gate_time) &
        (df["timestamp"] <= first_work)
    ]

    result = {
        "gate_to_outdoor": None,
        "gap_gate_outdoor": 0.0,
        "outdoor_to_hoist": None,
        "gap_outdoor_hoist": 0.0,
        "hoist_to_fab": None,
        "gap_hoist_fab": 0.0,
        "gate_to_fab_total": round(total_min, 1),
    }

    # 야외 기록 (타각기 이후)
    outdoor_recs = between[
        (between["locus_token"].isin(OUTDOOR_TOKENS)) &
        (between["timestamp"] > last_gate_time)
    ]
    if not outdoor_recs.empty:
        first_outdoor = outdoor_recs["timestamp"].min()
        result["gate_to_outdoor"] = round(
            (first_outdoor - last_gate_time).total_seconds() / 60, 1
        )
        result["gap_gate_outdoor"] = _count_seg_gap(between, last_gate_time, first_outdoor)

        hoist_recs = between[
            (between["locus_token"].isin(TRANSIT_TOKENS)) &
            (between["timestamp"] > first_outdoor)
        ]
        if not hoist_recs.empty:
            first_hoist = hoist_recs["timestamp"].min()
            result["outdoor_to_hoist"] = round(
                (first_hoist - first_outdoor).total_seconds() / 60, 1
            )
            result["gap_outdoor_hoist"] = _count_seg_gap(between, first_outdoor, first_hoist)
            result["hoist_to_fab"] = round(
                (first_work - first_hoist).total_seconds() / 60, 1
            )
            result["gap_hoist_fab"] = _count_seg_gap(between, first_hoist, first_work)
        else:
            result["hoist_to_fab"] = round(
                (first_work - first_outdoor).total_seconds() / 60, 1
            )
            result["gap_hoist_fab"] = _count_seg_gap(between, first_outdoor, first_work)
    else:
        hoist_recs = between[
            (between["locus_token"].isin(TRANSIT_TOKENS)) &
            (between["timestamp"] > last_gate_time)
        ]
        if not hoist_recs.empty:
            first_hoist = hoist_recs["timestamp"].min()
            result["outdoor_to_hoist"] = round(
                (first_hoist - last_gate_time).total_seconds() / 60, 1
            )
            result["gap_outdoor_hoist"] = _count_seg_gap(between, last_gate_time, first_hoist)
            result["hoist_to_fab"] = round(
                (first_work - first_hoist).total_seconds() / 60, 1
            )
            result["gap_hoist_fab"] = _count_seg_gap(between, first_hoist, first_work)

    return result


def _calc_eod_segments_for_worker(df: pd.DataFrame, cfg_values: dict) -> dict | None:
    """
    퇴근 (EOD) 경로 구간별 소요시간 분석 -- 단일 작업자용.

    기준: BLE 마지막 work_zone → 중간 경유지 → AccessLog out_datetime
    - 출발: T-Ward BLE FAB 작업층 마지막 감지
    - 중간: T-Ward BLE 감지 (호이스트, 공사현장)
    - 도착: AccessLog 공식 퇴장 시각

    Returns:
        dict with keys: fab_to_hoist, hoist_to_outdoor, outdoor_to_gate, fab_to_gate_total
        또는 None
    """
    # AccessLog 퇴장 시각 (공식 기준)
    out_time = df["out_datetime"].iloc[0] if "out_datetime" in df.columns else None
    if pd.isna(out_time):
        return None

    # 마지막 work_zone BLE 기록
    work_records = df[
        (df["locus_token"].isin(WORK_ZONE_TOKENS)) & (df["timestamp"] < out_time)
    ]
    if work_records.empty:
        return None
    last_work_time = work_records["timestamp"].max()

    total_min = (out_time - last_work_time).total_seconds() / 60
    if total_min > 60 or total_min < 0:   # EOD 구간 분석 상한 60분
        return None

    # work_zone ~ out_time 사이의 BLE 기록 (경유지 추출)
    between = df[
        (df["timestamp"] >= last_work_time) &
        (df["timestamp"] <= out_time)
    ]

    result = {
        "fab_to_hoist": None,
        "gap_fab_hoist": 0.0,
        "hoist_to_outdoor": None,
        "gap_hoist_outdoor": 0.0,
        "outdoor_to_gate": None,
        "gap_outdoor_gate": 0.0,
        "fab_to_gate_total": round(total_min, 1),
    }

    # 호이스트 첫 기록 (work_zone 이후)
    hoist_recs = between[
        (between["locus_token"].isin(TRANSIT_TOKENS)) &
        (between["timestamp"] > last_work_time)
    ]
    if not hoist_recs.empty:
        first_hoist = hoist_recs["timestamp"].min()
        result["fab_to_hoist"] = round(
            (first_hoist - last_work_time).total_seconds() / 60, 1
        )
        result["gap_fab_hoist"] = _count_seg_gap(between, last_work_time, first_hoist)

        outdoor_recs = between[
            (between["locus_token"].isin(OUTDOOR_TOKENS)) &
            (between["timestamp"] > first_hoist)
        ]
        if not outdoor_recs.empty:
            first_outdoor = outdoor_recs["timestamp"].min()
            result["hoist_to_outdoor"] = round(
                (first_outdoor - first_hoist).total_seconds() / 60, 1
            )
            result["gap_hoist_outdoor"] = _count_seg_gap(between, first_hoist, first_outdoor)
            result["outdoor_to_gate"] = round(
                (out_time - first_outdoor).total_seconds() / 60, 1
            )
            result["gap_outdoor_gate"] = _count_seg_gap(between, first_outdoor, out_time)
        else:
            result["outdoor_to_gate"] = round(
                (out_time - first_hoist).total_seconds() / 60, 1
            )
            result["gap_outdoor_gate"] = _count_seg_gap(between, first_hoist, out_time)

    return result


def _calc_lmt_for_worker(df: pd.DataFrame, cfg_values: dict) -> float | None:
    """
    LMT (Lunch Meal Time): 맥락 기반 외부 체류 시간 추정.

    단순 룰이 아닌 4가지 맥락 조건을 모두 충족해야 LMT로 인정한다:

    [조건 1] 점심 전 현장 내 작업 확인
        - 점심 시간대(11~14시) 이전에 work_zone BLE 감지가 있어야 함

    [조건 2] 현장 전체 BLE 무신호 구간 존재 (진짜 외부 이탈의 핵심 증거)
        - 타각기만이 아니라 현장 어떤 S-Ward에서도 감지되지 않는 공백이 있어야 함
        - 현장 내 어딘가에 있으면 최소 1개의 S-Ward가 1분 단위로 감지함
        - 무신호 구간 = 물리적으로 현장 외부에 있는 것

    [조건 3] 무신호 구간이 타각기(경계)를 통한 진출입으로 포장되었는가
        - 무신호 시작 직전(±5분 이내) 타각기 감지 → 정문 통과 확인
        - 무신호 종료 직후(±5분 이내) 타각기 감지 → 재진입 확인

    [조건 4] 복귀 후 work_zone에 돌아왔는가
        - 무신호 구간 이후 work_zone BLE 감지 있어야 함

    Returns:
        LMT (분) = 가장 긴 유효 무신호 구간, 또는 None
    """
    lunch_start = cfg_values["lunch_start"]
    lunch_end   = cfg_values["lunch_end"]

    # 탐색 범위: 점심 시작 1시간 전 ~ 점심 종료 2시간 후 (이탈/복귀 여유)
    df_window = df[
        (df["timestamp"].dt.hour >= lunch_start - 1) &
        (df["timestamp"].dt.hour < lunch_end + 2)
    ].sort_values("timestamp")
    if df_window.empty:
        return None

    df_lunch = df_window[
        (df_window["timestamp"].dt.hour >= lunch_start) &
        (df_window["timestamp"].dt.hour < lunch_end)
    ]

    # ── [조건 1] 점심 시간대 work_zone 작업 확인 ──────────────────────
    work_in_lunch = df_lunch[df_lunch["locus_token"].isin(WORK_ZONE_TOKENS)]
    if work_in_lunch.empty:
        return None
    last_work_before = work_in_lunch["timestamp"].max()

    # ── [조건 2] 현장 전체 BLE 무신호 구간 탐지 ──────────────────────
    # ★ shadow_zone(보정 기록)을 제외하고 실제 센서 감지 레코드만 사용
    #   shadow_zone은 BLE 미감지 구간을 파이프라인이 채운 값이므로
    #   외부 이탈 감지 시 간섭을 일으킴 → is_imputed=False OR non-shadow 토큰만
    REAL_TOKENS = WORK_ZONE_TOKENS | GATE_TOKENS | {
        "breakroom", "restroom", "smoking_area",
        "outdoor_work", "transit",
    }
    after_work_all = df_window[df_window["timestamp"] > last_work_before]
    after_work = after_work_all[
        after_work_all["locus_token"].isin(REAL_TOKENS)
    ].sort_values("timestamp")

    if after_work.empty:
        return None

    all_times = after_work["timestamp"].tolist()
    if len(all_times) < 2:
        return None

    # 연속 실제-감지 사이의 간격 = BLE 무신호 구간 (= 건물 밖 가능성)
    gaps = []
    for i in range(len(all_times) - 1):
        gap_start = all_times[i]
        gap_end   = all_times[i + 1]
        gap_min   = (gap_end - gap_start).total_seconds() / 60
        if gap_min >= 15:  # 15분 이상 무신호만 "외부 가능성" 후보
            gaps.append((gap_start, gap_end, gap_min))

    if not gaps:
        return None  # 유의미한 BLE 공백 없음 → 현장 내 이탈 없음

    # ── [조건 3] 퇴장 타각기 통과 확인 ─────────────────────────────────
    # ★ M15X 현장 특성: 복귀 시 타각기가 재감지되지 않는 경우 많음
    #   → 퇴장 측(gap_start 직전) 타각기 확인만 필수,
    #     재입장 측(gap_end 직후)은 work_zone 복귀(조건 4)로 대체
    GATE_WINDOW_MIN = 10  # 타각기 ↔ 무신호 시작 허용 시간차 (분, 여유 확보)

    gate_times = set(
        df_window[df_window["locus_token"].isin(GATE_TOKENS)]["timestamp"].tolist()
    )

    def has_gate_near(target_ts, window_min=GATE_WINDOW_MIN) -> bool:
        """target_ts 기준 ±window_min 이내에 타각기 감지가 있는지."""
        return any(
            abs((g - target_ts).total_seconds()) <= window_min * 60
            for g in gate_times
        )

    valid_lmt = None
    for gap_start, gap_end, gap_min in gaps:
        if gap_min > 150:
            continue  # 2.5시간 초과 = 비현실적 (점심시간 최대 여유 반영)

        # 조건 3: 퇴장 타각기 확인 (재입장은 work_zone 복귀로 대체)
        gate_exit_confirmed = has_gate_near(gap_start)

        if not gate_exit_confirmed:
            continue  # 타각기 통과 없이 사라진 것 → 단순 BLE 음영

        # ── [조건 4] 무신호 이후 work_zone 복귀 확인 ──────────────
        work_after = df_window[
            (df_window["locus_token"].isin(WORK_ZONE_TOKENS)) &
            (df_window["timestamp"] > gap_end)
        ]
        if work_after.empty:
            continue  # 복귀 없음 → 퇴근 동선일 가능성

        # 4가지 조건 모두 충족 → 유효한 LMT 후보
        if valid_lmt is None or gap_min > valid_lmt:
            valid_lmt = gap_min

    if valid_lmt is None:
        return None

    return round(valid_lmt, 1)


def _calc_lbt_for_worker(df: pd.DataFrame, cfg_values: dict) -> dict | None:
    """
    LBT (Lunch Break Transit) 계산 - 단일 작업자용.

    정의: 중식 왕복 이동 시간 합계 = LBT-Out + LBT-In
    - LBT-Out: 마지막 work_zone → 타각기 첫 감지 (FAB 출발 ~ 타각기 도달)
    - LBT-In:  타각기 마지막 감지 → 첫 work_zone 복귀 (타각기 출발 ~ FAB 복귀)
    - LBT = LBT-Out + LBT-In (왕복 이동 총합)

    Note: out/in 컴포넌트를 분리 저장하여 lbt_out_minutes + lbt_in_minutes = lbt_minutes
          가 동일 집단 기준으로 항상 성립하도록 보장.

    Returns:
        {"total": float, "out": float, "in": float | None} 또는 None (중식 이동 없음)
    """
    lunch_start = cfg_values["lunch_start"]
    lunch_end = cfg_values["lunch_end"]

    # 중식 시간대 데이터 필터
    df_lunch = df[
        (df["timestamp"].dt.hour >= lunch_start) &
        (df["timestamp"].dt.hour < lunch_end)
    ]
    if df_lunch.empty:
        return None

    # 중식 시간대 work_zone 마지막 기록
    work_in_lunch = df_lunch[df_lunch["locus_token"].isin(WORK_ZONE_TOKENS)]
    if work_in_lunch.empty:
        return None
    last_work_time = work_in_lunch["timestamp"].max()

    # LBT-Out: 중식 시간대 마지막 work_zone → 타각기 첫 감지 (퇴장)
    gate_in_lunch = df_lunch[
        (df_lunch["locus_token"].isin(GATE_TOKENS)) &
        (df_lunch["timestamp"] > last_work_time)
    ]
    if gate_in_lunch.empty:
        return None
    first_gate_time = gate_in_lunch["timestamp"].min()  # 퇴장 도달
    last_gate_time  = gate_in_lunch["timestamp"].max()  # 재입장 출발

    lbt_out = (first_gate_time - last_work_time).total_seconds() / 60

    # LBT-In: 타각기 마지막 감지 → 점심 후 첫 work_zone 복귀
    # 탐색 범위: last_gate_time + 45분 이내 (비현실적 복귀 제외)
    _lbt_in_search_limit = last_gate_time + pd.Timedelta(minutes=45)
    work_after_gate = df[
        (df["locus_token"].isin(WORK_ZONE_TOKENS)) &
        (df["timestamp"] > last_gate_time) &
        (df["timestamp"] <= _lbt_in_search_limit)
    ]
    if work_after_gate.empty:
        # LBT-In 감지 안 됨 → LBT-Out만으로 부분값 반환
        if lbt_out > 45 or lbt_out < 0:
            return None
        return {"total": round(lbt_out, 1), "out": round(lbt_out, 1), "in": None}

    first_work_return = work_after_gate["timestamp"].min()
    lbt_in = (first_work_return - last_gate_time).total_seconds() / 60

    lbt_total = lbt_out + lbt_in

    # 비현실적인 값 필터: 왕복 90분 초과는 제외 (M15X 현장 기준)
    if lbt_total > 90 or lbt_total < 0:
        return None

    return {
        "total": round(lbt_total, 1),
        "out":   round(lbt_out, 1),
        "in":    round(lbt_in, 1),
    }


def _calc_eod_for_worker(df: pd.DataFrame, cfg_values: dict) -> tuple[float | None, bool]:
    """
    EOD (End-of-Day Transit) 계산 - 단일 작업자용.

    정의: 마지막 work_zone BLE 기록 → AccessLog out_datetime (분)
    - 출발: T-Ward BLE의 FAB 작업층 마지막 감지 (gap-filled 포함)
    - 도착: AccessLog의 공식 퇴장 기록 (정확한 기준)

    Returns:
        (EOD 분, is_estimated) — is_estimated=True이면 gap-filled 레코드 기반 추정값
    """
    out_time = df["out_datetime"].iloc[0] if "out_datetime" in df.columns else None
    if pd.isna(out_time):
        return None, False

    work_records = df[
        (df["locus_token"].isin(WORK_ZONE_TOKENS)) & (df["timestamp"] < out_time)
    ]
    if work_records.empty:
        return None, False
    last_work_time = work_records["timestamp"].max()

    eod_minutes = (out_time - last_work_time).total_seconds() / 60

    if eod_minutes > 60 or eod_minutes < 0:   # FAB→게이트 현실 상한 60분
        return None, False

    # 마지막 work_zone 레코드가 gap-filled인지 확인
    last_rec = work_records[work_records["timestamp"] == last_work_time].iloc[0]
    is_estimated = bool(last_rec.get("is_gap_filled", False))

    return round(eod_minutes, 1), is_estimated


def calc_worker_transit(
    journey_df: pd.DataFrame,
    date_str: str | None = None,
) -> pd.DataFrame:
    """
    작업자별 MAT/LBT/EOD 대기시간 계산.

    Args:
        journey_df: 전처리된 journey DataFrame (processor.py 출력)
            필수 컬럼: user_no, user_name, company_name, timestamp, locus_token
        date_str: 날짜 문자열 (옵션, 로깅용)

    Returns:
        DataFrame with columns:
        - user_no, user_name, company_name
        - mat_minutes: 출근 대기시간 (분)
        - lbt_minutes: 중식 대기시간 (분)
        - eod_minutes: 퇴근 대기시간 (분)
        - total_wait_minutes: MAT + LBT + EOD 합계
        - work_minutes: 총 근무시간 (첫 기록 ~ 마지막 기록)
        - transit_ratio: total_wait / work_minutes (%)
    """
    if journey_df.empty:
        logger.warning("calc_worker_transit: empty journey_df")
        return pd.DataFrame()

    # 필수 컬럼 검증
    required = ["user_no", "user_name", "company_name", "timestamp", "locus_token"]
    missing = [c for c in required if c not in journey_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cfg_values = _get_config_values()

    # timestamp 타입 확인 및 변환
    df = journey_df.copy()

    # NOTE: has_tward 필터 제거 — TWardData에 BLE 기록이 있는 모든 작업자 대상
    # AccessLog의 twardid 유무와 관계없이, TWardData에 위치 데이터가 있으면
    # T-Ward를 착용한 것이므로 transit 분석 대상에 포함
    before_count = df["user_no"].nunique()
    logger.info("calc_worker_transit: %d workers (all BLE-tracked)", before_count)

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 시간순 정렬
    df = df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    # 작업자별 그룹핑
    results = []
    for user_no, grp in df.groupby("user_no"):
        user_name = grp["user_name"].iloc[0]
        company_name = grp["company_name"].iloc[0]

        # 각 지표 계산 (is_estimated 튜플 언팩)
        mat, mat_est = _calc_mat_for_worker(grp)
        lbt_result = _calc_lbt_for_worker(grp, cfg_values)
        lbt     = lbt_result["total"] if lbt_result else None
        lbt_out = lbt_result["out"]   if lbt_result else None
        lbt_in  = lbt_result["in"]    if lbt_result else None
        lmt = _calc_lmt_for_worker(grp, cfg_values)
        eod, eod_est = _calc_eod_for_worker(grp, cfg_values)

        # 경로 구간별 소요시간 (4가지)
        route_mat = _calc_route_segments_for_worker(grp)
        route_lbt_out = _calc_lbt_out_segments_for_worker(grp, cfg_values)
        route_lbt_in = _calc_lbt_in_segments_for_worker(grp, cfg_values)
        route_eod = _calc_eod_segments_for_worker(grp, cfg_values)

        # 총 대기시간 (None은 0으로 처리)
        mat_val = mat if mat is not None else 0
        lbt_val = lbt if lbt is not None else 0
        eod_val = eod if eod is not None else 0
        total_wait = mat_val + lbt_val + eod_val

        # 근무시간 (첫 기록 ~ 마지막 기록)
        first_time = grp["timestamp"].min()
        last_time = grp["timestamp"].max()
        work_minutes = (last_time - first_time).total_seconds() / 60
        work_minutes = round(work_minutes, 1) if work_minutes > 0 else 0

        # Transit Ratio (%)
        transit_ratio = round(total_wait / work_minutes * 100, 2) if work_minutes > 0 else 0

        row = {
            "user_no": user_no,
            "user_name": user_name,
            "company_name": company_name,
            "mat_minutes": mat,
            "mat_is_estimated": mat_est,   # gap-filled 레코드 기반 추정 여부
            "lbt_minutes":     lbt,
            "lbt_out_minutes": lbt_out,    # LBT-Out 편도 (FAB → 타각기) — lbt_out + lbt_in = lbt_minutes
            "lbt_in_minutes":  lbt_in,     # LBT-In  편도 (타각기 → FAB) — 동일 집단 기준
            "lmt_minutes": lmt,            # 점심식사 전체 시간 (이동 포함)
            "eod_minutes": eod,
            "eod_is_estimated": eod_est,   # gap-filled 레코드 기반 추정 여부
            "total_wait_minutes": round(total_wait, 1),
            "work_minutes": work_minutes,
            "transit_ratio": transit_ratio,
            # 출근 (MAT) 경로 구간별 소요시간 + 미감지 gap
            "seg_gate_to_outdoor": route_mat["gate_to_outdoor"] if route_mat else None,
            "seg_gap_gate_outdoor": route_mat["gap_gate_outdoor"] if route_mat else 0.0,
            "seg_outdoor_to_hoist": route_mat["outdoor_to_hoist"] if route_mat else None,
            "seg_gap_outdoor_hoist": route_mat["gap_outdoor_hoist"] if route_mat else 0.0,
            "seg_hoist_to_fab": route_mat["hoist_to_fab"] if route_mat else None,
            "seg_gap_hoist_fab": route_mat["gap_hoist_fab"] if route_mat else 0.0,
            "seg_total": route_mat["gate_to_fab_total"] if route_mat else None,
            # 점심 외출 (LBT-Out) 경로 구간별 소요시간 + 미감지 gap
            "seg_lbt_out_fab_to_hoist": route_lbt_out["fab_to_hoist"] if route_lbt_out else None,
            "seg_gap_lbt_out_fab_hoist": route_lbt_out["gap_fab_hoist"] if route_lbt_out else 0.0,
            "seg_lbt_out_hoist_to_outdoor": route_lbt_out["hoist_to_outdoor"] if route_lbt_out else None,
            "seg_gap_lbt_out_hoist_outdoor": route_lbt_out["gap_hoist_outdoor"] if route_lbt_out else 0.0,
            "seg_lbt_out_outdoor_to_gate": route_lbt_out["outdoor_to_gate"] if route_lbt_out else None,
            "seg_gap_lbt_out_outdoor_gate": route_lbt_out["gap_outdoor_gate"] if route_lbt_out else 0.0,
            "seg_lbt_out_total": route_lbt_out["fab_to_gate_total"] if route_lbt_out else None,
            # 점심 복귀 (LBT-In) 경로 구간별 소요시간 + 미감지 gap
            "seg_lbt_in_gate_to_outdoor": route_lbt_in["gate_to_outdoor"] if route_lbt_in else None,
            "seg_gap_lbt_in_gate_outdoor": route_lbt_in["gap_gate_outdoor"] if route_lbt_in else 0.0,
            "seg_lbt_in_outdoor_to_hoist": route_lbt_in["outdoor_to_hoist"] if route_lbt_in else None,
            "seg_gap_lbt_in_outdoor_hoist": route_lbt_in["gap_outdoor_hoist"] if route_lbt_in else 0.0,
            "seg_lbt_in_hoist_to_fab": route_lbt_in["hoist_to_fab"] if route_lbt_in else None,
            "seg_gap_lbt_in_hoist_fab": route_lbt_in["gap_hoist_fab"] if route_lbt_in else 0.0,
            "seg_lbt_in_total": route_lbt_in["gate_to_fab_total"] if route_lbt_in else None,
            # 퇴근 (EOD) 경로 구간별 소요시간 + 미감지 gap
            "seg_eod_fab_to_hoist": route_eod["fab_to_hoist"] if route_eod else None,
            "seg_gap_eod_fab_hoist": route_eod["gap_fab_hoist"] if route_eod else 0.0,
            "seg_eod_hoist_to_outdoor": route_eod["hoist_to_outdoor"] if route_eod else None,
            "seg_gap_eod_hoist_outdoor": route_eod["gap_hoist_outdoor"] if route_eod else 0.0,
            "seg_eod_outdoor_to_gate": route_eod["outdoor_to_gate"] if route_eod else None,
            "seg_gap_eod_outdoor_gate": route_eod["gap_outdoor_gate"] if route_eod else 0.0,
            "seg_eod_total": route_eod["fab_to_gate_total"] if route_eod else None,
        }
        results.append(row)

    result_df = pd.DataFrame(results)

    if date_str:
        logger.info(
            "calc_worker_transit [%s]: %d workers, avg_mat=%.1f, avg_lbt=%.1f, avg_eod=%.1f",
            date_str,
            len(result_df),
            result_df["mat_minutes"].dropna().mean() if len(result_df) else 0,
            result_df["lbt_minutes"].dropna().mean() if len(result_df) else 0,
            result_df["eod_minutes"].dropna().mean() if len(result_df) else 0,
        )

    return result_df


def calc_bp_daily_transit(
    worker_transit_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    BP(업체)별 일별 대기시간 집계.

    Args:
        worker_transit_df: calc_worker_transit() 출력 DataFrame

    Returns:
        DataFrame with columns:
        - company_name (BP)
        - worker_count: 작업자 수
        - avg_mat, avg_lbt, avg_eod, avg_total_wait: 평균 대기시간
        - median_mat, median_lbt, median_eod: 중앙값 대기시간
        - max_mat, max_lbt, max_eod: 최대 대기시간
    """
    if worker_transit_df.empty:
        logger.warning("calc_bp_daily_transit: empty worker_transit_df")
        return pd.DataFrame()

    def _safe_mean(s: pd.Series) -> float:
        """NaN 무시한 평균 (전부 NaN이면 NaN 반환)."""
        valid = s.dropna()
        return round(valid.mean(), 1) if len(valid) > 0 else np.nan

    def _safe_median(s: pd.Series) -> float:
        """NaN 무시한 중앙값."""
        valid = s.dropna()
        return round(valid.median(), 1) if len(valid) > 0 else np.nan

    def _safe_max(s: pd.Series) -> float:
        """NaN 무시한 최대값."""
        valid = s.dropna()
        return round(valid.max(), 1) if len(valid) > 0 else np.nan

    # 업체별 집계
    agg = worker_transit_df.groupby("company_name").agg(
        worker_count=("user_no", "count"),
        # 평균
        avg_mat=("mat_minutes", _safe_mean),
        avg_lbt=("lbt_minutes", _safe_mean),
        avg_lmt=("lmt_minutes", _safe_mean),
        avg_eod=("eod_minutes", _safe_mean),
        avg_total_wait=("total_wait_minutes", "mean"),
        # 중앙값
        median_mat=("mat_minutes", _safe_median),
        median_lbt=("lbt_minutes", _safe_median),
        median_eod=("eod_minutes", _safe_median),
        # 최대값
        max_mat=("mat_minutes", _safe_max),
        max_lbt=("lbt_minutes", _safe_max),
        max_eod=("eod_minutes", _safe_max),
    ).reset_index()

    # 평균 총 대기시간 반올림
    agg["avg_total_wait"] = agg["avg_total_wait"].round(1)

    # 작업자 수 기준 정렬
    agg = agg.sort_values("worker_count", ascending=False).reset_index(drop=True)

    return agg


def calc_transit_summary(
    journey_df: pd.DataFrame,
    date_str: str | None = None,
    worker_df: pd.DataFrame | None = None,
    coverage_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 대기시간 분석 실행 (작업자별 + BP별).

    v2: Coverage threshold 기반 필터링 지원.
    worker_df와 coverage_threshold가 제공되면 BLE coverage 미달 작업자를 제외하고 집계.

    Args:
        journey_df: 전처리된 journey DataFrame
        date_str: 날짜 문자열 (옵션)
        worker_df: 작업자별 집계 DataFrame (coverage 계산용, 옵션)
        coverage_threshold: BLE coverage 최소 기준 0-1 (None이면 필터링 없음)

    Returns:
        (worker_transit_df, bp_transit_df)
        - worker_transit_df: coverage_ratio, coverage_pct, is_coverage_excluded 컬럼 포함
        - bp_transit_df: coverage 통과 작업자만 집계
    """
    # 작업자별 대기시간 계산
    worker_transit_df = calc_worker_transit(journey_df, date_str)

    # Coverage 필터링 (worker_df 제공 시)
    if worker_df is not None and not worker_df.empty and coverage_threshold is not None:
        try:
            from src.pipeline.journey_reconstructor import (
                reconstruct_all_journeys,
                filter_by_coverage,
            )
            _, coverage_df = reconstruct_all_journeys(
                journey_df, worker_df, coverage_threshold=coverage_threshold
            )
            worker_transit_df, cov_stats = filter_by_coverage(
                worker_transit_df, coverage_df, coverage_threshold
            )
            logger.info(
                "Coverage filter [%s]: %d/%d workers included (threshold=%.0f%%)",
                date_str or "?",
                cov_stats["n_included"],
                cov_stats["n_total"],
                cov_stats["threshold_pct"],
            )
            # 모든 작업자에 커버리지 정보 저장 (제외된 것 포함)
            cov_all = coverage_df[["user_no", "coverage_ratio", "coverage_pct", "coverage_label", "excluded"]].copy()
            cov_all = cov_all.rename(columns={"excluded": "is_coverage_excluded"})
            worker_transit_df = worker_transit_df.merge(
                cov_all[["user_no", "is_coverage_excluded"]],
                on="user_no", how="left"
            )
            worker_transit_df["is_coverage_excluded"] = (
                worker_transit_df["is_coverage_excluded"].fillna(False)
            )
        except Exception as e:
            logger.warning("Coverage 필터링 실패 (전체 사용): %s", e)
            if "coverage_ratio" not in worker_transit_df.columns:
                worker_transit_df["coverage_ratio"] = None
                worker_transit_df["coverage_pct"] = None
                worker_transit_df["coverage_label"] = None
                worker_transit_df["is_coverage_excluded"] = False
    else:
        # coverage 컬럼 추가 (없으면)
        if "coverage_ratio" not in worker_transit_df.columns:
            # worker_df에서 ble_coverage_pct 가져오기 시도
            if worker_df is not None and "ble_coverage_pct" in worker_df.columns:
                cov_info = worker_df[["user_no", "ble_coverage_pct"]].copy()
                cov_info = cov_info.rename(columns={"ble_coverage_pct": "coverage_pct"})
                cov_info["coverage_ratio"] = cov_info["coverage_pct"] / 100
                cov_info["coverage_label"] = pd.cut(
                    cov_info["coverage_ratio"],
                    bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
                    labels=["불충분(<30%)", "부분(30-50%)", "양호(50-70%)", "충분(>70%)"],
                ).astype(str)
                worker_transit_df = worker_transit_df.merge(
                    cov_info, on="user_no", how="left"
                )
            else:
                worker_transit_df["coverage_ratio"] = None
                worker_transit_df["coverage_pct"] = None
                worker_transit_df["coverage_label"] = None
            worker_transit_df["is_coverage_excluded"] = False

    # BP(업체)별 집계 (coverage 통과 작업자만)
    transit_for_bp = worker_transit_df[
        ~worker_transit_df.get("is_coverage_excluded", pd.Series(False, index=worker_transit_df.index))
    ] if "is_coverage_excluded" in worker_transit_df.columns else worker_transit_df
    bp_transit_df = calc_bp_daily_transit(transit_for_bp)

    return worker_transit_df, bp_transit_df


# ─── 추가 분석 함수들 ──────────────────────────────────────────────────

def calc_transit_by_hour(
    journey_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    시간대별 호이스트/타각기 혼잡도 분석.

    Returns:
        DataFrame with columns:
        - hour: 시간대 (0~23)
        - transit_records: 호이스트 기록 수
        - gate_records: 타각기 기록 수
        - unique_workers: 고유 작업자 수
    """
    if journey_df.empty:
        return pd.DataFrame()

    df = journey_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour

    # 호이스트 기록
    transit_df = df[df["locus_token"].isin(TRANSIT_TOKENS)]
    transit_agg = transit_df.groupby("hour").agg(
        transit_records=("user_no", "count"),
        transit_workers=("user_no", "nunique"),
    ).reset_index()

    # 타각기 기록
    gate_df = df[df["locus_token"].isin(GATE_TOKENS)]
    gate_agg = gate_df.groupby("hour").agg(
        gate_records=("user_no", "count"),
        gate_workers=("user_no", "nunique"),
    ).reset_index()

    # 병합
    hours = pd.DataFrame({"hour": range(24)})
    result = hours.merge(transit_agg, on="hour", how="left")
    result = result.merge(gate_agg, on="hour", how="left")
    result = result.fillna(0).astype({
        "transit_records": int,
        "transit_workers": int,
        "gate_records": int,
        "gate_workers": int,
    })

    return result


def calc_floor_transit_matrix(
    journey_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    층간 이동 매트릭스 (From-To Matrix).

    Returns:
        DataFrame (pivot) - index: from_floor, columns: to_floor, values: count
    """
    if journey_df.empty:
        return pd.DataFrame()

    # locus_token에서 층 정보 추출 (work_zone만 해당)
    # M15X는 locus_id로 층 구분: L-M15X-001 (5F), 002 (6F), 003 (7F), 004 (RF)
    df = journey_df.copy()
    df = df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    # 이전 locus_token
    df["prev_token"] = df.groupby("user_no")["locus_token"].shift(1)

    # 작업구역 간 이동만 필터 (work_zone -> work_zone)
    work_transitions = df[
        (df["locus_token"].isin(WORK_ZONE_TOKENS)) &
        (df["prev_token"].isin(WORK_ZONE_TOKENS)) &
        (df["locus_token"] != df["prev_token"])
    ]

    if work_transitions.empty:
        return pd.DataFrame()

    # locus_id에서 층 추출 (L-M15X-001 -> 5F)
    floor_map = {
        "L-M15X-001": "5F",
        "L-M15X-002": "6F",
        "L-M15X-003": "7F",
        "L-M15X-004": "RF",
    }

    # prev_locus_id가 없으면 prev_locus 컬럼 사용
    if "prev_locus" not in work_transitions.columns:
        work_transitions["prev_locus"] = work_transitions.groupby("user_no")["locus_id"].shift(1)

    transitions = work_transitions.copy()
    transitions["from_floor"] = transitions["prev_locus"].map(floor_map)
    transitions["to_floor"] = transitions["locus_id"].map(floor_map)

    # NaN 제거
    transitions = transitions.dropna(subset=["from_floor", "to_floor"])

    if transitions.empty:
        return pd.DataFrame()

    # 피벗 테이블
    matrix = transitions.groupby(["from_floor", "to_floor"]).size().unstack(fill_value=0)

    return matrix


def get_transit_report(
    journey_df: pd.DataFrame,
    date_str: str,
) -> dict:
    """
    대기시간 분석 전체 리포트 생성.

    Returns:
        dict with keys:
        - date: 날짜
        - worker_transit: 작업자별 DataFrame
        - bp_transit: 업체별 DataFrame
        - hourly_congestion: 시간대별 혼잡도 DataFrame
        - summary: 요약 통계
    """
    worker_df, bp_df = calc_transit_summary(journey_df, date_str)
    hourly_df = calc_transit_by_hour(journey_df)

    # 요약 통계
    summary = {
        "total_workers": len(worker_df),
        "workers_with_mat": worker_df["mat_minutes"].notna().sum(),
        "workers_with_lbt": worker_df["lbt_minutes"].notna().sum(),
        "workers_with_eod": worker_df["eod_minutes"].notna().sum(),
        "avg_mat": round(worker_df["mat_minutes"].dropna().mean(), 1) if worker_df["mat_minutes"].notna().any() else None,
        "avg_lbt": round(worker_df["lbt_minutes"].dropna().mean(), 1) if worker_df["lbt_minutes"].notna().any() else None,
        "avg_eod": round(worker_df["eod_minutes"].dropna().mean(), 1) if worker_df["eod_minutes"].notna().any() else None,
        "avg_transit_ratio": round(worker_df["transit_ratio"].mean(), 2) if len(worker_df) > 0 else None,
        "bp_count": len(bp_df),
    }

    return {
        "date": date_str,
        "worker_transit": worker_df,
        "bp_transit": bp_df,
        "hourly_congestion": hourly_df,
        "summary": summary,
    }
