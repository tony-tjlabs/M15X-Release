"""DeepCon-M15X Metrics — EWI, CRE, SII (v2.0)
========================================
DeepCon-M15X 데이터 구조에 맞춘 생산성·안전 지표 계산 모듈.
DeepCon_SOIF v6.9 알고리즘을 DeepCon-M15X locus 기반으로 이식.

v2.0 (2026-04-11): EWI v2 — Space-Aware Effective Work Intensity
  - 공간 가중치(SPACE_WEIGHTS) × 활동 가중치(ACTIVITY_WEIGHTS) 교차 분류
  - 5분 슬라이딩 윈도우 기반 이동 중(in-transit) 감지
  - ewi(v2 기본값) + ewi_v1(하위호환) 동시 출력
  - 새 분류 컬럼: intense_work_min, light_work_min, idle_at_work_min,
    rest_min, transit_min, in_transit_min, unknown_min
v1.2 (2026-04-06): v2 Locus (GW-XXX, 213개) 호환
  - 토큰 기반 분류 → locus CSV 메타데이터 기반 분류로 전환
  - v1 하드코딩 토큰은 fallback으로 유지
v1.1 (2026-03-28): enriched locus 데이터 통합
  - hazard_level/hazard_grade 기반 static_risk 계산
  - locus_dict 파라미터 추가 (하위 호환 유지)

DeepCon_SOIF vs DeepCon-M15X 주요 차이:
  DeepCon_SOIF                  DeepCon-M15X
  ─────────────────────────── ──────────────────────────────
  PLACE_TYPE / SPACE_FUNCTION  locus_token (공간 역할 토큰)
  accel_state (moving/static)  active_ratio (BLE 신호 활성도)
  detect_work_shift()          AccessLog in_datetime/out_datetime
  IS_TRANSIT_EPISODE           locus_token in TRANSIT_TOKENS
  corrected_place              locus_id (spot_name → locus_id 매핑)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── 활성도 임계값 ──────────────────────────────────────────────────
HIGH_ACTIVE_THRESHOLD = 0.90   # 고활성: 수신신호 대부분 활성(10초 발신) → 실질 작업 중
LOW_ACTIVE_THRESHOLD  = 0.40   # 저활성 하한: 이 미만은 대기로 분류 (DeepCon 통일)

# ─── EWI 가중치 ──────────────────────────────────────────────────────
HIGH_WORK_WEIGHT = 1.0    # 고활성 작업
LOW_WORK_WEIGHT  = 0.5    # 저활성 작업
STANDBY_WEIGHT   = 0.2    # 대기 (크레인 대기·감독 등 실질 작업 포함)

# 음영지역 비율이 이 이상이면 recorded_min을 분모로 대체
EWI_GAP_RELIABLE_THRESHOLD = 0.20

# ─── EWI v2: Space-Aware Classification ─────────────────────────────
# 공간 가중치: 생산적 작업 기여도 (0.0 = 비생산, 1.0 = 완전 생산)
SPACE_WEIGHTS: dict[str, float] = {
    # PRODUCTIVE (실제 작업 공간)
    "work_zone":    1.0,
    "outdoor_work": 0.0,   # 이동 경로로 분류 — EWI 기여 없음
    # REST (휴식 공간 — 활성도 무관하게 비생산)
    "breakroom":    0.0,
    "smoking_area": 0.0,
    "dining_hall":  0.0,
    "restroom":     0.0,
    "parking_lot":  0.0,
    # TRANSPORT (대기/이동 — 작업 아님)
    "transit":      0.0,
    # GATE (출입 절차)
    "timeclock":    0.0,
    "main_gate":    0.0,
    "sub_gate":     0.0,
    # ADMIN (사무 작업 — 부분 인정)
    "office":       0.3,
    "facility":     0.3,
    # UNKNOWN / unmapped — 보수적 중간값
    "unknown":      0.3,
    "unmapped":     0.3,
    # SHADOW ZONE — BLE 30분+ 미감지 (EWI 기여 없음)
    "shadow_zone":  0.0,
}

# 활동 가중치 (v1과 동일 임계값 기반)
ACTIVITY_WEIGHTS = {
    "high":    1.0,   # active_ratio >= 0.60
    "low":     0.5,   # 0.15 <= active_ratio < 0.60
    "standby": 0.2,   # active_ratio < 0.15
}

# In-transit 감지 (이동 중 판정)
TRANSIT_WINDOW_SIZE = 5        # 5분 슬라이딩 윈도우
TRANSIT_THRESHOLD = 3          # 윈도우 내 >= 3회 locus 변경 = 이동 중
IN_TRANSIT_WEIGHT = 0.05       # 이동 중 기여도 (거의 0이지만 완전 0은 아님)

# PRODUCTIVE 공간 토큰 집합 (v2 분류용)
_PRODUCTIVE_TOKENS = {"work_zone"}

# ─── 공간 토큰 분류 ──────────────────────────────────────────────────
# v1 fallback 토큰 (LOCUS_VERSION="v1" 또는 locus CSV 로드 실패 시)
_V1_WORK_TOKENS = {
    "work_zone", "mechanical_room",
    "confined_space", "high_voltage", "transit",
}
_V1_TRANSIT_TOKENS = {"timeclock", "main_gate", "sub_gate", "outdoor_work", "transit"}
_V1_REST_TOKENS = {"breakroom", "smoking_area", "dining_hall", "restroom", "parking_lot"}
_V1_ADMIN_TOKENS = {"office", "facility"}

# v2 호환: locus CSV에서 토큰 분류 집합을 동적으로 로드
_cached_token_sets: dict | None = None


def _get_token_sets() -> dict[str, set[str]]:
    """
    LOCUS_VERSION에 따라 토큰 분류 집합 반환.

    v1: 하드코딩된 영문 토큰 집합
    v2: locus_v2.csv의 locus_type/function/locus_name 기반 동적 분류

    Returns:
        {"work": set, "transit": set, "rest": set, "admin": set}
    """
    global _cached_token_sets
    if _cached_token_sets is not None:
        return _cached_token_sets

    try:
        import config as cfg
    except ImportError:
        _cached_token_sets = _v1_token_sets()
        return _cached_token_sets

    if cfg.LOCUS_VERSION != "v2":
        _cached_token_sets = _v1_token_sets()
        return _cached_token_sets

    # v2: locus CSV 기반 동적 분류
    try:
        paths = cfg.get_sector_paths(cfg.SECTOR_ID)
        csv_path = paths.get("locus_v2_csv")
        if not csv_path or not csv_path.exists():
            logger.warning("locus_v2.csv 없음, v1 fallback")
            _cached_token_sets = _v1_token_sets()
            return _cached_token_sets

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        name = df["locus_name"].fillna("").str.lower()
        ltype = df.get("locus_type", pd.Series(dtype=str)).fillna("").str.upper()
        func = df.get("function", pd.Series(dtype=str)).fillna("").str.upper()

        # Gate/Transit: GATE 타입 또는 ACCESS 기능
        gate_mask = (ltype == "GATE") | (func == "ACCESS")
        transit_tokens = set(df.loc[gate_mask, "locus_name"].dropna())

        # 휴식 공간: locus_name에 키워드 포함
        rest_keywords = ["휴게", "흡연", "식당", "화장실", "restroom", "breakroom",
                         "smoking", "dining", "휴계"]
        rest_mask = pd.Series(False, index=df.index)
        for kw in rest_keywords:
            rest_mask = rest_mask | name.str.contains(kw)
        rest_tokens = set(df.loc[rest_mask, "locus_name"].dropna())

        # 관리/시설: locus_name에 키워드 포함
        admin_keywords = ["사무실", "office", "관리"]
        admin_mask = pd.Series(False, index=df.index)
        for kw in admin_keywords:
            admin_mask = admin_mask | name.str.contains(kw)
        admin_tokens = set(df.loc[admin_mask, "locus_name"].dropna())

        # 작업 공간: WORK_AREA/WORK/VERTICAL 중 휴식/관리 아닌 것
        work_mask = ((ltype == "WORK_AREA") | (func == "WORK") | (ltype == "VERTICAL")) & ~rest_mask & ~admin_mask & ~gate_mask
        work_tokens = set(df.loc[work_mask, "locus_name"].dropna())

        _cached_token_sets = {
            "work": work_tokens,
            "transit": transit_tokens,
            "rest": rest_tokens,
            "admin": admin_tokens,
        }
    except Exception as e:
        logger.warning("v2 토큰 분류 로드 실패: %s — v1 fallback", e)
        _cached_token_sets = _v1_token_sets()

    return _cached_token_sets


def _v1_token_sets() -> dict[str, set[str]]:
    return {
        "work": _V1_WORK_TOKENS,
        "transit": _V1_TRANSIT_TOKENS,
        "rest": _V1_REST_TOKENS,
        "admin": _V1_ADMIN_TOKENS,
    }


# 동적 접근을 위한 래퍼 (기존 코드에서 WORK_TOKENS 등을 직접 참조하므로)
def _init_module_token_sets():
    ts = _get_token_sets()
    return ts["work"], ts["transit"], ts["rest"], ts["admin"]

WORK_TOKENS, TRANSIT_TOKENS, REST_TOKENS, ADMIN_TOKENS = _init_module_token_sets()
NON_WORK_TOKENS = TRANSIT_TOKENS | REST_TOKENS | ADMIN_TOKENS

# ─── 공간별 정적 위험도 (Static Risk) ────────────────────────────────
# v1 영문 토큰 기반 (v2에서는 locus_dict 경로 우선 사용)
STATIC_RISK_BY_TOKEN: dict[str, float] = {
    # 최고 위험 (hazard_5)
    "confined_space": 2.0,
    "high_voltage":   2.0,
    # 고위험 (hazard_4)
    "mechanical_room": 1.8,
    "transit":         1.5,   # 호이스트·클라이머 (수직 이동 설비)
    # 중위험 (hazard_3)
    "outdoor_work": 1.3,
    "work_zone":    1.2,
    # 저위험 / 게이트
    "timeclock":  0.5,
    "main_gate":  0.3,
    "sub_gate":   0.3,
    # 비작업
    "breakroom":    0.2,
    "smoking_area": 0.2,
    "dining_hall":  0.2,
    "restroom":     0.2,
    "parking_lot":  0.2,
    "office":       0.3,
    "facility":     0.3,
    # ★ unmapped/unknown — 보수적 중간값
    "unknown":      0.5,
    "unmapped":     0.5,
}
_STATIC_RISK_MIN = 0.2
_STATIC_RISK_MAX = 2.0

# ─── CRE 가중치 ──────────────────────────────────────────────────────
CRE_W_PERSONAL = 0.45
CRE_W_STATIC   = 0.40
CRE_W_DYNAMIC  = 0.15

_DENSITY_SCALE  = 30.0   # 정규화 기준 (30명/슬롯 → dynamic_norm = 1.0)
_DENSITY_CAP    = 1.0    # 상한

# ─── Enriched Locus 기반 Static Risk 헬퍼 (2026-03-28 추가) ──────────

def _calc_locus_static_risk(
    locus_id: str,
    token: str,
    locus_dict: dict | None,
) -> float:
    """
    Locus별 static_risk 계산.

    우선순위:
    1. locus_dict에 hazard_level/hazard_grade 있으면 사용
    2. Fallback: token 기반 STATIC_RISK_BY_TOKEN

    Args:
        locus_id: Locus ID
        token: locus_token
        locus_dict: enriched locus 정보 (없으면 None)

    Returns:
        float: 0.2 ~ 2.0 범위의 static_risk
    """
    # Fallback: token 기반
    if locus_dict is None or not locus_id:
        return STATIC_RISK_BY_TOKEN.get(token, 1.0)

    info = locus_dict.get(locus_id, {})
    hazard_level = info.get("hazard_level")
    hazard_grade = info.get("hazard_grade")

    # hazard_level이 없으면 token fallback
    if not hazard_level:
        return STATIC_RISK_BY_TOKEN.get(token, 1.0)

    # hazard_level 문자열 정규화
    level_str = str(hazard_level).lower().strip()

    # hazard_grade 파싱
    try:
        grade = float(hazard_grade) if hazard_grade is not None else 2.0
    except (ValueError, TypeError):
        grade = 2.0

    # hazard_level 기반 기본값
    level_base = {
        "critical": 1.8,
        "high": 1.4,
        "medium": 1.0,
        "low": 0.6,
    }.get(level_str, 1.0)

    # grade 가중 (1~5 → x0.7~x1.1)
    grade_mult = 0.7 + (grade / 5) * 0.4

    return min(max(level_base * grade_mult, _STATIC_RISK_MIN), _STATIC_RISK_MAX)


# ─── EWI 계산 ────────────────────────────────────────────────────────

def calc_ewi_all_workers(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    locus_dict: dict | None = None,
) -> pd.DataFrame:
    """
    전체 작업자 EWI 계산 — v1 + v2 동시 출력.

    EWI v1 (ewi_v1): 기존 공간 무관 활성도 기반
      = (고활성×1.0 + 저활성×0.5 + 대기×0.2) / 근무시간(분)

    EWI v2 (ewi): Space-Aware 분류
      = SUM(space_weight × activity_weight) / 근무시간(분)
      - 이동 중(in_transit) 감지 시 기여도 0.05 적용
      - 비생산 공간(휴게/이동/출입) 기여도 0.0

    Args:
        journey_df : is_work_hour, locus_token, active_ratio, user_no,
                    locus_changed (v2 transit 감지용) 포함
        worker_df  : user_no, work_minutes 포함
        locus_dict : enriched locus 정보 (hazard_level/hazard_grade 기반 static_risk)
                    None이면 token 기반 fallback

    Returns:
        DataFrame with columns:
            user_no,
            ewi (v2), ewi_v1,
            intense_work_min, light_work_min, idle_at_work_min,
            rest_min, transit_min, in_transit_min, unknown_min,
            high_active_min, low_active_min, standby_min (v1 호환),
            recorded_work_min,
            ewi_reliable, ewi_calculable, gap_ratio, gap_min, static_risk
    """
    wdf = journey_df[journey_df["is_work_hour"]].copy()
    if wdf.empty:
        return _empty_ewi_df(worker_df)

    ratio = wdf["active_ratio"].fillna(0)
    token = wdf["locus_token"].fillna("work_zone")

    # ── v1 분류 (하위호환 유지) ──────────────────────────────────────
    is_non_work = token.isin(NON_WORK_TOKENS)
    is_transit_v1 = token.isin(TRANSIT_TOKENS)
    is_rest_v1 = token.isin(REST_TOKENS | ADMIN_TOKENS)
    is_work = ~is_non_work

    wdf["_v1_high"]    = is_work & (ratio >= HIGH_ACTIVE_THRESHOLD)
    wdf["_v1_low"]     = is_work & (ratio >= LOW_ACTIVE_THRESHOLD) & (ratio < HIGH_ACTIVE_THRESHOLD)
    wdf["_v1_standby"] = is_work & (ratio < LOW_ACTIVE_THRESHOLD)
    wdf["_v1_rest"]    = is_rest_v1
    wdf["_v1_transit"] = is_transit_v1

    # ── v2 분류: In-Transit 감지 (5분 롤링 윈도우) ────────────────────
    has_locus_changed = "locus_changed" in wdf.columns
    if has_locus_changed:
        wdf_sorted = wdf.sort_values(["user_no", "timestamp"])
        lc = wdf_sorted["locus_changed"].fillna(False).astype(int)
        rolling_trans = lc.groupby(wdf_sorted["user_no"]).rolling(
            window=TRANSIT_WINDOW_SIZE, min_periods=1
        ).sum().reset_index(level=0, drop=True)
        wdf["_is_in_transit"] = rolling_trans.reindex(wdf.index) >= TRANSIT_THRESHOLD
    else:
        # locus_changed 없으면 in-transit 감지 불가 → 전부 False
        logger.warning("locus_changed 컬럼 없음 — in-transit 감지 비활성")
        wdf["_is_in_transit"] = False

    # ── v2 분류: 공간 가중치 + 활동 가중치 ────────────────────────────
    wdf["_space_w"] = token.map(SPACE_WEIGHTS).fillna(0.3)

    wdf["_act_w"] = np.where(
        ratio >= HIGH_ACTIVE_THRESHOLD, ACTIVITY_WEIGHTS["high"],
        np.where(ratio >= LOW_ACTIVE_THRESHOLD, ACTIVITY_WEIGHTS["low"],
                 ACTIVITY_WEIGHTS["standby"])
    )

    # per-minute contribution: 이동 중이면 IN_TRANSIT_WEIGHT, 아니면 space × activity
    wdf["_v2_contrib"] = np.where(
        wdf["_is_in_transit"],
        IN_TRANSIT_WEIGHT,
        wdf["_space_w"] * wdf["_act_w"]
    )

    # ── v2 상세 분류 컬럼 (이동 중이 아닌 경우만 각 카테고리에 배정) ──
    not_transit = ~wdf["_is_in_transit"]
    is_productive = token.isin(_PRODUCTIVE_TOKENS)

    wdf["_intense_work"] = is_productive & (ratio >= HIGH_ACTIVE_THRESHOLD) & not_transit
    wdf["_light_work"]   = is_productive & (ratio >= LOW_ACTIVE_THRESHOLD) & (ratio < HIGH_ACTIVE_THRESHOLD) & not_transit
    wdf["_idle_at_work"] = is_productive & (ratio < LOW_ACTIVE_THRESHOLD) & not_transit
    wdf["_rest"]         = is_rest_v1 & not_transit
    wdf["_transport"]    = is_transit_v1 & not_transit
    wdf["_unknown"]      = token.isin({"unknown", "unmapped"}) & not_transit
    # shadow_zone: BLE 30분+ 미감지 — unknown과 별도 집계
    wdf["_shadow"]       = token.isin({"shadow_zone"}) & not_transit

    # ── static_risk: enriched locus 기반 또는 token 기반 ──────────────
    if locus_dict and "locus_id" in wdf.columns:
        wdf["_static_risk"] = wdf.apply(
            lambda r: _calc_locus_static_risk(
                r.get("locus_id", ""),
                r.get("locus_token", "unknown"),
                locus_dict,
            ),
            axis=1,
        )
    else:
        wdf["_static_risk"] = token.map(STATIC_RISK_BY_TOKEN).fillna(1.0)

    # ── 작업자별 집계 ────────────────────────────────────────────────
    agg = wdf.groupby("user_no").agg(
        # v1 호환 컬럼
        high_active_min   = ("_v1_high",        "sum"),
        low_active_min    = ("_v1_low",         "sum"),
        standby_min       = ("_v1_standby",     "sum"),
        # v2 상세 분류 컬럼
        intense_work_min  = ("_intense_work",   "sum"),
        light_work_min    = ("_light_work",     "sum"),
        idle_at_work_min  = ("_idle_at_work",   "sum"),
        rest_min          = ("_rest",           "sum"),
        transit_min       = ("_transport",      "sum"),
        in_transit_min    = ("_is_in_transit",  "sum"),
        unknown_min       = ("_unknown",        "sum"),
        shadow_min        = ("_shadow",         "sum"),
        # v2 EWI 분자
        _v2_sum           = ("_v2_contrib",     "sum"),
        # v1 rest/transit (하위호환)
        _v1_rest_min      = ("_v1_rest",        "sum"),
        _v1_transit_min   = ("_v1_transit",     "sum"),
        # 공통
        recorded_work_min = ("user_no",         "count"),
        mean_static_risk  = ("_static_risk",    "mean"),
        p90_static_risk   = ("_static_risk",    lambda x: float(np.percentile(x, 90))),
    ).reset_index()

    # ── work_minutes 병합 + 음영지역 처리 (v1/v2 공통) ────────────────
    agg = agg.merge(
        worker_df[["user_no", "work_minutes"]].dropna(subset=["work_minutes"]),
        on="user_no", how="left",
    )
    agg["work_minutes"] = agg["work_minutes"].fillna(agg["recorded_work_min"])

    agg["gap_min"] = (agg["work_minutes"] - agg["recorded_work_min"]).clip(lower=0)
    agg["gap_ratio"] = (
        agg["gap_min"] / agg["work_minutes"].replace(0, np.nan)
    ).fillna(1.0)
    agg["ewi_reliable"] = agg["gap_ratio"] <= EWI_GAP_RELIABLE_THRESHOLD
    agg["ewi_denom"] = np.where(
        agg["ewi_reliable"],
        agg["work_minutes"],
        agg["recorded_work_min"],
    )
    agg["ewi_calculable"] = agg["work_minutes"] > 0

    # ── EWI v1 계산 ──────────────────────────────────────────────────
    agg["_v1_num"] = (
        agg["high_active_min"] * HIGH_WORK_WEIGHT
        + agg["low_active_min"] * LOW_WORK_WEIGHT
        + agg["standby_min"]   * STANDBY_WEIGHT
    )
    agg["ewi_v1"] = (agg["_v1_num"] / agg["ewi_denom"].replace(0, np.nan)).fillna(0).clip(0, 1).round(4)
    agg.loc[~agg["ewi_calculable"], "ewi_v1"] = 0.0

    # ── EWI v2 계산 (기본값) ─────────────────────────────────────────
    agg["ewi"] = (agg["_v2_sum"] / agg["ewi_denom"].replace(0, np.nan)).fillna(0).clip(0, 1).round(4)
    agg.loc[~agg["ewi_calculable"], "ewi"] = 0.0
    agg.loc[~agg["ewi_calculable"], "ewi_reliable"] = False

    # Static risk: 평균 70% + 90th percentile 30% (위험 공간 희석 방지)
    agg["static_risk"] = (0.7 * agg["mean_static_risk"] + 0.3 * agg["p90_static_risk"]).fillna(1.0)

    # v1 rest_min/transit_min 덮어쓰기 방지: v2의 rest_min/transit_min은 이미 집계됨
    # _v1_rest_min/_v1_transit_min은 내부용 — 필요 시 참조 가능

    return agg[[
        "user_no",
        # v2 기본
        "ewi", "ewi_v1",
        "intense_work_min", "light_work_min", "idle_at_work_min",
        "rest_min", "transit_min", "in_transit_min", "unknown_min",
        "shadow_min",   # BLE 30분+ 미감지 (Phase 2 보간 불가 구간)
        # v1 호환
        "high_active_min", "low_active_min", "standby_min",
        # 공통
        "recorded_work_min",
        "ewi_reliable", "ewi_calculable", "gap_ratio", "gap_min", "static_risk",
    ]]


def _empty_ewi_df(worker_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "user_no",
        "ewi", "ewi_v1",
        "intense_work_min", "light_work_min", "idle_at_work_min",
        "rest_min", "transit_min", "in_transit_min", "unknown_min",
        "shadow_min",
        "high_active_min", "low_active_min", "standby_min",
        "recorded_work_min",
        "ewi_reliable", "ewi_calculable", "gap_ratio", "gap_min", "static_risk",
    ]
    df = pd.DataFrame(columns=cols)
    df["user_no"] = worker_df["user_no"]
    return df.fillna(0)


# ─── CRE 계산 ────────────────────────────────────────────────────────

def _calc_fatigue_scores(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    작업자별 피로도 점수 계산 (연속 활성 작업 기반).

    피로도 = 최대 연속 활성 시간(분) × 0.7 + 평균 연속 활성 시간(분) × 0.3
    → 연속 4시간+ 작업 시 fatigue_score ≥ 0.5 수준으로 설계

    Returns: DataFrame(user_no, fatigue_score, max_active_streak, avg_active_streak)
    """
    wdf = journey_df[journey_df["is_work_hour"]].copy()
    if wdf.empty:
        return pd.DataFrame(columns=["user_no", "fatigue_score", "max_active_streak", "avg_active_streak"])

    wdf = wdf.sort_values(["user_no", "timestamp"]).reset_index(drop=True)
    wdf["_is_active"] = (wdf["active_ratio"].fillna(0) >= 0.30)

    # 연속 run 감지 (user 경계 또는 활성 상태 변화 시 새 run)
    prev_user    = wdf["user_no"].shift()
    prev_active  = wdf["_is_active"].shift()
    run_break    = (wdf["user_no"] != prev_user) | (wdf["_is_active"] != prev_active)
    wdf["_run_id"] = run_break.cumsum()

    run_info = wdf.groupby(["user_no", "_run_id"]).agg(
        run_len        = ("_run_id",    "count"),
        is_active_run  = ("_is_active", "first"),
    ).reset_index()

    active_runs = run_info[run_info["is_active_run"]]
    if active_runs.empty:
        return pd.DataFrame({
            "user_no":           wdf["user_no"].unique(),
            "fatigue_score":     0.0,
            "max_active_streak": 0,
            "avg_active_streak": 0.0,
        })

    streak_agg = active_runs.groupby("user_no").agg(
        max_active_streak = ("run_len", "max"),
        avg_active_streak = ("run_len", "mean"),
    ).reset_index()

    # 피로도 정규화: 4시간(240분) 연속 = fatigue_score 1.0 기준
    _FATIGUE_NORM = 240.0
    streak_agg["fatigue_score"] = (
        (0.7 * streak_agg["max_active_streak"] + 0.3 * streak_agg["avg_active_streak"])
        / _FATIGUE_NORM
    ).clip(0, 1).round(4)

    return streak_agg[["user_no", "fatigue_score", "max_active_streak", "avg_active_streak"]]


def _calc_dynamic_pressure(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    작업자별 동적 밀집 압력 계산.

    동적 압력 = 작업 시간 중 해당 작업자가 머문 locus의 평균 동시 작업자 수 / DENSITY_SCALE

    Returns: DataFrame(user_no, dynamic_norm)
    """
    wdf = journey_df[journey_df["is_work_hour"] & journey_df["locus_token"].isin(WORK_TOKENS)].copy()
    if wdf.empty or "locus_id" not in wdf.columns:
        return pd.DataFrame(columns=["user_no", "dynamic_norm"])

    # ★ Performance: transform → merge 제거 (메모리 + 속도 개선)
    wdf["_cnt"] = wdf.groupby(["timestamp", "locus_id"])["user_no"].transform("nunique")

    worker_density = (
        wdf.groupby("user_no")["_cnt"]
        .mean()
        .reset_index()
        .rename(columns={"_cnt": "_mean_density"})
    )
    worker_density["dynamic_norm"] = (
        worker_density["_mean_density"] / _DENSITY_SCALE
    ).clip(0, _DENSITY_CAP).round(4)

    return worker_density[["user_no", "dynamic_norm"]]


def calc_cre_all_workers(
    journey_df: pd.DataFrame,
    ewi_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    전체 작업자 CRE (Combined Risk Exposure) 계산.

    CRE = 0.45 × personal_norm + 0.40 × static_norm + 0.15 × dynamic_norm

    Personal  = 피로도 × 0.5 + 단독작업 근사 × 0.5
    Static    = locus_token 기반 고유 공간 위험도
    Dynamic   = 해당 작업 구역 동시 밀집 수준

    Args:
        journey_df : is_work_hour, locus_token, active_ratio, user_no, timestamp 포함
        ewi_df     : calc_ewi_all_workers() 결과 (static_risk 포함)

    Returns:
        DataFrame(user_no, cre, personal_norm, static_norm, dynamic_norm,
                  fatigue_score, max_active_streak)
    """
    # ── 피로도 ────────────────────────────────────────────────────────
    fatigue_df = _calc_fatigue_scores(journey_df)

    # ── 단독 작업 근사 ────────────────────────────────────────────────
    # 고위험 구역에서 작업자 혼자인 시간 비율로 근사
    # M15X 고위험 구역:
    #   transit      = 호이스트 (수직이동 중 단독 탑승)
    #   outdoor_work = 공사현장 (외부 이동 중 단독 체류)
    #   confined_space, high_voltage, mechanical_room = 범용 (M15X에 없으나 하위호환 유지)
    _HIGH_RISK_LONE_TOKENS = {
        "confined_space", "high_voltage", "mechanical_room",
        "transit", "outdoor_work",  # M15X 현장 실정 반영
    }
    wdf_high = journey_df[
        journey_df["is_work_hour"]
        & journey_df["locus_token"].isin(_HIGH_RISK_LONE_TOKENS)
    ].copy()
    if not wdf_high.empty and "locus_id" in wdf_high.columns:
        coloc_hi = (
            wdf_high.groupby(["timestamp", "locus_id"])["user_no"]
            .nunique()
            .reset_index(name="_cnt")
        )
        merged_hi = wdf_high.merge(coloc_hi, on=["timestamp", "locus_id"], how="left")
        merged_hi["_alone"] = (merged_hi["_cnt"].fillna(1) <= 1).astype(int)
        alone_df = (
            merged_hi.groupby("user_no")["_alone"]
            .mean()
            .reset_index()
            .rename(columns={"_alone": "alone_ratio"})
        )
    else:
        alone_df = pd.DataFrame(columns=["user_no", "alone_ratio"])

    # ── 동적 압력 ─────────────────────────────────────────────────────
    dynamic_df = _calc_dynamic_pressure(journey_df)

    # ── 합산 ──────────────────────────────────────────────────────────
    cre_df = ewi_df[["user_no", "static_risk"]].copy()
    cre_df = cre_df.merge(fatigue_df, on="user_no", how="left")
    cre_df = cre_df.merge(alone_df,   on="user_no", how="left")
    cre_df = cre_df.merge(dynamic_df, on="user_no", how="left")

    cre_df["fatigue_score"] = cre_df["fatigue_score"].fillna(0)
    cre_df["alone_ratio"]   = cre_df["alone_ratio"].fillna(0)
    cre_df["dynamic_norm"]  = cre_df["dynamic_norm"].fillna(0)
    cre_df["max_active_streak"] = cre_df.get("max_active_streak", pd.Series(0, index=cre_df.index)).fillna(0)

    # Personal risk (0~1)
    cre_df["personal_norm"] = (
        0.5 * cre_df["fatigue_score"] + 0.5 * cre_df["alone_ratio"]
    ).clip(0, 1)

    # Static risk → normalize to 0~1
    cre_df["static_norm"] = (
        (cre_df["static_risk"] - _STATIC_RISK_MIN)
        / (_STATIC_RISK_MAX - _STATIC_RISK_MIN)
    ).clip(0, 1)

    # CRE
    cre_df["cre"] = (
        CRE_W_PERSONAL * cre_df["personal_norm"]
        + CRE_W_STATIC  * cre_df["static_norm"]
        + CRE_W_DYNAMIC * cre_df["dynamic_norm"]
    ).clip(0, 1).round(4)

    return cre_df[[
        "user_no", "cre", "personal_norm", "static_norm", "dynamic_norm",
        "fatigue_score", "max_active_streak", "alone_ratio",
    ]]


# ─── 전체 지표 통합 ───────────────────────────────────────────────────

def add_metrics_to_worker(
    journey_df: pd.DataFrame,
    worker_df:  pd.DataFrame,
    locus_dict: dict | None = None,
) -> pd.DataFrame:
    """
    worker_df에 EWI / CRE / SII 컬럼을 추가하여 반환.

    Args:
        journey_df: Journey 데이터
        worker_df: 작업자 데이터
        locus_dict: enriched locus 정보 (hazard_level/hazard_grade 기반 static_risk)
                   None이면 token 기반 fallback (하위 호환)

    신규 컬럼 (v2):
        ewi (v2 기본), ewi_v1 (하위호환),
        intense_work_min, light_work_min, idle_at_work_min,
        rest_min, transit_min, in_transit_min, unknown_min,
        high_active_min, low_active_min, standby_min (v1 호환),
        recorded_work_min, ewi_reliable, gap_ratio, gap_min,
        cre, personal_norm, static_norm, dynamic_norm,
        fatigue_score, max_active_streak, alone_ratio,
        sii  (= ewi × static_norm, 고강도+고위험 작업자 탐지)
    """
    ewi_df = calc_ewi_all_workers(journey_df, worker_df, locus_dict)
    cre_df = calc_cre_all_workers(journey_df, ewi_df)

    result = worker_df.merge(ewi_df, on="user_no", how="left")
    result = result.merge(
        cre_df.drop(columns=["static_norm"], errors="ignore"),
        on="user_no", how="left",
    )

    # SII: EWI × static_norm (고강도 작업 + 위험 공간 동시 해당 작업자 탐지)
    static_norm_col = ewi_df.merge(
        cre_df[["user_no", "static_norm"]], on="user_no", how="left"
    )
    result = result.merge(
        static_norm_col[["user_no", "static_norm"]], on="user_no", how="left"
    )
    result["sii"] = (result["ewi"] * result["static_norm"]).clip(0, 1).round(4)

    # 기본값 채우기
    for col in ["ewi", "ewi_v1", "cre", "sii", "fatigue_score", "gap_ratio"]:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    # ★ BLE 커버리지 등급 — 데이터 품질 투명성
    # gap_ratio: 0=완벽 커버리지, 1=완전 음영
    gr = result["gap_ratio"].fillna(1.0)
    result["ble_coverage"] = np.where(
        gr <= 0.2, "정상",
        np.where(gr <= 0.5, "부분음영",
                 np.where(gr <= 0.8, "음영", "미측정"))
    )
    result["ble_coverage_pct"] = ((1 - gr) * 100).clip(0, 100).round(1)

    # ★ 시간 분류 집계 파생 컬럼 (Time Breakdown)
    #   work_dwell_min    : 작업 공간 체류 = intense + light + idle_at_work
    #   transit_dwell_min : 이동 = transit_min + in_transit_min
    #   rest_dwell_min    : 휴식 = rest_min
    work_cols = [c for c in ["intense_work_min", "light_work_min", "idle_at_work_min"] if c in result.columns]
    result["work_dwell_min"] = result[work_cols].sum(axis=1) if work_cols else 0

    transit_cols = [c for c in ["transit_min", "in_transit_min"] if c in result.columns]
    result["transit_dwell_min"] = result[transit_cols].sum(axis=1) if transit_cols else 0

    result["rest_dwell_min"] = result["rest_min"] if "rest_min" in result.columns else 0

    return result


# ─── 지표 등급 판정 ───────────────────────────────────────────────────

def ewi_grade(v: float) -> str:
    if v >= 0.6: return "고강도"
    if v >= 0.2: return "보통"
    return "저강도"

def cre_grade(v: float) -> str:
    if v >= 0.6: return "고위험"
    if v >= 0.3: return "주의"
    return "정상"

def sii_grade(v: float) -> str:
    if v >= 0.5: return "집중관리"
    if v >= 0.25: return "주의"
    return "정상"


# ─── Gap Analyzer 연동 지표 ───────────────────────────────────────────────────

def enrich_worker_with_gap_stats(
    worker_df: pd.DataFrame,
    filled_journey_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    gap_analyzer.analyze_gaps() 결과로 생성된 filled_journey_df에서
    작업자별 활성 레벨 집계 지표를 계산하여 worker_df에 병합한다.

    EWI/CRE는 원본 레코드 기반 유지. 이 함수는 추가 지표만 병합.

    추가 컬럼:
        ga_high_active_min  : gap_analyzer 기준 고활성 (ar≥0.75, sc≥3) 시간
        ga_active_min       : gap_analyzer 기준 활성 (ar≥0.5, sc≥2) 시간
        ga_inactive_min     : 비활성 (ar≥0.2) 시간
        ga_deep_inactive_min: 완전 정지 (ar<0.2) 시간
        ga_estimated_min    : GAP 채워진 (음영) 시간
        ga_gap_ratio_pct    : 음영 비율 (%)
        ga_dominant_activity: 지배적 활성 레벨
        ga_low_conf_min     : low_confidence 원본 레코드 수
    """
    if filled_journey_df.empty or "activity_level" not in filled_journey_df.columns:
        return worker_df

    from src.pipeline.gap_analyzer import aggregate_activity_by_worker

    act_df = aggregate_activity_by_worker(filled_journey_df)

    # low_confidence 원본 레코드 수 집계
    if "is_low_confidence" in filled_journey_df.columns:
        low_conf = (
            filled_journey_df[filled_journey_df["is_low_confidence"]]
            .groupby("user_no")
            .size()
            .reset_index(name="ga_low_conf_min")
        )
        act_df = act_df.merge(low_conf, on="user_no", how="left")
        act_df["ga_low_conf_min"] = act_df["ga_low_conf_min"].fillna(0).astype(int)
    else:
        act_df["ga_low_conf_min"] = 0

    act_df = act_df.rename(columns={
        "high_active_min":   "ga_high_active_min",
        "active_min":        "ga_active_min",
        "inactive_min":      "ga_inactive_min",
        "deep_inactive_min": "ga_deep_inactive_min",
        "estimated_min":     "ga_estimated_min",
        "gap_ratio_pct":     "ga_gap_ratio_pct",
        "dominant_activity": "ga_dominant_activity",
    })

    result = worker_df.merge(act_df, on="user_no", how="left")

    # 기본값 채우기
    for col in ["ga_high_active_min", "ga_active_min", "ga_inactive_min",
                "ga_deep_inactive_min", "ga_estimated_min", "ga_gap_ratio_pct",
                "ga_low_conf_min"]:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    logger.info(
        "enrich_worker_with_gap_stats: merged %d workers' gap stats",
        act_df["user_no"].nunique(),
    )
    return result
