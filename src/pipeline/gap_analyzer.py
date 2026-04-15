"""
Gap Analyzer — T-Ward 음영 구간 탐지 · 분류 · 채우기 + 활성 레벨 분류
=======================================================================

건설현장 BLE 환경 특성상 T-Ward 작업자의 journey에는 음영 구간(GAP)이 빈번하다.
이 모듈은 GAP을 탐지·분류하고 합리적인 추정값으로 채워 transit 분석의
정확도를 높인다.

주요 함수:
  detect_gaps(journey_df)         → gap_df (GAP 목록 + 분류)
  fill_gaps(journey_df, gap_df)   → gap-filled journey_df
  classify_activity(journey_df)   → activity_level 컬럼 추가
  analyze_gaps(journey_df)        → 위 3단계 통합 실행 + stats 반환
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 임계값 상수 ───────────────────────────────────────────────────────────────
GAP_MIN_THRESHOLD       = 1.5   # 이 분(min) 초과 = GAP으로 판정
GAP_SHORT_MAX           = 5.0   # 짧은 GAP 상한 (분)
GAP_MEDIUM_MAX          = 20.0  # 중간 GAP 상한 (분)
GAP_LONG_MAX            = 60.0  # 긴 GAP 상한 (분)  — 초과 시 미채움
LOW_CONFIDENCE_SIGNAL   = 2     # signal_count ≤ 이 값이면 low_confidence

# 활성 레벨 임계값
AR_HIGH_ACTIVE          = 0.75  # active_ratio ≥ 이 값 = HIGH_ACTIVE
AR_ACTIVE               = 0.50  # active_ratio ≥ 이 값 = ACTIVE
AR_INACTIVE             = 0.20  # active_ratio ≥ 이 값 = INACTIVE (미만 = DEEP_INACTIVE)
SC_HIGH_ACTIVE_MIN      = 3     # HIGH_ACTIVE 판정 최소 signal_count
SC_ACTIVE_MIN           = 2     # ACTIVE 판정 최소 signal_count

# GAP 유형 레이블
GAP_TYPE_SHORT_SAME     = "shadow_same_short"    # <5분, 같은 locus
GAP_TYPE_SHORT_DIFF     = "shadow_diff_short"    # <5분, 다른 locus
GAP_TYPE_MEDIUM_SAME    = "shadow_same_medium"   # 5~20분, 같은 locus
GAP_TYPE_MEDIUM_DIFF    = "shadow_diff_medium"   # 5~20분, 다른 locus
GAP_TYPE_LONG           = "shadow_long"          # 20~60분
GAP_TYPE_VERY_LONG      = "shadow_very_long"     # >60분 — 채우지 않음

# 채우기 신뢰도
CONF_HIGH   = "high"
CONF_MEDIUM = "medium"
CONF_LOW    = "low"
CONF_NONE   = "none"   # 채우지 않음


# ── 타입 정의 ─────────────────────────────────────────────────────────────────
class GapStats(TypedDict):
    total_gaps: int
    filled_gaps: int
    skipped_gaps: int
    filled_records: int
    gap_workers: int
    avg_gap_min: float
    gap_type_dist: dict[str, int]
    low_confidence_records: int


# ── 1. GAP 탐지 ───────────────────────────────────────────────────────────────

def detect_gaps(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    T-Ward 작업자 journey에서 GAP 구간을 탐지하고 분류한다.

    Args:
        journey_df: journey.parquet 형식 DataFrame
            필수 컬럼: user_no, timestamp, locus_id, has_tward

    Returns:
        gap_df: 각 GAP을 한 행으로 표현한 DataFrame
            컬럼: user_no, gap_start, gap_end, gap_min,
                   locus_before, locus_after, same_locus,
                   gap_type, confidence
    """
    if journey_df.empty:
        return pd.DataFrame()

    # T-Ward 착용자만 대상
    if "has_tward" in journey_df.columns:
        df = journey_df[journey_df["has_tward"] == True].copy()
    else:
        df = journey_df.copy()

    if df.empty:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    rows = []
    for user_no, grp in df.groupby("user_no", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        ts      = grp["timestamp"]
        loci    = grp["locus_id"]

        for i in range(1, len(grp)):
            gap_min = (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() / 60
            if gap_min <= GAP_MIN_THRESHOLD:
                continue

            locus_b = loci.iloc[i - 1]
            locus_a = loci.iloc[i]
            same    = locus_b == locus_a
            gap_type, confidence = _classify_gap(gap_min, same)

            rows.append(
                {
                    "user_no":      user_no,
                    "gap_start":    ts.iloc[i - 1],
                    "gap_end":      ts.iloc[i],
                    "gap_min":      round(gap_min, 1),
                    "locus_before": locus_b,
                    "locus_after":  locus_a,
                    "same_locus":   same,
                    "gap_type":     gap_type,
                    "confidence":   confidence,
                }
            )

    if not rows:
        return pd.DataFrame()

    gap_df = pd.DataFrame(rows)
    logger.info(
        "detect_gaps: %d gaps detected across %d workers",
        len(gap_df),
        gap_df["user_no"].nunique(),
    )
    return gap_df


def _classify_gap(gap_min: float, same_locus: bool) -> tuple[str, str]:
    """GAP 크기 + locus 동일 여부 → (gap_type, confidence)"""
    if gap_min <= GAP_SHORT_MAX:
        if same_locus:
            return GAP_TYPE_SHORT_SAME, CONF_HIGH
        else:
            return GAP_TYPE_SHORT_DIFF, CONF_MEDIUM
    elif gap_min <= GAP_MEDIUM_MAX:
        if same_locus:
            return GAP_TYPE_MEDIUM_SAME, CONF_MEDIUM
        else:
            return GAP_TYPE_MEDIUM_DIFF, CONF_LOW
    elif gap_min <= GAP_LONG_MAX:
        return GAP_TYPE_LONG, CONF_LOW
    else:
        return GAP_TYPE_VERY_LONG, CONF_NONE


# ── 2. GAP 채우기 ─────────────────────────────────────────────────────────────

def fill_gaps(
    journey_df: pd.DataFrame,
    gap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    탐지된 GAP을 분 단위 레코드로 채운다.
    채워진 레코드에는 is_gap_filled=True, gap_confidence 컬럼이 추가된다.

    채우기 전략:
      shadow_same_short   : 출발 locus로 전체 채움
      shadow_diff_short   : GAP 절반 지점에서 locus 전환
      shadow_same_medium  : 출발 locus로 전체 채움
      shadow_diff_medium  : 앞 60% 출발, 뒤 40% 도착 locus
      shadow_long         : 앞 50% 출발, 뒤 50% 도착 locus (low confidence)
      shadow_very_long    : 채우지 않음

    Args:
        journey_df: 원본 journey DataFrame
        gap_df: detect_gaps() 결과

    Returns:
        gap-filled journey DataFrame (원본 + 채워진 레코드, 시간순 정렬)
    """
    if gap_df.empty:
        journey_df = journey_df.copy()
        journey_df["is_gap_filled"] = False
        journey_df["gap_confidence"] = CONF_NONE
        journey_df["is_low_confidence"] = _flag_low_confidence(journey_df)
        return journey_df

    # 채울 GAP만 필터링
    fillable = gap_df[gap_df["confidence"] != CONF_NONE].copy()

    synthetic_rows = []
    for _, gap in fillable.iterrows():
        new_rows = _generate_fill_records(gap, journey_df)
        synthetic_rows.extend(new_rows)

    if not synthetic_rows:
        journey_df = journey_df.copy()
        journey_df["is_gap_filled"] = False
        journey_df["gap_confidence"] = CONF_NONE
        journey_df["is_low_confidence"] = _flag_low_confidence(journey_df)
        return journey_df

    # 원본에 메타 컬럼 추가
    orig = journey_df.copy()
    orig["is_gap_filled"] = False
    orig["gap_confidence"] = CONF_NONE

    # 합성 레코드 DataFrame
    synth_df = pd.DataFrame(synthetic_rows)

    # 합치기
    result = pd.concat([orig, synth_df], ignore_index=True)
    result = result.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    # low_confidence 플래그 (원본 + 합성 모두)
    result["is_low_confidence"] = _flag_low_confidence(result)

    logger.info(
        "fill_gaps: %d synthetic records added for %d gaps",
        len(synthetic_rows),
        len(fillable),
    )
    return result


def _generate_fill_records(
    gap: pd.Series,
    journey_df: pd.DataFrame,
) -> list[dict]:
    """
    단일 GAP에 대해 분 단위 채우기 레코드 목록을 생성한다.
    """
    gap_type    = gap["gap_type"]
    locus_b     = gap["locus_before"]
    locus_a     = gap["locus_after"]
    confidence  = gap["confidence"]
    user_no     = gap["user_no"]

    # GAP 내 분 단위 타임스탬프 (출발+1분 ~ 도착-1분)
    start_ts = gap["gap_start"] + pd.Timedelta(minutes=1)
    end_ts   = gap["gap_end"]   - pd.Timedelta(minutes=1)

    if start_ts >= end_ts:
        return []

    minutes = pd.date_range(start=start_ts, end=end_ts, freq="1min")
    n       = len(minutes)
    if n == 0:
        return []

    # locus 시퀀스 결정
    locus_seq = _assign_locus_sequence(gap_type, locus_b, locus_a, n)

    # 원본에서 해당 작업자의 평균 active_ratio 참조 (채우기용)
    user_rows = journey_df[journey_df["user_no"] == user_no]
    avg_ar = user_rows["active_ratio"].mean() if len(user_rows) > 0 else 0.5

    # 기준 컬럼 복사용: 첫 번째 레코드에서 공통 필드 추출
    base = user_rows.iloc[0].to_dict() if len(user_rows) > 0 else {}

    records = []
    for i, ts in enumerate(minutes):
        locus = locus_seq[i]
        row = {
            # 작업자 식별
            "user_no":          user_no,
            "timestamp":        ts,
            "user_name":        base.get("user_name", ""),
            "company_name":     base.get("company_name", ""),
            "company_code":     base.get("company_code", ""),
            # 위치
            "locus_id":         locus,
            "locus_token":      base.get("locus_token", locus),  # 임시, 재계산됨
            "building_name":    base.get("building_name", ""),
            "floor_name":       base.get("floor_name", ""),
            "spot_name":        "",
            "x":                np.nan,
            "y":                np.nan,
            # 신호 (채운 값은 신호 없음)
            "signal_count":     0,
            "active_count":     0,
            "active_ratio":     avg_ar,  # 평균으로 보간
            # 출입 정보 (원본과 동일)
            "in_datetime":      base.get("in_datetime"),
            "out_datetime":     base.get("out_datetime"),
            "has_tward":        True,
            "twardid":          base.get("twardid"),
            "is_work_hour":     base.get("is_work_hour", True),
            "missing_exit":     base.get("missing_exit", False),
            "shift_type":       base.get("shift_type", ""),
            "exit_source":      base.get("exit_source", ""),
            "work_minutes":     base.get("work_minutes", np.nan),
            # GAP 메타
            "is_gap_filled":    True,
            "gap_confidence":   confidence,
        }
        records.append(row)

    return records


def _assign_locus_sequence(
    gap_type: str,
    locus_b: str,
    locus_a: str,
    n: int,
) -> list[str]:
    """GAP 유형에 따라 n개 분의 locus 시퀀스를 반환한다."""
    if gap_type in (GAP_TYPE_SHORT_SAME, GAP_TYPE_MEDIUM_SAME):
        # 같은 locus 전체 채움
        return [locus_b] * n

    elif gap_type == GAP_TYPE_SHORT_DIFF:
        # 절반 지점에서 locus 전환
        split = n // 2
        return [locus_b] * split + [locus_a] * (n - split)

    elif gap_type == GAP_TYPE_MEDIUM_DIFF:
        # 앞 60% 출발, 뒤 40% 도착
        split = int(n * 0.6)
        return [locus_b] * split + [locus_a] * (n - split)

    elif gap_type == GAP_TYPE_LONG:
        # 앞 50% 출발, 뒤 50% 도착 (low confidence)
        split = n // 2
        return [locus_b] * split + [locus_a] * (n - split)

    else:
        # 채우지 않음 (여기까지 오면 안 됨)
        return [locus_b] * n


def _flag_low_confidence(df: pd.DataFrame) -> pd.Series:
    """
    signal_count ≤ LOW_CONFIDENCE_SIGNAL 인 원본 레코드를 low_confidence로 표시.
    채워진 레코드(signal_count=0)는 별도로 gap_confidence로 판단.
    """
    if "signal_count" not in df.columns:
        return pd.Series(False, index=df.index)

    # 채워진 레코드가 아닌 원본 중 signal_count가 낮은 것
    is_orig = ~df.get("is_gap_filled", pd.Series(False, index=df.index))
    low_sig = df["signal_count"].fillna(0) <= LOW_CONFIDENCE_SIGNAL
    return is_orig & low_sig


# ── 3. 활성 레벨 분류 ─────────────────────────────────────────────────────────

def classify_activity(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 레코드에 activity_level 컬럼을 추가한다.

    레벨 정의:
      HIGH_ACTIVE   : active_ratio ≥ 0.75 AND signal_count ≥ 3
      ACTIVE        : active_ratio ≥ 0.50 AND signal_count ≥ 2
      INACTIVE      : active_ratio ≥ 0.20
      DEEP_INACTIVE : active_ratio < 0.20

    채워진 레코드(is_gap_filled=True)는 "ESTIMATED"로 별도 표시.

    Returns:
        activity_level 컬럼이 추가된 DataFrame
    """
    df = journey_df.copy()

    ar = df["active_ratio"].fillna(0.0)
    sc = df["signal_count"].fillna(0)
    is_filled = df.get("is_gap_filled", pd.Series(False, index=df.index))

    conditions = [
        is_filled,                                       # GAP 채운 레코드
        (ar >= AR_HIGH_ACTIVE) & (sc >= SC_HIGH_ACTIVE_MIN),
        (ar >= AR_ACTIVE) & (sc >= SC_ACTIVE_MIN),
        ar >= AR_INACTIVE,
    ]
    choices = ["ESTIMATED", "HIGH_ACTIVE", "ACTIVE", "INACTIVE"]
    df["activity_level"] = np.select(conditions, choices, default="DEEP_INACTIVE")

    return df


# ── 4. 작업자별 활성 집계 ─────────────────────────────────────────────────────

def aggregate_activity_by_worker(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    작업자별 activity_level 집계 지표를 계산한다.

    Returns:
        DataFrame with columns:
          user_no, high_active_min, active_min, inactive_min,
          deep_inactive_min, estimated_min, gap_ratio_pct,
          dominant_activity
    """
    if "activity_level" not in journey_df.columns:
        journey_df = classify_activity(journey_df)

    rows = []
    for user_no, grp in journey_df.groupby("user_no"):
        level_counts = grp["activity_level"].value_counts()
        total_min    = len(grp)

        high_active_min   = int(level_counts.get("HIGH_ACTIVE",   0))
        active_min        = int(level_counts.get("ACTIVE",        0))
        inactive_min      = int(level_counts.get("INACTIVE",      0))
        deep_inactive_min = int(level_counts.get("DEEP_INACTIVE", 0))
        estimated_min     = int(level_counts.get("ESTIMATED",     0))

        # 음영 비율: 채워진 레코드 / 전체
        gap_ratio = round(estimated_min / total_min * 100, 1) if total_min > 0 else 0.0

        # 지배적 활성 레벨 (ESTIMATED 제외)
        real_counts = level_counts.drop("ESTIMATED", errors="ignore")
        dominant = real_counts.idxmax() if len(real_counts) > 0 else "UNKNOWN"

        rows.append(
            {
                "user_no":           user_no,
                "high_active_min":   high_active_min,
                "active_min":        active_min,
                "inactive_min":      inactive_min,
                "deep_inactive_min": deep_inactive_min,
                "estimated_min":     estimated_min,
                "gap_ratio_pct":     gap_ratio,
                "dominant_activity": dominant,
            }
        )

    return pd.DataFrame(rows)


# ── 5. Locus별 활성 집계 ─────────────────────────────────────────────────────

def aggregate_activity_by_locus(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    Locus별 activity_level 분포 및 평균 active_ratio 집계.

    Returns:
        DataFrame with columns:
          locus_id, total_min, high_active_pct, active_pct,
          inactive_pct, deep_inactive_pct, avg_active_ratio
    """
    if "activity_level" not in journey_df.columns:
        journey_df = classify_activity(journey_df)

    # GAP 채운 레코드 제외 (실측만)
    real = journey_df[~journey_df.get("is_gap_filled", pd.Series(False, index=journey_df.index))]

    rows = []
    for locus_id, grp in real.groupby("locus_id"):
        total = len(grp)
        if total == 0:
            continue

        counts = grp["activity_level"].value_counts()
        rows.append(
            {
                "locus_id":          locus_id,
                "total_min":         total,
                "high_active_pct":   round(counts.get("HIGH_ACTIVE",   0) / total * 100, 1),
                "active_pct":        round(counts.get("ACTIVE",        0) / total * 100, 1),
                "inactive_pct":      round(counts.get("INACTIVE",      0) / total * 100, 1),
                "deep_inactive_pct": round(counts.get("DEEP_INACTIVE", 0) / total * 100, 1),
                "avg_active_ratio":  round(grp["active_ratio"].mean(), 3),
            }
        )

    return pd.DataFrame(rows)


# ── 6. 통합 실행 ─────────────────────────────────────────────────────────────

def analyze_gaps(journey_df: pd.DataFrame) -> tuple[pd.DataFrame, GapStats]:
    """
    GAP 탐지 → 채우기 → 활성 분류 3단계를 통합 실행한다.

    Args:
        journey_df: 원본 journey DataFrame (T-Ward 포함 전체 가능)

    Returns:
        (filled_df, stats)
          filled_df : gap-filled + activity_level 추가된 journey DataFrame
          stats     : GapStats 딕셔너리
    """
    # Step 1: GAP 탐지
    gap_df = detect_gaps(journey_df)

    # Step 2: GAP 채우기
    filled_df = fill_gaps(journey_df, gap_df)

    # Step 3: 활성 레벨 분류
    filled_df = classify_activity(filled_df)

    # 통계 수집
    stats = _build_stats(gap_df, filled_df)

    logger.info(
        "analyze_gaps complete: %d gaps → %d filled records added, "
        "%d low_confidence original records",
        stats["total_gaps"],
        stats["filled_records"],
        stats["low_confidence_records"],
    )
    return filled_df, stats


def _build_stats(gap_df: pd.DataFrame, filled_df: pd.DataFrame) -> GapStats:
    if gap_df.empty:
        return GapStats(
            total_gaps=0,
            filled_gaps=0,
            skipped_gaps=0,
            filled_records=0,
            gap_workers=0,
            avg_gap_min=0.0,
            gap_type_dist={},
            low_confidence_records=0,
        )

    fillable  = gap_df[gap_df["confidence"] != CONF_NONE]
    skipped   = gap_df[gap_df["confidence"] == CONF_NONE]
    n_filled_rec = int(filled_df.get("is_gap_filled", pd.Series(False)).sum())
    n_low_conf   = int(filled_df.get("is_low_confidence", pd.Series(False)).sum())

    return GapStats(
        total_gaps=len(gap_df),
        filled_gaps=len(fillable),
        skipped_gaps=len(skipped),
        filled_records=n_filled_rec,
        gap_workers=int(gap_df["user_no"].nunique()),
        avg_gap_min=round(float(gap_df["gap_min"].mean()), 1),
        gap_type_dist=gap_df["gap_type"].value_counts().to_dict(),
        low_confidence_records=n_low_conf,
    )
