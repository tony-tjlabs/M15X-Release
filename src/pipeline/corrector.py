"""
Journey Corrector — BLE 이동 노이즈 보정 모듈
===============================================

BLE 신호 특성상 발생하는 순간적인 위치 튐(노이즈)을 제거하고
작업자의 실제 동선(Journey)을 추정한다.

알고리즘:
  Phase 1: Run-Length 기반 플리커 보정 (전체 데이터 단일 패스, 완전 벡터화)
    - 연속 동일 locus 구간(Run) 식별 (user_no 경계에서 강제 분리)
    - Run 길이 ≤ MAX_FLICKER_RUN 이고 앞뒤 locus가 동일하면 플리커 → 앞 locus로 교체
    - 앵커 토큰(timeclock/breakroom/smoking_area)은 보호
    - 작업자별 for 루프 없이 groupby + map으로 처리

  Phase 2: 음영 구간 보간 (Shadow Zone Gap Imputation)
    - BLE 미감지 구간(detected row 間 시간 공백)을 앞뒤 맥락으로 채움
    - gap < MAX_SHADOW_IMPUTE_MIN (30분): 앞뒤 locus 기반 보간 → 해당 locus로 귀속
      * 앞뒤 동일 token: 해당 token 유지
      * 앞뒤 모두 work_zone: work_zone으로 귀속 (FAB 내 음영 → 작업 중으로 추정)
      * 앞뒤 모두 REST: 앞 REST token으로 유지 (짧은 음영은 같은 공간)
      * 그 외: 앞 token 보수적 유지
    - gap >= MAX_SHADOW_IMPUTE_MIN: "shadow_zone" 토큰으로 표시 (BLE 미감지 확정)
    - 보간된 행에는 is_imputed=True 플래그 설정

설계 원칙 (DeepCon_SOIF v6.1 Multi-Pass 철학):
  "앵커 공간 보호 → Run-Level 플리커 → 보수적 보정"
  - 실제 이동(A→B→C)은 건드리지 않음
  - 오직 노이즈성 짧은 튐(A→[short B]→A)만 제거
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── 보정 상수 ────────────────────────────────────────────────────────────
# Run 기반 플리커: 이 길이(분) 이하이고 앞뒤가 동일 locus이면 노이즈로 판단
MAX_FLICKER_RUN: int = 2

# 앵커 locus_token — 짧아도 보정하지 않는 실제 이벤트 공간
ANCHOR_TOKENS: frozenset[str] = frozenset({
    "timeclock",    # 타각기/게이트 — 출입 확인 이벤트
    "breakroom",    # 휴게실
    "smoking_area", # 흡연 구역
})

# Phase 2: 음영 보간 상수
MAX_SHADOW_IMPUTE_MIN: int = 30   # 이 미만 gap은 앞뒤 맥락으로 보간
SHADOW_TOKEN: str = "shadow_zone" # 30분 이상 gap에 할당되는 토큰
SHADOW_LOCUS_ID: str = "shadow_zone"

# REST 토큰 집합 (음영 보간 시 REST ↔ REST 구간 판별용)
_REST_TOKENS_IMPUTE: frozenset[str] = frozenset({
    "breakroom", "smoking_area", "dining_hall", "restroom", "parking_lot",
})


# ─── 공개 진입점 ──────────────────────────────────────────────────────────
def correct_journeys(journey_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    모든 작업자에 대해 Journey 보정 수행.

    단일 패스 벡터화 — for 루프 없이 전체 데이터를 한 번에 처리.

    Args:
        journey_df: apply_locus_mapping + build_worker_journeys 이후 DataFrame
                    필수 컬럼: user_no, timestamp, locus_id, locus_token

    Returns:
        (corrected_df, stats)
        stats = {
            "corrected_records":  int,   # 보정된 행 수
            "correction_ratio":   float, # 보정 비율 (%)
            "workers_corrected":  int,   # 보정된 작업자 수
        }
    """
    if journey_df.empty:
        return journey_df, _empty_stats()

    # 정렬 보장 (user_no + timestamp)
    df = journey_df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)
    original_loci = df["locus_id"].copy()

    # Phase 1: Run-Length 기반 플리커 보정 (전체 단일 패스)
    df = _run_flicker_global(df)

    # Phase 1 통계
    changed_mask   = df["locus_id"].values != original_loci.values
    total_changed  = int(changed_mask.sum())
    total_records  = len(df)
    workers_changed = (
        int(df.loc[changed_mask, "user_no"].nunique())
        if total_changed > 0 else 0
    )
    correction_ratio = round(total_changed / total_records * 100, 2) if total_records > 0 else 0.0

    # Phase 2: 음영 구간 보간
    df, shadow_stats = impute_shadow_gaps(df, max_impute_min=MAX_SHADOW_IMPUTE_MIN)

    stats = {
        "corrected_records": total_changed,
        "correction_ratio":  correction_ratio,
        "workers_corrected": workers_changed,
        "imputed_rows":      shadow_stats["imputed_rows"],
        "shadow_rows":       shadow_stats["shadow_rows"],
    }
    logger.info(
        f"Journey 보정 완료: {total_changed:,}건 ({correction_ratio:.2f}%) "
        f"/ 작업자 {workers_changed}명"
    )
    logger.info(
        f"음영 보간: 맥락 보간 {shadow_stats['imputed_rows']:,}분 "
        f"/ BLE 미감지(30분+) {shadow_stats['shadow_rows']:,}분"
    )
    return df, stats


# ─── Phase 1: Run-Length 기반 플리커 보정 (전체 데이터 단일 패스) ─────────
def _run_flicker_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 데이터프레임에 대해 단일 패스로 Run-Length 플리커 보정 수행.

    핵심 원리:
      A, A, A, [B, B], A, A  →  B 구간(Run 길이 2)의 앞뒤가 A로 동일 → 플리커 제거

    구현 (for 루프 없이 완전 벡터화):
      1. user_no 경계 또는 locus_id 변경 시 새 Run 시작 → run_id 생성
      2. run별 agg: locus_id, run_len, is_anchor (첫 토큰 기준)
      3. 플리커 조건 판별 (pandas 벡터 연산)
      4. 플리커 run_id에 해당하는 행의 locus_id를 이전 run locus로 교체
         → map() 사용으로 루프 없이 처리
    """
    result = df.copy()
    loci      = result["locus_id"].values
    user_nos  = result["user_no"].values

    # ── 1. Run ID 생성 ─────────────────────────────────────────────────
    # user_no 경계 또는 locus_id 변경 시 새 Run
    is_change = np.concatenate([[True], (
        (loci[1:] != loci[:-1]) | (user_nos[1:] != user_nos[:-1])
    )])
    run_id_arr = np.cumsum(is_change).astype(np.int32)
    result["_run_id"] = run_id_arr

    # ── 2. Run별 집계 ──────────────────────────────────────────────────
    agg_cols = {
        "locus_id":  ("locus_id",  "first"),
        "run_len":   ("_run_id",   "count"),
        "user_no":   ("user_no",   "first"),   # user_no 경계 체크용
    }
    if "locus_token" in result.columns:
        agg_cols["first_token"] = ("locus_token", "first")

    run_info = result.groupby("_run_id", sort=False).agg(**agg_cols).reset_index()

    # 앵커 여부 (토큰 기반)
    if "first_token" in run_info.columns:
        run_info["is_anchor"] = run_info["first_token"].isin(ANCHOR_TOKENS)
    else:
        run_info["is_anchor"] = False

    # ── 3. 플리커 조건 판별 ────────────────────────────────────────────
    # 이전/다음 Run의 locus_id 및 user_no (shift는 정렬된 run_id 순서 기준)
    run_info["prev_locus"]   = run_info["locus_id"].shift(1)
    run_info["next_locus"]   = run_info["locus_id"].shift(-1)
    run_info["prev_user_no"] = run_info["user_no"].shift(1)
    run_info["next_user_no"] = run_info["user_no"].shift(-1)

    flicker_mask = (
        (run_info["run_len"] <= MAX_FLICKER_RUN)
        & run_info["prev_locus"].notna()
        & run_info["next_locus"].notna()
        & (run_info["prev_locus"] == run_info["next_locus"])   # 앞뒤 동일 locus
        & (run_info["prev_locus"] != run_info["locus_id"])     # 현재 locus와 다름
        & ~run_info["is_anchor"]                               # 앵커 보호
        # ★ user_no 경계 보호 — 이전/다음 Run이 같은 작업자여야 함
        & (run_info["prev_user_no"] == run_info["user_no"])
        & (run_info["next_user_no"] == run_info["user_no"])
    )

    flicker_rows = run_info[flicker_mask]

    if flicker_rows.empty:
        result.drop(columns=["_run_id"], inplace=True)
        return result

    # ── 4. 플리커 run → 이전 locus로 교체 (map 사용, 루프 없음) ─────────
    # flicker run_id → 교체할 locus_id 딕셔너리
    replace_map = dict(zip(flicker_rows["_run_id"], flicker_rows["prev_locus"]))

    # _run_id로 교체할 locus_id 조회
    is_flicker_row = result["_run_id"].isin(replace_map)
    result.loc[is_flicker_row, "locus_id"] = (
        result.loc[is_flicker_row, "_run_id"].map(replace_map).values
    )

    result.drop(columns=["_run_id"], inplace=True)
    return result


# ─── Phase 2: 음영 구간 보간 ──────────────────────────────────────────────
def impute_shadow_gaps(
    df: pd.DataFrame,
    max_impute_min: int = MAX_SHADOW_IMPUTE_MIN,
) -> tuple[pd.DataFrame, dict]:
    """
    BLE 미감지 구간(detected row 間 시간 공백)을 앞뒤 맥락으로 채운다.

    알고리즘:
      1. 작업자별 연속 행 간 timestamp 차이 계산 (gap_min)
      2. gap > 0인 구간을 분류:
         - gap < max_impute_min (30분): 앞뒤 locus 맥락으로 보간
         - gap >= max_impute_min: "shadow_zone" 토큰 할당 (BLE 미감지)
      3. np.repeat + 타임스탬프 오프셋으로 완전 벡터화 (row 단위 루프 없음)

    보간 토큰 결정 규칙:
      - prev == next (동일 공간): 해당 토큰 유지
      - prev == next == work_zone: work_zone (FAB 내 음영 → 작업 중 추정)
      - prev & next 모두 REST: prev REST 토큰 유지 (짧은 음영도 같은 휴식 공간)
      - 그 외: prev 토큰 보수적 유지 (마지막 감지 위치 기준)

    Returns:
        (결과 df, stats)
        stats = {"imputed_rows": int, "shadow_rows": int}
    """
    if df.empty:
        df = df.copy()
        df["is_imputed"] = False
        return df, {"imputed_rows": 0, "shadow_rows": 0}

    df = df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    # ── 1. 다음 행 정보 계산 (같은 user_no 내) ─────────────────────
    grp = df.groupby("user_no", sort=False)
    df["_next_ts"]    = grp["timestamp"].shift(-1)
    df["_next_token"] = grp["locus_token"].shift(-1)
    df["_next_id"]    = grp["locus_id"].shift(-1)
    df["_next_ratio"] = grp["active_ratio"].shift(-1) if "active_ratio" in df.columns else 0.0

    # gap 계산: 현재 행 다음부터 다음 행 이전까지 (분)
    df["_gap_min"] = (
        (df["_next_ts"] - df["timestamp"]).dt.total_seconds() / 60 - 1
    ).fillna(0).clip(lower=0).astype(int)

    # ── 2. gap > 0인 행만 추출 ──────────────────────────────────────
    gap_mask = (df["_gap_min"] > 0) & df["_next_ts"].notna()
    gap_rows = df[gap_mask].copy()

    if gap_rows.empty:
        df["is_imputed"] = False
        df.drop(columns=["_next_ts", "_next_token", "_next_id", "_next_ratio", "_gap_min"],
                inplace=True, errors="ignore")
        return df, {"imputed_rows": 0, "shadow_rows": 0}

    # ── 3. 보간 토큰 결정 (벡터화) ──────────────────────────────────
    prev_tok = gap_rows["locus_token"].fillna("unknown")
    next_tok = gap_rows["_next_token"].fillna("unknown")
    gap_min  = gap_rows["_gap_min"]

    # 조건: 30분 미만 보간 대상
    is_short_gap = gap_min < max_impute_min

    # 토큰 결정 로직 (우선순위 순)
    same_token  = prev_tok == next_tok
    both_work   = (prev_tok == "work_zone") & (next_tok == "work_zone")
    both_rest   = prev_tok.isin(_REST_TOKENS_IMPUTE) & next_tok.isin(_REST_TOKENS_IMPUTE)

    # short gap: 앞뒤 동일 → 해당 토큰 / 모두 work_zone → work_zone / 모두 REST → prev / 나머지 → prev
    fill_token_short = prev_tok.copy()  # 기본: prev (보수적)
    # 앞뒤 동일 토큰 (work/rest/transit 모두 포함)
    fill_token_short = fill_token_short.where(~same_token, prev_tok)
    # 명시적으로 both_work 는 work_zone (이미 same_token에 포함되지만 명확성 위해)
    fill_token_short = fill_token_short.where(~both_work, pd.Series("work_zone", index=gap_rows.index))

    # long gap: shadow_zone
    fill_token = pd.Series(SHADOW_TOKEN, index=gap_rows.index)
    fill_token = fill_token.where(~is_short_gap, fill_token_short)

    # locus_id: short gap → prev locus_id / long gap → SHADOW_LOCUS_ID
    fill_locus_id = gap_rows["locus_id"].copy()
    fill_locus_id = fill_locus_id.where(~is_short_gap, fill_locus_id)  # short: prev id
    fill_locus_id[~is_short_gap] = SHADOW_LOCUS_ID

    gap_rows["_fill_token"]    = fill_token.values
    gap_rows["_fill_locus_id"] = fill_locus_id.values

    # ── 4. np.repeat으로 synthetic rows 완전 벡터화 생성 ─────────────
    repeats = gap_rows["_gap_min"].values
    expanded = gap_rows.loc[gap_rows.index.repeat(repeats)].copy()

    # 각 반복 행에 분 오프셋 추가
    offsets = np.concatenate([np.arange(1, g + 1) for g in repeats])
    expanded["timestamp"]    = expanded["timestamp"] + pd.to_timedelta(offsets, unit="m")
    expanded["locus_token"]  = expanded["_fill_token"].values
    expanded["locus_id"]     = expanded["_fill_locus_id"].values

    # active_ratio: prev + next 평균 (활동 수준 추정)
    if "active_ratio" in expanded.columns and "_next_ratio" in expanded.columns:
        expanded["active_ratio"] = (
            (expanded["active_ratio"].fillna(0) + expanded["_next_ratio"].fillna(0)) / 2
        )
    expanded["signal_count"] = 0          # 실제 감지 아님
    expanded["is_imputed"]   = True

    # 임시 컬럼 정리
    tmp_cols = ["_next_ts", "_next_token", "_next_id", "_next_ratio",
                "_gap_min", "_fill_token", "_fill_locus_id"]
    expanded.drop(columns=tmp_cols, inplace=True, errors="ignore")

    # ── 5. 원본 df 정리 후 합치기 ────────────────────────────────────
    df.drop(columns=["_next_ts", "_next_token", "_next_id", "_next_ratio", "_gap_min"],
            inplace=True, errors="ignore")
    df["is_imputed"] = False

    result = pd.concat([df, expanded], ignore_index=True)
    result = result.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    imputed_rows = int((expanded["locus_token"] != SHADOW_TOKEN).sum())
    shadow_rows  = int((expanded["locus_token"] == SHADOW_TOKEN).sum())

    return result, {"imputed_rows": imputed_rows, "shadow_rows": shadow_rows}


def _empty_stats() -> dict:
    return {
        "corrected_records": 0, "correction_ratio": 0.0, "workers_corrected": 0,
        "imputed_rows": 0, "shadow_rows": 0,
    }
