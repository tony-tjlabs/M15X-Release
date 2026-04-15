"""
Data Validator — 데이터 정합성 검증 모듈
========================================
파이프라인 처리 후 데이터 품질을 검증하여
신뢰할 수 있는 분석 결과를 보장한다.

검증 항목:
  1. BLE 커버리지: 근무 시간 대비 BLE 감지 비율
  2. Journey 연속성: Adjacency 기반 유효 이동 비율
  3. 출퇴근 일관성: AccessLog vs TWardData 시간 오차

사용법:
    from src.pipeline.validator import run_all_validations

    results = run_all_validations(journey_df, worker_df, access_df, spatial_graph)
    report = generate_quality_report(results)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.spatial.graph import SpatialGraph

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """검증 결과 등급."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationResult:
    """검증 결과."""

    check_name: str
    level: ValidationLevel
    metric: float
    threshold_pass: float
    threshold_warning: float
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "level": self.level.value,
            "metric": round(self.metric, 4),
            "threshold_pass": self.threshold_pass,
            "threshold_warning": self.threshold_warning,
            "message": self.message,
            **self.details,
        }


# ─── BLE 커버리지 검증 ─────────────────────────────────────────────────────


def validate_ble_coverage(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
) -> ValidationResult:
    """
    BLE 커버리지 검증.

    계산: 평균 (1 - gap_ratio) = BLE 감지 비율
    임계값:
      - PASS: >= 70%
      - WARNING: 50-70%
      - FAIL: < 50%
    """
    THRESHOLD_PASS = 0.70
    THRESHOLD_WARNING = 0.50

    if worker_df.empty or "gap_ratio" not in worker_df.columns:
        return ValidationResult(
            check_name="BLE 커버리지",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="gap_ratio 데이터 없음 — 검증 불가",
            details={"reason": "missing_column"},
        )

    # gap_ratio가 NaN인 경우 제외 (work_minutes=0인 작업자)
    valid_workers = worker_df[worker_df["gap_ratio"].notna()]
    if valid_workers.empty:
        return ValidationResult(
            check_name="BLE 커버리지",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="유효한 gap_ratio 데이터 없음",
            details={"reason": "no_valid_data"},
        )

    # 평균 커버리지 = 1 - 평균 gap_ratio
    avg_gap_ratio = valid_workers["gap_ratio"].mean()
    coverage = 1.0 - avg_gap_ratio

    # 등급 판정
    if coverage >= THRESHOLD_PASS:
        level = ValidationLevel.PASS
        message = f"BLE 커버리지 양호 ({coverage * 100:.1f}%)"
    elif coverage >= THRESHOLD_WARNING:
        level = ValidationLevel.WARNING
        message = f"BLE 커버리지 주의 ({coverage * 100:.1f}%) — 부분 음영 구역 존재"
    else:
        level = ValidationLevel.FAIL
        message = f"BLE 커버리지 부족 ({coverage * 100:.1f}%) — 음영 구역 점검 필요"

    # 상세 정보
    coverage_dist = {
        "정상 (>=70%)": int((valid_workers["gap_ratio"] <= 0.30).sum()),
        "부분음영 (50-70%)": int(((valid_workers["gap_ratio"] > 0.30) & (valid_workers["gap_ratio"] <= 0.50)).sum()),
        "음영 (30-50%)": int(((valid_workers["gap_ratio"] > 0.50) & (valid_workers["gap_ratio"] <= 0.70)).sum()),
        "미측정 (<30%)": int((valid_workers["gap_ratio"] > 0.70).sum()),
    }

    return ValidationResult(
        check_name="BLE 커버리지",
        level=level,
        metric=coverage,
        threshold_pass=THRESHOLD_PASS,
        threshold_warning=THRESHOLD_WARNING,
        message=message,
        details={
            "avg_gap_ratio": round(avg_gap_ratio, 4),
            "total_workers": len(valid_workers),
            "coverage_distribution": coverage_dist,
        },
    )


# ─── Journey 연속성 검증 ──────────────────────────────────────────────────


def validate_journey_continuity(
    journey_df: pd.DataFrame,
    spatial_graph: "SpatialGraph | None" = None,
) -> ValidationResult:
    """
    Journey 연속성 검증 (비인접 직접 이동 비율).

    계산: 비인접 직접 이동 수 / 전체 이동 수
    임계값:
      - PASS: < 5%
      - WARNING: 5-15%
      - FAIL: >= 15%
    """
    THRESHOLD_PASS = 0.05
    THRESHOLD_WARNING = 0.15

    if journey_df.empty:
        return ValidationResult(
            check_name="Journey 연속성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="Journey 데이터 없음",
            details={"reason": "empty_data"},
        )

    # is_valid_transition 컬럼이 있으면 사용
    if "is_valid_transition" in journey_df.columns:
        # 이동 기록만 필터 (locus_changed=True)
        if "locus_changed" in journey_df.columns:
            transitions = journey_df[journey_df["locus_changed"]]
        else:
            # locus_changed 없으면 prev_locus와 비교
            transitions = journey_df[journey_df["locus_id"] != journey_df.get("prev_locus", "")]

        if len(transitions) == 0:
            return ValidationResult(
                check_name="Journey 연속성",
                level=ValidationLevel.PASS,
                metric=1.0,
                threshold_pass=THRESHOLD_PASS,
                threshold_warning=THRESHOLD_WARNING,
                message="이동 기록 없음 (단일 위치 체류)",
                details={"total_transitions": 0},
            )

        invalid_count = (~transitions["is_valid_transition"]).sum()
        total_transitions = len(transitions)
        invalid_ratio = invalid_count / total_transitions

    elif spatial_graph is not None and "prev_locus" in journey_df.columns:
        # SpatialGraph로 직접 검증
        invalid_df = spatial_graph.detect_impossible_transitions(journey_df)
        invalid_count = len(invalid_df)

        # 전체 이동 수
        if "locus_changed" in journey_df.columns:
            total_transitions = journey_df["locus_changed"].sum()
        else:
            total_transitions = len(journey_df[journey_df["locus_id"] != journey_df["prev_locus"]])

        if total_transitions == 0:
            return ValidationResult(
                check_name="Journey 연속성",
                level=ValidationLevel.PASS,
                metric=1.0,
                threshold_pass=THRESHOLD_PASS,
                threshold_warning=THRESHOLD_WARNING,
                message="이동 기록 없음",
                details={"total_transitions": 0},
            )

        invalid_ratio = invalid_count / total_transitions

    else:
        return ValidationResult(
            check_name="Journey 연속성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="검증 불가 — is_valid_transition 또는 SpatialGraph 필요",
            details={"reason": "missing_dependency"},
        )

    # 연속성 = 1 - 비정상 비율
    continuity = 1.0 - invalid_ratio

    # 등급 판정
    if invalid_ratio < THRESHOLD_PASS:
        level = ValidationLevel.PASS
        message = f"Journey 연속성 양호 (비정상 {invalid_ratio * 100:.1f}%)"
    elif invalid_ratio < THRESHOLD_WARNING:
        level = ValidationLevel.WARNING
        message = f"Journey 연속성 주의 (비정상 {invalid_ratio * 100:.1f}%) — BLE 음영 또는 보정 필요"
    else:
        level = ValidationLevel.FAIL
        message = f"Journey 연속성 부족 (비정상 {invalid_ratio * 100:.1f}%) — 데이터 품질 점검 필요"

    return ValidationResult(
        check_name="Journey 연속성",
        level=level,
        metric=continuity,
        threshold_pass=1.0 - THRESHOLD_PASS,
        threshold_warning=1.0 - THRESHOLD_WARNING,
        message=message,
        details={
            "total_transitions": int(total_transitions),
            "invalid_transitions": int(invalid_count),
            "invalid_ratio": round(invalid_ratio, 4),
        },
    )


# ─── 출퇴근 일관성 검증 ───────────────────────────────────────────────────


def validate_work_hours_consistency(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
) -> ValidationResult:
    """
    출퇴근 기록 일관성 검증.

    계산: AccessLog 첫/마지막 기록 시간 vs TWardData in/out_datetime 오차
    임계값:
      - PASS: 평균 오차 < 10분
      - WARNING: 10-30분
      - FAIL: >= 30분
    """
    THRESHOLD_PASS = 10.0  # 분
    THRESHOLD_WARNING = 30.0

    if journey_df.empty or worker_df.empty:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="데이터 없음",
            details={"reason": "empty_data"},
        )

    # 필요한 컬럼 확인
    if "in_datetime" not in worker_df.columns or "timestamp" not in journey_df.columns:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="검증 불가 — in_datetime 또는 timestamp 없음",
            details={"reason": "missing_column"},
        )

    # AccessLog 기준 첫/마지막 기록 시간
    access_times = journey_df.groupby("user_no")["timestamp"].agg(["min", "max"])
    access_times.columns = ["access_first", "access_last"]

    # 병합
    merged = worker_df.merge(access_times, left_on="user_no", right_index=True, how="inner")

    if merged.empty:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="매칭된 작업자 없음",
            details={"reason": "no_match"},
        )

    # 시간 오차 계산 (분)
    errors = []

    for _, row in merged.iterrows():
        in_dt = row.get("in_datetime")
        access_first = row.get("access_first")

        if pd.notna(in_dt) and pd.notna(access_first):
            try:
                # datetime 변환
                if isinstance(in_dt, str):
                    in_dt = pd.to_datetime(in_dt)
                if isinstance(access_first, str):
                    access_first = pd.to_datetime(access_first)

                error_min = abs((access_first - in_dt).total_seconds() / 60)
                errors.append(error_min)
            except Exception:
                continue

    if not errors:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="유효한 비교 데이터 없음",
            details={"reason": "no_valid_comparison"},
        )

    avg_error = np.mean(errors)
    median_error = np.median(errors)

    # 등급 판정
    if avg_error < THRESHOLD_PASS:
        level = ValidationLevel.PASS
        message = f"출퇴근 기록 일관성 양호 (평균 오차 {avg_error:.1f}분)"
    elif avg_error < THRESHOLD_WARNING:
        level = ValidationLevel.WARNING
        message = f"출퇴근 기록 일관성 주의 (평균 오차 {avg_error:.1f}분)"
    else:
        level = ValidationLevel.FAIL
        message = f"출퇴근 기록 불일치 (평균 오차 {avg_error:.1f}분) — 데이터 점검 필요"

    return ValidationResult(
        check_name="출퇴근 일관성",
        level=level,
        metric=avg_error,
        threshold_pass=THRESHOLD_PASS,
        threshold_warning=THRESHOLD_WARNING,
        message=message,
        details={
            "avg_error_min": round(avg_error, 2),
            "median_error_min": round(median_error, 2),
            "sample_count": len(errors),
        },
    )


# ─── 통합 검증 ────────────────────────────────────────────────────────────


def run_all_validations(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    access_df: pd.DataFrame | None = None,
    spatial_graph: "SpatialGraph | None" = None,
) -> list[ValidationResult]:
    """
    모든 검증 실행.

    Args:
        journey_df: Journey 데이터
        worker_df: 작업자 데이터 (EWI/CRE 포함)
        access_df: AccessLog 원본 (선택)
        spatial_graph: SpatialGraph (선택)

    Returns:
        ValidationResult 리스트
    """
    results = [
        validate_ble_coverage(journey_df, worker_df),
        validate_journey_continuity(journey_df, spatial_graph),
        validate_work_hours_consistency(journey_df, worker_df),
    ]

    return results


def generate_quality_report(results: list[ValidationResult]) -> dict:
    """
    일별 품질 리포트 생성.

    Returns:
        {
            "overall_score": float (0~1),
            "overall_level": str (pass/warning/fail),
            "checks": [ValidationResult.to_dict(), ...],
            "summary": str,
        }
    """
    if not results:
        return {
            "overall_score": 0.0,
            "overall_level": "fail",
            "checks": [],
            "summary": "검증 결과 없음",
        }

    # 점수 계산 (PASS=1.0, WARNING=0.5, FAIL=0.0)
    level_scores = {
        ValidationLevel.PASS: 1.0,
        ValidationLevel.WARNING: 0.5,
        ValidationLevel.FAIL: 0.0,
    }

    scores = [level_scores[r.level] for r in results]
    overall_score = np.mean(scores)

    # 전체 등급
    if overall_score >= 0.8:
        overall_level = "pass"
    elif overall_score >= 0.5:
        overall_level = "warning"
    else:
        overall_level = "fail"

    # 요약 생성
    pass_count = sum(1 for r in results if r.level == ValidationLevel.PASS)
    warn_count = sum(1 for r in results if r.level == ValidationLevel.WARNING)
    fail_count = sum(1 for r in results if r.level == ValidationLevel.FAIL)

    summary_parts = []
    if pass_count > 0:
        summary_parts.append(f"통과 {pass_count}개")
    if warn_count > 0:
        summary_parts.append(f"주의 {warn_count}개")
    if fail_count > 0:
        summary_parts.append(f"실패 {fail_count}개")

    summary = " / ".join(summary_parts) if summary_parts else "검증 항목 없음"

    return {
        "overall_score": round(overall_score, 2),
        "overall_level": overall_level,
        "checks": [r.to_dict() for r in results],
        "summary": summary,
    }
