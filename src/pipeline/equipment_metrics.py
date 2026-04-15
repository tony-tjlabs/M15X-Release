"""
Equipment Metrics Calculator - Weekly BP-level OPR Aggregation
==============================================================
M15X FAB 건설현장 테이블 리프트(고소작업대) 가동률 계산 모듈.

고객 핵심 요구:
> "주 단위 BP별 가동률 평균" Data 요청 드립니다.
> PH4 와드 분출 시점부터 3월 2주차까지 자료가 필요합니다.

지표 정의:
- OPR (Operational Rate): (active_ratio >= 0.5인 분) / 600분 (10시간 고정)
  * 분모 기준: 실측 작업자 평균 근무시간 10시간(600분) — AccessLog 기반 확인값
  * active_ratio = activesignal_count / signal_count
  * T-Ward 진동 감지 원리:
    - 진동 있음(가동 중): 10초마다 신호 발신 → active_count = signal_count
    - 진동 없음(정지):   1분마다  신호 발신 → active_count = 0
    - 50% 이상 활성신호 = 해당 분은 "가동 중"으로 판정
  * 100%의 의미: 600분 중 600분 가동 — 현실적으로 불가능하므로 이상값으로 처리
- BP: Business Partner = company_name (업체)

Author: developer (agent)
Created: 2026-04-07
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import config as cfg
from src.pipeline.equipment_loader import (
    load_equipment_master,
    load_equipment_tward_daily,
    load_equipment_tward_weekly,
    aggregate_device_data_daily,
    detect_equipment_week_range,
)

logger = logging.getLogger(__name__)


# ============================================================
# Daily OPR Calculation
# ============================================================
def calc_equipment_daily_opr(
    tward_df: pd.DataFrame,
    master_df: pd.DataFrame,
    work_start_hour: int = cfg.WORK_START_HOUR,
    work_end_hour: int = cfg.WORK_END_HOUR,
) -> pd.DataFrame:
    """
    일별 장비별 가동률(OPR) 계산.

    OPR = (active_ratio >= 0.5 인 분) / 600분
    - 분모: 현장 실측 평균 근무시간 10시간(600분) 고정 기준
      → "하루 600분 중 몇 분이나 가동했는가"
      → 600분 초과 가동 불가 → OPR 100% = 물리적으로 불가능 → 센서 이상 판정
    - active_ratio = activesignal_count / signal_count (진동 감지 비율)
    - work_start_hour / work_end_hour 는 노이즈 필터용 (심야 데이터 제거)

    Args:
        tward_df: load_equipment_tward_daily() 결과
        master_df: load_equipment_master() 결과
        work_start_hour: 근무 시작 시간 필터 (기본: 5)
        work_end_hour: 근무 종료 시간 필터 (기본: 23)

    Returns:
        DataFrame with columns:
        - equipment_no: int
        - equipment_name: str
        - company_name: str
        - date: date
        - date_str: str (YYYYMMDD)
        - total_minutes: int (기록된 분 — 참고용)
        - active_minutes: int (가동 판정된 분)
        - opr: float (가동률 0~1, 분모=600분)
        - floor_main: str (가장 많이 머문 층)
    """
    if tward_df.empty:
        logger.warning("Empty tward_df provided")
        return pd.DataFrame()

    # 근무시간 필터
    df = tward_df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df = df[(df["hour"] >= work_start_hour) & (df["hour"] < work_end_hour)]

    if df.empty:
        logger.warning("No data within work hours")
        return pd.DataFrame()

    # 날짜 추출
    df["date"] = df["timestamp"].dt.date
    date_val = df["date"].iloc[0] if len(df) > 0 else None

    # 장비별 집계
    results = []
    for eq_no, group in df.groupby("equipment_no"):
        # 총 분 수 (고유 timestamp 수)
        total_minutes = group["timestamp"].dt.floor("min").nunique()

        # 활성 분 수: active_ratio >= 0.5 (진동센서 기반)
        # T-Ward 물리 원리:
        #   - 진동 있음(가동) → 10초마다 발신 → active_count ≈ signal_count (ratio ≈ 1.0)
        #   - 진동 없음(정지) → 1분마다 발신  → active_count = 0 (ratio = 0.0)
        #   → 50% 이상이면 해당 분을 "가동 중"으로 판정
        if "active_ratio" not in group.columns:
            group = group.copy()
            group["active_ratio"] = group["active_count"] / group["signal_count"].replace(0, 1)
        active_threshold = getattr(cfg, "EQUIPMENT_ACTIVE_THRESHOLD", 0.5)
        active_group = group[group["active_ratio"] >= active_threshold]
        active_minutes = active_group["timestamp"].dt.floor("min").nunique()

        # OPR 계산 — 분모: 600분 고정 (10시간 = 실측 평균 근무시간)
        OPR_DENOMINATOR = 600
        opr = min(active_minutes / OPR_DENOMINATOR, 1.0)

        # 가장 많이 머문 층
        floor_counts = group["floor"].value_counts()
        floor_main = floor_counts.index[0] if len(floor_counts) > 0 else ""

        results.append({
            "equipment_no": eq_no,
            "total_minutes": total_minutes,
            "active_minutes": active_minutes,
            "opr": opr,
            "floor_main": floor_main,
        })

    opr_df = pd.DataFrame(results)

    if opr_df.empty:
        return pd.DataFrame()

    # 마스터 정보 조인
    opr_df = opr_df.merge(
        master_df[["equipment_no", "equipment_name", "company_name"]],
        on="equipment_no",
        how="left"
    )

    # 날짜 추가
    opr_df["date"] = date_val
    opr_df["date_str"] = date_val.strftime("%Y%m%d") if date_val else ""

    # 컬럼 순서 정리
    result_cols = [
        "equipment_no", "equipment_name", "company_name",
        "date", "date_str",
        "total_minutes", "active_minutes", "opr",
        "floor_main"
    ]

    return opr_df[result_cols].sort_values("equipment_no").reset_index(drop=True)


# ============================================================
# Weekly BP OPR Aggregation (Customer Core Requirement)
# ============================================================
def calc_weekly_bp_opr(daily_opr_df: pd.DataFrame) -> pd.DataFrame:
    """
    *** 고객 핵심 요구: 주간 BP별 가동률 평균 ***

    BP = company_name (업체)
    집계: 주차 x BP별 평균 OPR

    Args:
        daily_opr_df: calc_equipment_daily_opr() 결과를 여러 날짜 합친 DataFrame

    Returns:
        DataFrame with columns:
        - year: int
        - week: int (ISO 주차)
        - week_label: str (예: "2026-W10")
        - company_name: str (BP)
        - equipment_count: int (해당 주 운영 장비 수)
        - avg_opr: float (평균 가동률)
        - median_opr: float (중앙값)
        - min_opr: float
        - max_opr: float
        - std_opr: float (표준편차)
    """
    if daily_opr_df.empty:
        logger.warning("Empty daily_opr_df provided")
        return pd.DataFrame()

    df = daily_opr_df.copy()

    # ★ OPR >= 99% 센서 이상 장비 제외 (통계 왜곡 방지)
    opr_threshold = getattr(cfg, "OPR_SENSOR_ANOMALY_THRESHOLD", 0.99)
    anomalous_mask = df["opr"] >= opr_threshold
    n_excluded = anomalous_mask.sum()
    if n_excluded > 0:
        excluded_equipments = df.loc[anomalous_mask, "equipment_no"].unique().tolist()
        logger.info(
            "calc_weekly_bp_opr: OPR 필터 — %d건 제외 (장비 %d대, OPR >= %.0f%%)",
            n_excluded, len(excluded_equipments), opr_threshold * 100,
        )
        df = df[~anomalous_mask].copy()

    # ISO 주차 추가
    df["iso_calendar"] = df["date"].apply(lambda d: d.isocalendar() if d else (None, None, None))
    df["year"] = df["iso_calendar"].apply(lambda x: x[0])
    df["week"] = df["iso_calendar"].apply(lambda x: x[1])

    # 주 x BP별 집계
    results = []
    for (year, week, company), group in df.groupby(["year", "week", "company_name"]):
        # 장비 수 (고유 equipment_no)
        equipment_count = group["equipment_no"].nunique()

        # 장비별 주간 평균 OPR 먼저 계산 (일별 → 장비별 평균 → BP 평균)
        equip_opr = group.groupby("equipment_no")["opr"].mean()

        results.append({
            "year": int(year),
            "week": int(week),
            "week_label": f"{year}-W{week:02d}",
            "company_name": company,
            "equipment_count": equipment_count,
            "avg_opr": equip_opr.mean(),
            "median_opr": equip_opr.median(),
            "min_opr": equip_opr.min(),
            "max_opr": equip_opr.max(),
            "std_opr": equip_opr.std() if len(equip_opr) > 1 else 0.0,
        })

    result = pd.DataFrame(results)

    if result.empty:
        return pd.DataFrame()

    # 정렬: 주차 → BP
    result = result.sort_values(["year", "week", "company_name"]).reset_index(drop=True)

    logger.info(
        f"Weekly BP OPR calculated: {len(result)} rows, "
        f"{result['week_label'].nunique()} weeks, "
        f"{result['company_name'].nunique()} companies"
    )

    return result


def calc_weekly_overall_opr(daily_opr_df: pd.DataFrame) -> pd.DataFrame:
    """
    주간 전체 가동률 요약 (BP 무관).

    Returns:
        DataFrame with columns:
        - year, week, week_label
        - total_equipment: int
        - avg_opr, median_opr, min_opr, max_opr, std_opr
    """
    if daily_opr_df.empty:
        return pd.DataFrame()

    df = daily_opr_df.copy()

    # ★ OPR >= 99% 센서 이상 장비 제외
    opr_threshold = getattr(cfg, "OPR_SENSOR_ANOMALY_THRESHOLD", 0.99)
    anomalous_mask = df["opr"] >= opr_threshold
    n_excluded = anomalous_mask.sum()
    if n_excluded > 0:
        excluded_equipments = df.loc[anomalous_mask, "equipment_no"].unique().tolist()
        logger.info(
            "calc_weekly_overall_opr: OPR 필터 — %d건 제외 (장비 %d대, OPR >= %.0f%%)",
            n_excluded, len(excluded_equipments), opr_threshold * 100,
        )
        df = df[~anomalous_mask].copy()

    df["iso_calendar"] = df["date"].apply(lambda d: d.isocalendar() if d else (None, None, None))
    df["year"] = df["iso_calendar"].apply(lambda x: x[0])
    df["week"] = df["iso_calendar"].apply(lambda x: x[1])

    results = []
    for (year, week), group in df.groupby(["year", "week"]):
        equip_count = group["equipment_no"].nunique()
        equip_opr = group.groupby("equipment_no")["opr"].mean()

        results.append({
            "year": int(year),
            "week": int(week),
            "week_label": f"{year}-W{week:02d}",
            "total_equipment": equip_count,
            "avg_opr": equip_opr.mean(),
            "median_opr": equip_opr.median(),
            "min_opr": equip_opr.min(),
            "max_opr": equip_opr.max(),
            "std_opr": equip_opr.std() if len(equip_opr) > 1 else 0.0,
        })

    return pd.DataFrame(results).sort_values(["year", "week"]).reset_index(drop=True)


# ============================================================
# Floor Distribution Analysis
# ============================================================
def calc_weekly_floor_distribution(daily_opr_df: pd.DataFrame) -> pd.DataFrame:
    """
    주간 층별 장비 분포.

    Returns:
        DataFrame with columns:
        - year: int
        - week: int
        - week_label: str
        - floor_name: str
        - equipment_count: int
        - avg_opr: float
    """
    if daily_opr_df.empty:
        return pd.DataFrame()

    df = daily_opr_df.copy()
    df["iso_calendar"] = df["date"].apply(lambda d: d.isocalendar() if d else (None, None, None))
    df["year"] = df["iso_calendar"].apply(lambda x: x[0])
    df["week"] = df["iso_calendar"].apply(lambda x: x[1])

    results = []
    for (year, week, floor), group in df.groupby(["year", "week", "floor_main"]):
        equip_count = group["equipment_no"].nunique()
        equip_opr = group.groupby("equipment_no")["opr"].mean()

        results.append({
            "year": int(year),
            "week": int(week),
            "week_label": f"{year}-W{week:02d}",
            "floor_name": floor if floor else "Unknown",
            "equipment_count": equip_count,
            "avg_opr": equip_opr.mean(),
        })

    result = pd.DataFrame(results)

    if result.empty:
        return pd.DataFrame()

    return result.sort_values(["year", "week", "floor_name"]).reset_index(drop=True)


# ============================================================
# Single Date Summary
# ============================================================
def calc_equipment_summary(date_str: str) -> dict[str, Any]:
    """
    단일 날짜 장비 요약 리포트.

    Returns:
        {
            "date": str,
            "total_equipment": int,
            "working_count": int,
            "avg_opr": float,
            "median_opr": float,
            "top_bp": str (가동률 최고 업체),
            "top_bp_opr": float,
            "bottom_bp": str (가동률 최저 업체),
            "bottom_bp_opr": float,
            "floor_distribution": dict[str, int],
        }
    """
    logger.info(f"Calculating equipment summary for {date_str}")

    # 데이터 로드
    master_df = load_equipment_master()
    tward_df = load_equipment_tward_daily(date_str)

    if tward_df.empty:
        return {
            "date": date_str,
            "total_equipment": len(master_df),
            "working_count": 0,
            "avg_opr": 0.0,
            "median_opr": 0.0,
            "top_bp": "",
            "top_bp_opr": 0.0,
            "bottom_bp": "",
            "bottom_bp_opr": 0.0,
            "floor_distribution": {},
        }

    # OPR 계산
    opr_df = calc_equipment_daily_opr(tward_df, master_df)

    if opr_df.empty:
        return {
            "date": date_str,
            "total_equipment": len(master_df),
            "working_count": 0,
            "avg_opr": 0.0,
            "median_opr": 0.0,
            "top_bp": "",
            "top_bp_opr": 0.0,
            "bottom_bp": "",
            "bottom_bp_opr": 0.0,
            "floor_distribution": {},
        }

    # 전체 통계
    working_count = opr_df["equipment_no"].nunique()
    avg_opr = opr_df["opr"].mean()
    median_opr = opr_df["opr"].median()

    # BP별 평균 OPR
    bp_opr = opr_df.groupby("company_name")["opr"].mean().sort_values(ascending=False)

    top_bp = bp_opr.index[0] if len(bp_opr) > 0 else ""
    top_bp_opr = bp_opr.iloc[0] if len(bp_opr) > 0 else 0.0
    bottom_bp = bp_opr.index[-1] if len(bp_opr) > 0 else ""
    bottom_bp_opr = bp_opr.iloc[-1] if len(bp_opr) > 0 else 0.0

    # 층별 분포
    floor_dist = opr_df["floor_main"].value_counts().to_dict()

    return {
        "date": date_str,
        "total_equipment": len(master_df),
        "working_count": working_count,
        "avg_opr": float(avg_opr),
        "median_opr": float(median_opr),
        "top_bp": top_bp,
        "top_bp_opr": float(top_bp_opr),
        "bottom_bp": bottom_bp,
        "bottom_bp_opr": float(bottom_bp_opr),
        "floor_distribution": floor_dist,
    }


# ============================================================
# Batch Processing
# ============================================================
def process_all_weeks(
    sector_id: str = "M15X_SKHynix",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전체 기간의 주간 BP별 가동률 일괄 처리.

    결과를 data/equipment/{sector_id}/weekly/ 에 Parquet으로 저장.

    Returns:
        (weekly_bp_opr_df, weekly_overall_df, weekly_floor_df) tuple
    """
    logger.info(f"Processing all weeks for {sector_id}")

    # 주차 범위 탐지
    weeks = detect_equipment_week_range()
    if not weeks:
        logger.warning("No weeks detected in equipment data")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info(f"Detected {len(weeks)} weeks: {weeks[0]} ~ {weeks[-1]}")

    # 마스터 로드 (1회)
    master_df = load_equipment_master()

    # 일별 OPR 수집
    all_daily_opr = []

    for year, week in weeks:
        logger.info(f"Processing {year}-W{week:02d}")

        # 해당 주의 날짜들 계산
        week_dates = _get_dates_in_week(year, week)

        for date_val in week_dates:
            date_str = date_val.strftime("%Y%m%d")
            tward_df = load_equipment_tward_daily(date_str)

            if tward_df.empty:
                continue

            daily_opr = calc_equipment_daily_opr(tward_df, master_df)
            if not daily_opr.empty:
                all_daily_opr.append(daily_opr)

    if not all_daily_opr:
        logger.warning("No daily OPR data collected")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 전체 일별 OPR 합치기
    daily_opr_df = pd.concat(all_daily_opr, ignore_index=True)
    logger.info(f"Total daily OPR records: {len(daily_opr_df)}")

    # 주간 집계
    weekly_bp_opr = calc_weekly_bp_opr(daily_opr_df)
    weekly_overall = calc_weekly_overall_opr(daily_opr_df)
    weekly_floor = calc_weekly_floor_distribution(daily_opr_df)

    # 저장
    if save:
        paths = cfg.get_sector_paths(sector_id)
        weekly_dir = paths["equipment_weekly"]
        weekly_dir.mkdir(parents=True, exist_ok=True)

        # BP별 OPR
        bp_path = weekly_dir / "weekly_bp_opr.parquet"
        weekly_bp_opr.to_parquet(bp_path, index=False)
        logger.info(f"Saved weekly BP OPR to {bp_path}")

        # 전체 OPR
        overall_path = weekly_dir / "weekly_overall_opr.parquet"
        weekly_overall.to_parquet(overall_path, index=False)
        logger.info(f"Saved weekly overall OPR to {overall_path}")

        # 층별 분포
        floor_path = weekly_dir / "weekly_floor_distribution.parquet"
        weekly_floor.to_parquet(floor_path, index=False)
        logger.info(f"Saved weekly floor distribution to {floor_path}")

        # 마스터도 저장
        master_path = paths["equipment_master"]
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_parquet(master_path, index=False)
        logger.info(f"Saved equipment master to {master_path}")

    return weekly_bp_opr, weekly_overall, weekly_floor


def _get_dates_in_week(year: int, week: int) -> list[date]:
    """ISO 주차의 모든 날짜 반환 (월~일)."""
    # ISO 주차 1의 첫 번째 목요일을 기준으로 계산
    jan4 = date(year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday())
    start_of_target_week = start_of_week1 + timedelta(weeks=week - 1)

    return [start_of_target_week + timedelta(days=i) for i in range(7)]


# ============================================================
# Cache Management
# ============================================================
def load_cached_weekly_bp_opr(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """캐시된 주간 BP별 OPR 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_weekly"] / "weekly_bp_opr.parquet"

    if not path.exists():
        logger.warning(f"Weekly BP OPR cache not found: {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_cached_weekly_overall_opr(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """캐시된 주간 전체 OPR 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_weekly"] / "weekly_overall_opr.parquet"

    if not path.exists():
        logger.warning(f"Weekly overall OPR cache not found: {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_cached_weekly_floor(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """캐시된 주간 층별 분포 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_weekly"] / "weekly_floor_distribution.parquet"

    if not path.exists():
        logger.warning(f"Weekly floor distribution cache not found: {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_cached_equipment_master(sector_id: str = "M15X_SKHynix") -> pd.DataFrame:
    """캐시된 장비 마스터 로드."""
    paths = cfg.get_sector_paths(sector_id)
    path = paths["equipment_master"]

    if not path.exists():
        logger.warning(f"Equipment master cache not found: {path}")
        return pd.DataFrame()

    return pd.read_parquet(path)


# ============================================================
# Export for Customer Report
# ============================================================
def export_weekly_bp_opr_to_excel(
    sector_id: str = "M15X_SKHynix",
    output_path: Path | None = None,
) -> Path:
    """
    주간 BP별 가동률을 엑셀로 내보내기.

    고객 납품용 형식:
    - Sheet1: 주간 BP별 가동률
    - Sheet2: 주간 전체 요약
    - Sheet3: 주간 층별 분포
    """
    weekly_bp = load_cached_weekly_bp_opr(sector_id)
    weekly_overall = load_cached_weekly_overall_opr(sector_id)
    weekly_floor = load_cached_weekly_floor(sector_id)

    if weekly_bp.empty:
        raise ValueError("No weekly BP OPR data available. Run process_all_weeks() first.")

    if output_path is None:
        paths = cfg.get_sector_paths(sector_id)
        output_path = paths["equipment_dir"] / f"Weekly_BP_OPR_{sector_id}.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # BP별 OPR (퍼센트로 변환)
        bp_export = weekly_bp.copy()
        for col in ["avg_opr", "median_opr", "min_opr", "max_opr", "std_opr"]:
            if col in bp_export.columns:
                bp_export[col] = (bp_export[col] * 100).round(1)
        bp_export.to_excel(writer, sheet_name="BP별 가동률", index=False)

        # 전체 요약
        if not weekly_overall.empty:
            overall_export = weekly_overall.copy()
            for col in ["avg_opr", "median_opr", "min_opr", "max_opr", "std_opr"]:
                if col in overall_export.columns:
                    overall_export[col] = (overall_export[col] * 100).round(1)
            overall_export.to_excel(writer, sheet_name="전체 요약", index=False)

        # 층별 분포
        if not weekly_floor.empty:
            floor_export = weekly_floor.copy()
            if "avg_opr" in floor_export.columns:
                floor_export["avg_opr"] = (floor_export["avg_opr"] * 100).round(1)
            floor_export.to_excel(writer, sheet_name="층별 분포", index=False)

    logger.info(f"Exported weekly BP OPR to {output_path}")
    return output_path


# ============================================================
# Test / Debug
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 마스터 로드
    master = load_equipment_master()
    print(f"\n=== Equipment Master ===")
    print(f"Working equipment: {len(master)}")
    print(f"Companies: {master['company_name'].nunique()}")

    # 단일 날짜 테스트 (있는 날짜로 변경)
    test_date = "20260310"
    print(f"\n=== Daily Summary for {test_date} ===")
    summary = calc_equipment_summary(test_date)
    for k, v in summary.items():
        print(f"  {k}: {v}")
