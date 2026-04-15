"""
Equipment Data Loader - Table Lift Master + Location/Detection Data
====================================================================
M15X FAB 건설현장 테이블 리프트(고소작업대) 데이터 로더.

파일 구조:
- TableLiftInfo: 628대 장비 마스터 정보
- TableLift_TWardData: 7.2M rows, 장비 위치 (1분 단위)
- TableLift_DeviceData: 27.8M rows, RSSI 감지 (초단위) - 청크 집계만

Author: developer (agent)
Created: 2026-04-07
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
CHUNK_SIZE = 500_000  # 50만 행씩 처리
DEVICE_CHUNK_SIZE = 1_000_000  # DeviceData는 100만 행씩 (더 큰 파일)


# ============================================================
# Helper Functions
# ============================================================
def _parse_timezone_datetime(series: pd.Series) -> pd.Series:
    """
    +0900 timezone suffix를 제거하고 datetime으로 파싱.

    예: "2026-03-13 23:58:00.000 +0900" -> datetime
    """
    return pd.to_datetime(
        series.astype(str)
        .str.replace(r"\s*[+-]\d{2}:?\d{2}$", "", regex=True)
        .str.replace(r"\.\d+$", "", regex=True),
        errors="coerce"
    )


def _normalize_company_name(name: str) -> str:
    """
    company_name 정규화: _PH4 접미사 제거.

    예: "세보엠이씨_F03_PH4" -> "세보엠이씨_F03"
    """
    if pd.isna(name):
        return ""
    return re.sub(r"_PH\d+$", "", str(name).strip())


def _extract_equipment_number(name: str) -> str:
    """
    equipment_name에서 #번호 추출.

    예: "세보엠이씨_F03_PH4 #28" -> "28"
    """
    if pd.isna(name):
        return ""
    match = re.search(r"#(\d+)$", str(name).strip())
    return match.group(1) if match else ""


# ============================================================
# Master Data Loader
# ============================================================
def load_equipment_master(
    file_path: Path | None = None,
    working_only: bool = True,
) -> pd.DataFrame:
    """
    TableLiftInfo 로드 + 정규화.

    Args:
        file_path: CSV 파일 경로 (기본: config에서 로드)
        working_only: True면 status='working'만 필터 (기본: True)

    Returns:
        DataFrame with columns:
        - equipment_no: int (고유 ID)
        - code: str (장비 코드)
        - equipment_name: str (장비명)
        - equipment_num: str (#번호 추출)
        - company_name: str (업체명, _PH4 제거됨)
        - company_name_raw: str (원본 업체명)
        - status: str (working/out)
        - manufacturer: str (제조사)
        - specification: str (규격)
        - inbound_date: date (반입일)
        - outbound_date: date (반출일, None 가능)
    """
    if file_path is None:
        file_path = cfg.RAW_EQUIP_INFO_FILE

    logger.info(f"Loading equipment master from {file_path}")

    df = pd.read_csv(file_path, encoding=cfg.RAW_ENCODING, low_memory=False)

    # 컬럼명 소문자 정규화
    df.columns = df.columns.str.lower().str.strip()

    # 필수 컬럼 확인
    required = ["equipment_no", "equipment_name", "company_name", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 시간 파싱
    if "inbound_datetime" in df.columns:
        df["inbound_date"] = _parse_timezone_datetime(df["inbound_datetime"]).dt.date
    else:
        df["inbound_date"] = None

    if "outbound_datetime" in df.columns:
        df["outbound_date"] = _parse_timezone_datetime(df["outbound_datetime"]).dt.date
    else:
        df["outbound_date"] = None

    # company_name 정규화 (원본 보존)
    df["company_name_raw"] = df["company_name"]
    df["company_name"] = df["company_name_raw"].apply(_normalize_company_name)

    # equipment_num 추출
    df["equipment_num"] = df["equipment_name"].apply(_extract_equipment_number)

    # status 정규화
    df["status"] = df["status"].str.lower().str.strip()

    # working_only 필터
    if working_only:
        df = df[df["status"] == "working"].copy()
        logger.info(f"Filtered to working equipment: {len(df)} (from total)")

    # 컬럼 선택 및 정렬
    result_cols = [
        "equipment_no", "code", "equipment_name", "equipment_num",
        "company_name", "company_name_raw", "status",
        "manufacturer", "specification",
        "inbound_date", "outbound_date"
    ]
    # 누락된 컬럼 처리
    for col in result_cols:
        if col not in df.columns:
            df[col] = None

    result = df[result_cols].copy()
    result = result.sort_values("equipment_no").reset_index(drop=True)

    logger.info(
        f"Equipment master loaded: {len(result)} equipment, "
        f"{result['company_name'].nunique()} companies"
    )

    return result


# ============================================================
# TWard Data Loader (Equipment Location)
# ============================================================
def iter_equipment_tward(
    file_path: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """
    TableLift_TWardData 청크 단위 순회.

    Yields:
        DataFrame chunks with parsed timestamps
    """
    if file_path is None:
        file_path = cfg.RAW_EQUIP_TWARD_FILE

    for chunk in pd.read_csv(
        file_path,
        encoding=cfg.RAW_ENCODING,
        low_memory=False,
        chunksize=chunk_size,
    ):
        # 컬럼명 소문자 정규화
        chunk.columns = chunk.columns.str.lower().str.strip()

        # 시간 파싱
        if "time" in chunk.columns:
            chunk["timestamp"] = _parse_timezone_datetime(chunk["time"])
            chunk["date"] = chunk["timestamp"].dt.date
            chunk["date_str"] = chunk["timestamp"].dt.strftime("%Y%m%d")

        yield chunk


def load_equipment_tward_daily(
    date_str: str,
    file_path: Path | None = None,
) -> pd.DataFrame:
    """
    특정 날짜의 장비 TWardData 추출 (청크 기반).

    Args:
        date_str: 날짜 문자열 (YYYYMMDD)
        file_path: CSV 파일 경로 (기본: config에서 로드)

    Returns:
        DataFrame with columns:
        - equipment_no: int
        - timestamp: datetime
        - equipment_name: str
        - building: str
        - level: str
        - place: str
        - x: float
        - y: float
        - signal_count: int
        - active_count: int
        - active_ratio: float
    """
    logger.info(f"Loading equipment TWard data for {date_str}")

    chunks = []
    total_rows = 0

    for chunk in iter_equipment_tward(file_path):
        # 날짜 필터
        filtered = chunk[chunk["date_str"] == date_str].copy()
        if len(filtered) > 0:
            chunks.append(filtered)
            total_rows += len(filtered)

    if not chunks:
        logger.warning(f"No equipment TWard data found for {date_str}")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # active_ratio 계산 (진동센서 기반 활성 비율)
    # T-Ward 물리 원리:
    #   - 진동 있음(가동) → 10초마다 발신, activesignal_count = signal_count → ratio ≈ 1.0
    #   - 진동 없음(정지) → 1분마다  발신, activesignal_count = 0             → ratio = 0.0
    #   → 여러 S-Ward에서 수신하므로 signal_count는 복수가 될 수 있음
    #   → ratio >= 0.5 이면 해당 분을 "가동 중"으로 판정 (equipment_metrics 기준)
    df["signal_count"] = pd.to_numeric(df.get("signal_count", 0), errors="coerce").fillna(0).astype(int)
    df["active_count"] = pd.to_numeric(df.get("activesignal_count", 0), errors="coerce").fillna(0).astype(int)
    df["active_ratio"] = df.apply(
        lambda r: r["active_count"] / r["signal_count"] if r["signal_count"] > 0 else 0.0,
        axis=1
    )

    # 컬럼 정리
    df = df.rename(columns={
        "level": "floor",
    })

    result_cols = [
        "equipment_no", "timestamp", "equipment_name",
        "building", "floor", "place", "x", "y",
        "signal_count", "active_count", "active_ratio"
    ]

    for col in result_cols:
        if col not in df.columns:
            df[col] = None

    result = df[result_cols].copy()
    result = result.sort_values(["equipment_no", "timestamp"]).reset_index(drop=True)

    logger.info(f"Loaded {len(result)} TWard records for {date_str}")
    return result


def load_equipment_tward_weekly(
    year: int,
    week: int,
    file_path: Path | None = None,
) -> pd.DataFrame:
    """
    특정 주의 장비 TWardData 추출 (7일 합산).

    Args:
        year: 연도 (예: 2026)
        week: ISO 주차 (1-53)
        file_path: CSV 파일 경로

    Returns:
        DataFrame with columns: 위와 동일 + date 컬럼
    """
    logger.info(f"Loading equipment TWard data for {year}-W{week:02d}")

    chunks = []
    total_rows = 0

    for chunk in iter_equipment_tward(file_path):
        # ISO 주차 필터
        chunk["iso_year"] = chunk["timestamp"].dt.isocalendar().year
        chunk["iso_week"] = chunk["timestamp"].dt.isocalendar().week

        filtered = chunk[(chunk["iso_year"] == year) & (chunk["iso_week"] == week)].copy()
        if len(filtered) > 0:
            chunks.append(filtered)
            total_rows += len(filtered)

    if not chunks:
        logger.warning(f"No equipment TWard data found for {year}-W{week:02d}")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # active_ratio 계산 (진동센서 기반 활성 비율)
    # T-Ward 물리 원리:
    #   - 진동 있음(가동) → 10초마다 발신, activesignal_count = signal_count → ratio ≈ 1.0
    #   - 진동 없음(정지) → 1분마다  발신, activesignal_count = 0             → ratio = 0.0
    df["signal_count"] = pd.to_numeric(df.get("signal_count", 0), errors="coerce").fillna(0).astype(int)
    df["active_count"] = pd.to_numeric(df.get("activesignal_count", 0), errors="coerce").fillna(0).astype(int)
    df["active_ratio"] = df.apply(
        lambda r: r["active_count"] / r["signal_count"] if r["signal_count"] > 0 else 0.0,
        axis=1
    )

    # 컬럼 정리
    df = df.rename(columns={"level": "floor"})

    result_cols = [
        "equipment_no", "timestamp", "date", "equipment_name",
        "building", "floor", "place", "x", "y",
        "signal_count", "active_count", "active_ratio"
    ]

    for col in result_cols:
        if col not in df.columns:
            df[col] = None

    result = df[result_cols].copy()
    result = result.sort_values(["equipment_no", "timestamp"]).reset_index(drop=True)

    logger.info(f"Loaded {len(result)} TWard records for {year}-W{week:02d}")
    return result


# ============================================================
# Device Data Aggregator (RSSI Detection)
# ============================================================
def aggregate_device_data_daily(
    date_str: str,
    file_path: Path | None = None,
) -> pd.DataFrame:
    """
    DeviceData를 일별 장비별 요약으로 집계.

    **청크 기반 처리 필수** (27.8M rows, 2.5GB -> 전체 로드 금지!)

    집계 항목:
    - equipment별 총 감지 건수
    - ttag_work_status=1 비율 (가동 시간 비율)
    - 평균 rssi, 배터리, 기압

    Args:
        date_str: 날짜 문자열 (YYYYMMDD)
        file_path: CSV 파일 경로

    Returns:
        DataFrame with columns:
        - equipment_name: str
        - total_records: int
        - active_records: int
        - active_ratio_device: float (ttag_work_status=1 비율)
        - avg_rssi: float
        - avg_batt: float
        - avg_pressure: float
    """
    if file_path is None:
        file_path = cfg.RAW_EQUIP_DEVICE_FILE

    logger.info(f"Aggregating device data for {date_str} (chunk-based)")

    # 장비별 집계 딕셔너리
    agg_dict: dict[str, dict] = {}
    chunks_processed = 0

    for chunk in pd.read_csv(
        file_path,
        encoding=cfg.RAW_ENCODING,
        low_memory=False,
        chunksize=DEVICE_CHUNK_SIZE,
    ):
        chunks_processed += 1

        # 컬럼명 소문자 정규화
        chunk.columns = chunk.columns.str.lower().str.strip()

        # 시간 파싱 및 날짜 필터
        if "insert_datetime" in chunk.columns:
            chunk["timestamp"] = _parse_timezone_datetime(chunk["insert_datetime"])
            chunk["date_str"] = chunk["timestamp"].dt.strftime("%Y%m%d")
            chunk = chunk[chunk["date_str"] == date_str]
        else:
            continue

        if len(chunk) == 0:
            continue

        # 장비별 집계
        for eq_name, group in chunk.groupby("equipment_name"):
            if eq_name not in agg_dict:
                agg_dict[eq_name] = {
                    "total_records": 0,
                    "active_records": 0,
                    "rssi_sum": 0.0,
                    "batt_sum": 0.0,
                    "pressure_sum": 0.0,
                }

            agg = agg_dict[eq_name]
            agg["total_records"] += len(group)

            # ttag_work_status=1 카운트
            if "ttag_work_status" in group.columns:
                agg["active_records"] += (group["ttag_work_status"] == cfg.EQUIPMENT_WORK_STATUS_ON).sum()

            # 평균 계산용 합산
            if "rssi" in group.columns:
                agg["rssi_sum"] += group["rssi"].sum()
            if "ttag_batt" in group.columns:
                agg["batt_sum"] += group["ttag_batt"].sum()
            if "pressure" in group.columns:
                agg["pressure_sum"] += group["pressure"].sum()

        if chunks_processed % 5 == 0:
            logger.debug(f"Processed {chunks_processed} chunks, {len(agg_dict)} equipment found")

    if not agg_dict:
        logger.warning(f"No device data found for {date_str}")
        return pd.DataFrame()

    # DataFrame 변환
    rows = []
    for eq_name, agg in agg_dict.items():
        total = agg["total_records"]
        rows.append({
            "equipment_name": eq_name,
            "total_records": total,
            "active_records": agg["active_records"],
            "active_ratio_device": agg["active_records"] / total if total > 0 else 0.0,
            "avg_rssi": agg["rssi_sum"] / total if total > 0 else 0.0,
            "avg_batt": agg["batt_sum"] / total if total > 0 else 0.0,
            "avg_pressure": agg["pressure_sum"] / total if total > 0 else 0.0,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("equipment_name").reset_index(drop=True)

    logger.info(
        f"Device data aggregated for {date_str}: "
        f"{len(result)} equipment, {chunks_processed} chunks processed"
    )

    return result


def aggregate_device_data_weekly(
    year: int,
    week: int,
    file_path: Path | None = None,
) -> pd.DataFrame:
    """
    DeviceData를 주간 장비별 요약으로 집계.

    Args:
        year: 연도
        week: ISO 주차
        file_path: CSV 파일 경로

    Returns:
        DataFrame (aggregate_device_data_daily와 동일 구조)
    """
    if file_path is None:
        file_path = cfg.RAW_EQUIP_DEVICE_FILE

    logger.info(f"Aggregating device data for {year}-W{week:02d} (chunk-based)")

    agg_dict: dict[str, dict] = {}
    chunks_processed = 0

    for chunk in pd.read_csv(
        file_path,
        encoding=cfg.RAW_ENCODING,
        low_memory=False,
        chunksize=DEVICE_CHUNK_SIZE,
    ):
        chunks_processed += 1

        chunk.columns = chunk.columns.str.lower().str.strip()

        if "insert_datetime" in chunk.columns:
            chunk["timestamp"] = _parse_timezone_datetime(chunk["insert_datetime"])
            chunk["iso_year"] = chunk["timestamp"].dt.isocalendar().year
            chunk["iso_week"] = chunk["timestamp"].dt.isocalendar().week
            chunk = chunk[(chunk["iso_year"] == year) & (chunk["iso_week"] == week)]
        else:
            continue

        if len(chunk) == 0:
            continue

        for eq_name, group in chunk.groupby("equipment_name"):
            if eq_name not in agg_dict:
                agg_dict[eq_name] = {
                    "total_records": 0,
                    "active_records": 0,
                    "rssi_sum": 0.0,
                    "batt_sum": 0.0,
                    "pressure_sum": 0.0,
                }

            agg = agg_dict[eq_name]
            agg["total_records"] += len(group)

            if "ttag_work_status" in group.columns:
                agg["active_records"] += (group["ttag_work_status"] == cfg.EQUIPMENT_WORK_STATUS_ON).sum()

            if "rssi" in group.columns:
                agg["rssi_sum"] += group["rssi"].sum()
            if "ttag_batt" in group.columns:
                agg["batt_sum"] += group["ttag_batt"].sum()
            if "pressure" in group.columns:
                agg["pressure_sum"] += group["pressure"].sum()

    if not agg_dict:
        logger.warning(f"No device data found for {year}-W{week:02d}")
        return pd.DataFrame()

    rows = []
    for eq_name, agg in agg_dict.items():
        total = agg["total_records"]
        rows.append({
            "equipment_name": eq_name,
            "total_records": total,
            "active_records": agg["active_records"],
            "active_ratio_device": agg["active_records"] / total if total > 0 else 0.0,
            "avg_rssi": agg["rssi_sum"] / total if total > 0 else 0.0,
            "avg_batt": agg["batt_sum"] / total if total > 0 else 0.0,
            "avg_pressure": agg["pressure_sum"] / total if total > 0 else 0.0,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("equipment_name").reset_index(drop=True)

    logger.info(
        f"Device data aggregated for {year}-W{week:02d}: "
        f"{len(result)} equipment, {chunks_processed} chunks processed"
    )

    return result


# ============================================================
# Utility Functions
# ============================================================
def detect_equipment_date_range(
    file_path: Path | None = None,
) -> tuple[str, str, list[str]]:
    """
    장비 TWardData에서 날짜 범위 탐지.

    Returns:
        (min_date, max_date, all_dates) tuple
    """
    if file_path is None:
        file_path = cfg.RAW_EQUIP_TWARD_FILE

    logger.info(f"Detecting date range from {file_path}")

    dates = set()
    for chunk in iter_equipment_tward(file_path):
        chunk_dates = chunk["date_str"].unique()
        dates.update(chunk_dates)

    sorted_dates = sorted(dates)

    if not sorted_dates:
        return "", "", []

    return sorted_dates[0], sorted_dates[-1], sorted_dates


def detect_equipment_week_range(
    file_path: Path | None = None,
) -> list[tuple[int, int]]:
    """
    장비 TWardData에서 주차 범위 탐지.

    Returns:
        [(year, week), ...] 리스트 (int 타입)
    """
    if file_path is None:
        file_path = cfg.RAW_EQUIP_TWARD_FILE

    logger.info(f"Detecting week range from {file_path}")

    weeks = set()
    for chunk in iter_equipment_tward(file_path):
        chunk["iso_year"] = chunk["timestamp"].dt.isocalendar().year
        chunk["iso_week"] = chunk["timestamp"].dt.isocalendar().week
        # numpy 타입을 int로 변환
        week_tuples = [(int(y), int(w)) for y, w in zip(chunk["iso_year"], chunk["iso_week"])]
        weeks.update(week_tuples)

    return sorted(weeks)


# ============================================================
# Test / Debug
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 마스터 로드 테스트
    master = load_equipment_master()
    print(f"\n=== Equipment Master ===")
    print(f"Working equipment: {len(master)}")
    print(f"Companies: {master['company_name'].nunique()}")
    print(f"\nTop 5 companies by equipment count:")
    print(master.groupby("company_name").size().sort_values(ascending=False).head())
    print(f"\nSample rows:")
    print(master.head())
