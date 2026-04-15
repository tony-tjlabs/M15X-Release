"""
Bulk Loader - 대용량 단일 CSV 날짜별 분리 모듈
=============================================
M15X 데이터는 전체 기간이 하나의 CSV 파일에 통합되어 있음.
이 모듈은 대용량 단일 CSV를 날짜별로 분리하는 유틸리티.

데이터 특성:
- TWardData: 11.3M rows, cp949, Time에 +0900 timezone
- AccessLog: 133K rows, cp949, Entry_time에 +0900, Exit_time은 naive

사용 예시:
>>> from src.pipeline.bulk_loader import detect_date_range, load_daily_data
>>> dates = detect_date_range(cfg.RAW_TWARD_SINGLE_FILE, "Time")
>>> df, meta = load_daily_data(dates[0])
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

# 청크 크기 (메모리 vs I/O 트레이드오프)
CHUNK_SIZE = 500_000  # 50만 행 = ~40MB

logger = logging.getLogger(__name__)


# =============================================================================
# Time Parsing Utilities
# =============================================================================

def parse_time_col(series: pd.Series, col_name: str = "Time") -> pd.Series:
    """
    시간 컬럼 파싱 (timezone strip + 다양한 형식 처리).

    지원 형식:
    - "2026-03-09 23:59:00.000 +0900" (TWardData Time)
    - "2026-02-09 08:11:35.000 +0900" (AccessLog Entry_time)
    - "2026-02-09 15:55:25" (AccessLog Exit_time - naive KST)
    - "2026.3.9 17:24" (AccessLog Exit_time 불규칙 형식 - 대비용)

    Args:
        series: 시간 문자열 Series
        col_name: 컬럼 이름 (로깅용)

    Returns:
        pd.Series: datetime64[ns] 타입 Series
    """
    if series.empty:
        return pd.Series(dtype="datetime64[ns]")

    # 문자열 변환
    str_series = series.astype(str)

    # Step 1: timezone offset 제거 (+0900, +09:00, -05:00 등)
    str_series = str_series.str.replace(
        r"\s*[+-]\d{2}:?\d{2}$", "", regex=True
    )

    # Step 2: 밀리초 제거 (.000 등)
    str_series = str_series.str.replace(r"\.\d+$", "", regex=True)

    # Step 3: 점(.) 구분 날짜 → 대시(-) 구분으로 통일
    # "2026.3.9 17:24" → "2026-03-09 17:24"
    def normalize_date_format(s: str) -> str:
        if pd.isna(s) or s in ("", "nan", "None", "NaT"):
            return ""
        # 점 구분 날짜 감지
        if "." in s.split(" ")[0]:
            parts = s.split(" ")
            date_part = parts[0]
            time_part = parts[1] if len(parts) > 1 else "00:00:00"
            try:
                date_components = date_part.split(".")
                if len(date_components) == 3:
                    year, month, day = date_components
                    # 패딩 추가
                    month = month.zfill(2)
                    day = day.zfill(2)
                    # 시간 부분에 초가 없으면 추가
                    if time_part.count(":") == 1:
                        time_part += ":00"
                    return f"{year}-{month}-{day} {time_part}"
            except Exception:
                pass
        return s

    # 불규칙 형식 처리 (점 구분 날짜가 있을 경우에만)
    if str_series.str.contains(r"^\d{4}\.\d+\.\d+", regex=True).any():
        str_series = str_series.apply(normalize_date_format)

    # Step 4: pandas datetime 변환 (errors='coerce' → 실패 시 NaT)
    result = pd.to_datetime(str_series, errors="coerce")

    # 파싱 실패 로깅
    null_count = result.isna().sum()
    original_null = series.isna().sum()
    new_nulls = null_count - original_null
    if new_nulls > 0:
        logger.warning(
            f"[{col_name}] {new_nulls}개 행 시간 파싱 실패 (총 {len(series)}행)"
        )

    return result


# =============================================================================
# Date Detection
# =============================================================================

def detect_date_range(
    file_path: Path | str,
    time_col: str = "Time",
    encoding: str = "cp949",
) -> list[str]:
    """
    CSV에서 고유 날짜 목록 추출.

    전체 파일을 청크 기반으로 스캔하여 모든 날짜 추출.

    Args:
        file_path: CSV 파일 경로
        time_col: 시간 컬럼명 (Time, Entry_time 등)
        encoding: 파일 인코딩 (기본 cp949)

    Returns:
        정렬된 날짜 문자열 리스트 ["20260226", "20260227", ...]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"파일 없음: {file_path}")
        return []

    dates: set[str] = set()

    try:
        for chunk in pd.read_csv(
            file_path,
            encoding=encoding,
            low_memory=False,
            chunksize=CHUNK_SIZE,
            usecols=[time_col],  # 시간 컬럼만 로드 (메모리 최적화)
        ):
            # 시간 파싱
            chunk[time_col] = parse_time_col(chunk[time_col], time_col)

            # 유효한 날짜만 추출
            valid_mask = chunk[time_col].notna()
            chunk_dates = (
                chunk.loc[valid_mask, time_col]
                .dt.strftime("%Y%m%d")
                .unique()
            )
            dates.update(chunk_dates)

    except Exception as e:
        logger.error(f"날짜 범위 감지 실패: {file_path}, 에러: {e}")
        return []

    result = sorted(dates)
    logger.info(f"감지된 날짜 {len(result)}개: {result[0]}~{result[-1]}")
    return result


# =============================================================================
# Split by Date
# =============================================================================

def split_by_date(
    file_path: Path | str,
    time_col: str = "Time",
    encoding: str = "cp949",
) -> dict[str, pd.DataFrame]:
    """
    단일 CSV -> 날짜별 dict[str, DataFrame].

    청크 기반 파싱으로 메모리 효율적 처리.
    각 날짜 DataFrame은 시간순 정렬됨.

    Args:
        file_path: CSV 파일 경로
        time_col: 시간 컬럼명
        encoding: 파일 인코딩

    Returns:
        {"20260226": df1, "20260227": df2, ...}
        빈 DataFrame인 날짜는 포함되지 않음.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"파일 없음: {file_path}")
        return {}

    result: dict[str, list[pd.DataFrame]] = {}

    try:
        for chunk in pd.read_csv(
            file_path,
            encoding=encoding,
            low_memory=False,
            chunksize=CHUNK_SIZE,
        ):
            # 시간 파싱
            chunk[time_col] = parse_time_col(chunk[time_col], time_col)

            # 날짜 컬럼 추가
            chunk["_date_str"] = chunk[time_col].dt.strftime("%Y%m%d")

            # 날짜별 그룹화
            for date_str, group in chunk.groupby("_date_str", dropna=True):
                if date_str not in result:
                    result[date_str] = []
                result[date_str].append(
                    group.drop(columns=["_date_str"]).copy()
                )

    except Exception as e:
        logger.error(f"날짜별 분리 실패: {file_path}, 에러: {e}")
        return {}

    # 날짜별 concat + 시간순 정렬
    final_result: dict[str, pd.DataFrame] = {}
    for date_str in sorted(result.keys()):
        df = pd.concat(result[date_str], ignore_index=True)
        # 시간순 정렬
        df = df.sort_values(time_col).reset_index(drop=True)
        final_result[date_str] = df
        logger.debug(f"  {date_str}: {len(df):,} rows")

    logger.info(f"총 {len(final_result)}개 날짜 분리 완료")
    return final_result


# =============================================================================
# Daily Data Loaders
# =============================================================================

def load_daily_tward(
    date_str: str,
    tward_file: Path | str | None = None,
) -> pd.DataFrame:
    """
    특정 날짜의 TWardData 추출 + 정규화.

    Args:
        date_str: 날짜 문자열 "YYYYMMDD"
        tward_file: TWardData CSV 경로 (None이면 config 사용)

    Returns:
        정규화된 DataFrame (컬럼명 소문자)

    컬럼 정규화:
    - User_no -> user_no
    - Time -> timestamp
    - Worker_name -> user_name
    - Building -> building
    - Level -> level
    - Place -> place
    - X -> x
    - Y -> y
    - Signal_count -> signal_count
    - ActiveSignal_count -> active_signal_count
    """
    # config import (순환 참조 방지)
    import config as cfg

    if tward_file is None:
        tward_file = cfg.RAW_TWARD_SINGLE_FILE
    tward_file = Path(tward_file)

    if not tward_file.exists():
        logger.error(f"TWardData 파일 없음: {tward_file}")
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []

    try:
        for chunk in pd.read_csv(
            tward_file,
            encoding=cfg.RAW_ENCODING,
            low_memory=False,
            chunksize=CHUNK_SIZE,
        ):
            # 시간 파싱
            chunk["Time"] = parse_time_col(chunk["Time"], "Time")

            # 날짜 필터링
            chunk["_date_str"] = chunk["Time"].dt.strftime("%Y%m%d")
            mask = chunk["_date_str"] == date_str
            if mask.any():
                chunks.append(chunk.loc[mask].drop(columns=["_date_str"]).copy())

    except Exception as e:
        logger.error(f"TWardData 로드 실패: {date_str}, 에러: {e}")
        return pd.DataFrame()

    if not chunks:
        logger.warning(f"TWardData 데이터 없음: {date_str}")
        return pd.DataFrame()

    # concat + 정렬
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("Time").reset_index(drop=True)

    # 컬럼 정규화
    df = df.rename(columns={
        "User_no": "user_no",
        "Time": "timestamp",
        "Worker_name": "user_name",
        "Building": "building",
        "Level": "level",
        "Place": "place",
        "X": "x",
        "Y": "y",
        "Signal_count": "signal_count",
        "ActiveSignal_count": "active_signal_count",
    })

    # user_no NaN 제거 후 타입 캐스팅
    before_count = len(df)
    df = df.dropna(subset=["user_no"])
    if len(df) < before_count:
        logger.warning(
            f"TWardData user_no NaN 제거: {before_count - len(df)}개"
        )
    df["user_no"] = df["user_no"].astype("int64")
    df["x"] = pd.to_numeric(df["x"], errors="coerce").astype("float32")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("float32")
    df["signal_count"] = pd.to_numeric(
        df["signal_count"], errors="coerce"
    ).fillna(0).astype("int32")
    df["active_signal_count"] = pd.to_numeric(
        df["active_signal_count"], errors="coerce"
    ).fillna(0).astype("int32")

    logger.info(f"TWardData 로드 완료: {date_str}, {len(df):,} rows")
    return df


def load_daily_access(
    date_str: str,
    access_file: Path | str | None = None,
) -> pd.DataFrame:
    """
    특정 날짜의 AccessLog 추출 + 정규화.

    Args:
        date_str: 날짜 문자열 "YYYYMMDD"
        access_file: AccessLog CSV 경로 (None이면 config 사용)

    Returns:
        정규화된 DataFrame

    컬럼 정규화:
    - User_no -> user_no
    - Worker_name -> user_name
    - Cellphone -> cellphone
    - User_record_id -> user_record_id
    - SCon_company_name -> company_name
    - SCon_company_code -> company_code
    - EmploymentStatus_Hycon -> employment_status
    - HyCon_company_name -> hycon_company_name
    - HyCon_company_code -> hycon_company_code
    - Entry_time -> entry_time
    - Exit_time -> exit_time
    - T-Ward ID -> tward_id
    """
    # config import (순환 참조 방지)
    import config as cfg

    if access_file is None:
        access_file = cfg.RAW_ACCESS_SINGLE_FILE
    access_file = Path(access_file)

    if not access_file.exists():
        logger.error(f"AccessLog 파일 없음: {access_file}")
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []

    try:
        for chunk in pd.read_csv(
            access_file,
            encoding=cfg.RAW_ENCODING,
            low_memory=False,
            chunksize=CHUNK_SIZE,
        ):
            # Entry_time 파싱 (기준 시간)
            chunk["Entry_time"] = parse_time_col(chunk["Entry_time"], "Entry_time")

            # 날짜 필터링 (Entry_time 기준)
            chunk["_date_str"] = chunk["Entry_time"].dt.strftime("%Y%m%d")
            mask = chunk["_date_str"] == date_str
            if mask.any():
                chunks.append(chunk.loc[mask].drop(columns=["_date_str"]).copy())

    except Exception as e:
        logger.error(f"AccessLog 로드 실패: {date_str}, 에러: {e}")
        return pd.DataFrame()

    if not chunks:
        logger.warning(f"AccessLog 데이터 없음: {date_str}")
        return pd.DataFrame()

    # concat
    df = pd.concat(chunks, ignore_index=True)

    # Exit_time 파싱 (별도 처리 - 형식 다름)
    if "Exit_time" in df.columns:
        df["Exit_time"] = parse_time_col(df["Exit_time"], "Exit_time")

    # 컬럼 정규화
    df = df.rename(columns={
        "User_no": "user_no",
        "Worker_name": "user_name",
        "Cellphone": "cellphone",
        "User_record_id": "user_record_id",
        "SCon_company_name": "company_name",
        "SCon_company_code": "company_code",
        "EmploymentStatus_Hycon": "employment_status",
        "HyCon_company_name": "hycon_company_name",
        "HyCon_company_code": "hycon_company_code",
        "Entry_time": "entry_time",
        "Exit_time": "exit_time",
        "T-Ward ID": "tward_id",
    })

    # user_no NaN 제거 후 타입 캐스팅
    before_count = len(df)
    df = df.dropna(subset=["user_no"])
    if len(df) < before_count:
        logger.warning(
            f"AccessLog user_no NaN 제거: {before_count - len(df)}개"
        )
    df["user_no"] = df["user_no"].astype("int64")

    # Entry_time 기준 정렬
    df = df.sort_values("entry_time").reset_index(drop=True)

    logger.info(f"AccessLog 로드 완료: {date_str}, {len(df):,} rows")
    return df


def load_daily_data(
    date_str: str,
    tward_file: Path | str | None = None,
    access_file: Path | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    TWardData + AccessLog 조인 (user_no 기준).

    Args:
        date_str: 날짜 문자열 "YYYYMMDD"
        tward_file: TWardData CSV 경로 (None이면 config 사용)
        access_file: AccessLog CSV 경로 (None이면 config 사용)

    Returns:
        (journey_df, meta_dict) 튜플

        journey_df: TWardData + AccessLog 조인 결과
            - user_no, timestamp, user_name, building, level, place, x, y
            - signal_count, active_signal_count
            - company_name, company_code, entry_time, exit_time, tward_id

        meta_dict: {
            "date": "YYYYMMDD",
            "total_workers": 작업자 수,
            "total_records": 위치 기록 수,
            "total_access": 출입 기록 수,
            "has_tward_count": T-Ward 착용자 수,
            "companies": 업체 수,
        }
    """
    # TWardData 로드
    tward_df = load_daily_tward(date_str, tward_file)

    # AccessLog 로드
    access_df = load_daily_access(date_str, access_file)

    # 메타 정보 초기화
    meta: dict = {
        "date": date_str,
        "total_workers": 0,
        "total_records": 0,
        "total_access": len(access_df),
        "has_tward_count": 0,
        "companies": 0,
    }

    # TWardData가 비어있으면 빈 결과 반환
    if tward_df.empty:
        logger.warning(f"TWardData 없음, 빈 결과 반환: {date_str}")
        return pd.DataFrame(), meta

    # 메타 업데이트
    meta["total_records"] = len(tward_df)
    meta["total_workers"] = tward_df["user_no"].nunique()

    # AccessLog와 조인 (user_no 기준, left join)
    # AccessLog에서 필요한 컬럼만 선택
    if not access_df.empty:
        access_cols = [
            "user_no", "company_name", "company_code",
            "entry_time", "exit_time", "tward_id",
        ]
        access_subset = access_df[[c for c in access_cols if c in access_df.columns]]

        # 중복 user_no 제거 (첫 번째 출입 기록 사용)
        access_subset = access_subset.drop_duplicates(subset=["user_no"], keep="first")

        # 조인
        journey_df = tward_df.merge(access_subset, on="user_no", how="left")

        # 메타 업데이트
        meta["has_tward_count"] = access_df["tward_id"].notna().sum()
        meta["companies"] = access_df["company_name"].nunique()
    else:
        journey_df = tward_df.copy()
        # 기본 컬럼 추가
        journey_df["company_name"] = None
        journey_df["company_code"] = None
        journey_df["entry_time"] = pd.NaT
        journey_df["exit_time"] = pd.NaT
        journey_df["tward_id"] = None

    # 시간순 정렬
    journey_df = journey_df.sort_values(
        ["user_no", "timestamp"]
    ).reset_index(drop=True)

    logger.info(
        f"Daily Data 로드 완료: {date_str}, "
        f"workers={meta['total_workers']}, records={meta['total_records']:,}"
    )

    return journey_df, meta


# =============================================================================
# Iterator (Memory Optimized)
# =============================================================================

def iter_by_date(
    file_path: Path | str,
    time_col: str = "Time",
    encoding: str = "cp949",
) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    날짜별 DataFrame Generator (메모리 최적화).

    2-pass 방식:
    1. 전체 날짜 범위 감지
    2. 각 날짜별로 청크 필터링 후 yield

    주의: 대용량 파일에서 각 날짜마다 전체 파일 스캔 필요.
    전체 로드가 메모리에 들어간다면 split_by_date() 권장.

    Yields:
        (date_str, daily_df) 튜플
    """
    file_path = Path(file_path)
    dates = detect_date_range(file_path, time_col, encoding)

    for target_date in dates:
        daily_chunks: list[pd.DataFrame] = []

        for chunk in pd.read_csv(
            file_path,
            encoding=encoding,
            low_memory=False,
            chunksize=CHUNK_SIZE,
        ):
            # 시간 파싱
            chunk[time_col] = parse_time_col(chunk[time_col], time_col)

            # 날짜 필터링
            mask = chunk[time_col].dt.strftime("%Y%m%d") == target_date
            if mask.any():
                daily_chunks.append(chunk.loc[mask].copy())

        if daily_chunks:
            df = pd.concat(daily_chunks, ignore_index=True)
            df = df.sort_values(time_col).reset_index(drop=True)
            yield target_date, df


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    import sys

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # config import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config as cfg

    print("=" * 60)
    print("Bulk Loader Test")
    print("=" * 60)

    # 1. 날짜 범위 감지
    print("\n[1] TWardData 날짜 범위 감지")
    tward_dates = detect_date_range(cfg.RAW_TWARD_SINGLE_FILE, "Time")
    print(f"    감지된 날짜: {len(tward_dates)}개")
    if tward_dates:
        print(f"    범위: {tward_dates[0]} ~ {tward_dates[-1]}")

    print("\n[2] AccessLog 날짜 범위 감지")
    access_dates = detect_date_range(cfg.RAW_ACCESS_SINGLE_FILE, "Entry_time")
    print(f"    감지된 날짜: {len(access_dates)}개")
    if access_dates:
        print(f"    범위: {access_dates[0]} ~ {access_dates[-1]}")

    # 2. 첫 날짜 로드 테스트
    if tward_dates:
        test_date = tward_dates[0]
        print(f"\n[3] Daily Data 로드 테스트: {test_date}")
        df, meta = load_daily_data(test_date)
        print(f"    DataFrame: {len(df):,} rows x {len(df.columns)} cols")
        print(f"    Meta: {meta}")
        if not df.empty:
            print(f"    Columns: {list(df.columns)}")
            print(f"    Sample:\n{df.head(3)}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
