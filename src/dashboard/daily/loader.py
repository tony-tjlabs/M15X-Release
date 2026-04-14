"""
Data Loader — Raw CSV 로딩 모듈 (v3, 2026-03-21)
===================================
AccessLog(출입 이력) + TWardData(T-Ward 이동 위치)를 로드하고 join하여
파이프라인이 사용할 수 있는 형태로 반환.

[v3 추가]
  - classify_shift(): Entry 시간 기준 주간/야간 분류
  - load_daily_data(): 야간 근무자 D+1 TWardData 자동 보완
  - 헬멧 방치 탐지를 위한 exit_source 컬럼 추가

Q. AccessLog vs TWardData 역할 분리는?
A.  AccessLog  = 생체인식 출입 이력 (in/out 시간, 업체, T-Ward ID)
                 → 총 출입자 카운트의 참값 (T-Ward 미착용자 포함)
    TWardData  = 헬멧 T-Ward BLE 이동 위치 (1분 단위, 좌표/신호)
                 → T-Ward 착용 작업자의 실제 이동 데이터
    join key   = User_no

Q. 주간/야간 분류 기준은?
A.  주간 (day):   Entry 04:00 ~ 16:59
    야간 (night): Entry 17:00 ~ 23:59 또는 00:00 ~ 03:59
    → 야간 작업자는 자정을 넘겨 다음날 퇴근하므로 D+1 TWardData 보완 필요

Q. 야간 근무자 BLE 보완은?
A.  out_datetime이 D+1인 야간 작업자의 경우
    D+1 TWardData에서 해당 user_no의 오전 데이터(~ exit_time)를 추가.
    → 이로써 is_work_hour 범위가 자정을 넘겨 정확하게 계산됨.

Q. 총 출입자 vs 이동 작업자 차이는?
A.  총 출입자    = AccessLog unique user_no 수 (참값, 생체인식 기준)
    이동 작업자  = TWardData unique user_no 수 (T-Ward 착용 + BLE 수신)
    차이 = T-Ward 미착용자 + BLE 음영 작업자
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── 날짜 포맷 헬퍼 ──────────────────────────────────────────────────
def _dot_date_repl(m: re.Match) -> str:
    """
    마침표 구분 날짜의 월/일을 제로패딩하여 대시 구분으로 변환.
    "2026.3.9 17:24" → regex 매치 ".3.9 " → "-03-09 "
    """
    month = m.group(1).zfill(2)
    day   = m.group(2).zfill(2)
    return f"-{month}-{day} "


# ─── 날짜 유틸 ─────────────────────────────────────────────────────
def _date_from_filename(fname: str) -> str | None:
    """파일명에서 날짜 추출 (YYYYMMDD 형식)."""
    m = re.search(r"(\d{8})", fname)
    return m.group(1) if m else None


def _adjacent_date_str(date_str: str, delta: int) -> str:
    """YYYYMMDD → ±delta일 YYYYMMDD 반환."""
    d = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=delta)
    return d.strftime("%Y%m%d")


def detect_raw_dates(raw_dir: Path, sector_id: str | None = None) -> list[str]:
    """
    raw_dir에서 처리 가능한 날짜 목록 반환.

    M15X 모드: 단일 통합 CSV 파일에서 날짜 추출 (bulk_loader 사용)
    Y1 모드:   날짜별 개별 CSV 파일 패턴 매칭 (기존 방식)

    반환: ["20260309", "20260310", ...]
    """
    import config as cfg

    # ── M15X 모드: 단일 통합 CSV 존재 시 bulk_loader 사용 ──────────────
    if cfg.RAW_TWARD_SINGLE_FILE.exists():
        try:
            from src.pipeline.bulk_loader import detect_date_range
            # TWardData와 AccessLog 모두에서 날짜 추출 후 교집합
            tward_dates = set(detect_date_range(cfg.RAW_TWARD_SINGLE_FILE, "Time"))
            access_dates = set(detect_date_range(cfg.RAW_ACCESS_SINGLE_FILE, "Entry_time"))
            both = tward_dates & access_dates
            result = sorted(both)
            logger.info(f"M15X 단일 CSV 모드: {len(result)}개 날짜 감지 ({result[0]}~{result[-1]})")
            return result
        except Exception as e:
            logger.warning(f"bulk_loader 사용 실패, 기존 방식으로 fallback: {e}")

    # ── Y1 모드 (fallback): 날짜별 개별 파일 패턴 매칭 ─────────────────
    access_dates = {
        _date_from_filename(f.name)
        for f in raw_dir.glob("*AccessLog*.csv")
        if _date_from_filename(f.name)
    }
    tward_dates = {
        _date_from_filename(f.name)
        for f in raw_dir.glob("*TWardData*.csv")
        if _date_from_filename(f.name)
    }
    both = access_dates & tward_dates
    return sorted(both)


# ─── Shift 분류 ────────────────────────────────────────────────────
def classify_shift(in_datetime: pd.Series) -> pd.Series:
    """
    Entry 시간(in_datetime) 기준 근무 Shift 분류.

    주간 (day):   04:00 ~ 16:59 입장
    야간 (night): 17:00 ~ 23:59 또는 00:00 ~ 03:59 입장
    unknown:      NaT

    Note:
      - 야간 작업자는 대부분 다음날 퇴근 (out_datetime = D+1)
      - EWI/CRE 분리 분석 및 D+1 BLE 보완 여부 판단에 사용
    """
    hour   = in_datetime.dt.hour
    result = pd.Series("unknown", index=in_datetime.index, dtype="object")
    known  = in_datetime.notna()
    result.loc[known & (hour >= 4) & (hour < 17)] = "day"
    result.loc[known & ((hour >= 17) | (hour < 4))] = "night"
    return result


# ─── AccessLog 로드 (출입 이력 — 생체인식 Gate) ────────────────────
def load_access_log(path: Path) -> pd.DataFrame:
    """
    AccessLog CSV 로드 → 출입 이력 + 업체 정보 정규화.

    원본 컬럼:
        User_no, Worker_name, Cellphone, User_record_id,
        SCon_company_name, SCon_company_code,
        EmploymentStatus_Hycon, HyCon_company_name, HyCon_company_code,
        Entry_time, Exit_time, SCON_record_id, T-Ward ID

    반환 컬럼:
        user_no (int), user_name (str),
        company_name (str), company_code (str),
        in_datetime (datetime, KST), out_datetime (datetime, KST),
        work_minutes (float),
        shift_type (str): "day" / "night" / "unknown"
        exit_source (str): "access_log" / "missing"
        twardid (str|None), has_tward (bool)

    Note:
      - 야간 작업자의 Exit_time은 D+1 날짜를 포함할 수 있음
        (e.g., "2026-03-21 06:00:00") → 올바르게 파싱됨
      - Exit NaT = 퇴근 미기록 (exit_source="missing")
    """
    df = pd.read_csv(path, encoding="cp949", low_memory=False)

    # ── user_no 정규화 ─────────────────────────────────────────────
    df["user_no"] = pd.to_numeric(df["User_no"], errors="coerce").astype("Int64")

    # ── 이름 ──────────────────────────────────────────────────────
    df["user_name"] = df["Worker_name"].fillna("").astype(str)

    # ── 업체명: HyCon 우선, 없으면 SCon fallback ─────────────────
    hcon = df["HyCon_company_name"].fillna("").astype(str)
    scon = df["SCon_company_name"].fillna("").astype(str)
    df["company_name"] = hcon.where(hcon != "", scon).where(hcon != "", "미확인")
    df["company_name"] = df["company_name"].replace("", "미확인")

    hcon_code = df["HyCon_company_code"].fillna("").astype(str)
    scon_code = df["SCon_company_code"].fillna("").astype(str)
    df["company_code"] = hcon_code.where(hcon_code != "", scon_code)

    # ── 출입 시간 파싱 (KST 벽시계 시간 유지) ────────────────────
    # 지원 포맷:
    #   Y1:   "2026-03-19 05:58:48.000 +0900" / "2026-03-19 17:10:17"
    #   M15X: "2026.3.9 17:24" / "2026.3.11 6:00" (마침표 구분, 제로패딩 없음)
    for src, dst in [("Entry_time", "in_datetime"), ("Exit_time", "out_datetime")]:
        if src in df.columns:
            cleaned = (
                df[src].astype(str)
                .str.replace(r"\s*[+-]\d{2}:?\d{2}$", "", regex=True)
                .str.replace(r"\.\d{3}$", "", regex=True)   # ms 제거 (.000)
                .str.strip()
            )
            # 1차: Y1 포맷 (YYYY-MM-DD HH:MM:SS)
            parsed = pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            # 2차: 마침표 구분 포맷 (YYYY.M.D H:MM 등) — 1차 실패분만
            nat_mask = parsed.isna() & (cleaned != "nan") & (cleaned != "")
            if nat_mask.any():
                # "2026.3.9 17:24" → "2026-03-09 17:24:00"
                dot_cleaned = (
                    cleaned[nat_mask]
                    .str.replace(r"\.(\d{1,2})\.(\d{1,2})\s", _dot_date_repl, regex=True)
                )
                parsed_dot = pd.to_datetime(dot_cleaned, format="%Y-%m-%d %H:%M:%S", errors="coerce")
                # HH:MM만 있는 경우 (초 없음)
                still_nat = parsed_dot.isna()
                if still_nat.any():
                    parsed_dot[still_nat] = pd.to_datetime(
                        dot_cleaned[still_nat], format="%Y-%m-%d %H:%M", errors="coerce"
                    )
                parsed[nat_mask] = parsed_dot
            df[dst] = parsed
        else:
            df[dst] = pd.NaT

    # ── 근무 시간 (분) ────────────────────────────────────────────
    # 야간 작업자: in=D 20:00, out=D+1 06:00 → 600분 (정확)
    # NaT out: clip → 0 (퇴근 미기록)
    delta = df["out_datetime"] - df["in_datetime"]
    df["work_minutes"] = (delta.dt.total_seconds() / 60).clip(lower=0).fillna(0)

    # ── Shift 분류 + exit_source ───────────────────────────────────
    df["shift_type"]  = classify_shift(df["in_datetime"])
    df["exit_source"] = np.where(df["out_datetime"].notna(), "access_log", "missing")

    # ── T-Ward ID ─────────────────────────────────────────────────
    tward_col = "T-Ward ID"
    if tward_col in df.columns:
        df["twardid"] = df[tward_col].astype(str).str.strip()
        df["twardid"] = df["twardid"].replace({"nan": None, "": None})
    else:
        df["twardid"] = None

    df["has_tward"] = df["twardid"].notna()

    # ── 중복 user_no 처리 ─────────────────────────────────────────
    # 같은 날 동일인 여러 출입 (야간→주간 교대 등)
    # → work_minutes가 가장 큰 레코드 유지 (가장 넓은 시간 범위 보존)
    # → BLE join 시 is_work_hour 판정이 정확해짐
    if df["user_no"].duplicated().any():
        df = df.sort_values("work_minutes", ascending=False)
        df = df.drop_duplicates(subset=["user_no"], keep="first")

    keep = ["user_no", "user_name", "company_name", "company_code",
            "in_datetime", "out_datetime", "work_minutes",
            "shift_type", "exit_source",
            "twardid", "has_tward"]
    return df[keep].reset_index(drop=True)


# ─── TWardData 로드 (T-Ward 이동 위치 — BLE) ──────────────────────
def load_tward_data(path: Path) -> pd.DataFrame:
    """
    TWardData CSV 로드 → 1분 단위 이동 위치 정규화.

    원본 컬럼:
        User_no, Time, Worker_name,
        Building, Level, Place, X, Y,
        Signal_count, ActiveSignal_count

    반환 컬럼:
        timestamp (datetime, 1분 단위),
        user_no (int), user_name (str),
        building_name (str), floor_name (str), spot_name (str),
        x (float), y (float),
        signal_count (int), active_count (int),
        active_ratio (float)
    """
    df = pd.read_csv(path, encoding="cp949", low_memory=False)

    # ── timestamp 파싱 ─────────────────────────────────────────────
    # 포맷: "2026-03-19 23:59:00.000 +0900"
    cleaned_time = (
        df["Time"].astype(str)
        .str.replace(r"\s*[+-]\d{2}:?\d{2}$", "", regex=True)
        .str.replace(r"\.\d+$", "", regex=True)
        .str.strip()
    )
    df["timestamp"] = pd.to_datetime(cleaned_time, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # ── 컬럼 정규화 ───────────────────────────────────────────────
    df["user_no"]       = pd.to_numeric(df["User_no"], errors="coerce").astype("Int64")
    df["user_name"]     = df["Worker_name"].fillna("").astype(str)
    df["building_name"] = df["Building"].fillna("").astype(str)
    df["floor_name"]    = df["Level"].fillna("").astype(str)
    df["spot_name"]     = df["Place"].fillna("unknown").astype(str)
    df["x"]             = pd.to_numeric(df["X"], errors="coerce")
    df["y"]             = pd.to_numeric(df["Y"], errors="coerce")
    df["signal_count"]  = pd.to_numeric(df["Signal_count"],       errors="coerce").fillna(0).astype(int)
    df["active_count"]  = pd.to_numeric(df["ActiveSignal_count"], errors="coerce").fillna(0).astype(int)

    # ── 활성 비율 (벡터화) ─────────────────────────────────────────
    mask = df["signal_count"] > 0
    df["active_ratio"] = 0.0
    df.loc[mask, "active_ratio"] = (
        df.loc[mask, "active_count"] / df.loc[mask, "signal_count"]
    )

    keep = ["timestamp", "user_no", "user_name",
            "building_name", "floor_name", "spot_name",
            "x", "y", "signal_count", "active_count", "active_ratio"]
    return df[keep].reset_index(drop=True)


# ─── 통합 로드 + Join ──────────────────────────────────────────────
def load_daily_data(raw_dir: Path, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    하루치 AccessLog(출입 이력) + TWardData(T-Ward 이동 위치) 로드 후 join.

    [v4 M15X 단일 CSV 지원]
    M15X 모드: bulk_loader를 사용하여 단일 CSV에서 날짜별 데이터 추출
    Y1 모드:   기존 날짜별 개별 파일 로드 방식 유지

    [v3 야간 근무 보완]
    야간 작업자(Entry >= 17:00)의 out_datetime이 D+1인 경우:
      → D+1 TWardData에서 해당 user_no의 exit_time 이전 BLE 기록을 추가
      → is_work_hour 필터가 자정을 넘겨 정확하게 적용됨

    반환:
        journey_df  : TWardData에 출입 정보(업체/시간)가 join된 이동 데이터
        access_df   : AccessLog 전체 (T-Ward 미착용자 포함)
        meta        : 로딩 통계 dict (shift 분류 포함)
    """
    import config as cfg

    # ── M15X 모드: 단일 통합 CSV에서 로드 ───────────────────────────
    if cfg.RAW_TWARD_SINGLE_FILE.exists():
        return _load_daily_data_m15x(date_str)

    # ── Y1 모드 (기존): 날짜별 개별 파일에서 로드 ──────────────────────
    # ── 파일 탐색 ──────────────────────────────────────────────────
    access_candidates = list(raw_dir.glob(f"*AccessLog*{date_str}*.csv"))
    tward_candidates  = list(raw_dir.glob(f"*TWardData*{date_str}*.csv"))

    if not access_candidates:
        raise FileNotFoundError(f"AccessLog 없음: {raw_dir}/*AccessLog*{date_str}*.csv")
    if not tward_candidates:
        raise FileNotFoundError(f"TWardData 없음: {raw_dir}/*TWardData*{date_str}*.csv")

    # ── D 날짜 로드 ────────────────────────────────────────────────
    access_df = load_access_log(access_candidates[0])
    tward_df  = load_tward_data(tward_candidates[0])

    # ── 야간 작업자 D+1 BLE 보완 ──────────────────────────────────
    # 야간 근무자 중 out_datetime이 D+1 날짜인 경우 (퇴근 시간이 자정 이후)
    # → D+1 TWardData에서 exit_time 이전 BLE 기록 추가
    next_date_str   = _adjacent_date_str(date_str, 1)
    next_date_pd    = pd.Timestamp(datetime.strptime(next_date_str, "%Y%m%d"))
    tward_next_candidates = list(raw_dir.glob(f"*TWardData*{next_date_str}*.csv"))

    night_with_next_exit = access_df[
        (access_df["shift_type"] == "night")
        & access_df["out_datetime"].notna()
        & (access_df["out_datetime"].dt.date == next_date_pd.date())
    ]

    night_supplement_count = 0
    incomplete_night_workers = 0
    if not night_with_next_exit.empty and not tward_next_candidates:
        # ★ D+1 TWardData 없음 → 야간 작업자 BLE 불완전 (경고)
        incomplete_night_workers = len(night_with_next_exit)
        logger.warning(
            f"D+1 TWardData({next_date_str}) 없음 — "
            f"야간 작업자 {incomplete_night_workers}명 BLE 불완전"
        )
    if not night_with_next_exit.empty and tward_next_candidates:
        tward_next = load_tward_data(tward_next_candidates[0])

        # 야간 작업자의 D+1 새벽 BLE만 추출 (exit_time 이전까지)
        exit_map = (
            night_with_next_exit
            .set_index("user_no")["out_datetime"]
            .to_dict()
        )
        night_nos = set(night_with_next_exit["user_no"].dropna())

        mask_night = tward_next["user_no"].isin(night_nos)
        tward_next_night = tward_next[mask_night].copy()

        # 각 작업자의 exit_time 이전 레코드만 유지 (벡터화)
        tward_next_night["_exit"] = tward_next_night["user_no"].map(exit_map)
        tward_next_night = tward_next_night[
            tward_next_night["timestamp"] <= tward_next_night["_exit"]
        ].drop(columns=["_exit"])

        night_supplement_count = len(tward_next_night)
        tward_df = pd.concat([tward_df, tward_next_night], ignore_index=True)

    # ── Join: TWardData ← AccessLog (user_no 기준, left join) ─────
    join_cols = ["user_no", "company_name", "company_code",
                 "in_datetime", "out_datetime", "work_minutes",
                 "shift_type", "exit_source",
                 "twardid", "has_tward"]
    join_cols = [c for c in join_cols if c in access_df.columns]

    journey_df = tward_df.merge(
        access_df[join_cols],
        on="user_no",
        how="left",
    )

    # ── 미매핑 처리 ───────────────────────────────────────────────
    journey_df["company_name"] = journey_df["company_name"].fillna("미확인")
    journey_df["has_tward"]    = journey_df["has_tward"].infer_objects(copy=False).fillna(False).astype(bool)
    journey_df["shift_type"]   = journey_df["shift_type"].fillna("unknown")
    journey_df["exit_source"]  = journey_df["exit_source"].fillna("missing")

    # ── is_work_hour 계산 ───────────────────────────────────────
    # ★ v2: out_datetime이 NaT인 경우 FAR_FUTURE로 채우면 모든 BLE가
    #        근무시간 판정 → 비정상 EWI. in/out 모두 있어야 판정.
    _FAR_PAST   = pd.Timestamp("1900-01-01")
    in_dt  = journey_df["in_datetime"].fillna(_FAR_PAST)
    out_dt = journey_df["out_datetime"].fillna(_FAR_PAST)  # ★ NaT → FAR_PAST (is_work_hour=False)
    ts     = journey_df["timestamp"].fillna(_FAR_PAST)
    journey_df["is_work_hour"] = (
        journey_df["in_datetime"].notna()
        & journey_df["out_datetime"].notna()  # ★ exit도 있어야 판정 가능
        & journey_df["timestamp"].notna()
        & (ts >= in_dt)
        & (ts <= out_dt)
    )
    journey_df["missing_exit"] = journey_df["out_datetime"].isna()

    # ── Shift별 통계 (메타) ───────────────────────────────────────
    day_workers   = int(access_df[access_df["shift_type"] == "day"  ]["user_no"].nunique())
    night_workers = int(access_df[access_df["shift_type"] == "night"]["user_no"].nunique())
    missing_exit  = int((access_df["exit_source"] == "missing").sum())

    meta = {
        "date_str":              date_str,
        "total_records":         len(journey_df),
        "total_workers_access":  access_df["user_no"].nunique(),
        "total_workers_move":    int(
            tward_df.groupby("user_no")["active_count"]
            .sum()
            .pipe(lambda s: (s >= 10).sum())
        ),  # ★ 활성 신호 10회 이상인 작업자만 카운팅 (비활성 헬멧 제외)
        "tward_holders":         int(access_df["has_tward"].sum()),
        "companies":             access_df["company_name"].nunique(),
        "spots":                 journey_df["spot_name"].nunique(),
        "day_workers":           day_workers,
        "night_workers":         night_workers,
        "missing_exit_workers":  missing_exit,
        "night_supplement_records": night_supplement_count,
        "incomplete_night_workers": incomplete_night_workers,
        "time_start":            str(journey_df["timestamp"].min()),
        "time_end":              str(journey_df["timestamp"].max()),
    }

    return journey_df, access_df, meta


# ─── M15X 단일 CSV 전용 로드 함수 ─────────────────────────────────────
def _load_daily_data_m15x(date_str: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    M15X 단일 CSV 파일에서 하루치 데이터 로드.

    bulk_loader를 사용하여 TWardData + AccessLog를 날짜별로 추출 후 join.
    Y1 방식(load_daily_data)과 동일한 출력 형식 유지.

    반환:
        journey_df  : TWardData에 출입 정보(업체/시간)가 join된 이동 데이터
        access_df   : AccessLog 전체 (T-Ward 미착용자 포함)
        meta        : 로딩 통계 dict
    """
    from src.pipeline.bulk_loader import load_daily_tward, load_daily_access

    logger.info(f"M15X 단일 CSV 모드: {date_str} 로드 시작")

    # ── TWardData + AccessLog 로드 ────────────────────────────────────
    tward_df = load_daily_tward(date_str)
    access_raw_df = load_daily_access(date_str)

    if tward_df.empty:
        logger.warning(f"TWardData 없음: {date_str}")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, {"date_str": date_str, "total_records": 0}

    # ── AccessLog 정규화 (Y1 방식 호환) ─────────────────────────────────
    # bulk_loader는 컬럼명이 다르므로 Y1 방식으로 재매핑
    access_df = _normalize_access_df(access_raw_df, date_str)

    # ── 야간 작업자 D+1 BLE 보완 ─────────────────────────────────────────
    # 야간 근무자(Entry ≥ 17:00)의 out_datetime이 D+1인 경우
    # → D+1 TWardData에서 exit_time 이전 BLE 기록 추가
    night_supplement_count = 0
    if "shift_type" in access_df.columns and "out_datetime" in access_df.columns:
        next_date_str = _adjacent_date_str(date_str, 1)
        next_date_pd  = pd.Timestamp(datetime.strptime(next_date_str, "%Y%m%d"))
        night_workers_df = access_df[
            (access_df["shift_type"] == "night")
            & access_df["out_datetime"].notna()
            & (access_df["out_datetime"].dt.date == next_date_pd.date())
        ]
        if not night_workers_df.empty:
            try:
                tward_next = load_daily_tward(next_date_str)
                if not tward_next.empty:
                    exit_map = night_workers_df.set_index("user_no")["out_datetime"].to_dict()
                    night_nos = set(night_workers_df["user_no"].dropna())
                    tward_next_night = tward_next[tward_next["user_no"].isin(night_nos)].copy()
                    tward_next_night["_exit"] = tward_next_night["user_no"].map(exit_map)
                    tward_next_night = tward_next_night[
                        tward_next_night["timestamp"] <= tward_next_night["_exit"]
                    ].drop(columns=["_exit"])
                    night_supplement_count = len(tward_next_night)
                    tward_df = pd.concat([tward_df, tward_next_night], ignore_index=True)
                    logger.info(
                        f"야간 작업자 D+1 BLE 보완: {len(night_workers_df)}명 / {night_supplement_count}건 추가"
                    )
            except Exception as e:
                logger.warning(f"D+1 BLE 보완 실패 ({next_date_str}): {e}")

    # ── TWardData 정규화 ───────────────────────────────────────────────
    # bulk_loader 출력: user_no, timestamp, user_name, building, level, place, x, y, signal_count, active_signal_count
    # Y1 방식 출력:     user_no, timestamp, user_name, building_name, floor_name, spot_name, x, y, signal_count, active_count, active_ratio
    tward_df = tward_df.rename(columns={
        "building": "building_name",
        "level": "floor_name",
        "place": "spot_name",
        "active_signal_count": "active_count",
    })
    # active_ratio 계산
    mask = tward_df["signal_count"] > 0
    tward_df["active_ratio"] = 0.0
    tward_df.loc[mask, "active_ratio"] = (
        tward_df.loc[mask, "active_count"] / tward_df.loc[mask, "signal_count"]
    )

    # ── Join: TWardData ← AccessLog (user_no 기준, left join) ──────────
    join_cols = ["user_no", "company_name", "company_code",
                 "in_datetime", "out_datetime", "work_minutes",
                 "shift_type", "exit_source",
                 "twardid", "has_tward"]
    join_cols = [c for c in join_cols if c in access_df.columns]

    journey_df = tward_df.merge(
        access_df[join_cols],
        on="user_no",
        how="left",
    )

    # ── 미매핑 처리 ───────────────────────────────────────────────────
    journey_df["company_name"] = journey_df["company_name"].fillna("미확인")
    journey_df["has_tward"] = journey_df.get("has_tward", pd.Series(False, index=journey_df.index)).infer_objects(copy=False).fillna(False).astype(bool)
    journey_df["shift_type"] = journey_df.get("shift_type", pd.Series("unknown", index=journey_df.index)).fillna("unknown")
    journey_df["exit_source"] = journey_df.get("exit_source", pd.Series("missing", index=journey_df.index)).fillna("missing")

    # ── is_work_hour 계산 ─────────────────────────────────────────────
    _FAR_PAST = pd.Timestamp("1900-01-01")
    in_dt = journey_df.get("in_datetime", pd.Series(pd.NaT, index=journey_df.index)).fillna(_FAR_PAST)
    out_dt = journey_df.get("out_datetime", pd.Series(pd.NaT, index=journey_df.index)).fillna(_FAR_PAST)
    ts = journey_df["timestamp"].fillna(_FAR_PAST)

    journey_df["is_work_hour"] = (
        journey_df.get("in_datetime", pd.Series(pd.NaT)).notna()
        & journey_df.get("out_datetime", pd.Series(pd.NaT)).notna()
        & journey_df["timestamp"].notna()
        & (ts >= in_dt)
        & (ts <= out_dt)
    )
    journey_df["missing_exit"] = journey_df.get("out_datetime", pd.Series(pd.NaT)).isna()

    # ── 메타 정보 ─────────────────────────────────────────────────────
    day_workers = int(access_df[access_df["shift_type"] == "day"]["user_no"].nunique()) if "shift_type" in access_df.columns else 0
    night_workers = int(access_df[access_df["shift_type"] == "night"]["user_no"].nunique()) if "shift_type" in access_df.columns else 0
    missing_exit = int((access_df["exit_source"] == "missing").sum()) if "exit_source" in access_df.columns else 0

    # 활성 신호 10회 이상인 작업자만 카운팅 (비활성 헬멧 제외)
    total_workers_move = int(
        tward_df.groupby("user_no")["active_count"]
        .sum()
        .pipe(lambda s: (s >= 10).sum())
    ) if "active_count" in tward_df.columns else tward_df["user_no"].nunique()

    meta = {
        "date_str": date_str,
        "total_records": len(journey_df),
        "total_workers_access": access_df["user_no"].nunique() if not access_df.empty else 0,
        "total_workers_move": total_workers_move,
        "tward_holders": int(access_df["has_tward"].sum()) if "has_tward" in access_df.columns else 0,
        "companies": access_df["company_name"].nunique() if not access_df.empty else 0,
        "spots": journey_df["spot_name"].nunique() if "spot_name" in journey_df.columns else 0,
        "day_workers": day_workers,
        "night_workers": night_workers,
        "missing_exit_workers": missing_exit,
        "night_supplement_records": night_supplement_count,
        "incomplete_night_workers": 0,
        "time_start": str(journey_df["timestamp"].min()),
        "time_end": str(journey_df["timestamp"].max()),
    }

    logger.info(
        f"M15X 로드 완료: {date_str}, "
        f"records={meta['total_records']:,}, "
        f"workers={meta['total_workers_move']}"
    )

    return journey_df, access_df, meta


def _normalize_access_df(access_raw_df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    bulk_loader의 AccessLog 출력을 Y1 방식 컬럼명으로 정규화.

    bulk_loader 출력:
        user_no, user_name, company_name, company_code, entry_time, exit_time, tward_id

    Y1 방식 출력:
        user_no, user_name, company_name, company_code,
        in_datetime, out_datetime, work_minutes,
        shift_type, exit_source, twardid, has_tward
    """
    if access_raw_df.empty:
        return pd.DataFrame()

    df = access_raw_df.copy()

    # 컬럼명 재매핑
    df = df.rename(columns={
        "entry_time": "in_datetime",
        "exit_time": "out_datetime",
        "tward_id": "twardid",
    })

    # 근무 시간 계산 (분)
    if "in_datetime" in df.columns and "out_datetime" in df.columns:
        delta = df["out_datetime"] - df["in_datetime"]
        df["work_minutes"] = (delta.dt.total_seconds() / 60).clip(lower=0).fillna(0)
    else:
        df["work_minutes"] = 0.0

    # Shift 분류
    df["shift_type"] = classify_shift(df["in_datetime"])

    # exit_source
    df["exit_source"] = np.where(df["out_datetime"].notna(), "access_log", "missing")

    # has_tward
    df["has_tward"] = df["twardid"].notna() & (df["twardid"] != "")

    # 중복 user_no 처리: work_minutes 기준 최대값 유지
    if df["user_no"].duplicated().any():
        df = df.sort_values("work_minutes", ascending=False)
        df = df.drop_duplicates(subset=["user_no"], keep="first")

    return df.reset_index(drop=True)
