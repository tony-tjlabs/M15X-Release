"""
DeepCon-M15X Configuration
==============================
모든 경로, 상수, 설정을 중앙 관리
"""
import os
from pathlib import Path

# ★ .env 파일 로드 (프로젝트 .env → 루트 .env 순서로 탐색)
def _load_env(path: Path, override: bool = False) -> None:
    """
    .env 파일을 로드한다.
    override=True: 기존 값을 덮어씀 (프로젝트별 설정)
    override=False: 빈 값 포함 미설정 키만 채움 (공유 secrets fallback)
    """
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=override)
    except ImportError:
        with open(path) as _f:
            for _line in _f:
                _line = _line.strip()
                if "=" in _line and not _line.startswith("#"):
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip()
                    if override:
                        # 값이 있을 때만 덮어씀 (빈값으로 유효 키 삭제 방지)
                        if _v:
                            os.environ[_k] = _v
                    else:
                        # 미설정이거나 빈 값이면 채움
                        if not os.environ.get(_k):
                            os.environ[_k] = _v

_load_env(Path(__file__).resolve().parent.parent.parent / ".env", override=False)  # 루트 .env: 공유 secrets
_load_env(Path(__file__).resolve().parent / ".env", override=True)                 # 프로젝트 .env: 프로젝트 설정 우선


def _get_secret(key: str, default: str = "") -> str:
    """환경변수 → Streamlit secrets → default 순으로 조회."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

# ─── 배포 모드 ───────────────────────────────────────────────────
# Streamlit Cloud 배포 시 True → 파이프라인 숨김, Drive에서 데이터 로드
CLOUD_MODE = _get_secret("CLOUD_MODE", "true").lower() == "true"

# ─── 기본 경로 ─────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPATIAL_DIR   = DATA_DIR / "spatial_model"
INDEX_DIR      = DATA_DIR / "index"
MODEL_DIR      = DATA_DIR / "model"          # Journey 임베딩 (하위호환)
EQUIPMENT_DIR  = DATA_DIR / "equipment"     # ★ 테이블 리프트 장비 캐시

# ─── M15X Raw 데이터 경로 ───────────────────────────────────────────
RAW_M15X_CONFIG_DIR    = RAW_DIR / "M15X_Configuration"
RAW_M15X_FLOW_DIR      = RAW_DIR / "M15X_FlowData_20260407"
RAW_M15X_EQUIPMENT_DIR = RAW_DIR / "M15X_TableLiftData_20260407"

# ─── M15X 단일 파일 경로 (날짜별 분리 필요) ─────────────────────────
RAW_TWARD_SINGLE_FILE  = RAW_M15X_FLOW_DIR / "M15X_TWardData_20260209~20260313.csv"
RAW_ACCESS_SINGLE_FILE = RAW_M15X_FLOW_DIR / "M15X_AccessLog_20260209~20260313.csv"
RAW_EQUIP_TWARD_FILE   = RAW_M15X_EQUIPMENT_DIR / "M15X_TableLift_TWardData_20260209~20260313.csv"
RAW_EQUIP_DEVICE_FILE  = RAW_M15X_EQUIPMENT_DIR / "M15X_TableLift_DeviceData_20260209~20260313.csv"
RAW_EQUIP_INFO_FILE    = RAW_M15X_EQUIPMENT_DIR / "M15X_TableLiftInfo_20260407.csv"

# ─── Raw 파일 패턴 ─────────────────────────────────────────────────
RAW_DATE_FORMAT  = "%Y%m%d"
RAW_ENCODING     = "cp949"

# ─── Processed 파일 패턴 ───────────────────────────────────────────
PROCESSED_ENCODING = "utf-8"

# ─── 데이터 처리 상수 ──────────────────────────────────────────────
MIN_SIGNAL_COUNT           = 1      # 최소 신호 수
MIN_ACTIVE_RATIO           = 0.0    # 최소 활성 비율
WORK_START_HOUR            = 5      # 05:00 이후 유효
WORK_END_HOUR              = 23     # 23:00 이전
JOURNEY_CORRECTION_ENABLED  = True   # Phase 1(슬라이딩 윈도우) + Phase 2(DBSCAN) 활성
JOURNEY_TOKENIZATION_ENABLED = True  # Journey 의미 블록 토큰화 활성
DATA_VALIDATION_ENABLED      = True  # 데이터 정합성 검증 활성

# ─── 대기시간 분석 임계값 ────────────────────────────────────────────
TRANSIT_GAP_THRESHOLD_MIN   = 5     # 5분 이상 간격 = 이동 구간
LUNCH_START_HOUR            = 11    # 중식 시작
LUNCH_END_HOUR              = 14    # 중식 종료
WORK_END_HOUR_THRESHOLD     = 17    # 퇴근 시작 판단

# ─── 가동률 계산 임계값 ──────────────────────────────────────────────
EQUIPMENT_ACTIVE_THRESHOLD  = 0.5   # active_ratio >= 0.5 = 가동 중 (진동센서 기반, 50% 이상 활성신호)
EQUIPMENT_WORK_STATUS_ON    = 1     # ttag_work_status = 1 = 작업 중

# ─── OPR 센서 이상 탐지 ─────────────────────────────────────────────
OPR_SENSOR_ANOMALY_THRESHOLD = 0.99  # OPR >= 99% = 통계에서 제외 (센서 이상 의심)

# ─── Locus 버전 ──────────────────────────────────────────────────────
# "v1" = 기존 58개 Locus (spot_name_map.json 기반)
# "v2" = 신규 213개 Locus (S-Ward 좌표 기반)
LOCUS_VERSION = "v1"  # v1: spot_name 기반 11개 Locus (M15X)

# ─── Sector 레지스트리 ─────────────────────────────────────────────
# 새 Sector 추가 시 이 딕셔너리에만 등록하면 됨
SECTOR_REGISTRY: dict[str, dict] = {
    "M15X_SKHynix": {
        "label":    "SK하이닉스 M15X",
        "subtitle": "반도체 FAB 건설현장",
        "icon":     "🏭",
        "domain":   "construction",
        "status":   "active",
        "access_log_prefix": "M15X_AccessLog",
        "tward_prefix":      "M15X_TWardData",
    },
}

# ─── 사용자(계정) 레지스트리 ────────────────────────────────────────
# sectors=None → 모든 Sector 접근 가능 (admin)
# sectors=[...] → 해당 Sector만 접근 가능 (client)
# ★ 비밀번호: .env 또는 환경변수/Streamlit secrets에서만 로드 (코드에 기본값 없음)

USER_REGISTRY: dict[str, dict] = {
    "administrator": {
        "label":    "Administrator",
        "icon":     "⚙️",
        "role":     "admin",
        "password": _get_secret("APOLLO_ADMIN_PASSWORD"),
        "sectors":  None,              # None = 전체 Sector
    },
    "M15X_SKHynix": {
        "label":    "SK하이닉스 M15X",
        "icon":     "🏭",
        "role":     "client",
        "password": _get_secret("APOLLO_M15X_PASSWORD"),
        "sectors":  ["M15X_SKHynix"],
    },
}

# ─── Sector별 경로 헬퍼 ────────────────────────────────────────────
def get_sector_paths(sector_id: str) -> dict:
    """
    Sector ID → 경로 딕셔너리 반환.

    반환 키:
        raw_dir, processed_dir, spatial_dir,
        ssmp_dir, locus_dir, adjacency_dir,
        spot_name_map, locus_csv, locus_xlsx, adjacency
    """
    raw_dir     = RAW_DIR       / sector_id
    proc_dir    = PROCESSED_DIR / sector_id
    spatial_dir = SPATIAL_DIR   / sector_id
    locus_dir   = spatial_dir   / "locus"
    adj_dir     = spatial_dir   / "adjacency"
    equip_dir = EQUIPMENT_DIR / sector_id
    return {
        "raw_dir":       raw_dir,
        "processed_dir": proc_dir,
        "spatial_dir":   spatial_dir,
        "ssmp_dir":      spatial_dir / "ssmp",
        "locus_dir":     locus_dir,
        "adjacency_dir": adj_dir,
        "spot_name_map": locus_dir   / "spot_name_map.json",
        "locus_csv":     locus_dir   / ("locus_v2.csv" if LOCUS_VERSION == "v2" else "locus.csv"),
        "locus_xlsx":    locus_dir   / "Locus.xlsx",
        "adjacency":     adj_dir     / "locus_adjacency.csv",
        # v2 경로
        "locus_v2_csv":  locus_dir   / "locus_v2.csv",
        "gateway_csv":   spatial_dir / "gateway" / "GatewayPoint.csv",
        "locus_map_xlsx": locus_dir  / "Locus_SWard_Map.xlsx",
        # M15X Configuration
        "gateway_raw_csv":  RAW_M15X_CONFIG_DIR / "M15X_GatewayPoint_20260407.csv",
        "active_area_csv":  RAW_M15X_CONFIG_DIR / "M15X_ActiveArea_20260407.csv",
        "map_point_csv":    RAW_M15X_CONFIG_DIR / "M15X_MapPoint_20260407.csv",
        "spot_point_csv":   RAW_M15X_CONFIG_DIR / "M15X_SpotPoint_20260407.csv",
        # Equipment
        "equipment_dir":    equip_dir,
        "equipment_master": equip_dir / "master.parquet",
        "equipment_weekly": equip_dir / "weekly",
    }


def get_allowed_sectors_for_user(user_id: str) -> list[str]:
    """해당 사용자가 접근 가능한 활성 Sector 목록."""
    user    = USER_REGISTRY.get(user_id, {})
    allowed = user.get("sectors")                        # None = admin
    active  = [sid for sid, info in SECTOR_REGISTRY.items()
               if info.get("status") == "active"]
    if allowed is None:
        return active                                     # 전체 접근
    return [s for s in allowed if s in SECTOR_REGISTRY
            and SECTOR_REGISTRY[s].get("status") == "active"]


# ─── 하위 호환 (기존 코드가 cfg.XXX_SECTOR_DIR 등을 직접 쓰는 경우 대비) ──
# 새 코드는 get_sector_paths() 사용 권장
SECTOR_ID            = "M15X_SKHynix"
SECTOR_LABEL         = "SK하이닉스 M15X FAB 건설현장"
RAW_SECTOR_DIR       = RAW_DIR       / SECTOR_ID
PROCESSED_SECTOR_DIR = PROCESSED_DIR / SECTOR_ID
SPATIAL_SECTOR_DIR   = SPATIAL_DIR   / SECTOR_ID
SSMP_DIR             = SPATIAL_SECTOR_DIR / "ssmp"
LOCUS_DIR            = SPATIAL_SECTOR_DIR / "locus"
ADJACENCY_DIR        = SPATIAL_SECTOR_DIR / "adjacency"
SPOT_NAME_MAP_FILE   = LOCUS_DIR / "spot_name_map.json"
LOCUS_CSV_FILE       = LOCUS_DIR / "locus.csv"
LOCUS_XLSX_FILE      = LOCUS_DIR / "Locus.xlsx"
ADJACENCY_FILE       = ADJACENCY_DIR / "locus_adjacency.csv"

# ─── Dashboard 설정 ────────────────────────────────────────────────
APP_TITLE     = "Deep Con at M15X"
APP_SUBTITLE  = "Spatial Data Analysis using Deep Con Prototype"
APP_ICON      = "🌐"
THEME_PRIMARY = "#1A3A5C"
THEME_ACCENT  = "#00AEEF"
THEME_BG      = "#0D1B2A"
THEME_CARD_BG = "#1A2A3A"
THEME_TEXT    = "#C8D6E8"
THEME_SUCCESS = "#00C897"
THEME_WARNING = "#FFB300"
THEME_DANGER  = "#FF4C4C"

# ─── 리포트 설정 ───────────────────────────────────────────────────
REPORT_OUTPUT_DIR = BASE_DIR / "output" / "reports"
WEEKLY_REPORT_DAY = 0   # 0=월요일

# ─── LLM 백엔드 설정 ─────────────────────────────────────────────
# "anthropic" (기본값) 또는 "bedrock" (AWS Bedrock)
LLM_BACKEND        = _get_secret("LLM_BACKEND", "anthropic")
AWS_REGION         = _get_secret("AWS_REGION", "ap-northeast-2")
AWS_BEDROCK_MODEL_ID = _get_secret("AWS_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-v1")

# ─── 익명화 설정 ─────────────────────────────────────────────────
# True(기본값): LLM 호출 전 작업자/업체/구역명 마스킹
ANONYMIZE_LLM = _get_secret("ANONYMIZE_LLM", "true").lower() == "true"
# True(기본값): LLM 호출 전 알고리즘 공식도 추상화
ANONYMIZE_LOGIC = _get_secret("ANONYMIZE_LOGIC", "true").lower() == "true"

# ─── 보안 모드 ───────────────────────────────────────────────────
# True(기본값): 방법론 탭에서 상세 알고리즘 숨김 (배포용)
# False: 개발 환경에서 상세 표시 (디버깅용)
SECURE_MODE = _get_secret("SECURE_MODE", "true").lower() == "true"

# ─── 버전 ──────────────────────────────────────────────────────────
APP_VERSION   = "0.1.0"
CACHE_VERSION = "v1"
