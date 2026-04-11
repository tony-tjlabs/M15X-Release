"""
DeepCon-M15X Anonymizer v2 — 데이터 + 로직 익명화
============================================
LLM 호출 시:
  1. 개인정보 마스킹 (작업자/업체/구역명)
  2. 알고리즘 공식 추상화 (EWI/CRE 수식 등)
  3. 수치 데이터의 스케일링/상대화 (절대 수치 대신 상대 비율)

사용:
    anon = Anonymizer(anonymize_logic=True)
    masked = anon.mask(text, worker_names=["홍길동"], ...)
    # masked: "Worker_001이 Company_A에서 ..."
    restored = anon.unmask(response_text)

    # 단일 이름 마스킹 (UI 표시용)
    from src.pipeline.anonymizer import mask_name
    masked = mask_name("홍길동")  # -> "홍**"

    # 수치 상대화 (LLM 전송용)
    relativized = relativize_metrics(worker_df, baseline_df)
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ─── 로직 추상화 매핑 ─────────────────────────────────────────────────────
# EWI/CRE 등 핵심 공식을 일반 용어로 변환
LOGIC_ABSTRACTIONS = {
    # EWI 관련 — 상세 수식 숨김
    r"EWI\s*=\s*[^,\n\]]+": "작업 집중도 지수(계산식 비공개)",
    r"H_high\s*[×x*]\s*[\d.]+\s*\+\s*H_low\s*[×x*]\s*[\d.]+": "가중 활동 시간 합계",
    r"H_high\s*[×x*]\s*1\.0\s*\+\s*H_low\s*[×x*]\s*0\.5\s*\+\s*H_standby\s*[×x*]\s*0\.2": "가중 활동 시간(비공개)",
    r"active_ratio\s*[>=<]+\s*0\.\d+": "활성도 기준 적용",
    r"active_ratio\s*≥\s*0\.\d+": "활성도 기준 적용",

    # CRE 관련 — 상세 수식 숨김
    r"CRE\s*=\s*[^,\n\]]+": "복합 위험 노출도(계산식 비공개)",
    r"P_norm\s*\+\s*S_norm\s*\+\s*D_norm": "개인/공간/밀집 위험 통합",
    r"0\.\d+\s*[×x*]\s*[PFSDR]_\w+": "가중치 적용(비공개)",
    r"\d+\.\d+\s*[×x*]\s*P_norm\s*\+\s*\d+\.\d+\s*[×x*]\s*S_norm\s*\+\s*\d+\.\d+\s*[×x*]\s*D_norm": "위험 통합(비공개)",

    # 보정 관련 — 알고리즘 이름 일반화
    r"DBSCAN": "클러스터링 알고리즘",
    r"Run-Length": "노이즈 보정",
    r"플리커\s*제거": "신호 보정",
    r"MAX_FLICKER_RUN\s*=\s*\d+": "보정 파라미터(비공개)",
    r"eps\s*=\s*[\d.]+": "파라미터(비공개)",
    r"min_samples\s*=\s*\d+": "파라미터(비공개)",

    # Word2Vec/K-means — 하이퍼파라미터 숨김
    r"dim\s*=\s*\d+": "벡터 차원(비공개)",
    r"window\s*=\s*\d+": "윈도우 크기(비공개)",
    r"n_clusters\s*=\s*\d+": "클러스터 수(비공개)",
    r"Word2Vec": "임베딩 모델",
    r"K-means|KMeans|k-means": "클러스터링 모델",

    # 공간 위험 가중치 — 구체적 수치 숨김
    r"confined_space.*?2\.0": "밀폐공간(고위험)",
    r"high_voltage.*?2\.0": "고압전(고위험)",
    r"mechanical_room.*?1\.8": "기계실(위험)",
    r"w_space\s*=\s*[\d.]+": "공간 위험 가중치(비공개)",
}


# ─── 단일 이름 마스킹 함수 (UI 표시용) ─────────────────────────────────


def mask_name(name: str | None) -> str | None:
    """
    작업자 이름을 마스킹. 성(첫 글자)만 남기고 나머지는 ** 로 표시.

    예시:
        "이택진" -> "이**"
        "홍길동" -> "홍**"
        "Kim" -> "K**"
        "J" -> "J**"  (1글자도 처리)
        None/NaN -> None
        "" -> ""

    Args:
        name: 원본 이름

    Returns:
        마스킹된 이름 또는 None/빈값 그대로
    """
    # None, NaN, 빈 문자열 처리
    if name is None:
        return None
    if pd.isna(name):
        return name
    name_str = str(name).strip()
    if not name_str:
        return name_str

    # 이미 마스킹된 경우 (** 포함) 그대로 반환
    if "**" in name_str:
        return name_str

    # 첫 글자만 남기고 ** 추가
    return f"{name_str[0]}**"


def mask_name_series(series: pd.Series) -> pd.Series:
    """
    pandas Series의 모든 이름을 마스킹.

    Args:
        series: user_name 등의 Series

    Returns:
        마스킹된 Series
    """
    return series.apply(mask_name)


class Anonymizer:
    """작업자/업체/구역명 + 알고리즘 로직을 익명화한다."""

    def __init__(self, anonymize_logic: bool = True) -> None:
        """
        Args:
            anonymize_logic: True면 알고리즘 공식도 추상적 설명으로 교체
        """
        self._map: dict[str, str] = {}        # original → masked
        self._reverse: dict[str, str] = {}    # masked → original
        self._counters = {"worker": 0, "company": 0, "zone": 0}
        self._anonymize_logic = anonymize_logic

    # ─── public ─────────────────────────────────────────────────────

    def mask(
        self,
        text: str,
        *,
        worker_names: Iterable[str] | None = None,
        company_names: Iterable[str] | None = None,
        zone_names: Iterable[str] | None = None,
    ) -> str:
        """
        텍스트 내 모든 알려진 이름을 익명 코드로 치환.

        치환 순서: 길이가 긴 이름부터 처리하여 부분 치환을 방지한다.
        예) "광건티앤씨(주)_IBL" 이 "광건" 보다 먼저 치환됨.
        """
        # 1) 새 이름 등록
        for name in _unique_sorted(worker_names):
            self._register(name, "worker")
        for name in _unique_sorted(company_names):
            self._register(name, "company")
        for name in _unique_sorted(zone_names):
            self._register(name, "zone")

        # 2) 길이 내림차순으로 치환 (부분 치환 방지)
        pairs = sorted(self._map.items(), key=lambda kv: len(kv[0]), reverse=True)
        for original, masked in pairs:
            text = text.replace(original, masked)

        # 3) 알고리즘 공식 추상화 (신규 v2)
        if self._anonymize_logic:
            text = self._mask_logic(text)

        return text

    def _mask_logic(self, text: str) -> str:
        """알고리즘 공식을 추상적 설명으로 교체."""
        for pattern, replacement in LOGIC_ABSTRACTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def unmask(self, text: str) -> str:
        """
        익명 코드를 원래 이름으로 복원.

        Note: 로직 추상화는 복원 불가 (의도적으로 비가역 처리)
        """
        # 길이 내림차순으로 복원 (Worker_001 → "홍길동" 이 Worker_00 보다 먼저)
        pairs = sorted(self._reverse.items(), key=lambda kv: len(kv[0]), reverse=True)
        for masked, original in pairs:
            text = text.replace(masked, original)
        return text

    def get_mapping(self) -> dict[str, str]:
        """현재 매핑 테이블 반환 (디버그/감사용)."""
        return dict(self._map)

    def reset(self) -> None:
        """매핑 초기화."""
        self._map.clear()
        self._reverse.clear()
        self._counters = {"worker": 0, "company": 0, "zone": 0}

    # ─── internal ───────────────────────────────────────────────────

    def _register(self, name: str, category: str) -> str:
        """이름을 카테고리별 익명 코드로 등록. 이미 등록된 이름은 스킵."""
        name = name.strip()
        if not name or name in self._map:
            return self._map.get(name, "")

        self._counters[category] += 1
        idx = self._counters[category]

        if category == "worker":
            code = f"Worker_{idx:03d}"
        elif category == "company":
            # A, B, C, ... Z, AA, AB, ...
            code = f"Company_{_alpha_code(idx)}"
        else:  # zone
            code = f"Zone_{idx:03d}"

        self._map[name] = code
        self._reverse[code] = name
        logger.debug("Anonymizer: %r → %s", name, code)
        return code


# ─── helpers ────────────────────────────────────────────────────────

def _unique_sorted(names: Iterable[str] | None) -> list[str]:
    """중복 제거 + 길이 내림차순 정렬."""
    if not names:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    result.sort(key=len, reverse=True)
    return result


def _alpha_code(n: int) -> str:
    """1→A, 2→B, ..., 26→Z, 27→AA, ..."""
    result = []
    while n > 0:
        n -= 1
        result.append(chr(65 + n % 26))
        n //= 26
    return "".join(reversed(result))


# ─── 수치 상대화 함수 (LLM 전송용) ──────────────────────────────────────


def relativize_metrics(
    current_avg: float,
    baseline_avg: float,
    metric_name: str = "지표",
) -> str:
    """
    절대 수치 대신 상대적 변화로 표현.

    LLM에 전송 시 구체적 수치보다 상대 비율을 제공하여
    핵심 알고리즘 파라미터 유추를 방지.

    Args:
        current_avg: 현재 평균값
        baseline_avg: 기준(평소) 평균값
        metric_name: 지표명 (예: "EWI", "CRE")

    Returns:
        상대적 변화 설명 문자열

    Examples:
        >>> relativize_metrics(0.65, 0.55, "EWI")
        "EWI가 평소 대비 약 18% 높음 (상위 수준)"
    """
    if baseline_avg == 0 or pd.isna(baseline_avg):
        if current_avg > 0.7:
            return f"{metric_name}가 높은 수준"
        elif current_avg > 0.4:
            return f"{metric_name}가 보통 수준"
        else:
            return f"{metric_name}가 낮은 수준"

    pct_change = (current_avg - baseline_avg) / baseline_avg * 100

    # 등급 판정 (절대 수치 대신 상대적 위치)
    if current_avg >= 0.7:
        level = "상위 수준"
    elif current_avg >= 0.5:
        level = "중상위 수준"
    elif current_avg >= 0.3:
        level = "중간 수준"
    else:
        level = "하위 수준"

    # 변화 방향
    if abs(pct_change) < 5:
        direction = "평소와 유사"
    elif pct_change > 0:
        direction = f"평소 대비 약 {abs(pct_change):.0f}% 높음"
    else:
        direction = f"평소 대비 약 {abs(pct_change):.0f}% 낮음"

    return f"{metric_name}가 {direction} ({level})"


def anonymize_metrics_for_llm(
    worker_count: int,
    avg_ewi: float,
    avg_cre: float,
    baseline_ewi: float = 0.5,
    baseline_cre: float = 0.35,
) -> str:
    """
    LLM 프롬프트용 메트릭 익명화 요약.

    절대 수치 대신 상대적 표현 + 등급으로 제공.

    Args:
        worker_count: 작업자 수
        avg_ewi: 평균 EWI
        avg_cre: 평균 CRE
        baseline_ewi: EWI 기준값 (기본 0.5)
        baseline_cre: CRE 기준값 (기본 0.35)

    Returns:
        익명화된 메트릭 요약 문자열
    """
    # 작업자 수는 범위로 표현 (정확한 수치 숨김)
    if worker_count < 100:
        worker_range = "소규모(100명 미만)"
    elif worker_count < 1000:
        worker_range = "중규모(100~1000명)"
    elif worker_count < 5000:
        worker_range = "대규모(1000~5000명)"
    else:
        worker_range = "초대규모(5000명 이상)"

    ewi_desc = relativize_metrics(avg_ewi, baseline_ewi, "작업 집중도")
    cre_desc = relativize_metrics(avg_cre, baseline_cre, "위험 노출도")

    return f"""
작업자 규모: {worker_range}
{ewi_desc}
{cre_desc}
    """.strip()


def anonymize_high_risk_list(
    high_risk_count: int,
    total_count: int,
) -> str:
    """
    고위험 작업자 수를 비율로 표현.

    Args:
        high_risk_count: 고위험 작업자 수
        total_count: 전체 작업자 수

    Returns:
        상대적 표현 문자열
    """
    if total_count == 0:
        return "데이터 없음"

    ratio = high_risk_count / total_count * 100

    if ratio < 1:
        return "고위험 작업자 비율 매우 낮음 (1% 미만)"
    elif ratio < 5:
        return f"고위험 작업자 비율 낮음 (약 {ratio:.1f}%)"
    elif ratio < 10:
        return f"고위험 작업자 비율 보통 (약 {ratio:.1f}%)"
    else:
        return f"고위험 작업자 비율 높음 (약 {ratio:.1f}%)"
