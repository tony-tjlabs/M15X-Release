"""
Anonymization Pipeline — LLM 전송 전 단일 보안 허브
====================================================
모든 LLM 호출은 반드시 이 파이프라인을 통과해야 한다.

보안 5단계:
  Step 1: k-Anonymity   — 집단 크기 < K_ANON_MIN 억제
  Step 2: 식별자 차단   — 이름/업체명/user_no → 세션 코드
  Step 3: 날짜 상대화   — "2026-03-10" → "분석 1일차(월)"
  Step 4: 수치 맥락화   — 절대값 + 상대 변화율 병기
  Step 5: 알고리즘 추상 — EWI/CRE 수식 등 제거 (ANONYMIZE_LOGIC)

사용:
    from src.intelligence.anonymization_pipeline import AnonymizationPipeline

    safe_text = AnonymizationPipeline.run(
        text,
        company_names=["A사", "B사"],
        date_list=["20260310", "20260311"],
    )
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# k-Anonymity 임계값: 이 미만 집단은 수치 억제
K_ANON_MIN = 10

# 요일 레이블
_DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _load_cfg() -> tuple[bool, bool]:
    """(ANONYMIZE_LLM, ANONYMIZE_LOGIC) 반환."""
    try:
        import config as cfg
        return (
            getattr(cfg, "ANONYMIZE_LLM", True),
            getattr(cfg, "ANONYMIZE_LOGIC", True),
        )
    except Exception:
        return True, True


def _get_session_anon():
    """
    세션 스코프 Anonymizer 반환.
    같은 세션 내에서 업체명 코드가 일관되게 유지된다.
    """
    try:
        import streamlit as st
        _, anonymize_logic = _load_cfg()
        key = f"_anon_pipeline_{anonymize_logic}"
        if key not in st.session_state:
            from src.pipeline.anonymizer import Anonymizer
            st.session_state[key] = Anonymizer(anonymize_logic=anonymize_logic)
        return st.session_state[key]
    except Exception:
        # Streamlit 컨텍스트 외부 (스크립트 실행 등) — 임시 인스턴스
        from src.pipeline.anonymizer import Anonymizer
        _, anonymize_logic = _load_cfg()
        return Anonymizer(anonymize_logic=anonymize_logic)


class AnonymizationPipeline:
    """
    LLM 전송 데이터의 단일 익명화 파이프라인.

    모든 메서드는 classmethods — 인스턴스화 불필요.
    """

    # ─── 메인 진입점 ─────────────────────────────────────────────────

    @classmethod
    def run(
        cls,
        text: str,
        *,
        company_names: list[str] | None = None,
        worker_names: list[str] | None = None,
        zone_names: list[str] | None = None,
        date_list: list[str] | None = None,
    ) -> str:
        """
        텍스트를 익명화 파이프라인에 통과시킨다.

        Args:
            text:          익명화할 원본 텍스트
            company_names: 업체명 목록 (Company_A, B사 등으로 치환)
            worker_names:  작업자명 목록 (Worker_001 등으로 치환)
            zone_names:    구역명 목록 (Zone_001 등으로 치환)
            date_list:     날짜 목록 ("20260310" 또는 "2026-03-10")

        Returns:
            익명화된 텍스트
        """
        anonymize_llm, _ = _load_cfg()
        if not anonymize_llm:
            return text  # 설정으로 비활성화 시 그대로 반환

        anon = _get_session_anon()

        # Step 2: 식별자 → 코드 치환
        text = anon.mask(
            text,
            worker_names=worker_names or [],
            company_names=company_names or [],
            zone_names=zone_names or [],
        )

        # Step 3: 날짜 상대화
        if date_list:
            text = cls._relativize_dates(text, date_list)

        # user_no 패턴 제거 (숫자 ID가 남아있을 경우 대비)
        text = re.sub(r"\buser_no\s*[:=]\s*\d+", "[작업자ID 제거됨]", text)

        return text

    @classmethod
    def get_company_code(cls, company_name: str) -> str:
        """업체명의 세션 내 익명 코드를 반환한다 (Company_A 형식)."""
        anon = _get_session_anon()
        return anon._register(company_name, "company")

    # ─── k-Anonymity ─────────────────────────────────────────────────

    @classmethod
    def filter_small_groups(
        cls,
        rows: list[dict],
        count_key: str = "n_workers",
    ) -> list[dict]:
        """
        집단 크기가 K_ANON_MIN 미만인 항목을 억제한다.

        수치 컬럼은 None으로, 레이블은 "인원 부족" 표시로 변경.
        """
        result = []
        for item in rows:
            count = item.get(count_key, 0)
            if isinstance(count, (int, float)) and count < K_ANON_MIN:
                suppressed = {}
                for k, v in item.items():
                    if k == count_key:
                        suppressed[k] = f"<{K_ANON_MIN}명 (비공개)"
                    elif isinstance(v, (int, float)):
                        suppressed[k] = None
                    else:
                        suppressed[k] = v
                result.append(suppressed)
            else:
                result.append(item)
        return result

    @classmethod
    def check_k_anonymity(cls, count: int) -> bool:
        """단일 집단의 k-Anonymity 충족 여부."""
        return count >= K_ANON_MIN

    # ─── 날짜 상대화 ─────────────────────────────────────────────────

    @classmethod
    def _relativize_dates(cls, text: str, date_list: list[str]) -> str:
        """
        절대 날짜를 상대적 표현으로 치환.

        "2026-03-10" → "분석 1일차(월)"
        "20260310"   → "분석 1일차(월)"
        """
        parsed: list[tuple[str, str, datetime]] = []  # (dash_form, plain_form, dt)

        for d in date_list:
            try:
                d = d.strip()
                if len(d) == 8 and d.isdigit():
                    dt = datetime.strptime(d, "%Y%m%d")
                    dash = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                    plain = d
                elif len(d) >= 10:
                    dt = datetime.strptime(d[:10], "%Y-%m-%d")
                    dash = d[:10]
                    plain = d[:10].replace("-", "")
                else:
                    continue
                parsed.append((dash, plain, dt))
            except Exception:
                continue

        # 날짜순 정렬 후 인덱스 부여
        parsed.sort(key=lambda x: x[2])
        for idx, (dash, plain, dt) in enumerate(parsed, 1):
            dow = _DOW_KR[dt.weekday()]
            label = f"분석 {idx}일차({dow})"
            text = text.replace(dash, label)
            text = text.replace(plain, label)

        return text

    # ─── 수치 맥락화 헬퍼 ────────────────────────────────────────────

    @staticmethod
    def relativize(
        value: float,
        baseline: float,
        label: str,
        unit: str = "",
    ) -> str:
        """
        절대값을 상대적 변화와 함께 표현.

        예: relativize(14.2, 12.5, "MAT", "분")
            → "MAT 14.2분 (기준 대비 +13.6%)"
        """
        if baseline and baseline != 0:
            pct = (value - baseline) / baseline * 100
            sign = "+" if pct >= 0 else ""
            return f"{label} {value:.1f}{unit} (기준 대비 {sign}{pct:.1f}%)"
        return f"{label} {value:.1f}{unit}"

    @staticmethod
    def level_label(value: float, thresholds: dict[str, float]) -> str:
        """
        값에 따른 레벨 레이블 반환.

        thresholds: {"높음": 0.7, "보통": 0.4, "낮음": 0.0}
        """
        for label, thresh in sorted(thresholds.items(), key=lambda x: -x[1]):
            if value >= thresh:
                return label
        return list(thresholds.keys())[-1]

    # ─── 사전 구조 데이터 익명화 ─────────────────────────────────────

    @classmethod
    def sanitize_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        딕셔너리에서 민감 키를 제거한다.

        제거 키: user_no, user_name, twardid, x, y, phone, email
        """
        BLOCKED_KEYS = {
            "user_no", "user_name", "twardid", "worker_name",
            "x", "y", "phone", "email", "cellphone",
        }
        return {k: v for k, v in data.items() if k.lower() not in BLOCKED_KEYS}
