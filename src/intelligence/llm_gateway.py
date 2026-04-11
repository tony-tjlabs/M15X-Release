"""
LLM Gateway — 모든 LLM 호출의 단일 진입점
==========================================
모든 탭의 AI 분석은 이 Gateway를 통과해야 한다.

흐름:
  1. DataPackager  → 탭별 풍부한 데이터 텍스트 조립
  2. AnonymizationPipeline → 익명화/보안 처리 (우회 불가)
  3. PromptBuilder → 시스템 프롬프트 + 유저 프롬프트 조립
  4. LLM 호출      → _call_claude (Anthropic 또는 Bedrock)
  5. AuditLog      → 전송된 내용 기록 (세션 내 감사)

탭 ID:
  - "transit"    : 대기/이동 시간 분석
  - "equipment"  : 테이블 리프트 가동률
  - "worker"     : 작업자 분석
  - "congestion" : 공간 혼잡도
  - "overview"   : 현장 개요

사용:
    from src.intelligence.llm_gateway import LLMGateway

    result = LLMGateway.analyze(
        tab_id="transit",
        packed_text=DataPackager.transit(worker_df, dates),
        company_names=list(worker_df["company_name"].dropna().unique()),
        date_list=sorted(worker_df["date"].unique()),
    )
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ─── 탭별 시스템 프롬프트 설정 ────────────────────────────────────────────────

_TAB_CONTEXT: dict[str, str] = {
    "transit": """
분석 주제: 작업자 대기/이동 시간 (MAT/LBT/EOD)
- MAT(출근 대기): 출입 → 첫 FAB 작업구역 도달 시간. 15분 초과 시 출입 병목 의심.
- LBT(중식 이동): 중식 이동 패턴. 요일별 변동이 클 수 있음 (금요일 단축).
- EOD(퇴근 이동): 마지막 작업구역 → 출입구 이동. 18분 초과 시 퇴근 혼잡 의심.
- TWT(총 대기): MAT+LBT+EOD 합계. 건설현장 기준 30분 초과 시 주목할 수준.
- 음영(GAP): 건설현장 BLE 특성상 평균 50% 음영은 정상. 음영 중 이동도 발생.
""".strip(),

    "equipment": """
분석 주제: 테이블 리프트 가동률 (OPR)
- OPR: 진동센서 기반 실제 가동 비율. 60% 미만 시 작업 흐름 개선 여지.
- 층별 편차: 공사 진행 단계에 따라 층별 가동률 차이 발생 (정상).
- 비가동 원인: 점검(정기), 대기(자재/인력), 미사용(공정 완료) 구분 필요.
- 5주 추이: 가동률 지속 하락은 공정 후반부 진입 또는 설비 노후 신호.
""".strip(),

    "worker": """
분석 주제: 작업자 작업강도 및 위험노출도 (EWI/CRE)
- EWI(작업집중도): 진동센서 기반 실제 작업 강도. 0.7 이상 = 고집중, 0.4 미만 = 저집중.
- CRE(위험노출도): 공간 위험 + 밀집도 + 개인 특성 통합. 0.5 이상 = 고위험 주의.
- 고위험 비율: 전체의 10% 초과 시 현장 안전 관리 강화 신호.
- 활성레벨: HIGH_ACTIVE=작업 중, ACTIVE=경작업, INACTIVE=대기, DEEP_INACTIVE=장기정지.
""".strip(),

    "congestion": """
분석 주제: 공간별 혼잡도 및 점유 패턴
- 피크 혼잡: 식사시간(11~13시), 출퇴근(07~09시, 17~19시) 집중 현상 정상.
- 작업구역 혼잡: 층간 병목 또는 특정 공정 집중이 원인일 수 있음.
- 이동구역(호이스트) 혼잡: 층간 이동 대기 → MAT/LBT에 직접 영향.
- 밀폐공간/고압전 구역 과밀: 안전 규정(2인1조) 준수 확인 필요.
""".strip(),

    "overview": """
분석 주제: 현장 일일 종합 현황
- 출입 인원과 T-Ward 비율: T-Ward 미착용자는 이동 경로 추적 불가.
- EWI 추이: 주초(월) 낮음 → 주중 상승 → 금요일 소폭 하락이 일반적 패턴.
- CRE 급등: 당일 특수 작업(고소/밀폐) 증가 또는 인력 집중 배치가 원인.
- 밀폐/고압전 구역: 진입 인원 증가 시 안전 관리 중점 구역으로 주목.
""".strip(),
}

_BASE_SYSTEM_PROMPT = """
당신은 반도체 FAB 건설현장 데이터 해석 보조입니다.
M15X 건설현장 (단일 FAB 건물, 5F/6F/7F/RF 4개 층)의 작업자 이동 데이터를 분석합니다.

[현장 기본 특성]
- 작업 시간: 07:00~19:00, 중식: 11:30~13:00
- BLE 음영: 건설현장 특성상 평균 50% 음영 — 정상 범위로 해석할 것
- T-Ward: 진동센서 내장 BLE 태그. active_ratio = 작업 강도의 proxy (비공개)
- 호이스트: 층간 수직 이동 설비 — 체류 시간 = 이동 대기

[해석 지침]
- MAT > 15분: 출입 병목 또는 이동 거리 과다 의심
- HIGH_ACTIVE < 30%: 실질 작업 시간 부족 신호
- 장비 가동률 < 60%: 작업 흐름 개선 여지
- CRE ≥ 0.5 비율 > 10%: 안전 관리 강화 필요 신호
- 요일별 차이 > 20%: 요일 패턴(주초 적응, 금요일 조기 종료) 고려

[역할 제한]
- 수치 요약 및 패턴 해석만 수행
- 원인 단정, 특정 조치 권고, "~해야 합니다" 사용 금지
- 알고리즘 수식/계산 방법 언급 금지
- 특정 작업자/업체 직접 언급 금지 (코드명만 허용)

[비교 컨텍스트 블록 해석 — 데이터에 포함된 경우 반드시 활용]
데이터에 "## 비교 컨텍스트" 섹션이 포함된 경우:
- [A] 기준선: 오늘 수치가 최근 N일 평균 대비 어떤 위치인지 WHAT에서 명시
  예: "오늘 EWI(0.52)는 최근 7일 평균(0.48) 대비 +0.04로 다소 높은 수준입니다."
- [B] 요일 패턴: 같은 요일 대비 특이점이 있으면 언급
- [C] 직전일 변화: 방향성(▲/▼)을 WHY 해석에 활용
- [D] 순위: "N일 중 X위" 정보를 상대적 위치 표현에 활용
컨텍스트가 없거나 N일 데이터가 적으면 해당 비교는 생략.

[출력 형식 — 반드시 준수]
[WHAT] 데이터가 보여주는 주요 현상 2~3문장 (수치 직접 인용, 비교 컨텍스트 활용)
[WHY] 패턴의 현장 맥락 해석 1~2문장 (원인 단정 금지, "~로 보임" 형식)
[NOTE] 해석 한계 또는 추가 확인 필요 사항 1문장

한국어, 간결체 사용.
""".strip()


def _build_prompt(tab_id: str, data_text: str) -> str:
    """탭별 시스템 컨텍스트 + 데이터를 조합한 유저 프롬프트 생성."""
    tab_ctx = _TAB_CONTEXT.get(tab_id, "")
    return f"""
{tab_ctx}

---

{data_text}

---

위 데이터를 바탕으로 건설현장 현장 소장 관점의 인사이트를 제공하세요.
출력 형식([WHAT]/[WHY]/[NOTE])을 반드시 사용하세요.
""".strip()


# ─── 감사 로그 ────────────────────────────────────────────────────────────────

def _audit_log(tab_id: str, prompt_text: str) -> None:
    """세션 내 전송 내용 감사 로그."""
    try:
        key = "_llm_audit_log"
        if key not in st.session_state:
            st.session_state[key] = []
        st.session_state[key].append({
            "time": datetime.now().isoformat(),
            "tab": tab_id,
            "chars": len(prompt_text),
            "preview": prompt_text[:200],
        })
    except Exception:
        pass


def get_audit_log() -> list[dict]:
    """세션 내 감사 로그 반환 (관리자 탭용)."""
    try:
        return st.session_state.get("_llm_audit_log", [])
    except Exception:
        return []


# ─── 메인 Gateway ─────────────────────────────────────────────────────────────

class LLMGateway:
    """
    모든 LLM 호출의 단일 진입점.

    1) packed_text 를 AnonymizationPipeline 으로 익명화
    2) 탭별 강화된 시스템 프롬프트 + 유저 프롬프트 조립
    3) _call_claude 호출
    4) 감사 로그 기록
    """

    @staticmethod
    def analyze(
        tab_id: str,
        packed_text: str,
        *,
        company_names: list[str] | None = None,
        worker_names: list[str] | None = None,
        zone_names: list[str] | None = None,
        date_list: list[str] | None = None,
        max_tokens: int = 900,
    ) -> str:
        """
        탭별 AI 분석 실행.

        Args:
            tab_id:        탭 식별자 ("transit"|"equipment"|"worker"|"congestion"|"overview")
            packed_text:   DataPackager가 생성한 풍부한 데이터 텍스트
            company_names: 익명화할 업체명 목록 (자동 코드로 치환)
            worker_names:  익명화할 작업자명 목록
            zone_names:    익명화할 구역명 목록
            date_list:     날짜 목록 (상대화 처리)
            max_tokens:    LLM 최대 토큰

        Returns:
            LLM 응답 텍스트 (빈 문자열 = 실패)
        """
        from src.intelligence.anonymization_pipeline import AnonymizationPipeline
        from src.dashboard.llm_apollo import _call_claude

        # Step 1: 익명화 파이프라인 (우회 불가)
        safe_text = AnonymizationPipeline.run(
            packed_text,
            company_names=company_names,
            worker_names=worker_names,
            zone_names=zone_names,
            date_list=date_list,
        )

        # Step 2: 프롬프트 조립
        user_prompt = _build_prompt(tab_id, safe_text)

        # Step 3: 감사 로그
        _audit_log(tab_id, user_prompt)

        logger.info(
            "LLMGateway.analyze: tab=%s, chars=%d, companies=%d",
            tab_id,
            len(user_prompt),
            len(company_names or []),
        )

        # Step 4: LLM 호출 (기존 _call_claude 재사용 — 시스템 프롬프트 주입됨)
        # 탭별 시스템 프롬프트를 _BASE_SYSTEM_PROMPT로 일시 오버라이드
        result = _call_claude_with_system(
            prompt=user_prompt,
            system=_BASE_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )

        return result

    @staticmethod
    def is_available() -> bool:
        """LLM 사용 가능 여부."""
        from src.dashboard.llm_apollo import is_llm_available
        return is_llm_available()


def _call_claude_with_system(prompt: str, system: str, max_tokens: int) -> str:
    """
    강화된 시스템 프롬프트로 Claude 호출.
    기존 llm_apollo._call_anthropic 패턴과 동일하되 system 파라미터 주입.
    """
    import os
    from pathlib import Path

    # API 키 로드
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)
    except ImportError:
        pass

    try:
        import config as cfg
        backend = getattr(cfg, "LLM_BACKEND", "anthropic")
        aws_region = getattr(cfg, "AWS_REGION", "ap-northeast-2")
        aws_model = getattr(cfg, "AWS_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-v1")
    except Exception:
        backend = os.getenv("LLM_BACKEND", "anthropic")
        aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
        aws_model = os.getenv("AWS_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-v1")

    if backend == "bedrock":
        return _bedrock_call(prompt, system, max_tokens, aws_region, aws_model)
    return _anthropic_call(prompt, system, max_tokens)


def _anthropic_call(prompt: str, system: str, max_tokens: int) -> str:
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY 미설정")
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning("Anthropic 호출 실패: %s", e)
        return ""


def _bedrock_call(prompt: str, system: str, max_tokens: int, region: str, model_id: str) -> str:
    import json
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=region)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(modelId=model_id, body=body)
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()
    except Exception as e:
        logger.warning("Bedrock 호출 실패: %s", e)
        return ""
