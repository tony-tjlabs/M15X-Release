"""
M15X 현장 컨텍스트 — LLM 프롬프트 기반 정보
==========================================
모든 LLM 호출에 이 컨텍스트를 첨부하여
Claude가 건설현장의 맥락을 이해한 상태에서 분석하도록 한다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.security.data_guard import DataGuard
from domain_packs.construction.prompts import (
    SYSTEM_CONTEXT as DOMAIN_SYSTEM_CONTEXT,
    build_insight_prompt as domain_build_insight_prompt,
    build_weekly_prompt as domain_build_weekly_prompt,
)

if TYPE_CHECKING:
    pass

# 모듈 레벨 DataGuard 인스턴스
_data_guard = DataGuard(audit_enabled=True)

SITE_CONTEXT = """
## 현장 개요
SK하이닉스 M15X는 경기도 이천시에 위치한 반도체 FAB 건설현장입니다.
FAB(Fabrication) 단일 건물 중심의 건설현장으로, 일일 약 1,500~2,000명의 작업자가 투입됩니다.

## 주요 건물 및 공간 (11개 Locus)
- **FAB 5F/6F/7F/RF (전체)**: FAB 건물 내부 작업구역. 4개 층으로 구성.
  클린룸 환경 구축, 고소작업, 중량물 설치, 정밀 전기/배관 작업이 집중됨.
- **FAB 건너 화장실/휴게실/흡연실**: FAB 건물 건너편 편의시설.
  작업자 휴식 및 편의 공간.
- **공사현장**: FAB 외부 야외 공사구역. 자재 이동, 장비 작업 등.
- **타각기 주변**: 출입 타각기 위치. 출퇴근 기록 지점.
- **호이스트2/호이스트3 주변**: FAB 건물 수직이동 설비. 층간 이동 경로.

## 작업 특성
- 다수의 협력업체가 동시 투입
- 주간(07:00~19:00)과 야간 교대 근무
- FAB 단일 건물 집중형 현장 (Y1 대비 소규모)
- 호이스트를 통한 층간 이동이 빈번

## 데이터 수집 시스템
- T-Ward: BLE 태그, 1분 단위 위치 기록
- 활성신호(가속도 기반): 실제 작업 활동 감지 (움직임 vs 정지)
- 11개 Locus(공간 단위)
- EWI(유효작업집중도): 고활성×1.0 + 저활성×0.5 + 대기×0.2 / 근무시간
- CRE(건설위험노출도): 공간위험(40%) + 동적밀집(15%) + 개인위험(45%)

## 분석 시 유의사항
- T-Ward 미착용자(twardid=NaN)는 이동 데이터 없음 → 분석 대상 제외
- FAB 단일 건물이므로 건물 간 이동 분석은 해당 없음
- 호이스트 주변 체류 = 층간 이동 대기 (작업 활동 아님)
""".strip()

# 하위호환: 기존 코드에서 Y1_SITE_CONTEXT 참조하는 경우
Y1_SITE_CONTEXT = SITE_CONTEXT


def build_space_context(locus_dict: dict) -> str:
    """
    Enriched locus 정보로 공간 맥락 문자열 생성.

    Args:
        locus_dict: enriched locus 정보

    Returns:
        공간 유형별 특성을 설명하는 마크다운 문자열
    """
    if not locus_dict:
        return ""

    lines = ["## 주요 공간 속성 정보 (데이터 기반)"]

    # 카테고리별 분류
    categories = {
        "HAZARD_ZONE": [],
        "LONG_STAY": [],
        "SHORT_STAY": [],
        "TRANSIT": [],
    }

    for locus_id, info in locus_dict.items():
        cat = info.get("dwell_category", "UNKNOWN")
        if cat in categories:
            categories[cat].append(info)

    # 고위험 구역 (HAZARD_ZONE)
    if categories["HAZARD_ZONE"]:
        lines.append("\n### 고위험 구역 (HAZARD_ZONE) - 장시간 체류 시 즉시 확인 필요")
        for info in categories["HAZARD_ZONE"][:5]:
            name = info.get("locus_name", "")
            avg_dwell = info.get("avg_dwell_minutes") or 0
            peak = info.get("peak_hour") or 0
            max_occ = info.get("max_concurrent_occupancy") or 0
            lines.append(
                f"- {name}: 평균 체류 {avg_dwell:.0f}분, 피크 {peak}시, 최대 동시 {max_occ}명"
            )

    # 주요 작업 구역 (LONG_STAY Top 5)
    long_stay = sorted(
        categories["LONG_STAY"],
        key=lambda x: x.get("avg_daily_visitors") or 0,
        reverse=True,
    )[:5]
    if long_stay:
        lines.append("\n### 주요 작업 구역 (LONG_STAY Top 5)")
        for info in long_stay:
            name = info.get("locus_name", "")
            avg_dwell = info.get("avg_dwell_minutes") or 0
            peak = info.get("peak_hour") or 0
            max_occ = info.get("max_concurrent_occupancy") or 0
            lines.append(
                f"- {name}: 평균 체류 {avg_dwell:.0f}분, 피크 {peak}시, 최대 동시 {max_occ}명"
            )

    # 공간 유형 설명
    lines.append("\n### 공간 유형별 특성")
    lines.append("- TRANSIT (통과형): 게이트, 타각기, 호이스트 — 체류 시간 짧음 (평균 <5분)")
    lines.append("- SHORT_STAY (단기체류): 휴게실, 흡연실 — 체류 시간 10~30분")
    lines.append("- LONG_STAY (작업구역): FAB, CUB, 야외 — 체류 시간 30분+")
    lines.append("- HAZARD_ZONE (고위험): 밀폐공간, 고압전 — 장시간 체류 시 경고 필요")

    return "\n".join(lines)


def build_analysis_prompt(
    date_str: str,
    insights_text: str,
    kpi_summary: str,
    locus_dict: dict | None = None,
) -> str:
    """
    인사이트 기반 LLM 마스터 프롬프트.

    v1.2: domain_packs/construction/prompts.py 템플릿 사용 + DataGuard 적용.
    민감 정보(작업자명, 회사명, 좌표)가 LLM에 전달되지 않도록 필터링.

    Args:
        date_str: 분석 날짜
        insights_text: 인사이트 요약
        kpi_summary: KPI 요약
        locus_dict: enriched locus 정보 (공간 맥락 추가용)
    """
    # DataGuard를 통해 KPI/인사이트 데이터 필터링
    safe_kpi = _data_guard.sanitize_for_llm({"kpi_summary": kpi_summary})
    safe_insights = _data_guard.sanitize_for_llm({"insights_text": insights_text})

    sanitized_kpi = safe_kpi.get("kpi_summary", kpi_summary)
    sanitized_insights = safe_insights.get("insights_text", insights_text)

    # 공간 맥락 추가 (기존 기능 유지)
    space_context = ""
    if locus_dict:
        space_context = "\n\n" + build_space_context(locus_dict)

    # domain_packs 프롬프트 템플릿 사용
    base_prompt = domain_build_insight_prompt(
        date_str=date_str,
        kpi_summary=sanitized_kpi,
        insights_text=sanitized_insights,
        include_space_context=True,
    )

    # 현장 컨텍스트 + 공간 맥락 추가
    return f"""{Y1_SITE_CONTEXT}{space_context}

{base_prompt}
""".strip()


def build_weekly_prompt(
    date_range: str,
    weekly_insights_text: str,
    trend_summary: str,
) -> str:
    """주간 트렌드 분석 LLM 프롬프트.

    v1.2: domain_packs/construction/prompts.py 템플릿 사용 + DataGuard 적용.
    """
    # DataGuard를 통해 데이터 필터링
    safe_data = _data_guard.sanitize_for_llm({
        "trend_summary": trend_summary,
        "weekly_insights_text": weekly_insights_text,
    })

    # domain_packs 프롬프트 템플릿 사용
    base_prompt = domain_build_weekly_prompt(
        date_range=date_range,
        trend_summary=safe_data.get("trend_summary", trend_summary),
        weekly_insights_text=safe_data.get("weekly_insights_text", weekly_insights_text),
    )

    return f"""{Y1_SITE_CONTEXT}

{base_prompt}
""".strip()
