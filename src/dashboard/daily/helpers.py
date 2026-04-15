"""
Daily Tab Helpers
=================
등급 판정 함수 + 공용 로더.
"""
from __future__ import annotations


def ewi_grade(v: float) -> str:
    """EWI 등급 판정.

    M15X 33일 17,469건 실측 P33/P66 사분위 기반:
      고강도 >= 0.52  (상위 34%)
      보통   0.29~0.52 (중위 33%)
      저강도 < 0.29   (하위 33%)
    """
    if v >= 0.52:
        return "고강도"
    if v >= 0.29:
        return "보통"
    return "저강도"


def cre_grade(v: float) -> str:
    """CRE 등급 판정."""
    if v >= 0.6:
        return "고위험"
    if v >= 0.3:
        return "주의"
    return "정상"


def sii_grade(v: float) -> str:
    """SII 등급 판정."""
    if v >= 0.5:
        return "집중관리"
    if v >= 0.25:
        return "주의"
    return "정상"
