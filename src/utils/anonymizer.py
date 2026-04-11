"""
작업자 이름 비식별화 유틸리티
==============================
성(첫 글자)만 남기고 나머지는 **로 마스킹.

- 한글: "이택진" -> "이**"
- 영문: "John Kim" -> "J** K**"
- NaN/빈값: 그대로 반환
"""
from __future__ import annotations

import pandas as pd


def mask_name(name) -> str:
    """단일 이름을 비식별화.

    Args:
        name: 원본 이름 (str, NaN, None 등)

    Returns:
        마스킹된 이름 문자열. NaN/빈값은 그대로 반환.

    Examples:
        >>> mask_name("이택진")
        '이**'
        >>> mask_name("홍길동")
        '홍**'
        >>> mask_name("John Kim")
        'J** K**'
        >>> mask_name("")
        ''
        >>> mask_name(None)
        ''
    """
    if pd.isna(name) or name is None:
        return ""
    name = str(name).strip()
    if not name:
        return ""

    # 공백으로 구분된 복수 단어 (영문 이름 등)
    parts = name.split()
    if len(parts) >= 2:
        masked_parts = []
        for part in parts:
            if len(part) <= 1:
                masked_parts.append(part)
            else:
                masked_parts.append(part[0] + "**")
        return " ".join(masked_parts)

    # 단일 단어 (한글 이름 등)
    if len(name) <= 1:
        return name
    return name[0] + "**"


def mask_names_in_df(df: pd.DataFrame, column: str = "user_name") -> pd.DataFrame:
    """DataFrame의 특정 컬럼에 이름 마스킹을 일괄 적용.

    원본 DataFrame을 변경하지 않고 복사본을 반환한다.

    Args:
        df: 대상 DataFrame
        column: 마스킹할 컬럼명 (기본: "user_name")

    Returns:
        마스킹이 적용된 DataFrame 복사본.
        해당 컬럼이 없으면 원본 그대로 반환.
    """
    if column not in df.columns:
        return df
    result = df.copy()
    result[column] = result[column].apply(mask_name)
    return result
