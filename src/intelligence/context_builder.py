"""
Context Builder — 역사적 맥락 데이터 구축
==========================================
특정 날짜(anchor_date)의 AI 분석 품질을 높이기 위해
과거 데이터로부터 4가지 비교 컨텍스트 블록을 생성한다.

[A] 최근 N일 기준선    — 평균/표준편차 범위
[B] 같은 요일 패턴     — 요일 습관성 파악
[C] 직전일 대비 변화   — 단기 트렌드 감지
[D] 전체 기간 내 순위  — 이날이 얼마나 특이한지

사용:
    from src.intelligence.context_builder import ContextBuilder

    ctx = ContextBuilder.build_worker("20260313", "M15X_SKHynix")
    ctx_t = ContextBuilder.build_transit("20260313", "M15X_SKHynix")
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

_WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

# 경량 로드용 컬럼 목록
_WORKER_COLS = [
    "ewi", "cre", "gap_ratio",
    "high_active_min", "low_active_min", "standby_min",
    "user_no",
]
_TRANSIT_COLS = [
    "mat_minutes", "lbt_minutes", "eod_minutes", "total_wait_minutes",
]

_METRIC_LABELS = {
    "ewi":               "EWI(작업집중도)",
    "cre":               "CRE(위험노출도)",
    "high_cre_pct":      "고위험 비율(%)",
    "n_workers":         "작업자 수(명)",
    "mat_minutes":       "MAT(분)",
    "lbt_minutes":       "LBT(분)",
    "eod_minutes":       "EOD(분)",
    "total_wait_minutes":"TWT 총대기(분)",
    "gap_ratio":         "GAP 비율",
}


# ── 경량 단일 날짜 로더 (캐시 적용) ─────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_worker_slim(date_str: str, sector_id: str) -> Optional[dict]:
    """
    worker.parquet 핵심 컬럼만 읽어 집계 dict 반환.
    DataFrame 전체가 아닌 float 집계값만 캐시 → 메모리 절약.
    """
    try:
        import config as cfg
        paths = cfg.get_sector_paths(sector_id)
        p = paths["processed_dir"] / date_str / "worker.parquet"
        if not p.exists():
            return None

        import pyarrow.parquet as pq
        schema = pq.read_schema(str(p))
        cols = [c for c in schema.names if c in _WORKER_COLS]
        if not cols:
            return None

        df = pd.read_parquet(p, columns=cols)
        if df.empty:
            return None

        rec: dict = {"n_workers": len(df)}
        for col in ["ewi", "cre", "gap_ratio"]:
            if col in df.columns:
                s = df[col].dropna()
                if len(s):
                    rec[col] = float(s.mean())
                    rec[f"{col}_std"] = float(s.std()) if len(s) > 1 else 0.0

        if "cre" in df.columns:
            rec["high_cre_pct"] = float((df["cre"] >= 0.5).mean() * 100)

        return rec
    except Exception as e:
        logger.debug("worker slim load failed %s: %s", date_str, e)
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _load_transit_slim(date_str: str, sector_id: str) -> Optional[dict]:
    """transit.parquet 핵심 컬럼만 읽어 집계 dict 반환."""
    try:
        import config as cfg
        paths = cfg.get_sector_paths(sector_id)
        p = paths["processed_dir"] / date_str / "transit.parquet"
        if not p.exists():
            return None

        import pyarrow.parquet as pq
        schema = pq.read_schema(str(p))
        cols = [c for c in schema.names if c in _TRANSIT_COLS]
        if not cols:
            return None

        df = pd.read_parquet(p, columns=cols)
        if df.empty:
            return None

        rec: dict = {}
        for col in _TRANSIT_COLS:
            if col in df.columns:
                s = df[col].dropna()
                if len(s):
                    rec[col] = float(s.mean())
                    rec[f"{col}_std"] = float(s.std()) if len(s) > 1 else 0.0

        return rec if rec else None
    except Exception as e:
        logger.debug("transit slim load failed %s: %s", date_str, e)
        return None


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _select_dates(
    anchor_date: str,
    sector_id: str,
    n_prior: int,
    n_same_dow: int,
) -> tuple[list[str], list[str], Optional[str]]:
    """
    최근 N일, 같은 요일 M개, 직전일 날짜 선택.
    Returns: (recent_dates, same_dow_dates, prev_date)
    """
    from src.pipeline.cache_manager import detect_processed_dates
    processed = sorted(detect_processed_dates(sector_id))
    prior = [d for d in processed if d < anchor_date]

    recent = prior[-n_prior:]
    prev_date = prior[-1] if prior else None

    try:
        anchor_dow = datetime.strptime(anchor_date, "%Y%m%d").weekday()
        same_dow = [
            d for d in prior
            if datetime.strptime(d, "%Y%m%d").weekday() == anchor_dow
        ][-n_same_dow:]
    except ValueError:
        same_dow = []

    return recent, same_dow, prev_date


def _stats_block(records: list[dict], key: str) -> Optional[dict]:
    """여러 날짜 집계 dict에서 특정 key의 기술통계 계산."""
    vals = [r[key] for r in records if key in r and r[key] is not None]
    if not vals:
        return None
    s = pd.Series(vals, dtype=float)
    return {
        "mean": round(float(s.mean()), 3),
        "std":  round(float(s.std()), 3) if len(vals) > 1 else 0.0,
        "min":  round(float(s.min()), 3),
        "max":  round(float(s.max()), 3),
        "n":    len(vals),
    }


def _rank_from_index(anchor_date: str, metric_key: str, sector_id: str) -> tuple[int, int]:
    """
    summary_index에서 metric_key의 전체 날짜 중 anchor_date 순위 반환.
    metric_key 예: "avg_ewi", "avg_cre", "high_cre_count"
    Returns: (rank, total)  rank은 1-based, 높을수록 높은 값
    """
    try:
        from src.pipeline.summary_index import load_summary_index
        idx = load_summary_index(sector_id)
        dates_data = idx.get("dates", {})
        pairs = [
            (d, v[metric_key])
            for d, v in dates_data.items()
            if metric_key in v and v[metric_key] is not None
        ]
        if not pairs:
            return 0, 0
        pairs.sort(key=lambda x: x[1], reverse=True)
        total = len(pairs)
        rank = next((i + 1 for i, (d, _) in enumerate(pairs) if d == anchor_date), 0)
        return rank, total
    except Exception as e:
        logger.debug("rank from index failed: %s", e)
        return 0, 0


def _fmt_baseline(records: list[dict], keys: list[str], n_days: int) -> list[str]:
    """최근 N일 기준선 텍스트 블록."""
    if not records:
        return []
    lines = [f"### [A] 최근 {n_days}일 기준선 ({len(records)}일 데이터)"]
    for key in keys:
        st_ = _stats_block(records, key)
        if st_:
            label = _METRIC_LABELS.get(key, key)
            lines.append(
                f"- {label}: 평균 {st_['mean']} (±{st_['std']}, "
                f"범위 {st_['min']}~{st_['max']})"
            )
    return lines


def _fmt_dow(records: list[dict], keys: list[str], weekday_str: str, n: int) -> list[str]:
    """같은 요일 패턴 텍스트 블록."""
    if not records:
        return []
    lines = [f"### [B] {weekday_str} 패턴 (최근 {len(records)}회 평균)"]
    for key in keys:
        st_ = _stats_block(records, key)
        if st_:
            label = _METRIC_LABELS.get(key, key)
            lines.append(f"- {label}: {st_['mean']} (±{st_['std']})")
    return lines


def _fmt_trend(anchor_rec: dict, prev_rec: dict, keys: list[str]) -> list[str]:
    """직전일 대비 변화 텍스트 블록."""
    if not anchor_rec or not prev_rec:
        return []
    lines = ["### [C] 직전일 대비 변화"]
    for key in keys:
        curr = anchor_rec.get(key)
        prev = prev_rec.get(key)
        if curr is None or prev is None:
            continue
        delta = curr - prev
        pct = (delta / abs(prev) * 100) if prev != 0 else 0.0
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        label = _METRIC_LABELS.get(key, key)
        lines.append(
            f"- {label}: {round(prev, 3)} → {round(curr, 3)} "
            f"({arrow}{abs(delta):.3f} / {pct:+.1f}%)"
        )
    return lines


def _fmt_rank(anchor_date: str, sector_id: str, rank_specs: list[tuple[str, str, bool]]) -> list[str]:
    """
    전체 기간 순위 텍스트 블록.
    rank_specs: [(idx_key, display_label, higher_is_better), ...]
    """
    lines = []
    for idx_key, label, higher_better in rank_specs:
        rank, total = _rank_from_index(anchor_date, idx_key, sector_id)
        if rank and total:
            if higher_better:
                note = "높을수록 좋음"
            else:
                note = "낮을수록 좋음"
            lines.append(f"- {label}: {total}일 중 {rank}위 ({note})")
    if lines:
        lines.insert(0, "### [D] 전체 기간 내 순위")
    return lines


# ── 공개 API ─────────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    anchor_date 기준 역사적 맥락 텍스트 생성기.
    결과는 DataPackager 텍스트에 이어붙여 LLMGateway로 전달한다.
    """

    @staticmethod
    def build_worker(
        anchor_date: str,
        sector_id: str,
        n_baseline: int = 7,
        n_same_dow: int = 3,
    ) -> str:
        """
        작업자 분석용 비교 컨텍스트 텍스트.

        [A] 최근 N일 EWI/CRE/고위험비율 기준선
        [B] 같은 요일 패턴
        [C] 직전일 대비 변화
        [D] 전체 기간 내 순위 (summary_index 기반, 추가 parquet 로드 없음)
        """
        try:
            anchor_dt = datetime.strptime(anchor_date, "%Y%m%d")
        except ValueError:
            return ""

        weekday_str = _WEEKDAY_KR[anchor_dt.weekday()] + "요일"
        recent, same_dow, prev_date = _select_dates(
            anchor_date, sector_id, n_baseline, n_same_dow
        )

        if not recent and not same_dow:
            return ""

        # 데이터 로드
        recent_recs = [r for r in (_load_worker_slim(d, sector_id) for d in recent) if r]
        dow_recs = [r for r in (_load_worker_slim(d, sector_id) for d in same_dow) if r]
        anchor_rec = _load_worker_slim(anchor_date, sector_id) or {}
        prev_rec = _load_worker_slim(prev_date, sector_id) if prev_date else {}

        metric_keys = ["n_workers", "ewi", "cre", "high_cre_pct"]

        sections: list[str] = [
            f"\n## 비교 컨텍스트 (기준일: {anchor_date} {weekday_str})"
        ]
        sections += _fmt_baseline(recent_recs, metric_keys, n_baseline)
        if dow_recs:
            sections += _fmt_dow(dow_recs, metric_keys, weekday_str, n_same_dow)
        if prev_rec:
            sections += _fmt_trend(anchor_rec, prev_rec, metric_keys)

        # 순위: summary_index에 있는 키 사용 (avg_ewi, avg_cre)
        rank_specs = [
            ("avg_ewi",         "EWI",         True),
            ("avg_cre",         "CRE",         False),
            ("high_cre_count",  "고위험자 수",  False),
        ]
        sections += _fmt_rank(anchor_date, sector_id, rank_specs)

        return "\n".join(sections)

    @staticmethod
    def build_transit(
        anchor_date: str,
        sector_id: str,
        n_baseline: int = 7,
        n_same_dow: int = 3,
    ) -> str:
        """
        대기/이동 시간 분석용 비교 컨텍스트 텍스트.

        [A] 최근 N일 MAT/LBT/EOD 기준선
        [B] 같은 요일 패턴
        [C] 직전일 대비 변화
        """
        try:
            anchor_dt = datetime.strptime(anchor_date, "%Y%m%d")
        except ValueError:
            return ""

        weekday_str = _WEEKDAY_KR[anchor_dt.weekday()] + "요일"
        recent, same_dow, prev_date = _select_dates(
            anchor_date, sector_id, n_baseline, n_same_dow
        )

        if not recent and not same_dow:
            return ""

        recent_recs = [r for r in (_load_transit_slim(d, sector_id) for d in recent) if r]
        dow_recs = [r for r in (_load_transit_slim(d, sector_id) for d in same_dow) if r]
        anchor_rec = _load_transit_slim(anchor_date, sector_id) or {}
        prev_rec = _load_transit_slim(prev_date, sector_id) if prev_date else {}

        metric_keys = ["mat_minutes", "lbt_minutes", "eod_minutes", "total_wait_minutes"]

        sections: list[str] = [
            f"\n## 비교 컨텍스트 (기준일: {anchor_date} {weekday_str})"
        ]
        sections += _fmt_baseline(recent_recs, metric_keys, n_baseline)
        if dow_recs:
            sections += _fmt_dow(dow_recs, metric_keys, weekday_str, n_same_dow)
        if prev_rec:
            sections += _fmt_trend(anchor_rec, prev_rec, metric_keys)
        # transit은 summary_index에 MAT 키 없으므로 순위 생략

        return "\n".join(sections)
