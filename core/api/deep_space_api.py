"""
DeepSpaceAPI — Deep Space 모델 추론 인터페이스
================================================
Deep Space Foundation Model의 통합 추론 API.

현재 상태: 스켈레톤 (모델 학습 전 인터페이스만 정의)
실제 구현은 src/model/ 마이그레이션 후 연결 예정.

사용:
    from core.api.deep_space_api import DeepSpaceAPI

    api = DeepSpaceAPI("M15X_SKHynix")
    predictions = api.predict_next(["L-Y1-001", "L-Y1-070"], top_k=5)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.registry.locus_registry import LocusRegistry

logger = logging.getLogger(__name__)


class DeepSpaceAPI:
    """Deep Space 모델 추론 통합 인터페이스.

    모든 Deep Space 기능(예측, 이상탐지, 임베딩, 유사도)을
    단일 API로 제공한다.

    Attributes:
        sector_id: Sector 식별자
        model_loaded: 모델 로드 여부
    """

    def __init__(
        self,
        sector_id: str,
        registry: Optional["LocusRegistry"] = None,
    ) -> None:
        """DeepSpaceAPI를 초기화한다.

        Args:
            sector_id: Sector 식별자
            registry: LocusRegistry 인스턴스 (None이면 나중에 자동 로드)
        """
        self.sector_id = sector_id
        self.model_loaded = False
        self._model = None
        self._tokenizer = None
        self._registry = registry
        logger.info(
            "DeepSpaceAPI 초기화: %s (registry=%s)",
            sector_id, "provided" if registry else "None",
        )

    def ensure_registry(self) -> None:
        """Registry가 없으면 로드한다."""
        if self._registry is None:
            try:
                from core.registry.locus_registry import get_registry
                self._registry = get_registry(self.sector_id)
                logger.info("DeepSpaceAPI: Registry 자동 로드 완료")
            except Exception as e:
                logger.warning("DeepSpaceAPI: Registry 로드 실패 — %s", e)

    @property
    def registry(self) -> Optional["LocusRegistry"]:
        """Registry 접근자 (읽기 전용)."""
        return self._registry

    def get_locus_context(self, locus_ids: list[str]) -> str:
        """Locus ID 목록을 자연어 컨텍스트로 변환한다.

        Args:
            locus_ids: Locus ID 리스트

        Returns:
            자연어 설명 문자열 (LLM용)
        """
        self.ensure_registry()
        if self._registry is None:
            return "(Registry 없음)"

        descriptions = []
        for lid in locus_ids:
            try:
                desc = self._registry.to_natural_language(lid)
                descriptions.append(desc)
            except Exception:
                descriptions.append(f"알 수 없는 장소 ({lid})")

        return "\n".join(descriptions)

    def load_model(self) -> bool:
        """Deep Space 모델과 토크나이저를 로드한다.

        Returns:
            로드 성공 여부
        """
        try:
            # 점진적 마이그레이션: src/model/에서 로드 시도
            # Phase 4에서 core/model/로 이동 후 경로 변경
            from src.model.tokenizer import get_tokenizer
            self._tokenizer = get_tokenizer(self.sector_id)
            if self._tokenizer is None:
                logger.warning("토크나이저 없음: %s", self.sector_id)
                return False

            # 모델 로드 (존재하는 경우)
            try:
                from src.model.transformer import load_model
                self._model = load_model(self.sector_id)
                self.model_loaded = True
                logger.info("Deep Space 모델 로드 완료: %s", self.sector_id)
            except (ImportError, FileNotFoundError):
                logger.info("Deep Space 모델 없음: %s (토크나이저만 로드)", self.sector_id)
                self.model_loaded = False

            return self.model_loaded
        except Exception as e:
            logger.warning("DeepSpaceAPI 모델 로드 실패: %s", e)
            return False

    def predict_next(
        self,
        journey: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """다음 이동 장소를 예측한다.

        Args:
            journey: Locus ID 시퀀스
            top_k: 상위 K개 예측

        Returns:
            [(locus_id, probability), ...] 확률 내림차순

        Raises:
            RuntimeError: 모델이 로드되지 않았을 때
        """
        if not self.model_loaded or self._model is None:
            raise RuntimeError("Deep Space 모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        # TODO: 실제 모델 추론 연결 (Phase 4)
        logger.warning("predict_next: 모델 추론 미구현 (스켈레톤)")
        return []

    def detect_anomaly(
        self,
        journey: list[str],
    ) -> list[dict]:
        """이상 이동을 탐지한다.

        Args:
            journey: Locus ID 시퀀스

        Returns:
            [{"position": int, "from": str, "to": str,
              "perplexity": float, "is_anomaly": bool}, ...]

        Raises:
            RuntimeError: 모델이 로드되지 않았을 때
        """
        if not self.model_loaded or self._model is None:
            raise RuntimeError("Deep Space 모델이 로드되지 않았습니다.")

        # TODO: 실제 이상 탐지 연결 (Phase 4)
        logger.warning("detect_anomaly: 모델 추론 미구현 (스켈레톤)")
        return []

    def get_embeddings(
        self,
        journeys: Optional[list[list[str]]] = None,
    ) -> np.ndarray:
        """Journey 임베딩 벡터를 추출한다.

        Args:
            journeys: Journey 시퀀스 리스트 (None이면 전체 Locus 임베딩)

        Returns:
            [n_journeys, d_model] 형태의 numpy 배열

        Raises:
            RuntimeError: 모델이 로드되지 않았을 때
        """
        if not self.model_loaded or self._model is None:
            raise RuntimeError("Deep Space 모델이 로드되지 않았습니다.")

        # TODO: 실제 임베딩 추출 연결 (Phase 4)
        logger.warning("get_embeddings: 모델 추론 미구현 (스켈레톤)")
        return np.array([])

    def get_locus_similarity(
        self,
        locus_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Locus 임베딩 기반 유사도를 조회한다.

        Args:
            locus_id: 기준 Locus ID
            top_k: 상위 K개

        Returns:
            [(locus_id, cosine_similarity), ...] 유사도 내림차순

        Raises:
            RuntimeError: 모델이 로드되지 않았을 때
        """
        if not self.model_loaded or self._model is None:
            raise RuntimeError("Deep Space 모델이 로드되지 않았습니다.")

        # TODO: 실제 유사도 계산 연결 (Phase 4)
        logger.warning("get_locus_similarity: 모델 추론 미구현 (스켈레톤)")
        return []

    def get_transition_matrix(self) -> tuple[np.ndarray, list[str]]:
        """전이 확률 매트릭스를 반환한다.

        Returns:
            (matrix[n_loci, n_loci], locus_ids) 튜플

        Raises:
            RuntimeError: 모델이 로드되지 않았을 때
        """
        if not self.model_loaded or self._model is None:
            raise RuntimeError("Deep Space 모델이 로드되지 않았습니다.")

        # TODO: 전이 매트릭스 계산 연결 (Phase 4)
        logger.warning("get_transition_matrix: 모델 추론 미구현 (스켈레톤)")
        return np.array([]), []

    @property
    def is_ready(self) -> bool:
        """모델이 추론 가능한 상태인지."""
        return self.model_loaded and self._model is not None
