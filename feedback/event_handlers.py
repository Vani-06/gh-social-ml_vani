import logging
import math
import uuid
from typing import Any, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import (
    DWELL_BASE_ALPHA,
    EMBEDDING_DIM,
    MAX_DWELL_SECONDS,
    MIN_DWELL_SECONDS,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
)
from scripts.user_onboarding import TARGET_VECTOR_NAME, USER_PROFILES_COLLECTION
from .interactions import INTERACTIONS, normalize_interaction

logger = logging.getLogger("pipeline.feedback")

ACTION_WEIGHTS = {
    **{name: definition.embedding_alpha for name, definition in INTERACTIONS.items()},
    # Compatibility aliases accepted only by the legacy handler.
    "click": 0.05,
    "skip": -0.10,
}


def _dwell_alpha(dwell_seconds: float) -> Optional[float]:
    if dwell_seconds < MIN_DWELL_SECONDS:
        return None
    ratio = math.log1p(dwell_seconds) / math.log1p(MAX_DWELL_SECONDS)
    return DWELL_BASE_ALPHA * min(ratio, 1.0)


def shift_vector(user_vec: List[float], repo_vec: List[float], alpha: float) -> List[float]:
    updated = np.asarray(user_vec, dtype=np.float32) + alpha * np.asarray(repo_vec, dtype=np.float32)
    norm = float(np.linalg.norm(updated))
    if norm > 0:
        updated = updated / norm
    return updated.tolist()


class FeedbackHandler:
    """Legacy v1 vector handler retained during rollout.

    Product counters, feedback state, event durability, and cache versioning are
    backend responsibilities. This handler has no PostgreSQL dependency.
    """

    def __init__(
        self,
        db_connector: Any = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.db = None
        self.qdrant_url = qdrant_url or QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or QDRANT_API_KEY
        self._qdrant_client: QdrantClient | None = None
        if self.qdrant_url:
            try:
                self._qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=10.0,
                )
            except Exception as exc:
                logger.error("Failed to configure Qdrant: %s", exc)

    @property
    def qdrant(self) -> QdrantClient | None:
        return self._qdrant_client

    def handle_feedback(
        self,
        user_id: str,
        repo_id: str,
        action: str,
        *,
        dwell_seconds: Optional[float] = None,
        message_id: Optional[str] = None,
    ) -> bool:
        # ``message_id`` remains in the v1 consumer call signature for retry
        # compatibility. Durable ordering and idempotency belong to feedback.v2.
        del message_id
        action = normalize_interaction(action)
        if action != "dwell" and action not in ACTION_WEIGHTS:
            logger.error("Unknown feedback action: %s", action)
            return False
        if action == "dwell":
            if dwell_seconds is None:
                return False
            alpha = _dwell_alpha(float(dwell_seconds))
            if alpha is None:
                return True
        else:
            alpha = ACTION_WEIGHTS[action]
        if not alpha:
            return True
        return self.update_user_embedding(user_id, repo_id, float(alpha))

    @staticmethod
    def _point_id(value: str, namespace: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{value}"))

    @staticmethod
    def _vector(value: Any, preferred: str | None = None) -> tuple[list[float], str | None]:
        if isinstance(value, dict):
            if preferred and preferred in value:
                return list(value[preferred]), preferred
            if not value:
                raise ValueError("empty named-vector mapping")
            name = next(iter(value))
            return list(value[name]), name
        if value is None:
            raise ValueError("missing vector")
        return list(value), None

    def update_user_embedding(self, user_id: str, repo_id: str, alpha: float) -> bool:
        if not self.qdrant:
            return False
        user_point_id = self._point_id(user_id, "user")
        repo_point_id = self._point_id(repo_id, "github")
        try:
            users = self.qdrant.retrieve(
                collection_name=USER_PROFILES_COLLECTION,
                ids=[user_point_id],
                with_vectors=True,
            )
            repos = self.qdrant.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[repo_point_id],
                with_vectors=True,
            )
            if not users or not repos:
                return False
            user_vector, vector_name = self._vector(users[0].vector, TARGET_VECTOR_NAME)
            repo_vector, _ = self._vector(repos[0].vector, "repo_embedding")
            if len(user_vector) != EMBEDDING_DIM or len(repo_vector) != EMBEDDING_DIM:
                return False
            updated = shift_vector(user_vector, repo_vector, alpha)
            stored: Any = {vector_name: updated} if vector_name else updated
            self.qdrant.upsert(
                collection_name=USER_PROFILES_COLLECTION,
                points=[PointStruct(id=user_point_id, vector=stored, payload=users[0].payload or {})],
                wait=True,
            )
            return True
        except Exception as exc:
            logger.error("Failed to apply legacy feedback vector update: %s", exc)
            return False
