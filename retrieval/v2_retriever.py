from __future__ import annotations

import math
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient

from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_VECTOR_NAME
from embedding.vector_contract import legacy_repository_point_id, user_point_ids
from inference.feed_assembly import FeedAssemblySystem
from scripts.user_onboarding import TARGET_VECTOR_NAME, USER_PROFILES_COLLECTION


@dataclass(frozen=True, slots=True)
class RankedRepository:
    repo_id: str
    score: float
    source: str


class QdrantV2Retriever:
    """Canonical-ID candidate retrieval with no PostgreSQL dependency."""

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        repository_collection: str = QDRANT_COLLECTION_NAME,
        user_collection: str = USER_PROFILES_COLLECTION,
        max_candidates: int = 500,
        assembler: FeedAssemblySystem | None = None,
    ) -> None:
        self.client = client or QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10")),
        )
        self.repository_collection = repository_collection
        self.user_collection = user_collection
        self.max_candidates = max(50, min(max_candidates, 2_000))
        self.assembler = assembler or FeedAssemblySystem()
        self.model_version = os.getenv("ML_MODEL_VERSION", "qdrant-hybrid-v2")
        self.embedding_version = os.getenv("REPOSITORY_EMBEDDING_VERSION", "repo-embedding-v2")

    @staticmethod
    def _canonical_id(point: Any) -> str | None:
        payload = point.payload or {}
        candidate = str(payload.get("repo_id") or point.id)
        try:
            canonical = str(uuid.UUID(candidate))
        except (ValueError, AttributeError):
            return None
        valid_point_ids = {canonical, legacy_repository_point_id(canonical)}
        has_canonical_payload = str(payload.get("repo_id")) == canonical
        return canonical if str(point.id) in valid_point_ids and has_canonical_payload else None

    @staticmethod
    def _vector(value: Any, preferred: str | None = None) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            if preferred and preferred in value:
                return list(value[preferred])
            if len(value) == 1:
                return list(next(iter(value.values())))
            return None
        return list(value)

    def _user_vector(self, user_id: str) -> list[float] | None:
        canonical, legacy = user_point_ids(user_id)
        points = self.client.retrieve(
            collection_name=self.user_collection,
            ids=[canonical, legacy],
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            return None
        by_id = {str(point.id): point for point in points}
        point = by_id.get(canonical) or by_id.get(legacy)
        return self._vector(point.vector, TARGET_VECTOR_NAME) if point else None

    def _semantic(self, vector: list[float], limit: int) -> list[tuple[Any, float]]:
        response = self.client.query_points(
            collection_name=self.repository_collection,
            query=vector,
            using=QDRANT_VECTOR_NAME,
            limit=min(limit, self.max_candidates),
            with_payload=True,
            with_vectors=False,
        )
        return [(point, float(point.score)) for point in response.points]

    def _discovery(self, limit: int) -> list[Any]:
        points: list[Any] = []
        offset = None
        while len(points) < min(limit, self.max_candidates):
            records, offset = self.client.scroll(
                collection_name=self.repository_collection,
                limit=min(100, limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(records)
            if not records or offset is None:
                break
        return points

    @staticmethod
    def _discovery_score(payload: dict[str, Any]) -> tuple[float, str]:
        stars = max(0, int(payload.get("star_count") or 0))
        velocity = max(0.0, float(payload.get("trend_velocity") or payload.get("delta_7d") or 0))
        activity = max(0.0, float(payload.get("activity_score") or 0))
        raw_pushed_days = payload.get("pushed_days_ago")
        pushed_days = 999 if raw_pushed_days is None else max(0, int(raw_pushed_days))
        freshness = math.exp(-pushed_days / 60)
        score = 0.35 * math.log1p(stars) + 0.35 * math.log1p(velocity) + 0.2 * activity + 0.1 * freshness
        source = "trending" if velocity > 0 else "fresh" if pushed_days <= 30 else "popular"
        return score, source

    def recommend(
        self,
        user_id: str,
        limit: int,
        exclude_repo_ids: list[str],
        generation_seed: str | None = None,
    ) -> list[RankedRepository]:
        excluded = {str(uuid.UUID(item)) for item in exclude_repo_ids}
        candidates: dict[str, RankedRepository] = {}
        metadata: dict[str, dict[str, Any]] = {}
        user_vector = self._user_vector(user_id)
        if user_vector:
            for point, score in self._semantic(user_vector, max(limit * 5, 100)):
                repo_id = self._canonical_id(point)
                if repo_id and repo_id not in excluded and math.isfinite(score):
                    candidates[repo_id] = RankedRepository(repo_id, score, "semantic")
                    metadata[repo_id] = dict(point.payload or {})

        discovery: list[tuple[str, float, str, dict[str, Any]]] = []
        for point in self._discovery(max(limit * 8, 200)):
            repo_id = self._canonical_id(point)
            if not repo_id or repo_id in excluded:
                continue
            payload = dict(point.payload or {})
            score, source = self._discovery_score(payload)
            if math.isfinite(score):
                discovery.append((repo_id, score, source, payload))

        maximum_discovery = max((item[1] for item in discovery), default=0.0)
        for repo_id, score, source, payload in discovery:
            normalized_discovery = score / maximum_discovery if maximum_discovery > 0 else 0.0
            current = candidates.get(repo_id)
            combined = 0.15 * normalized_discovery
            if current is not None:
                combined += current.score
            if math.isfinite(combined):
                candidates[repo_id] = RankedRepository(repo_id, combined, current.source if current else source)
                metadata[repo_id] = payload

        ranked = sorted(candidates.values(), key=lambda item: (-item.score, item.repo_id))
        assembly_input = [
            {
                "repo_id": item.repo_id,
                "score": item.score,
                "final_score": item.score,
                "source": item.source,
                "primary_language": metadata.get(item.repo_id, {}).get("primary_language"),
                "created_at": metadata.get(item.repo_id, {}).get("created_at"),
            }
            for item in ranked
        ]
        shaped = self.assembler.shape_batch(
            assembly_input,
            seen_repo_ids=excluded,
            randomizer=random.Random(generation_seed),
        )
        return [
            RankedRepository(
                repo_id=str(item["repo_id"]),
                score=round(float(item["final_score"]), 6),
                source=str(item["source"]),
            )
            for item in shaped[:limit]
        ]

    def health(self) -> dict[str, Any]:
        info = self.client.get_collection(self.repository_collection)
        return {"qdrant": "healthy", "repository_points": int(info.points_count or 0)}
