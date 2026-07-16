from __future__ import annotations

import json
import logging
import math
import os
import socket
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_VECTOR_NAME
from scripts.user_onboarding import TARGET_VECTOR_NAME, USER_PROFILES_COLLECTION

logger = logging.getLogger(__name__)

STREAM = "ml:feedback:v2"
GROUP = "ml-feedback-v2"
ACCEPT_LUA = """
if redis.call('set', KEYS[1], '1', 'NX', 'EX', ARGV[1]) then
  return redis.call('xadd', KEYS[2], '*', unpack(ARGV, 2))
end
return 'duplicate'
"""
RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end
return 0
"""

ALPHAS = {
    "readme_open": 0.05,
    "github_open": 0.07,
    "share": 0.10,
    "like": 0.15,
    "dislike": -0.15,
    "save": 0.20,
    "unlike": 0.0,
    "undislike": 0.0,
    "unsave": 0.0,
}


def _redis_client(redis_url: str | None = None):
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("redis>=5 is required for the production v2 feedback boundary") from exc
    url = redis_url or os.getenv("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is required for durable v2 feedback")
    client = redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        health_check_interval=30,
    )
    client.ping()
    return client


class DurableFeedbackProducer:
    def __init__(self, redis_client=None) -> None:
        self.redis = redis_client or _redis_client()

    def enqueue(self, events: Iterable[dict[str, Any]]) -> tuple[int, int]:
        accepted = 0
        duplicates = 0
        for event in events:
            fields: list[str] = []
            for key, value in event.items():
                fields.extend([key, json.dumps(value) if isinstance(value, (dict, list)) else str(value or "")])
            result = self.redis.eval(
                ACCEPT_LUA,
                2,
                f"ml:feedback:v2:accepted:{event['event_id']}",
                STREAM,
                str(30 * 24 * 60 * 60),
                *fields,
            )
            if result == "duplicate":
                duplicates += 1
            else:
                accepted += 1
        return accepted, duplicates

    def health(self) -> dict[str, Any]:
        self.redis.ping()
        try:
            groups = self.redis.xinfo_groups(STREAM)
            group = next((item for item in groups if item.get("name") == GROUP), {})
        except Exception:
            group = {}
        return {"redis": "healthy", "feedback_pending": int(group.get("pending", 0)), "feedback_lag": int(group.get("lag", 0) or 0)}


@dataclass(frozen=True, slots=True)
class ApplyResult:
    status: str
    last_feedback_version: int


class OrderedFeedbackApplier:
    def __init__(self, qdrant: QdrantClient | None = None) -> None:
        self.qdrant = qdrant or QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)

    @staticmethod
    def _vector(value: Any, preferred: str | None = None) -> tuple[list[float], str | None]:
        if isinstance(value, dict):
            if preferred and preferred in value:
                return list(value[preferred]), preferred
            if len(value) == 1:
                name, vector = next(iter(value.items()))
                return list(vector), name
            raise ValueError("ambiguous named vector")
        if value is None:
            raise ValueError("missing vector")
        return list(value), None

    @staticmethod
    def _alpha(event: dict[str, Any]) -> float:
        if event["event_type"] != "dwell":
            return ALPHAS[event["event_type"]]
        dwell = min(300_000, max(3_000, int(event.get("dwell_ms") or 3_000)))
        return 0.15 * math.log1p(dwell) / math.log1p(300_000)

    def apply(self, event: dict[str, Any]) -> ApplyResult:
        user_id = str(uuid.UUID(event["user_id"]))
        repo_id = str(uuid.UUID(event["repo_id"]))
        version = int(event["feedback_version"])
        users = self.qdrant.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[user_id],
            with_payload=True,
            with_vectors=True,
        )
        if not users:
            raise LookupError(f"user vector {user_id} is not indexed")
        user = users[0]
        payload = dict(user.payload or {})
        last = int(payload.get("last_feedback_version") or 0)
        if version <= last:
            return ApplyResult("duplicate", last)
        if version != last + 1:
            return ApplyResult("gap", last)

        repos = self.qdrant.retrieve(
            collection_name=QDRANT_COLLECTION_NAME,
            ids=[repo_id],
            with_payload=False,
            with_vectors=True,
        )
        if not repos:
            raise LookupError(f"repository vector {repo_id} is not indexed")
        user_vector, user_vector_name = self._vector(user.vector, TARGET_VECTOR_NAME)
        repo_vector, _ = self._vector(repos[0].vector, QDRANT_VECTOR_NAME)
        if len(user_vector) != len(repo_vector):
            raise ValueError("user and repository vector dimensions differ")
        alpha = self._alpha(event)
        shifted = np.asarray(user_vector, dtype=np.float64) + alpha * np.asarray(repo_vector, dtype=np.float64)
        norm = float(np.linalg.norm(shifted))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("feedback produced an invalid vector")
        vector = (shifted / norm).tolist()
        payload["last_feedback_version"] = version
        payload["last_feedback_event_id"] = event["event_id"]
        stored_vector: Any = vector if user_vector_name is None else {user_vector_name: vector}
        self.qdrant.upsert(
            collection_name=USER_PROFILES_COLLECTION,
            points=[PointStruct(id=user_id, vector=stored_vector, payload=payload)],
            wait=True,
        )
        return ApplyResult("applied", version)


class OrderedFeedbackConsumer:
    def __init__(self, redis_client=None, applier: OrderedFeedbackApplier | None = None) -> None:
        self.redis = redis_client or _redis_client()
        self.applier = applier or OrderedFeedbackApplier()
        self.consumer = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4()}"
        try:
            self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _messages(self):
        try:
            claimed = self.redis.xautoclaim(STREAM, GROUP, self.consumer, 30_000, "0-0", count=20)
            for message_id, payload in claimed[1] if len(claimed) > 1 else []:
                yield message_id, payload
        except Exception as exc:
            logger.warning("pending feedback reclaim failed: %s", exc)
        for _, messages in self.redis.xreadgroup(GROUP, self.consumer, {STREAM: ">"}, count=20, block=1_000):
            yield from messages

    def run_once(self) -> int:
        processed = 0
        for message_id, payload in self._messages():
            user_id = payload.get("user_id")
            if not user_id:
                self.redis.xack(STREAM, GROUP, message_id)
                continue
            token = str(uuid.uuid4())
            lock = f"ml:feedback:v2:user-lock:{user_id}"
            if not self.redis.set(lock, token, nx=True, px=30_000):
                continue
            try:
                result = self.applier.apply(payload)
                if result.status != "gap":
                    self.redis.xack(STREAM, GROUP, message_id)
                    processed += 1
            except (ValueError, KeyError, LookupError) as exc:
                logger.error("feedback %s rejected: %s", message_id, exc)
                self.redis.xadd(f"{STREAM}:dead", {**payload, "error": str(exc)})
                self.redis.xack(STREAM, GROUP, message_id)
            finally:
                self.redis.eval(RELEASE_LOCK_LUA, 1, lock, token)
        return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    consumer = OrderedFeedbackConsumer()
    while True:
        consumer.run_once()
