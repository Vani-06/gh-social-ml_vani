import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.v2 import FeedbackBatch, RecommendationRequest, router
from embedding.qdrant_store import QdrantRepositoryStore


def test_recommendation_contract_rejects_duplicate_exclusions():
    item = uuid.uuid4()
    with pytest.raises(ValidationError):
        RecommendationRequest(
            schema_version=2,
            generation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            feed_version=1,
            limit=45,
            exclude_repo_ids=[item, item],
            context={"cold_start": False},
        )


def test_feedback_contract_enforces_dwell_and_unique_events():
    base = {
        "event_id": uuid.uuid4(), "user_id": uuid.uuid4(), "repo_id": uuid.uuid4(),
        "feedback_version": 1, "event_type": "dwell", "occurred_at": "2026-07-14T00:00:00Z",
    }
    with pytest.raises(ValidationError):
        FeedbackBatch(schema_version=2, events=[{**base, "dwell_ms": 2_999}])
    valid = {**base, "dwell_ms": 3_000}
    with pytest.raises(ValidationError):
        FeedbackBatch(schema_version=2, events=[valid, valid])


def test_repository_point_id_is_the_canonical_backend_uuid():
    repo_id = str(uuid.uuid4())
    assert QdrantRepositoryStore._point_id(repo_id) == repo_id
    with pytest.raises(ValueError):
        QdrantRepositoryStore._point_id("owner/repository")


def test_v2_health_requires_internal_auth(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    response = client.get("/api/v2/health")
    assert response.status_code == 401

    monkeypatch.delenv("INTERNAL_API_SECRET")
    response = client.get("/api/v2/health")
    assert response.status_code == 503
