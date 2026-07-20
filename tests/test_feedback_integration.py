"""Opt-in end-to-end verification against real Redis and Qdrant services."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import os
import uuid

import pytest
from qdrant_client import QdrantClient, models

from embedding.vector_contract import repository_point_id, user_point_id
from feedback.consumer import FeedbackConsumer
from feedback.event_handlers import FeedbackHandler, PROCESSED_KEY
from feedback.producer import FeedbackProducer, create_redis_client
from feedback.settings import FeedbackSettings


pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _integration_enabled() -> bool:
    return os.getenv("RUN_FEEDBACK_INTEGRATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_feedback_event_round_trip_through_real_redis_and_qdrant(anyio_backend):
    if not _integration_enabled():
        pytest.skip("set RUN_FEEDBACK_INTEGRATION=true to use real Redis and Qdrant")

    base = FeedbackSettings.from_env()
    if not base.redis_url:
        pytest.skip("REDIS_URL is required for the feedback integration test")
    if base.vector_dimension < 2:
        pytest.skip("VECTOR_DIMENSION must be at least 2 for the integration test")

    suffix = uuid.uuid4().hex
    settings = replace(
        base,
        allow_memory_fallback=False,
        stream_name=f"feedback-integration:{suffix}",
        consumer_group=f"feedback-integration:{suffix}",
        consumer_name_prefix=f"feedback-integration-{suffix}",
        dead_letter_stream=f"feedback-integration-dlq:{suffix}",
        repository_collection=f"feedback_integration_repositories_{suffix}",
        user_collection=f"feedback_integration_users_{suffix}",
        user_vector_name=None,
    )
    redis_client = create_redis_client(settings)
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30.0,
    )
    created_collections: list[str] = []
    event_id = f"feedback-integration-{suffix}"

    try:
        try:
            await asyncio.to_thread(redis_client.ping)
            await asyncio.to_thread(qdrant.get_collections)
        except Exception as exc:
            pytest.skip(f"real Redis/Qdrant services are unavailable: {exc}")

        qdrant.create_collection(
            collection_name=settings.user_collection,
            vectors_config=models.VectorParams(
                size=settings.vector_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        created_collections.append(settings.user_collection)
        qdrant.create_collection(
            collection_name=settings.repository_collection,
            vectors_config={
                settings.repository_vector_name: models.VectorParams(
                    size=settings.vector_dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )
        created_collections.append(settings.repository_collection)

        user_id = str(uuid.uuid4())
        repo_id = str(uuid.uuid4())
        user_point_id_value = user_point_id(user_id)
        repo_point_id_value = repository_point_id(repo_id)
        user_vector = [1.0, 0.0] + [0.0] * (settings.vector_dimension - 2)
        repo_vector = [0.0, 1.0] + [0.0] * (settings.vector_dimension - 2)
        qdrant.upsert(
            collection_name=settings.user_collection,
            points=[
                models.PointStruct(
                    id=user_point_id_value,
                    vector=user_vector,
                    payload={"user_id": user_id},
                )
            ],
            wait=True,
        )
        qdrant.upsert(
            collection_name=settings.repository_collection,
            points=[
                models.PointStruct(
                    id=repo_point_id_value,
                    vector={settings.repository_vector_name: repo_vector},
                    payload={"repo_id": repo_id},
                )
            ],
            wait=True,
        )

        handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
        producer = FeedbackProducer(redis_client=redis_client, settings=settings)
        consumer = FeedbackConsumer(
            handler=handler,
            redis_client=redis_client,
            settings=settings,
        )
        await producer.start()
        await consumer._ensure_group()
        await producer.submit_feedback(
            user_id=user_id,
            repo_id=repo_id,
            action="like",
            event_id=event_id,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )

        response = await asyncio.to_thread(
            redis_client.xreadgroup,
            settings.consumer_group,
            consumer.consumer_name,
            {settings.stream_name: ">"},
            count=1,
            block=2_000,
        )
        assert response and response[0][1]
        message_id, payload = response[0][1][0]
        await consumer._process_message(str(message_id), payload)

        assert redis_client.exists(f"feedback:processed:{event_id}") == 1
        updated = qdrant.retrieve(
            collection_name=settings.user_collection,
            ids=[user_point_id_value],
            with_payload=True,
            with_vectors=True,
        )
        assert updated[0].payload[PROCESSED_KEY] == [event_id]
        assert updated[0].vector != pytest.approx(user_vector)
    finally:
        if redis_client is not None:
            try:
                redis_client.delete(
                    settings.stream_name,
                    settings.dead_letter_stream,
                    f"feedback:processed:{event_id}",
                    f"feedback:attempts:{event_id}",
                )
            except Exception:
                pass
        for collection_name in reversed(created_collections):
            try:
                qdrant.delete_collection(collection_name=collection_name)
            except Exception:
                pass
        close_redis = getattr(redis_client, "close", None)
        if close_redis:
            try:
                close_redis()
            except Exception:
                pass
        try:
            qdrant.close()
        except Exception:
            pass
