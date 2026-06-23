import os
import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger("pipeline.feedback.producer")

# Global in-memory queue fallback for non-Redis environments
_in_memory_queue: asyncio.Queue = asyncio.Queue()


def get_in_memory_queue() -> asyncio.Queue:
    return _in_memory_queue


class FeedbackProducer:
    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.redis_client = None

        if self.redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis at %s for feedback streaming", self.redis_url)
            except Exception as exc:
                logger.warning("Redis connection failed: %s. Falling back to In-Memory Queue.", exc)
                self.redis_client = None

    async def submit_feedback(self, user_id: str, repo_id: str, action: str) -> bool:
        """Submit feedback event to the queue.

        Pushes to Redis Stream if available, otherwise falls back to the in-memory queue.
        """
        event = {
            "user_id": user_id,
            "repo_id": repo_id,
            "action": action,
        }

        if self.redis_client:
            try:
                # Add to Redis Stream
                self.redis_client.xadd("feedback_stream", event)
                logger.info("Published event to Redis Stream: %s", event)
                return True
            except Exception as exc:
                logger.error("Failed to publish to Redis Stream: %s. Falling back to In-Memory Queue.", exc)

        # Fallback to in-memory queue
        await _in_memory_queue.put(event)
        logger.info("Enqueued event to In-Memory Queue: %s", event)
        return True
