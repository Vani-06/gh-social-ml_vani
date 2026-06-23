import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from feedback.producer import FeedbackProducer
from feedback.consumer import FeedbackConsumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline.api")

# Global instances
producer: FeedbackProducer | None = None
consumer: FeedbackConsumer | None = None


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., description="Unique ID of the user performing the action")
    repo_id: str = Field(..., description="Full name or UUID of the repository")
    action: str = Field(..., description="Interaction action type: click, like, save, or skip")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize components and background worker tasks."""
    global producer, consumer
    logger.info("Initializing API components...")
    producer = FeedbackProducer()
    consumer = FeedbackConsumer()
    
    # Start background event consume worker loop
    await consumer.start()
    logger.info("Feedback Ingestion API and Background Consumer started successfully.")
    
    yield
    
    # Shutdown components
    logger.info("Shutting down API components...")
    if consumer:
        consumer.stop()
    logger.info("API components shut down.")


app = FastAPI(
    title="Git Social ML - Feedback Ingestion API",
    description="Real-time ingestion endpoint for user feedback events to update recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/api/v1/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(request: FeedbackRequest):
    """Endpoint to submit a user interaction event.

    Pushes the event to the processing queue and returns 202 Accepted.
    """
    action = request.action.lower()
    valid_actions = {"click", "like", "save", "skip"}
    if action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: '{request.action}'. Supported actions are: {list(valid_actions)}",
        )

    try:
        # Enqueue the event
        success = await producer.submit_feedback(
            user_id=request.user_id,
            repo_id=request.repo_id,
            action=action,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue feedback event.",
            )

        return {
            "status": "accepted",
            "message": "Feedback event received and queued successfully.",
            "data": {
                "user_id": request.user_id,
                "repo_id": request.repo_id,
                "action": action,
            },
        }

    except HTTPException:
        # Re-raise HTTPExceptions explicitly to prevent wrapping them in 500
        raise
    except Exception as exc:
        logger.error("Failed to process feedback submission: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal API error: {str(exc)}",
        )


@app.get("/api/v1/health")
async def health_check():
    """Basic service health check."""
    return {
        "status": "healthy",
        "consumer_running": consumer.running if consumer else False,
    }
