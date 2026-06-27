import math
import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, conlist, field_validator

from inference.feed_assembly import FeedAssemblySystem

logger = logging.getLogger(__name__)


class FeedCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    repo_id: str = Field(..., min_length=1)
    final_score: float | None = None
    score: float | None = None
    created_at: datetime | None = None

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        repo_id = value.strip()
        if not repo_id:
            raise ValueError("repo_id must not be blank")
        return repo_id

    @field_validator("final_score", "score")
    @classmethod
    def validate_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score values must be finite")
        return value


class FeedAssemblyRequest(BaseModel):
    candidates: conlist(FeedCandidate, min_length=15, max_length=15)


def _candidate_to_dict(candidate: FeedCandidate) -> dict:
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump(mode="python")
    return candidate.dict()


# 1. Initialize the FastAPI Web Service Application Space
app = FastAPI(
    title="GH-Social ML Assembly Engine",
    description="Internal ML service serving freshness and exploration injections.",
    version="1.0.0"
)

# 2. Add safe local and production guardrails (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/internal/ml/assemble-feed", status_code=status.HTTP_200_OK)
async def assemble_feed_endpoint(request: FeedAssemblyRequest):
    """
    Receives the 15 pre-ranked JSON objects from the main backend,
    applies time-decay freshness boosts and bottom-tier exploration shuffling,
    and returns the final ordered sequence of string IDs.
    """
    candidate_dicts = []

    try:
        candidate_dicts = [_candidate_to_dict(candidate) for candidate in request.candidates]

        # Pass the candidates block down to your verified module
        ordered_ids = FeedAssemblySystem.process_feed_assembly(candidate_dicts, target_size=15)
        return {"rankedRepoIds": ordered_ids}
        
    except Exception as err:
        # Fail-soft fallback so an unexpected parsing error won't crash the web worker process
        logger.exception("Feed assembly failed: %s", err)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Processing Exception",
                "details": "Feed assembly failed",
                "rankedRepoIds": [
                    str(item.get("repo_id"))
                    for item in candidate_dicts[:15]
                    if item.get("repo_id")
                ]
            },
        )

if __name__ == "__main__":
    # Run the Uvicorn worker locally on Port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
