import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from retrieval_engine import RetrievalEngine


@pytest.fixture
def mock_retrieval_dependencies():
    """Mocks Qdrant and Postgres databases for RetrievalEngine."""
    with patch("retrieval_engine.QdrantClient") as mock_qdrant_cls, \
         patch("retrieval_engine.QdrantRepositoryStore") as mock_repo_store_cls, \
         patch("database.PostgreSQLConnector") as mock_db_cls:
        
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        
        mock_repo_store = MagicMock()
        mock_repo_store_cls.return_value = mock_repo_store
        
        mock_db = MagicMock()
        mock_db.enabled = True
        mock_db_cls.return_value = mock_db
        
        yield mock_qdrant, mock_repo_store, mock_db


def test_retrieval_engine_lazy_loading(mock_retrieval_dependencies):
    """Verify that RankerService and PostgreSQLConnector are loaded lazily."""
    _, _, _ = mock_retrieval_dependencies
    
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    
    # Internal references should start as None/unloaded
    assert engine._db is None
    assert engine._ranker is None
    
    # Trigger lazy loading properties
    db = engine.db
    assert db is not None
    assert engine._db is not None
    
    # Mocking RankerService instantiation to avoid loading actual torch weights in unit tests
    with patch("inference.ranker_service.RankerService") as mock_ranker_cls:
        ranker = engine.ranker
        assert ranker is not None
        assert engine._ranker is not None


def test_retrieval_engine_fetch_and_rank(mock_retrieval_dependencies):
    """Verify RetrievalEngine fetches user interest profile and ranks them using RankerService."""
    mock_qdrant, mock_repo_store, mock_db = mock_retrieval_dependencies
    
    # 1. Setup mock user profile with interests
    mock_user_point = MagicMock()
    mock_user_point.vector = [0.1] * 384
    mock_user_point.payload = {
        "user_id": "user_456",
        "skills": ["Python", "AI/ML"],
    }
    mock_qdrant.retrieve.return_value = [mock_user_point]
    
    # 2. Setup mock corpus search returning 3 candidate points
    mock_repo_store.search.return_value = [
        {
            "id": "point_1",
            "score": 0.85,
            "repo_id": "owner/repo1",
            "vector": [0.2] * 384,
            "payload": {
                "repo_id": "owner/repo1",
                "star_count": 100,
                "primary_language": "Python",
            }
        },
        {
            "id": "point_2",
            "score": 0.80,
            "repo_id": "owner/repo2",
            "vector": [0.3] * 384,
            "payload": {
                "repo_id": "owner/repo2",
                "star_count": 200,
                "primary_language": "Python",
            }
        },
        {
            "id": "point_3",
            "score": 0.75,
            "repo_id": "owner/repo3",
            "vector": [0.4] * 384,
            "payload": {
                "repo_id": "owner/repo3",
                "star_count": 300,
                "primary_language": "JavaScript",
            }
        }
    ]
    
    # 3. Setup mock RankerService that ranks repo2 as #1, repo3 as #2, and repo1 as #3
    mock_ranker = MagicMock()
    mock_ranker.score_batch.return_value = [
        {"repo_id": "owner/repo2", "final_score": 10.5},
        {"repo_id": "owner/repo3", "final_score": 5.2},
        {"repo_id": "owner/repo1", "final_score": 1.1},
    ]
    
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = mock_ranker
    
    # Mock postgres connection
    mock_conn = MagicMock()
    mock_db.connect.return_value = mock_conn
    
    # Call fetch onboarding batches
    batches = engine.fetch_onboarding_batches("user_456")
    
    # Assert Qdrant corpus search is queried with with_vectors=True
    mock_repo_store.search.assert_called_once_with(
        [0.1] * 384,
        limit=45,
        exact=True,
        with_vectors=True
    )
    
    # Assert RankerService.score_batch was called with the candidates
    assert mock_ranker.score_batch.call_count == 1
    call_args = mock_ranker.score_batch.call_args[0]
    assert call_args[0] == [0.1] * 384  # user vector
    assert call_args[1] == ["Python", "AI/ML"]  # user skills
    
    # Assert batches are constructed in the ranker-sorted order
    batch_1 = batches["batch_1"]
    assert len(batch_1) == 3
    assert batch_1[0]["repo_id"] == "owner/repo2"
    assert batch_1[1]["repo_id"] == "owner/repo3"
    assert batch_1[2]["repo_id"] == "owner/repo1"
    
    # Assert scores are updated to the ranker score
    assert batch_1[0]["cosine_score"] == 10.5
    assert batch_1[1]["cosine_score"] == 5.2
    assert batch_1[2]["cosine_score"] == 1.1
