import pytest
from unittest.mock import MagicMock, patch

from retrieval.candidate_retriever import CandidateRetriever
from retrieval.config import (
    SEMANTIC_LIMIT,
    TRENDING_LIMIT,
    TOTAL_CANDIDATE_POOL,
    FALLBACK_REPOS,
    EMBEDDING_DIM
)

@pytest.fixture
def mock_db_connector():
    db = MagicMock()
    db.enabled = True
    
    # Mock connection and cursor
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    db.connect.return_value = conn
    
    return db, cursor

@pytest.fixture
def retriever(mock_db_connector):
    db, _ = mock_db_connector
    
    # Mock Qdrant instance
    mock_qdrant_instance = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 1000
    mock_qdrant_instance.get_collection.return_value = mock_collection_info
    
    r = CandidateRetriever(db_connector=db)
    # Manually inject the mocked qdrant client
    r._qdrant = mock_qdrant_instance
    return r

def test_semantic_retrieval(retriever):
    mock_qdrant = retriever._qdrant
    
    # Mock search results
    hit1 = MagicMock()
    hit1.id = "uuid-1"
    hit1.score = 0.95
    hit1.payload = {"repo_id": "repo-uuid-1"}
    
    hit2 = MagicMock()
    hit2.id = "uuid-2"
    hit2.score = 0.88
    hit2.payload = {"repo_id": "repo-uuid-2"}
    
    mock_qdrant.search.return_value = [hit1, hit2]
    
    embedding = [0.1] * EMBEDDING_DIM
    results = retriever._retrieve_semantic(embedding, quota=10)
    
    assert len(results) == 2
    assert results[0]["repo_id"] == "repo-uuid-1"
    assert results[0]["score"] == 0.95
    assert results[0]["source"] == "semantic"
    mock_qdrant.search.assert_called_once()

def test_trending_retrieval(retriever, mock_db_connector):
    db, cursor = mock_db_connector
    
    # Mock DB rows (repo_id, full_name, star_count)
    cursor.fetchall.return_value = [
        ("repo-uuid-3", "org/repo3", 5000),
        ("repo-uuid-4", "org/repo4", 4000),
    ]
    
    results = retriever._retrieve_trending(quota=10)
    
    assert len(results) == 2
    assert results[0]["repo_id"] == "repo-uuid-3"
    assert results[0]["full_name"] == "org/repo3"
    assert results[0]["star_count"] == 5000
    assert results[0]["source"] == "trending"
    cursor.execute.assert_called_once()

def test_merge_and_deduplicate(retriever):
    semantic = [
        {"repo_id": "r1", "score": 0.9},
        {"repo_id": "r2", "score": 0.8},
    ]
    trending = [
        {"repo_id": "r2", "star_count": 1000},  # Duplicate
        {"repo_id": "r3", "star_count": 500},
        {"repo_id": "r4", "star_count": 100},
    ]
    
    merged = retriever._merge_and_deduplicate(semantic, trending, pool_limit=3)
    
    assert len(merged) == 3
    # r1 from semantic
    assert merged[0]["repo_id"] == "r1"
    # r2 from semantic (priority)
    assert merged[1]["repo_id"] == "r2"
    assert "score" in merged[1]
    assert "star_count" not in merged[1]
    # r3 from trending
    assert merged[2]["repo_id"] == "r3"

def test_fallback_repos(retriever, mock_db_connector):
    # Disable both sources
    retriever._qdrant = None
    retriever.db.enabled = False
    
    results = retriever.retrieve_candidates(user_embedding=[0.1] * EMBEDDING_DIM)
    
    assert len(results) == len(FALLBACK_REPOS)
    assert results[0]["retrieval_source"] == "fallback"
    assert results[0]["repo_embedding"] == [0.0] * EMBEDDING_DIM

def test_end_to_end_retrieval(retriever, mock_db_connector):
    db, cursor = mock_db_connector
    mock_qdrant = retriever._qdrant
    
    # Mock Semantic Search
    hit = MagicMock()
    hit.id = "uuid-1"
    hit.score = 0.9
    hit.payload = {"repo_id": "repo-1"}
    mock_qdrant.search.return_value = [hit]
    
    # Mock Trending
    cursor.fetchall.side_effect = [
        # Call 1: Trending search
        [("repo-2", "org/repo2", 100)],
        # Call 2: Metadata hydration
        [("repo-1", "url", "owner", "name", "org/repo1", "desc", "lang", "[]", "[]", "readme", 50, 10, 5, 20, 0, 0, 0, None, None),
         ("repo-2", "url", "owner", "name", "org/repo2", "desc", "lang", "[]", "[]", "readme", 100, 10, 5, 20, 0, 0, 0, None, None)]
    ]
    
    # Mock Embedding hydration
    point1 = MagicMock()
    point1.id = "uuid-1"
    point1.vector = [0.5] * EMBEDDING_DIM
    
    point2 = MagicMock()
    point2.id = "uuid-2"
    point2.vector = [0.2] * EMBEDDING_DIM
    
    mock_qdrant.retrieve.return_value = [point1, point2]
    
    candidates = retriever.retrieve_candidates(user_embedding=[0.1] * EMBEDDING_DIM)
    
    assert len(candidates) == 2
    assert candidates[0]["repo_id"] == "repo-1"
    assert candidates[0]["retrieval_source"] == "semantic"
    assert candidates[0]["repo_embedding"] == [0.5] * EMBEDDING_DIM
    
    assert candidates[1]["repo_id"] == "repo-2"
    assert candidates[1]["retrieval_source"] == "trending"
    assert candidates[1]["star_count"] == 100
    assert "repo_embedding" in candidates[1]
