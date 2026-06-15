import pytest
import numpy as np
import threading
from ingestion_engine import (
    extract_tags, classify_category, ingest_repository, CorpusStore, IngestionResult
)


# 1. extract_tags repository-name parsing
def test_extract_tags_repository_name_parsing():
    # duplicate words
    tags = extract_tags("my-react-react-app", [])
    assert tags.count("react") == 1
    assert "my" not in tags
    assert "app" not in tags

    # mixed case words
    tags = extract_tags("Vue-VuE-vUe", [])
    assert tags.count("vue") == 1

    # stop words
    tags = extract_tags("the-demo-project", [])
    assert len(tags) == 0

    # hyphenated repository names
    tags = extract_tags("react-admin-dashboard", [])
    assert set(tags) == {"react", "admin", "dashboard"}

    # underscore-separated names
    tags = extract_tags("python_machine_learning", [])
    assert set(tags) == {"python", "machine", "learning"}


# M1: Phrase deduplication
def test_extract_tags_phrase_deduplication():
    # test compound semantic concepts
    paragraphs = ["This is an AI Assistant that acts as a local AI."]
    tags = extract_tags("test-repo", paragraphs)
    assert tags.count("Ai Assistant") == 1
    assert tags.count("Local Ai") == 1
    
    # Should not add duplicates if lowercase exists
    paragraphs = ["The AI Assistant uses LLM."]
    # The title will have 'Ai Assistant', the tokens won't capture it as a single token but 'ai' and 'assistant'
    # Wait, 'Ai Assistant' should just be appended once.
    tags2 = extract_tags("ai-assistant", paragraphs)
    assert tags2.count("Ai Assistant") == 1


# 7. React vs ReAct classification
def test_react_classification():
    # Frontend react
    repo_ui = {
        "id": "facebook/react",
        "primary_language": "JavaScript",
        "extracted_paragraphs": ["React is a library for building user interfaces."]
    }
    cat_ui = classify_category(repo_ui, ["react", "ui", "javascript"])
    assert cat_ui == "Web/Frontend Frameworks"

    # AI Agent ReAct
    repo_agent = {
        "id": "reasoning/react-agent",
        "primary_language": "Python",
        "extracted_paragraphs": ["An autonomous agent using ReAct reasoning framework with LLM thought and action."]
    }
    cat_agent = classify_category(repo_agent, ["react", "agent", "llm"])
    assert cat_agent == "Data Engineering & AI/ML Pipelines"


# 8. concurrent ingestion behavior
def test_concurrent_ingestion():
    store = CorpusStore()
    
    def worker(i):
        store.add_node({"repo_id": str(i)}, 1.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.size() == 10
    assert len(store.get_timeline()) == 10

