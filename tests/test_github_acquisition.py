from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any

import ingestion_engine
from utils.readme_processor import process_markdown, process_readme_payload


def test_readme_processor_decodes_and_extracts_meaningful_paragraphs() -> None:
    markdown = """
# Project
![badge](https://img.shields.io/badge/build-passing.svg)

This repository provides a robust automation framework for running developer workflows across cloud systems and local machines.

```python
print("not semantic docs")
```

![screenshot](docs/screenshot.png)

## Install
pip install example

The architecture includes a scheduler, plugin runtime, event router, and integration adapters for production operations.
"""
    payload = {"encoding": "base64", "content": base64.b64encode(markdown.encode()).decode()}

    document = process_readme_payload(payload)

    assert document.readme_length == len(markdown)
    assert len(document.extracted_paragraphs) == 2
    assert "badge" not in document.clean_text.lower()
    assert "screenshot" not in document.clean_text.lower()


def test_ingestion_engine_pre_filters() -> None:
    store = ingestion_engine.CorpusStore()
    # Passes pre-filters and blended score (0.6206 >= 0.60)
    first = ingestion_engine.ingest_repository(_osiris_repo("owner/seed"), corpus_store=store, auto_index=True)
    # Low star repo - rejected by pre-filter
    low_stars = _osiris_repo("owner/low-stars")
    low_stars["star_count"] = 10
    second = ingestion_engine.ingest_repository(low_stars, corpus_store=store, auto_index=True)
    # Low readme repo - rejected by pre-filter
    low_readme = _osiris_repo("owner/low-readme")
    low_readme["readme_length"] = 50
    third = ingestion_engine.ingest_repository(low_readme, corpus_store=store, auto_index=True)


    assert first.decision == "APPROVED"
    assert first.novelty.final == 1.0
    assert second.decision == "REJECTED"
    assert "star count (10) is below minimum threshold" in second.rejection_reason
    assert third.decision == "REJECTED"
    assert "README length (50 chars) is below minimum threshold" in third.rejection_reason


def _osiris_repo(repo_id: str) -> dict[str, Any]:
    return {
        "id": repo_id,
        "star_count": 100,
        "pushed_days_ago": 1,
        "mentionable_users_count": 5,
        "primary_language": "Python",
        "readme_length": 2000,
        "readme_to_codebase_ratio": 0.05,
        "extracted_paragraphs": [
            "This Python AI agent framework provides retrieval augmented generation workflows, tool calling, and production automation for developer systems."
        ],
        "delta_3d": 3,
        "delta_7d": 7,
        "delta_30d": 20,
    }

