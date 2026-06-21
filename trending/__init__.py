"""GitHub Trending Repository Ingestion Engine.

This module provides a high-performance ingestion service that fetches the top 30
repositories from GitHub's Trending page, bypasses standard quality filters, and
refreshes this list every 24 hours.

Architecture:
- fetcher: Fetches trending repositories from GitHub Trending page via HTML parsing
- scheduler: Manages 24-hour refresh scheduling
- storage: Handles PostgreSQL storage for trending repos
- config: Trending-specific configuration
- logger: Centralized logging setup
"""

from .fetcher import TrendingFetcher
from .storage import TrendingStorage
from .scheduler import TrendingScheduler
from . import config

__all__ = [
    "TrendingFetcher",
    "TrendingStorage",
    "TrendingScheduler",
    "config",
]
