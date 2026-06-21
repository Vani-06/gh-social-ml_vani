"""Trending-specific configuration for the GitHub Trending ingestion engine.

This module contains all configuration parameters for the trending repository
ingestion service, including database settings, API limits, and scheduling parameters.
"""

import os
import re


# ── Core Configuration ─────────────────────────────────────────────────────────────

# Number of trending repositories to fetch per refresh cycle
TRENDING_REPO_LIMIT: int = int(os.getenv("TRENDING_REPO_LIMIT", "30"))

# Refresh interval in hours (default: 24 hours)
TRENDING_REFRESH_HOURS: int = int(os.getenv("TRENDING_REFRESH_HOURS", "24"))

# PostgreSQL table names for trending repositories
TRENDING_TABLE_NAME: str = os.getenv("TRENDING_TABLE_NAME", "trending_repositories")
TRENDING_METADATA_TABLE_NAME: str = os.getenv("TRENDING_METADATA_TABLE_NAME", "trending_metadata")


# ── GitHub API Configuration ───────────────────────────────────────────────────────

# Request timeout in seconds (for HTTP requests to GitHub Trending page)
GITHUB_TIMEOUT_SECONDS: float = float(os.getenv("GITHUB_TIMEOUT_SECONDS", "30.0"))

# Maximum retry attempts for failed requests
GITHUB_MAX_RETRIES: int = int(os.getenv("GITHUB_MAX_RETRIES", "4"))


# ── Data Enrichment Configuration ───────────────────────────────────────────────────

# Whether to fetch README content for trending repositories
FETCH_README: bool = os.getenv("FETCH_README", "true").lower() == "true"

# Whether to fetch repository topics/tags
FETCH_TOPICS: bool = os.getenv("FETCH_TOPICS", "true").lower() == "true"

# Maximum README length to store (in characters)
README_MAX_LENGTH: int = int(os.getenv("README_MAX_LENGTH", "10000"))


# ── Database Configuration ─────────────────────────────────────────────────────────

# PostgreSQL database URL (shared with main application)
DATABASE_URL: str | None = os.getenv("DATABASE_URL")

# Connection pool size for database operations
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))


# ── Logging Configuration ───────────────────────────────────────────────────────────

# Log level for trending service
LOG_LEVEL: str = os.getenv("TRENDING_LOG_LEVEL", "INFO")

# Log file path (if not set, logs to stdout)
LOG_FILE: str | None = os.getenv("TRENDING_LOG_FILE")

# Log format
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ── Error Handling Configuration ───────────────────────────────────────────────────

# Whether to continue processing on individual repository failures
CONTINUE_ON_ERROR: bool = os.getenv("CONTINUE_ON_ERROR", "true").lower() == "true"

# Maximum number of consecutive failures before stopping the pipeline
MAX_CONSECUTIVE_FAILURES: int = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "10"))


# ── Validation ─────────────────────────────────────────────────────────────────────

def validate_config() -> list[str]:
    """Validate trending configuration and return list of errors.

    Returns:
        List of error messages. Empty list if configuration is valid.
    """
    errors: list[str] = []

    if TRENDING_REPO_LIMIT <= 0:
        errors.append(f"TRENDING_REPO_LIMIT must be positive, got {TRENDING_REPO_LIMIT}")

    if TRENDING_REFRESH_HOURS <= 0:
        errors.append(f"TRENDING_REFRESH_HOURS must be positive, got {TRENDING_REFRESH_HOURS}")

    # Validate table names to prevent SQL injection
    table_name_pattern = r'^[a-z_][a-z0-9_]*$'
    if not re.match(table_name_pattern, TRENDING_TABLE_NAME):
        errors.append(f"TRENDING_TABLE_NAME must match pattern {table_name_pattern}, got '{TRENDING_TABLE_NAME}'")
    if not re.match(table_name_pattern, TRENDING_METADATA_TABLE_NAME):
        errors.append(f"TRENDING_METADATA_TABLE_NAME must match pattern {table_name_pattern}, got '{TRENDING_METADATA_TABLE_NAME}'")

    if not DATABASE_URL:
        errors.append("DATABASE_URL is not set. Database integration will be disabled.")

    return errors
