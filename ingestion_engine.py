"""Compatibility entrypoint for the canonical repository-ingestion worker.

Production ingestion is owned by :mod:`main`: GitHub acquisition publishes to
the authenticated Node v2 ingestion API, which creates canonical repository
identities and durable ML indexing outbox jobs. This module intentionally has
no direct PostgreSQL or Qdrant access.
"""

from __future__ import annotations

import sys

from main import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
