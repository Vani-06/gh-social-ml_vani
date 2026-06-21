#!/usr/bin/env python3
"""Main entry point for the GitHub Trending ingestion service.

This script provides a command-line interface for running the trending repository
ingestion engine. It supports both scheduled (daemon) mode and one-time execution.

Usage:
    # Run as a scheduled service (refreshes every 24 hours)
    python trending_service.py --scheduled

    # Run a single refresh cycle
    python trending_service.py --once

    # Run with custom configuration
    python trending_service.py --once --limit 50 --refresh-hours 12
"""

import argparse
import logging
import os
import sys

# Load environment variables from .env file before importing config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional, continue without it

from trending.scheduler import run_scheduler, run_once
from trending.config import validate_config, TRENDING_REPO_LIMIT, TRENDING_REFRESH_HOURS
from trending.logger import setup_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GitHub Trending Repository Ingestion Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scheduled      Run as a scheduled service (24-hour refresh cycle)
  %(prog)s --once           Run a single refresh cycle
  %(prog)s --once --limit 50  Fetch 50 repositories instead of default 30
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--scheduled",
        action="store_true",
        help="Run as a scheduled service (refreshes every TRENDING_REFRESH_HOURS)",
    )
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run a single refresh cycle and exit",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Number of repositories to fetch (default: {TRENDING_REPO_LIMIT})",
    )

    parser.add_argument(
        "--refresh-hours",
        type=int,
        default=None,
        help=f"Refresh interval in hours (default: {TRENDING_REFRESH_HOURS})",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: stdout)",
    )

    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit without running",
    )

    return parser.parse_args()


def main():
    """Main entry point for the trending service."""
    args = parse_args()

    # Setup logging
    logger = setup_logger(
        name="trending_service",
        level=args.log_level,
        log_file=args.log_file,
    )

    logger.info("GitHub Trending Ingestion Service")
    logger.info("=" * 60)

    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        logger.error("Configuration validation failed:")
        for error in config_errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    if args.validate_config:
        logger.info("Configuration validation passed.")
        if not args.scheduled and not args.once:
            sys.exit(0)

    # Check if mode is specified
    if not args.scheduled and not args.once:
        logger.error("Either --scheduled or --once must be specified (unless using --validate-config)")
        sys.exit(1)

    # Override configuration if command-line arguments provided
    if args.limit:
        import trending.config as config
        config.TRENDING_REPO_LIMIT = args.limit
        logger.info(f"Override: TRENDING_REPO_LIMIT = {args.limit}")

    if args.refresh_hours:
        import trending.config as config
        config.TRENDING_REFRESH_HOURS = args.refresh_hours
        logger.info(f"Override: TRENDING_REFRESH_HOURS = {args.refresh_hours}")

    # Run in requested mode
    try:
        if args.scheduled:
            logger.info("Starting scheduled mode...")
            run_scheduler()
        elif args.once:
            logger.info("Starting single refresh cycle...")
            success = run_once(force=True)
            if success:
                logger.info("Single refresh cycle completed successfully.")
                sys.exit(0)
            else:
                logger.error("Single refresh cycle failed.")
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
