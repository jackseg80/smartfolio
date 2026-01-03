#!/usr/bin/env python3
"""
Portfolio History Migration Script

PERFORMANCE FIX (Dec 2025): Migrate from monolithic portfolio_history.json
to partitioned structure for O(1) access.

Usage:
    python scripts/migrate_portfolio_history.py [--dry-run] [--force]

Options:
    --dry-run    Show what would be migrated without actually doing it
    --force      Skip backup prompts (use with caution)

Example:
    # Dry run to see migration plan
    python scripts/migrate_portfolio_history.py --dry-run

    # Actual migration with prompts
    python scripts/migrate_portfolio_history.py

    # Force migration without prompts
    python scripts/migrate_portfolio_history.py --force
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.portfolio_history_storage import PartitionedPortfolioStorage
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run migration script."""
    parser = argparse.ArgumentParser(
        description='Migrate portfolio history to partitioned structure'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show migration plan without executing'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip backup prompts (use with caution)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Portfolio History Migration Script")
    logger.info("=" * 80)

    # Initialize storage
    storage = PartitionedPortfolioStorage(retention_days=365)

    # Check if legacy file exists
    if not storage.legacy_file.exists():
        logger.warning(f"Legacy file not found: {storage.legacy_file}")
        logger.info("Migration not needed - system already using partitioned storage")
        return 0

    # Load legacy file to show stats
    import json
    try:
        with open(storage.legacy_file, 'r', encoding='utf-8') as f:
            legacy_data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to read legacy file: {e}")
        return 1

    if not legacy_data:
        logger.warning("Legacy file is empty - nothing to migrate")
        return 0

    # Calculate migration stats
    from collections import defaultdict
    stats_by_user = defaultdict(lambda: defaultdict(int))

    for snapshot in legacy_data:
        user_id = snapshot.get('user_id', 'demo')
        source = snapshot.get('source', 'cointracking')
        stats_by_user[user_id][source] += 1

    # Display migration plan
    logger.info("")
    logger.info("Migration Plan:")
    logger.info("-" * 80)
    logger.info(f"Legacy file: {storage.legacy_file}")
    logger.info(f"Total snapshots: {len(legacy_data)}")
    logger.info(f"Users: {len(stats_by_user)}")
    logger.info("")

    for user_id, sources in stats_by_user.items():
        logger.info(f"  User: {user_id}")
        for source, count in sources.items():
            logger.info(f"    └─ {source}: {count} snapshots")

    logger.info("")
    logger.info("Target structure:")
    logger.info("  data/portfolio_history/")
    logger.info("    {user_id}/")
    logger.info("      {source}/")
    logger.info("        {YYYY}/")
    logger.info("          {MM}/")
    logger.info("            snapshots.json  # Max 31 snapshots per file")
    logger.info("")

    # Dry run mode - exit here
    if args.dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN - No changes made")
        logger.info("=" * 80)
        logger.info("")
        logger.info("To execute migration, run without --dry-run:")
        logger.info("  python scripts/migrate_portfolio_history.py")
        return 0

    # Prompt for confirmation (unless --force)
    if not args.force:
        logger.warning("=" * 80)
        logger.warning("WARNING: This will migrate your portfolio history data")
        logger.warning("=" * 80)
        logger.warning("")
        logger.warning(f"The legacy file will be backed up to: {storage.legacy_file}.backup")
        logger.warning("")

        response = input("Continue with migration? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            logger.info("Migration cancelled by user")
            return 0

    # Execute migration
    logger.info("")
    logger.info("=" * 80)
    logger.info("Starting migration...")
    logger.info("=" * 80)

    migration_stats = storage.migrate_from_legacy()

    # Display results
    logger.info("")
    logger.info("=" * 80)
    logger.info("Migration Complete!")
    logger.info("=" * 80)
    logger.info(f"Snapshots migrated: {migration_stats['snapshots_migrated']}")
    logger.info(f"Users: {migration_stats['users']}")
    logger.info(f"Sources: {migration_stats['sources']}")
    logger.info(f"Backup file: {storage.legacy_file}.backup")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify partitioned data: ls data/portfolio_history/")
    logger.info("  2. Test application endpoints")
    logger.info("  3. Monitor logs for any issues")
    logger.info("")
    logger.info("If you need to rollback:")
    logger.info(f"  mv {storage.legacy_file}.backup {storage.legacy_file}")
    logger.info("  rm -rf data/portfolio_history/")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
