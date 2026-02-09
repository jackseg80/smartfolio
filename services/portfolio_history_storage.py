"""
Portfolio History Storage - Partitioned File Structure

PERFORMANCE FIX (Dec 2025): Migrate from single JSON file to partitioned structure.

Previous (SLOW):
  - Single file: data/portfolio_history.json
  - Load ALL snapshots for ALL users → O(n) scan
  - File grows indefinitely (100MB+ after 1 year)

New (FAST):
  - Partitioned: data/portfolio_history/{user_id}/{source}/{YYYY}/{MM}/snapshots.json
  - Load only relevant month → O(1) access
  - Max 31 snapshots per file (1 month)
  - Automatic cleanup of old data

Architecture:
    data/portfolio_history/
        demo/
            cointracking/
                2025/
                    12/
                        snapshots.json  # Dec 2025 snapshots only
                    11/
                        snapshots.json  # Nov 2025 snapshots only
            saxobank/
                2025/
                    12/
                        snapshots.json
        jack/
            cointracking/
                2025/
                    12/
                        snapshots.json
"""

import json
import os
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

from filelock import FileLock

logger = logging.getLogger(__name__)

TZ = ZoneInfo("Europe/Zurich")
BASE_DIR = Path("data/portfolio_history")


class PartitionedPortfolioStorage:
    """
    Partitioned portfolio history storage for O(1) access.

    Features:
        - Automatic partitioning by user_id/source/year/month
        - O(1) read/write (no full file scan)
        - Automatic retention (365 days default)
        - Backward compatible with legacy portfolio_history.json
        - Thread-safe / Coroutine-safe via asyncio locks
    """

    def __init__(self, retention_days: int = 365):
        """
        Initialize partitioned storage.

        Args:
            retention_days: Number of days to retain snapshots (default: 365)
        """
        self.retention_days = retention_days
        self.legacy_file = Path("data/portfolio_history.json")
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create an asyncio lock for a specific file path."""
        if path not in self._locks:
            self._locks[path] = asyncio.Lock()
        return self._locks[path]

    def _get_partition_path(self, user_id: str, source: str, date: datetime) -> Path:
        """
        Get partition file path for a given date.

        Args:
            user_id: User identifier
            source: Data source (e.g., "cointracking", "saxobank")
            date: Snapshot date

        Returns:
            Path to partition file (e.g., data/portfolio_history/demo/cointracking/2025/12/snapshots.json)
        """
        year = str(date.year)
        month = f"{date.month:02d}"

        partition_dir = BASE_DIR / user_id / source / year / month
        return partition_dir / "snapshots.json"

    async def save_snapshot(
        self,
        snapshot: Dict[str, Any],
        user_id: str,
        source: str
    ) -> bool:
        """
        Save portfolio snapshot to partitioned storage.
        Async method with partition-level locking to prevent race conditions.

        Args:
            snapshot: Snapshot data (must contain 'date' field)
            user_id: User identifier
            source: Data source

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Parse snapshot date
            snapshot_date = datetime.fromisoformat(snapshot["date"])
            if snapshot_date.tzinfo is None:
                snapshot_date = snapshot_date.replace(tzinfo=TZ)

            # Get partition path
            partition_path = self._get_partition_path(user_id, source, snapshot_date)
            lock_key = str(partition_path)

            # Create directory if needed
            partition_path.parent.mkdir(parents=True, exist_ok=True)

            # Critical Section: Read -> Modify -> Write
            async with self._get_lock(lock_key):
                # Load existing snapshots for this month
                month_snapshots = []
                if partition_path.exists():
                    try:
                        # Note: Blocking I/O in async method, but acceptable inside lock for small JSONs
                        # For very large files, consider run_in_executor
                        with open(partition_path, 'r', encoding='utf-8') as f:
                            month_snapshots = json.load(f)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Corrupted partition file {partition_path}, resetting: {e}")
                        month_snapshots = []

                # Upsert snapshot (replace if same date exists)
                snapshot_date_str = snapshot["date"]
                existing_idx = next(
                    (i for i, s in enumerate(month_snapshots) if s.get("date") == snapshot_date_str),
                    None
                )

                if existing_idx is not None:
                    # Update existing snapshot
                    month_snapshots[existing_idx] = snapshot
                    logger.debug(f"Updated existing snapshot for {snapshot_date_str}")
                else:
                    # Append new snapshot
                    month_snapshots.append(snapshot)
                    logger.debug(f"Added new snapshot for {snapshot_date_str}")

                # Sort by date
                month_snapshots.sort(key=lambda x: x.get("date", ""))

                # Save atomically with filelock (protects multi-process)
                temp_path = partition_path.with_suffix('.tmp')
                with FileLock(str(partition_path) + ".lock", timeout=5):
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(month_snapshots, f, indent=2)
                    temp_path.replace(partition_path)

            logger.info(
                f"Snapshot saved: user={user_id}, source={source}, "
                f"date={snapshot_date_str}, partition={partition_path.relative_to(BASE_DIR)}"
            )
            return True

        except KeyError as e:
            logger.error(f"Missing required field in snapshot: {e}")
            return False
        except (OSError, PermissionError) as e:
            logger.error(f"I/O error saving snapshot: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving snapshot: {e}", exc_info=True)
            return False

    def load_snapshots(
        self,
        user_id: str,
        source: str,
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load portfolio snapshots for a user/source.

        Args:
            user_id: User identifier
            source: Data source
            days: Number of days to load (None = all within retention period)

        Returns:
            List of snapshots, sorted by date (oldest first)
        """
        try:
            # Calculate date range
            now = datetime.now(TZ)
            if days is not None:
                start_date = now - timedelta(days=days)
            else:
                start_date = now - timedelta(days=self.retention_days)

            # Find all partition files to load
            partitions_to_load = self._get_partitions_in_range(user_id, source, start_date, now)

            # Load all snapshots from partitions
            all_snapshots = []
            for partition_path in partitions_to_load:
                if not partition_path.exists():
                    continue

                try:
                    with open(partition_path, 'r', encoding='utf-8') as f:
                        month_snapshots = json.load(f)
                        all_snapshots.extend(month_snapshots)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Corrupted partition {partition_path}, skipping: {e}")
                    continue

            # Filter by date range
            filtered_snapshots = [
                s for s in all_snapshots
                if start_date <= datetime.fromisoformat(s.get("date", "")) <= now
            ]

            # Sort by date
            filtered_snapshots.sort(key=lambda x: x.get("date", ""))

            logger.debug(
                f"Loaded {len(filtered_snapshots)} snapshots for user={user_id}, source={source}, "
                f"days={days or self.retention_days}"
            )

            return filtered_snapshots

        except Exception as e:
            logger.error(f"Error loading snapshots: {e}", exc_info=True)
            return []

    def _get_partitions_in_range(
        self,
        user_id: str,
        source: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Path]:
        """
        Get all partition files within a date range.

        Args:
            user_id: User identifier
            source: Data source
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of partition file paths to load
        """
        partitions = []

        # Iterate through months in range
        current = start_date.replace(day=1)
        while current <= end_date:
            partition_path = self._get_partition_path(user_id, source, current)
            partitions.append(partition_path)

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return partitions

    def cleanup_old_snapshots(self, user_id: Optional[str] = None) -> int:
        """
        Remove snapshots older than retention period.

        Args:
            user_id: User to cleanup (None = all users)

        Returns:
            Number of partitions removed
        """
        try:
            cutoff_date = datetime.now(TZ) - timedelta(days=self.retention_days)
            removed_count = 0

            # Determine users to cleanup
            if user_id:
                users_to_check = [user_id]
            else:
                if not BASE_DIR.exists():
                    return 0
                users_to_check = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]

            for user in users_to_check:
                user_dir = BASE_DIR / user
                if not user_dir.exists():
                    continue

                # Iterate through sources
                for source_dir in user_dir.iterdir():
                    if not source_dir.is_dir():
                        continue

                    # Iterate through years
                    for year_dir in source_dir.iterdir():
                        if not year_dir.is_dir():
                            continue

                        # Iterate through months
                        for month_dir in year_dir.iterdir():
                            if not month_dir.is_dir():
                                continue

                            # Check if month is outside retention period
                            try:
                                year = int(year_dir.name)
                                month = int(month_dir.name)
                                month_date = datetime(year, month, 1, tzinfo=TZ)

                                if month_date < cutoff_date.replace(day=1):
                                    # Remove entire month directory
                                    import shutil
                                    shutil.rmtree(month_dir)
                                    removed_count += 1
                                    logger.info(f"Removed old partition: {month_dir.relative_to(BASE_DIR)}")
                            except (ValueError, OSError) as e:
                                logger.warning(f"Error removing partition {month_dir}: {e}")
                                continue

            if removed_count > 0:
                logger.info(f"Cleanup complete: removed {removed_count} old partitions")

            return removed_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            return 0

    async def migrate_from_legacy(self) -> Dict[str, int]:
        """
        Migrate data from legacy portfolio_history.json to partitioned structure.

        Returns:
            Migration stats: {"snapshots_migrated": N, "users": N, "sources": N}
        """
        if not self.legacy_file.exists():
            logger.info("No legacy file to migrate")
            return {"snapshots_migrated": 0, "users": 0, "sources": 0}

        try:
            logger.info(f"Starting migration from {self.legacy_file}")

            # Load legacy file
            with open(self.legacy_file, 'r', encoding='utf-8') as f:
                legacy_snapshots = json.load(f)

            if not legacy_snapshots:
                logger.info("Legacy file is empty, nothing to migrate")
                return {"snapshots_migrated": 0, "users": 0, "sources": 0}

            # Group snapshots by (user_id, source)
            from collections import defaultdict
            grouped = defaultdict(list)
            for snapshot in legacy_snapshots:
                user_id = snapshot.get("user_id", "demo")
                source = snapshot.get("source", "cointracking")
                grouped[(user_id, source)].append(snapshot)

            # Migrate each group
            migrated_count = 0
            for (user_id, source), snapshots in grouped.items():
                for snapshot in snapshots:
                    if await self.save_snapshot(snapshot, user_id, source):
                        migrated_count += 1

            # Rename legacy file to .backup
            backup_path = self.legacy_file.with_suffix('.json.backup')
            self.legacy_file.rename(backup_path)
            logger.info(f"Legacy file backed up to {backup_path}")

            stats = {
                "snapshots_migrated": migrated_count,
                "users": len(set(user_id for user_id, _ in grouped.keys())),
                "sources": len(set(source for _, source in grouped.keys()))
            }

            logger.info(
                f"Migration complete: {stats['snapshots_migrated']} snapshots migrated "
                f"({stats['users']} users, {stats['sources']} sources)"
            )

            return stats

        except (OSError, PermissionError) as e:
            logger.error(f"I/O error during migration: {e}")
            return {"snapshots_migrated": 0, "users": 0, "sources": 0}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON error during migration: {e}")
            return {"snapshots_migrated": 0, "users": 0, "sources": 0}
        except Exception as e:
            logger.error(f"Unexpected error during migration: {e}", exc_info=True)
            return {"snapshots_migrated": 0, "users": 0, "sources": 0}
