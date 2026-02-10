"""
Tests for BackupManager — backup creation, retention, restore, verification.
"""
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from services.backup_manager import (
    BackupManager,
    RETENTION_DAILY,
    RETENTION_WEEKLY,
    RETENTION_MONTHLY,
    BACKUP_SUBDIRS,
    BACKUP_FILES,
    EXCLUDED_FILES,
)


@pytest.fixture
def tmp_env(tmp_path):
    """Create a realistic data/users + data/backups structure."""
    data_dir = tmp_path / "users"
    backup_dir = tmp_path / "backups"

    # Create two users with realistic data
    for uid in ["jack", "demo"]:
        user_dir = data_dir / uid
        user_dir.mkdir(parents=True)

        # config.json
        (user_dir / "config.json").write_text(
            json.dumps({"user_id": uid, "theme": "dark"})
        )

        # secrets.json (should be excluded by default)
        (user_dir / "secrets.json").write_text(
            json.dumps({"api_key": "super_secret_123"})
        )

        # config/sources.json
        config_dir = user_dir / "config"
        config_dir.mkdir()
        (config_dir / "sources.json").write_text(
            json.dumps({"primary": "cointracking"})
        )

        # cointracking data
        ct_data = user_dir / "cointracking" / "data"
        ct_data.mkdir(parents=True)
        (ct_data / "20260210_120000_trades.csv").write_text("date,amount\n2026-01-01,100")
        (ct_data / "20260209_120000_trades.csv").write_text("date,amount\n2026-01-02,200")

        # cointracking snapshots
        ct_snap = user_dir / "cointracking" / "snapshots"
        ct_snap.mkdir(parents=True)
        (ct_snap / "snapshot_20260210.json").write_text(json.dumps({"total": 50000}))

        # wealth
        wealth_dir = user_dir / "wealth"
        wealth_dir.mkdir()
        (wealth_dir / "patrimoine.json").write_text(json.dumps({"net_worth": 100000}))

        # banks
        banks_dir = user_dir / "banks"
        banks_dir.mkdir()
        (banks_dir / "snapshot.json").write_text(json.dumps({"accounts": []}))

    return data_dir, backup_dir


@pytest.fixture
def mgr(tmp_env):
    """Create a BackupManager with temp directories."""
    data_dir, backup_dir = tmp_env
    return BackupManager(data_dir=data_dir, backup_dir=backup_dir)


# ─── User Discovery ─────────────────────────────────────────────────


class TestGetUserIds:
    def test_lists_existing_users(self, mgr):
        users = mgr.get_user_ids()
        assert "jack" in users
        assert "demo" in users

    def test_sorted_alphabetically(self, mgr):
        users = mgr.get_user_ids()
        assert users == sorted(users)

    def test_ignores_hidden_dirs(self, mgr):
        (mgr.data_dir / ".hidden").mkdir()
        users = mgr.get_user_ids()
        assert ".hidden" not in users

    def test_empty_if_no_dir(self, tmp_path):
        mgr = BackupManager(data_dir=tmp_path / "nonexistent")
        assert mgr.get_user_ids() == []


# ─── Backup Creation ────────────────────────────────────────────────


class TestCreateBackup:
    def test_creates_zip_for_all_users(self, mgr):
        result = mgr.create_backup()
        assert result["total_ok"] == 2
        assert result["total_failed"] == 0
        assert "jack" in result["users"]
        assert "demo" in result["users"]

    def test_creates_zip_for_specific_user(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        assert result["total_ok"] == 1
        assert "jack" in result["users"]
        assert "demo" not in result["users"]

    def test_zip_file_exists(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        zip_path = result["users"]["jack"]["file"]
        assert Path(zip_path).exists()

    def test_zip_contains_config(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        zip_path = result["users"]["jack"]["file"]
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "config.json" in names

    def test_zip_contains_subdirs(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        zip_path = result["users"]["jack"]["file"]
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # Should contain cointracking CSV files
            csv_files = [n for n in names if n.endswith(".csv")]
            assert len(csv_files) >= 2

    def test_excludes_secrets_by_default(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        zip_path = result["users"]["jack"]["file"]
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "secrets.json" not in names

    def test_includes_secrets_when_requested(self, mgr):
        result = mgr.create_backup(user_ids=["jack"], include_secrets=True)
        zip_path = result["users"]["jack"]["file"]
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "secrets.json" in names

    def test_returns_checksum(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        checksum = result["users"]["jack"]["checksum_sha256"]
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex

    def test_returns_file_count(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        count = result["users"]["jack"]["file_count"]
        assert count >= 5  # config + 2 csv + snapshot + patrimoine + banks

    def test_returns_sizes(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        user_result = result["users"]["jack"]
        assert user_result["zip_size"] > 0
        assert user_result["original_size"] > 0

    def test_nonexistent_user_returns_error(self, mgr):
        result = mgr.create_backup(user_ids=["nonexistent"])
        assert result["total_failed"] == 1
        assert not result["users"]["nonexistent"]["ok"]

    def test_backup_dir_created_automatically(self, mgr):
        assert not mgr.backup_dir.exists()
        mgr.create_backup(user_ids=["jack"])
        assert (mgr.backup_dir / "jack").exists()

    def test_timestamp_in_filename(self, mgr):
        result = mgr.create_backup(user_ids=["jack"])
        filename = Path(result["users"]["jack"]["file"]).name
        assert filename.startswith("backup_")
        assert filename.endswith(".zip")
        # Format: backup_YYYYMMDD_HHMMSS.zip
        parts = filename.replace("backup_", "").replace(".zip", "")
        assert len(parts) == 15  # YYYYMMDD_HHMMSS

    def test_multiple_backups_different_files(self, mgr):
        import time
        r1 = mgr.create_backup(user_ids=["jack"])
        time.sleep(1.1)  # ensure different timestamp
        r2 = mgr.create_backup(user_ids=["jack"])
        assert r1["users"]["jack"]["file"] != r2["users"]["jack"]["file"]


# ─── Backup Listing ─────────────────────────────────────────────────


class TestListBackups:
    def test_empty_initially(self, mgr):
        assert mgr.list_backups() == []

    def test_lists_created_backups(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups()
        assert len(backups) == 1
        assert backups[0]["user_id"] == "jack"

    def test_filter_by_user(self, mgr):
        mgr.create_backup()  # all users
        jack_backups = mgr.list_backups(user_id="jack")
        assert all(b["user_id"] == "jack" for b in jack_backups)

    def test_includes_metadata(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups()
        b = backups[0]
        assert "file" in b
        assert "size" in b
        assert "date" in b
        assert b["size"] > 0

    def test_sorted_by_date_descending(self, mgr):
        import time
        mgr.create_backup(user_ids=["jack"])
        time.sleep(1.1)
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups()
        dates = [b["date"] for b in backups]
        assert dates == sorted(dates, reverse=True)


# ─── Status ──────────────────────────────────────────────────────────


class TestGetStatus:
    def test_empty_status(self, mgr):
        s = mgr.get_status()
        assert s["total_backups"] == 0
        assert s["total_size_bytes"] == 0

    def test_status_after_backup(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        s = mgr.get_status()
        assert s["total_backups"] == 1
        assert s["total_size_bytes"] > 0
        assert "jack" in s["users"]
        assert s["users"]["jack"]["count"] == 1

    def test_includes_retention_policy(self, mgr):
        s = mgr.get_status()
        assert s["retention_policy"]["daily"] == RETENTION_DAILY
        assert s["retention_policy"]["weekly"] == RETENTION_WEEKLY
        assert s["retention_policy"]["monthly"] == RETENTION_MONTHLY

    def test_total_size_mb(self, mgr):
        mgr.create_backup()
        s = mgr.get_status()
        assert isinstance(s["total_size_mb"], float)


# ─── Retention ───────────────────────────────────────────────────────


class TestRetention:
    def _create_fake_backup(self, backup_dir: Path, user_id: str, dt: datetime):
        """Create a fake backup file with a specific date in its name."""
        user_backup_dir = backup_dir / user_id
        user_backup_dir.mkdir(parents=True, exist_ok=True)
        filename = f"backup_{dt.strftime('%Y%m%d_%H%M%S')}.zip"
        fpath = user_backup_dir / filename
        # Create a minimal valid ZIP
        with zipfile.ZipFile(fpath, "w") as zf:
            zf.writestr("dummy.txt", f"backup from {dt}")
        return fpath

    def test_no_deletion_under_daily_limit(self, mgr):
        now = datetime.now()
        for i in range(5):  # 5 < 7 daily
            self._create_fake_backup(mgr.backup_dir, "jack", now - timedelta(days=i))
        deleted = mgr.apply_retention("jack")
        assert deleted.get("jack", 0) == 0

    def test_deletes_beyond_daily_limit(self, mgr):
        now = datetime.now()
        for i in range(15):  # 15 > 7 daily
            self._create_fake_backup(
                mgr.backup_dir, "jack", now - timedelta(days=i, hours=12)
            )
        deleted = mgr.apply_retention("jack")
        # Should keep 7 daily + some monthly, delete the rest
        assert deleted["jack"] > 0
        remaining = mgr._list_backup_files(mgr.backup_dir / "jack")
        assert len(remaining) <= RETENTION_DAILY + RETENTION_MONTHLY

    def test_keeps_monthly_backups(self, mgr):
        # Create one backup per month for 14 months
        now = datetime.now()
        for i in range(14):
            dt = now - timedelta(days=30 * i)
            self._create_fake_backup(mgr.backup_dir, "jack", dt)
        deleted = mgr.apply_retention("jack")
        remaining = mgr._list_backup_files(mgr.backup_dir / "jack")
        # Should keep at most daily + monthly
        assert len(remaining) <= RETENTION_DAILY + RETENTION_MONTHLY

    def test_empty_dir_no_error(self, mgr):
        deleted = mgr.apply_retention("jack")
        assert deleted == {}

    def test_all_users_retention(self, mgr):
        now = datetime.now()
        for uid in ["jack", "demo"]:
            for i in range(10):
                self._create_fake_backup(
                    mgr.backup_dir, uid, now - timedelta(days=i, hours=6)
                )
        deleted = mgr.apply_retention()
        assert "jack" in deleted
        assert "demo" in deleted


# ─── Verify ──────────────────────────────────────────────────────────


class TestVerifyBackup:
    def test_verify_valid_backup(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups(user_id="jack")
        result = mgr.verify_backup(backups[0]["path"])
        assert result["ok"] is True
        assert result["file_count"] > 0
        assert len(result["checksum_sha256"]) == 64

    def test_verify_nonexistent_file(self, mgr):
        result = mgr.verify_backup("/nonexistent/file.zip")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    def test_verify_corrupted_file(self, tmp_path, mgr):
        fake = tmp_path / "corrupt.zip"
        fake.write_text("not a zip file")
        result = mgr.verify_backup(str(fake))
        assert result["ok"] is False

    def test_checksum_consistent(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups(user_id="jack")
        r1 = mgr.verify_backup(backups[0]["path"])
        r2 = mgr.verify_backup(backups[0]["path"])
        assert r1["checksum_sha256"] == r2["checksum_sha256"]


# ─── Restore ─────────────────────────────────────────────────────────


class TestRestore:
    def test_dry_run_lists_files(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups(user_id="jack")
        result = mgr.restore_backup(
            zip_path=backups[0]["path"],
            user_id="jack",
            dry_run=True,
        )
        assert result["ok"] is True
        assert result["dry_run"] is True
        assert result["file_count"] > 0
        assert isinstance(result["files_to_restore"], list)

    def test_real_restore_writes_files(self, mgr, tmp_path):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups(user_id="jack")

        # Restore to a new user
        result = mgr.restore_backup(
            zip_path=backups[0]["path"],
            user_id="restored_jack",
            dry_run=False,
        )
        assert result["ok"] is True
        assert result["dry_run"] is False
        assert result["file_count"] > 0

        # Verify files exist
        restored_dir = mgr.data_dir / "restored_jack"
        assert restored_dir.exists()
        assert (restored_dir / "config.json").exists()

    def test_restored_data_matches_original(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups(user_id="jack")

        # Restore to new user
        mgr.restore_backup(
            zip_path=backups[0]["path"],
            user_id="clone_jack",
            dry_run=False,
        )

        # Compare config.json
        original = json.loads((mgr.data_dir / "jack" / "config.json").read_text())
        restored = json.loads((mgr.data_dir / "clone_jack" / "config.json").read_text())
        assert original == restored

    def test_restore_nonexistent_zip(self, mgr):
        result = mgr.restore_backup(
            zip_path="/nonexistent.zip",
            user_id="jack",
            dry_run=True,
        )
        assert result["ok"] is False

    def test_restore_corrupted_zip(self, tmp_path, mgr):
        fake = tmp_path / "bad.zip"
        fake.write_text("corrupted")
        result = mgr.restore_backup(
            zip_path=str(fake),
            user_id="jack",
            dry_run=True,
        )
        assert result["ok"] is False

    def test_restore_creates_target_dir(self, mgr):
        mgr.create_backup(user_ids=["jack"])
        backups = mgr.list_backups(user_id="jack")

        new_user = "brand_new_user"
        assert not (mgr.data_dir / new_user).exists()

        mgr.restore_backup(
            zip_path=backups[0]["path"],
            user_id=new_user,
            dry_run=False,
        )
        assert (mgr.data_dir / new_user).exists()


# ─── Helpers ─────────────────────────────────────────────────────────


class TestHelpers:
    def test_compute_checksum(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        checksum = BackupManager._compute_checksum(f)
        assert isinstance(checksum, str)
        assert len(checksum) == 64
        # Deterministic
        assert BackupManager._compute_checksum(f) == checksum

    def test_parse_backup_date_valid(self):
        dt = BackupManager._parse_backup_date("backup_20260210_153045.zip")
        assert dt == datetime(2026, 2, 10, 15, 30, 45)

    def test_parse_backup_date_invalid(self):
        assert BackupManager._parse_backup_date("not_a_backup.zip") is None
        assert BackupManager._parse_backup_date("backup_invalid.zip") is None

    def test_list_backup_files_filters(self, tmp_path):
        (tmp_path / "backup_20260210_120000.zip").write_text("a")
        (tmp_path / "backup_20260211_120000.zip").write_text("b")
        (tmp_path / "not_a_backup.txt").write_text("c")
        (tmp_path / "readme.md").write_text("d")
        files = BackupManager._list_backup_files(tmp_path)
        assert len(files) == 2
        assert all(f.name.startswith("backup_") for f in files)

    def test_list_backup_files_empty_dir(self, tmp_path):
        assert BackupManager._list_backup_files(tmp_path) == []

    def test_list_backup_files_nonexistent(self, tmp_path):
        assert BackupManager._list_backup_files(tmp_path / "nope") == []


# ─── Constants ───────────────────────────────────────────────────────


class TestConstants:
    def test_retention_values(self):
        assert RETENTION_DAILY == 7
        assert RETENTION_WEEKLY == 4
        assert RETENTION_MONTHLY == 12

    def test_excluded_files(self):
        assert "secrets.json" in EXCLUDED_FILES

    def test_backup_files_includes_config(self):
        assert "config.json" in BACKUP_FILES

    def test_backup_subdirs_not_empty(self):
        assert len(BACKUP_SUBDIRS) > 0
        assert "config" in BACKUP_SUBDIRS
        assert "cointracking/data" in BACKUP_SUBDIRS
