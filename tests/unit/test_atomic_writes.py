"""
Tests for atomic write patterns in user_management and alert_storage.
Ensures temp+os.replace pattern prevents file corruption.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from services.user_management import UserManagementService
from services.alerts.alert_storage import AlertStorage


# ─── UserManagement atomic writes ────────────────────────────────────────


class TestUserManagementAtomicWrite:
    """Tests for _save_users_config atomic write pattern."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.tmp_dir) / "users.json"
        self.config_path.write_text(json.dumps({
            "users": {"test": {"role": "viewer"}}
        }))
        self.mgr = UserManagementService()
        self.mgr.users_config_path = self.config_path

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_saves_valid_json(self):
        config = {"users": {"test": {"role": "admin"}}}
        self.mgr._save_users_config(config)
        result = json.loads(self.config_path.read_text())
        assert result["users"]["test"]["role"] == "admin"

    def test_preserves_unicode(self):
        config = {"users": {"test": {"name": "Jean-François"}}}
        self.mgr._save_users_config(config)
        result = json.loads(self.config_path.read_text(encoding="utf-8"))
        assert result["users"]["test"]["name"] == "Jean-François"

    def test_no_temp_file_left_on_success(self):
        config = {"users": {"test": {"role": "admin"}}}
        self.mgr._save_users_config(config)
        tmp_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_original_untouched_on_write_error(self):
        original_content = self.config_path.read_text()
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                self.mgr._save_users_config({"users": {}})
        assert self.config_path.read_text() == original_content

    def test_temp_file_cleaned_on_error(self):
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                self.mgr._save_users_config({"users": {}})
        tmp_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_file_created_if_not_exists(self):
        new_path = Path(self.tmp_dir) / "new_users.json"
        self.mgr.users_config_path = new_path
        self.mgr._save_users_config({"users": {"new": {"role": "viewer"}}})
        assert new_path.exists()
        result = json.loads(new_path.read_text())
        assert "new" in result["users"]

    def test_concurrent_writes_dont_corrupt(self):
        """Simulate rapid sequential writes — file should always be valid JSON."""
        for i in range(20):
            self.mgr._save_users_config({"users": {"test": {"iteration": i}}})
        result = json.loads(self.config_path.read_text())
        assert result["users"]["test"]["iteration"] == 19

    def test_large_config_write(self):
        config = {"users": {f"user_{i}": {"role": "viewer", "data": "x" * 1000} for i in range(100)}}
        self.mgr._save_users_config(config)
        result = json.loads(self.config_path.read_text())
        assert len(result["users"]) == 100

    def test_uses_os_replace(self):
        """Verify the atomic os.replace is actually called."""
        with patch("os.replace", wraps=os.replace) as mock_replace:
            self.mgr._save_users_config({"users": {}})
            mock_replace.assert_called_once()

    def test_permission_error_cleanup(self):
        with patch("os.replace", side_effect=PermissionError("access denied")):
            with pytest.raises(PermissionError):
                self.mgr._save_users_config({"users": {}})
        tmp_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0


# ─── AlertStorage atomic writes ─────────────────────────────────────────


class TestAlertStorageAtomicWrite:
    """Tests for _save_json_data atomic write pattern."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.json_path = Path(self.tmp_dir) / "alerts.json"
        initial = {"alerts": [], "metadata": {"created": "2026-01-01"}}
        self.json_path.write_text(json.dumps(initial))
        self.storage = AlertStorage.__new__(AlertStorage)
        self.storage.json_file = self.json_path
        self.storage.lock_file = self.json_path.with_suffix(".lock")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_saves_valid_json(self):
        data = {"alerts": [{"id": "a1", "type": "test"}], "metadata": {}}
        self.storage._save_json_data(data)
        result = json.loads(self.json_path.read_text())
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["id"] == "a1"

    def test_no_temp_file_left_on_success(self):
        self.storage._save_json_data({"alerts": [], "metadata": {}})
        tmp_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_original_untouched_on_write_error(self):
        original = self.json_path.read_text()
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                self.storage._save_json_data({"alerts": [{"id": "new"}], "metadata": {}})
        assert self.json_path.read_text() == original

    def test_temp_cleaned_on_error(self):
        with patch("os.replace", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                self.storage._save_json_data({"alerts": [], "metadata": {}})
        tmp_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_handles_enum_serialization(self):
        """_save_json_data uses _serialize_for_json for enums/datetime."""
        from enum import Enum

        class Severity(Enum):
            HIGH = "high"

        data = {"alerts": [{"severity": Severity.HIGH}], "metadata": {}}
        self.storage._save_json_data(data)
        result = json.loads(self.json_path.read_text())
        assert result["alerts"][0]["severity"] == "high"

    def test_concurrent_writes_valid_json(self):
        for i in range(20):
            data = {"alerts": [{"id": f"alert_{j}"} for j in range(i + 1)], "metadata": {}}
            self.storage._save_json_data(data)
        result = json.loads(self.json_path.read_text())
        assert len(result["alerts"]) == 20

    def test_large_data_write(self):
        alerts = [{"id": f"a_{i}", "message": "x" * 500} for i in range(200)]
        self.storage._save_json_data({"alerts": alerts, "metadata": {}})
        result = json.loads(self.json_path.read_text())
        assert len(result["alerts"]) == 200

    def test_uses_os_replace(self):
        with patch("os.replace", wraps=os.replace) as mock_replace:
            self.storage._save_json_data({"alerts": [], "metadata": {}})
            mock_replace.assert_called_once()

    def test_permission_error_cleanup(self):
        with patch("os.replace", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                self.storage._save_json_data({"alerts": [], "metadata": {}})
        tmp_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_empty_data_write(self):
        self.storage._save_json_data({"alerts": [], "metadata": {}})
        result = json.loads(self.json_path.read_text())
        assert result["alerts"] == []
