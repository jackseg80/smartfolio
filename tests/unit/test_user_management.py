"""
Tests for services/user_management.py - UserManagementService
Covers: folder creation, config CRUD, user listing, deletion, role management.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.user_management import UserManagementService, get_user_management_service


def _make_users_config(users=None, default="demo", roles=None):
    """Helper to build a users.json dict."""
    if users is None:
        users = [
            {"id": "demo", "label": "Demo", "roles": ["viewer"], "status": "active"},
            {"id": "jack", "label": "Jack", "roles": ["admin"], "status": "active"},
        ]
    if roles is None:
        roles = {
            "admin": "Full access",
            "viewer": "Read-only",
            "governance_admin": "Governance management",
            "ml_admin": "ML management",
        }
    return {"default": default, "users": users, "roles": roles}


@pytest.fixture()
def svc(tmp_path):
    """Create a UserManagementService wired to tmp_path."""
    users_json = tmp_path / "config" / "users.json"
    users_json.parent.mkdir(parents=True, exist_ok=True)
    users_json.write_text(json.dumps(_make_users_config()), encoding="utf-8")
    data_users = tmp_path / "data" / "users"
    data_users.mkdir(parents=True, exist_ok=True)
    service = UserManagementService()
    service.users_config_path = users_json
    service.data_users_path = data_users
    return service


@pytest.fixture()
def svc_empty(tmp_path):
    """Service with an empty user list."""
    users_json = tmp_path / "config" / "users.json"
    users_json.parent.mkdir(parents=True, exist_ok=True)
    users_json.write_text(json.dumps(_make_users_config(users=[])), encoding="utf-8")
    data_users = tmp_path / "data" / "users"
    data_users.mkdir(parents=True, exist_ok=True)
    service = UserManagementService()
    service.users_config_path = users_json
    service.data_users_path = data_users
    return service

class TestLoadSaveConfig:
    def test_load_config_success(self, svc):
        config = svc._load_users_config()
        assert "users" in config
        assert len(config["users"]) == 2

    def test_load_config_missing_file(self, tmp_path):
        service = UserManagementService()
        service.users_config_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            service._load_users_config()

    @patch("services.user_management.clear_users_cache")
    def test_save_config_clears_cache(self, mock_clear, svc):
        config = svc._load_users_config()
        svc._save_users_config(config)
        mock_clear.assert_called_once()

    @patch("services.user_management.clear_users_cache")
    def test_save_config_writes_json(self, mock_clear, svc):
        config = svc._load_users_config()
        config["extra_key"] = "test"
        svc._save_users_config(config)
        reloaded = json.loads(svc.users_config_path.read_text(encoding="utf-8"))
        assert reloaded["extra_key"] == "test"


class TestFolderStructure:
    @patch("services.user_management.clear_users_cache")
    def test_creates_main_directory(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        assert (svc.data_users_path / "testuser").is_dir()

    @patch("services.user_management.clear_users_cache")
    def test_creates_config_json(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        cfg = svc.data_users_path / "testuser" / "config.json"
        assert cfg.exists()
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["theme"] == "dark"
        assert "sources" in data

    @patch("services.user_management.clear_users_cache")
    def test_creates_secrets_json(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        sec = svc.data_users_path / "testuser" / "secrets.json"
        assert sec.exists()
        data = json.loads(sec.read_text(encoding="utf-8"))
        assert "coingecko" in data
        assert "saxo" in data

    @patch("services.user_management.clear_users_cache")
    def test_creates_audit_log(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        assert (svc.data_users_path / "testuser" / "audit.log").exists()

    @patch("services.user_management.clear_users_cache")
    def test_creates_cointracking_dirs(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        assert (svc.data_users_path / "testuser" / "cointracking" / "data").is_dir()
        assert (svc.data_users_path / "testuser" / "cointracking" / "api_cache").is_dir()

    @patch("services.user_management.clear_users_cache")
    def test_creates_saxobank_dirs(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        assert (svc.data_users_path / "testuser" / "saxobank" / "data").is_dir()

    @patch("services.user_management.clear_users_cache")
    def test_creates_sources_json(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        src = svc.data_users_path / "testuser" / "config" / "sources.json"
        assert src.exists()
        data = json.loads(src.read_text(encoding="utf-8"))
        assert "modules" in data

    @patch("services.user_management.clear_users_cache")
    def test_idempotent_no_overwrite(self, _mock, svc):
        svc._create_user_folder_structure("testuser")
        cfg = svc.data_users_path / "testuser" / "config.json"
        cfg.write_text(json.dumps({"custom": True}), encoding="utf-8")
        svc._create_user_folder_structure("testuser")
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data == {"custom": True}

class TestCreateUser:
    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_create_user_success(self, _val, _cache, svc):
        result = svc.create_user("NewUser", label="New User")
        assert result["id"] == "newuser"
        assert result["label"] == "New User"
        assert result["roles"] == ["viewer"]
        assert result["status"] == "active"
        assert "created_at" in result

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_create_user_custom_roles(self, _val, _cache, svc):
        result = svc.create_user("admin2", label="Admin Two", roles=["admin", "ml_admin"])
        assert result["roles"] == ["admin", "ml_admin"]

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_create_user_duplicate_raises(self, _val, _cache, svc):
        with pytest.raises(ValueError, match="User already exists"):
            svc.create_user("demo", label="Duplicate")

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_create_user_creates_folder(self, _val, _cache, svc):
        svc.create_user("foldertest", label="Folder Test")
        assert (svc.data_users_path / "foldertest").is_dir()

    @patch("services.user_management.validate_user_id", side_effect=ValueError("Invalid user"))
    def test_create_user_invalid_id_raises(self, _val, svc):
        with pytest.raises(ValueError, match="Invalid user"):
            svc.create_user("bad", label="Bad")

class TestUpdateUser:
    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_update_label(self, _val, _cache, svc):
        result = svc.update_user("jack", {"label": "Jacques"})
        assert result["label"] == "Jacques"

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_update_roles(self, _val, _cache, svc):
        result = svc.update_user("jack", {"roles": ["viewer"]})
        assert result["roles"] == ["viewer"]

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_update_ignores_disallowed_fields(self, _val, _cache, svc):
        result = svc.update_user("jack", {"id": "hacked", "label": "Valid"})
        assert result["id"] == "jack"
        assert result["label"] == "Valid"

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_update_nonexistent_raises(self, _val, _cache, svc):
        with pytest.raises(ValueError, match="User not found"):
            svc.update_user("ghost", {"label": "Phantom"})

class TestDeleteUser:
    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_soft_delete_marks_inactive(self, _val, _cache, svc):
        result = svc.delete_user("jack")
        assert result["delete_type"] == "soft"
        assert result["deleted"] is True
        config = svc._load_users_config()
        jack = next(u for u in config["users"] if u["id"] == "jack")
        assert jack["status"] == "inactive"

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_soft_delete_renames_folder(self, _val, _cache, svc):
        user_dir = svc.data_users_path / "jack"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "marker.txt").write_text("exists")
        result = svc.delete_user("jack")
        assert result["delete_type"] == "soft"
        assert not user_dir.exists()
        renamed = list(svc.data_users_path.glob("jack_deleted_*"))
        assert len(renamed) == 1

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_hard_delete_removes_from_config(self, _val, _cache, svc):
        result = svc.delete_user("jack", hard_delete=True)
        assert result["delete_type"] == "hard"
        config = svc._load_users_config()
        ids = [u["id"] for u in config["users"]]
        assert "jack" not in ids

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_hard_delete_removes_folder(self, _val, _cache, svc):
        user_dir = svc.data_users_path / "jack"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "file.txt").write_text("data")
        svc.delete_user("jack", hard_delete=True)
        assert not user_dir.exists()

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_delete_default_user_raises(self, _val, _cache, svc):
        with pytest.raises(ValueError, match="Cannot delete default user"):
            svc.delete_user("demo")

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_delete_nonexistent_raises(self, _val, _cache, svc):
        with pytest.raises(ValueError, match="User not found"):
            svc.delete_user("ghost")

    @patch("services.user_management.shutil.rmtree", side_effect=OSError("perm denied"))
    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_hard_delete_folder_error_logged(self, _val, _cache, _rmtree, svc):
        user_dir = svc.data_users_path / "jack"
        user_dir.mkdir(parents=True, exist_ok=True)
        result = svc.delete_user("jack", hard_delete=True)
        assert result["deleted"] is True

    @patch("services.user_management.shutil.move", side_effect=OSError("perm denied"))
    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_soft_delete_rename_error_logged(self, _val, _cache, _move, svc):
        user_dir = svc.data_users_path / "jack"
        user_dir.mkdir(parents=True, exist_ok=True)
        result = svc.delete_user("jack")
        assert result["deleted"] is True

class TestAssignRoles:
    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_assign_valid_roles(self, _val, _cache, svc):
        result = svc.assign_roles("jack", ["viewer", "ml_admin"])
        assert result["roles"] == ["viewer", "ml_admin"]

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_assign_invalid_role_raises(self, _val, _cache, svc):
        with pytest.raises(ValueError, match="Invalid roles"):
            svc.assign_roles("jack", ["superadmin"])

    @patch("services.user_management.clear_users_cache")
    @patch("services.user_management.validate_user_id", side_effect=lambda x: x.strip().lower())
    def test_assign_mixed_valid_invalid_raises(self, _val, _cache, svc):
        with pytest.raises(ValueError, match="Invalid roles"):
            svc.assign_roles("jack", ["admin", "nonexistent_role"])


class TestGetAllRoles:
    def test_get_all_roles_returns_dict(self, svc):
        roles = svc.get_all_roles()
        assert isinstance(roles, dict)
        assert "admin" in roles
        assert "viewer" in roles

    def test_get_all_roles_values_are_strings(self, svc):
        roles = svc.get_all_roles()
        for desc in roles.values():
            assert isinstance(desc, str)


class TestSingleton:
    @patch("services.user_management._user_management_service", None)
    def test_get_user_management_service_creates_singleton(self):
        svc1 = get_user_management_service()
        svc2 = get_user_management_service()
        assert svc1 is svc2
        assert isinstance(svc1, UserManagementService)
