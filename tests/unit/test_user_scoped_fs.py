"""
Tests unitaires pour UserScopedFS - Protection anti-path traversal.
Vérifie que tous les chemins malveillants sont bloqués.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from api.services.user_fs import UserScopedFS


class TestUserScopedFSPathTraversal:
    """Tests de sécurité anti-path traversal"""

    @pytest.fixture
    def temp_project(self):
        """Crée un projet temporaire pour les tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def user_fs(self, temp_project):
        """UserScopedFS pour utilisateur de test"""
        return UserScopedFS(temp_project, "test_user")

    def test_user_root_created(self, user_fs, temp_project):
        """Vérifie que le répertoire utilisateur est créé"""
        expected_root = Path(temp_project) / "data" / "users" / "test_user"
        assert Path(user_fs.get_user_root()).resolve() == expected_root.resolve()
        assert expected_root.exists()

    def test_valid_relative_path(self, user_fs):
        """Chemin relatif valide doit fonctionner"""
        path = user_fs.get_path("config.json")
        assert "test_user" in path
        assert ".." not in path

    def test_block_parent_directory_traversal(self, user_fs):
        """Bloque ../../../etc/passwd"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.get_path("../../../etc/passwd")

    def test_block_absolute_path_escape(self, user_fs):
        """Bloque /etc/passwd (chemin absolu)"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.get_path("/etc/passwd")

    def test_block_multiple_parent_refs(self, user_fs):
        """Bloque ../../other_user/secrets.json"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.get_path("../../other_user/secrets.json")

    def test_block_dot_dot_in_middle(self, user_fs):
        """Bloque cointracking/../../../etc/passwd"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.get_path("cointracking/../../../etc/passwd")

    def test_block_windows_path_traversal(self, user_fs):
        """Bloque ..\\..\\..\\Windows\\System32 (Windows)"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.get_path("..\\..\\..\\Windows\\System32")

    def test_block_mixed_slashes(self, user_fs):
        """Bloque ../../../etc/passwd avec backslashes"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.get_path("..\\../..\\etc/passwd")

    def test_allow_subdirectory_navigation(self, user_fs):
        """Permet navigation dans sous-dossiers légitimes"""
        path = user_fs.get_path("cointracking/data/balances.csv")
        assert "test_user" in path
        assert "cointracking" in path

    def test_exists_blocks_traversal(self, user_fs):
        """exists() doit aussi bloquer path traversal"""
        result = user_fs.exists("../../../etc/passwd")
        assert result is False  # Retourne False au lieu de lever exception

    def test_list_files_blocks_traversal(self, user_fs):
        """list_files() doit bloquer path traversal"""
        files = user_fs.list_files("../../../etc", "*.conf")
        assert files == []  # Retourne liste vide au lieu de lever exception

    def test_glob_files_validates_results(self, user_fs, temp_project):
        """glob_files() doit valider que tous les résultats sont dans le scope"""
        # Créer fichiers de test
        user_root = Path(user_fs.get_user_root())
        test_file = user_root / "test.csv"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test data")

        # Glob devrait trouver le fichier
        files = user_fs.glob_files("*.csv")
        assert len(files) == 1
        assert "test_user" in files[0]

    def test_read_json_blocks_traversal(self, user_fs):
        """read_json() doit bloquer path traversal"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.read_json("../../../etc/passwd")

    def test_write_json_blocks_traversal(self, user_fs):
        """write_json() doit bloquer path traversal"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.write_json("../../../tmp/malicious.json", {"data": "evil"})

    def test_delete_file_blocks_traversal(self, user_fs):
        """delete_file() doit bloquer path traversal"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.delete_file("../../../tmp/important.txt")

    def test_symlink_traversal_blocked(self, user_fs, temp_project):
        """Vérifie que les symlinks ne permettent pas d'échapper au scope"""
        user_root = Path(user_fs.get_user_root())

        # Créer un fichier en dehors du scope utilisateur
        outside_file = Path(temp_project) / "outside.txt"
        outside_file.write_text("sensitive data")

        # Tenter de créer un symlink vers le fichier externe
        symlink_path = user_root / "symlink.txt"
        try:
            symlink_path.symlink_to(outside_file)
        except OSError:
            # Symlinks peuvent ne pas être supportés sur Windows sans droits admin
            pytest.skip("Symlinks not supported on this system")

        # Le chemin résolu devrait détecter la sortie du scope
        with pytest.raises(ValueError, match="Path traversal detected"):
            user_fs.get_path("symlink.txt")


class TestUserScopedFSFunctionality:
    """Tests fonctionnels de UserScopedFS"""

    @pytest.fixture
    def temp_project(self):
        """Crée un projet temporaire pour les tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def user_fs(self, temp_project):
        """UserScopedFS pour utilisateur de test"""
        return UserScopedFS(temp_project, "test_user")

    def test_read_write_json(self, user_fs):
        """Test lecture/écriture JSON"""
        test_data = {"key": "value", "number": 42}
        user_fs.write_json("test.json", test_data)

        read_data = user_fs.read_json("test.json")
        assert read_data == test_data

    def test_get_most_recent_file(self, user_fs):
        """Test récupération du fichier le plus récent"""
        import time

        # Créer plusieurs fichiers avec délai
        user_root = Path(user_fs.get_user_root())
        file1 = user_root / "data1.csv"
        file1.write_text("data1")

        time.sleep(0.01)

        file2 = user_root / "data2.csv"
        file2.write_text("data2")

        # Le plus récent devrait être data2.csv
        most_recent = user_fs.get_most_recent_file("*.csv")
        assert most_recent is not None
        assert "data2.csv" in most_recent

    def test_list_files_filters_directories(self, user_fs):
        """list_files() ne doit retourner que les fichiers, pas les dossiers"""
        user_root = Path(user_fs.get_user_root())

        # Créer fichier et dossier
        (user_root / "file.txt").write_text("content")
        (user_root / "directory").mkdir()

        files = user_fs.list_files("", "*.txt")
        assert len(files) == 1
        assert "file.txt" in files[0]

    def test_user_isolation(self, temp_project):
        """Vérifie que deux utilisateurs sont isolés"""
        user1_fs = UserScopedFS(temp_project, "user1")
        user2_fs = UserScopedFS(temp_project, "user2")

        # Écrire données pour user1
        user1_fs.write_json("private.json", {"secret": "user1_data"})

        # User2 ne doit pas pouvoir accéder aux données de user1
        with pytest.raises(ValueError, match="Path traversal detected"):
            user2_fs.read_json("../user1/private.json")

        # Vérifier que chaque utilisateur a son propre espace
        assert "user1" in user1_fs.get_user_root()
        assert "user2" in user2_fs.get_user_root()
        assert user1_fs.get_user_root() != user2_fs.get_user_root()
