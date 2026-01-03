"""
Script pour ajouter les auth guards à toutes les pages HTML.

Usage:
    python scripts/add_auth_guards.py                # Dry-run (preview)
    python scripts/add_auth_guards.py --apply        # Appliquer les changements
    python scripts/add_auth_guards.py --file dashboard.html  # Fichier spécifique
"""
import sys
import re
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

STATIC_DIR = Path(__file__).parent.parent / "static"

# Pages à protéger (toutes sauf login.html)
PAGES_TO_PROTECT = [
    "dashboard.html",
    "analytics-unified.html",
    "risk-dashboard.html",
    "cycle-analysis.html",
    "rebalance.html",
    "execution.html",
    "simulations.html",
    "wealth-dashboard.html",
    "monitoring.html",
    "admin-dashboard.html",
    "saxo-dashboard.html",
    "settings.html",
    "alias-manager.html",
    "ai-dashboard.html",
]

# Auth guard snippet à ajouter
AUTH_GUARD_SNIPPET = """
    // ===== AUTH GUARD (Dec 2025) =====
    import { checkAuth } from './core/auth-guard.js';
    await checkAuth();  // Vérifie authentification + redirect si nécessaire
    // =================================
"""

def find_script_module_tag(content):
    """
    Trouve la première balise <script type="module"> dans le HTML.

    Returns:
        tuple: (start_index, end_index) ou (None, None) si non trouvé
    """
    # Chercher <script type="module">
    pattern = r'<script\s+type=["\']module["\']>'
    match = re.search(pattern, content, re.IGNORECASE)

    if not match:
        return None, None

    script_start = match.end()  # Position après >

    # Chercher le </script> correspondant
    script_end = content.find('</script>', script_start)

    if script_end == -1:
        return None, None

    return script_start, script_end


def has_auth_guard(content):
    """
    Vérifie si le fichier a déjà l'auth guard.
    """
    return 'AUTH GUARD' in content or 'checkAuth()' in content


def add_auth_guard_to_file(file_path, apply=False):
    """
    Ajoute l'auth guard à un fichier HTML.

    Args:
        file_path: Path du fichier HTML
        apply: Si True, applique les changements, sinon dry-run

    Returns:
        bool: True si modifié, False sinon
    """
    if not file_path.exists():
        print(f"⏭️  Skip: {file_path.name} (not found)")
        return False

    # Lire le contenu
    content = file_path.read_text(encoding='utf-8')

    # Vérifier si déjà protégé
    if has_auth_guard(content):
        print(f"⏭️  Skip: {file_path.name} (already protected)")
        return False

    # Trouver la balise <script type="module">
    script_start, script_end = find_script_module_tag(content)

    if script_start is None:
        print(f"⚠️  Warning: {file_path.name} has no <script type=\"module\">")
        return False

    # Insérer l'auth guard au début du script
    new_content = (
        content[:script_start] +
        AUTH_GUARD_SNIPPET +
        content[script_start:]
    )

    if apply:
        file_path.write_text(new_content, encoding='utf-8')
        print(f"✅ Protected: {file_path.name}")
    else:
        print(f"🔍 Would protect: {file_path.name}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Add auth guards to HTML pages"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default: dry-run preview)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific file to process (e.g., dashboard.html)"
    )

    args = parser.parse_args()

    if not args.apply:
        print("=" * 60)
        print("DRY-RUN MODE (preview only)")
        print("Use --apply to actually modify files")
        print("=" * 60)
        print()

    # Déterminer les fichiers à traiter
    if args.file:
        files_to_process = [args.file]
    else:
        files_to_process = PAGES_TO_PROTECT

    # Traiter chaque fichier
    modified_count = 0
    for filename in files_to_process:
        file_path = STATIC_DIR / filename

        if add_auth_guard_to_file(file_path, apply=args.apply):
            modified_count += 1

    print()
    print("=" * 60)
    if args.apply:
        print(f"✅ {modified_count} file(s) protected with auth guards")
    else:
        print(f"🔍 {modified_count} file(s) would be protected")
        print("Run with --apply to apply changes")
    print("=" * 60)


if __name__ == "__main__":
    main()
