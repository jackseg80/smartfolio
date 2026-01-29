#!/usr/bin/env python3
"""
Script de linting pour SmartFolio
Usage: python scripts/lint.py [--fix] [--check] [path]

Options:
  --check    : Vérifier sans modifier (défaut)
  --fix      : Appliquer les corrections automatiques
  --stats    : Afficher les statistiques sans corrections
  path       : Chemin spécifique (défaut: api/ services/)
"""

import subprocess
import sys
from pathlib import Path

# Couleurs pour output terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def run_command(cmd: list[str], description: str) -> tuple[int, str]:
    """Exécute une commande et retourne le code de sortie + output."""
    print(f"\n{BLUE}▶ {description}{RESET}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    return result.returncode, output


def main():
    args = sys.argv[1:]
    fix_mode = "--fix" in args
    check_mode = "--check" in args or not fix_mode
    stats_mode = "--stats" in args

    # Chemins à analyser (par défaut: api/ et services/)
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        paths = ["api/", "services/"]

    print(f"{BLUE}{'='*60}")
    print(f"SmartFolio - Linting Python")
    print(f"{'='*60}{RESET}")
    print(f"Mode: {'🔧 FIX' if fix_mode else '✓ CHECK'}")
    print(f"Paths: {', '.join(paths)}")

    results = {}

    # 1. Black (code formatting)
    if fix_mode:
        code, output = run_command(
            [".venv/Scripts/black"] + paths, "Black: Reformatage du code"
        )
    else:
        code, output = run_command(
            [".venv/Scripts/black", "--check"] + paths,
            "Black: Vérification du formatage",
        )
    results["black"] = code == 0
    if code != 0:
        print(output)

    # 2. Isort (import sorting)
    if fix_mode:
        code, output = run_command(
            [".venv/Scripts/isort"] + paths, "Isort: Tri des imports"
        )
    else:
        code, output = run_command(
            [".venv/Scripts/isort", "--check-only"] + paths,
            "Isort: Vérification des imports",
        )
    results["isort"] = code == 0
    if code != 0:
        print(output)

    # 3. Flake8 (linting)
    code, output = run_command(
        [".venv/Scripts/flake8"] + paths, "Flake8: Analyse de code"
    )
    results["flake8"] = code == 0
    if code != 0:
        print(output)

    # Résumé
    print(f"\n{BLUE}{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}{RESET}")

    for tool, passed in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"{tool:12s} : {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n{GREEN}✓ Tous les checks sont passés !{RESET}")
        return 0
    else:
        print(
            f"\n{YELLOW}⚠ Certains checks ont échoué. "
            f"Utilisez --fix pour corriger automatiquement.{RESET}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
