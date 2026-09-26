#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de migration pour ajouter toast.js dans toutes les pages HTML
Ajoute automatiquement le script après debug-logger.js
"""

import sys
import os
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

STATIC_DIR = Path("static")
INSERTION_LINE = '<script src="debug-logger.js"></script>'
TOAST_SCRIPT = '    <script src="components/toast.js" type="module"></script>'

def migrate_html_files():
    """Ajoute toast.js dans toutes les pages HTML"""

    if not STATIC_DIR.exists():
        print(f"[Error] Dossier {STATIC_DIR} introuvable")
        return

    html_files = list(STATIC_DIR.glob("*.html"))
    print(f" Trouvé {len(html_files)} fichiers HTML\n")

    updated = 0
    skipped = 0
    no_debug_logger = 0

    for file in html_files:
        print(f" {file.name}...", end=" ")

        try:
            content = file.read_text(encoding='utf-8')

            # Skip si toast.js déjà présent
            if 'toast.js' in content:
                print(f"  Already has toast.js")
                skipped += 1
                continue

            # Vérifier si debug-logger.js est présent
            if INSERTION_LINE not in content:
                print(f"  No debug-logger.js found")
                no_debug_logger += 1
                continue

            # Insérer toast.js après debug-logger.js
            new_content = content.replace(
                INSERTION_LINE,
                f"{INSERTION_LINE}\n{TOAST_SCRIPT}"
            )

            # Sauvegarder
            file.write_text(new_content, encoding='utf-8')
            updated += 1
            print(f"[OK] Toast script added")

        except Exception as e:
            print(f" Error: {e}")

    # Résumé
    print(f"\n{'='*60}")
    print(f" RÉSUMÉ")
    print(f"{'='*60}")
    print(f"[OK] Fichiers mis à jour: {updated}")
    print(f"  Fichiers ignorés (déjà à jour): {skipped}")
    print(f"[Warning]  Fichiers sans debug-logger: {no_debug_logger}")
    print(f" Total fichiers traités: {len(html_files)}")

    if updated > 0:
        print(f"\n Migration réussie! {updated} fichiers mis à jour.")
        print(f" Veuillez redémarrer le serveur et tester les pages.")
    elif skipped == len(html_files):
        print(f"\n[OK] Tous les fichiers sont déjà à jour!")
    else:
        print(f"\n[Warning]  Aucun fichier n'a été mis à jour. Vérifiez la structure des fichiers.")

if __name__ == '__main__':
    print("="*60)
    print(" MIGRATION TOAST.JS")
    print("="*60)
    print(f"Dossier cible: {STATIC_DIR.resolve()}")
    print(f"Action: Ajouter toast.js après debug-logger.js\n")

    response = input("Continuer? (y/n): ")
    if response.lower() != 'y':
        print("[Error] Migration annulée")
        exit(0)

    print()
    migrate_html_files()
