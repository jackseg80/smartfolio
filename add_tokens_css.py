#!/usr/bin/env python3
"""
Script pour ajouter tokens.css dans toutes les pages HTML
Ajoute <link rel="stylesheet" href="css/tokens.css"> AVANT shared-theme.css
"""
import re
from pathlib import Path

def add_tokens_css_to_file(html_path):
    """Ajoute tokens.css à un fichier HTML si pas déjà présent"""
    content = html_path.read_text(encoding='utf-8')

    # Vérifier si tokens.css est déjà présent
    if 'tokens.css' in content:
        return False, "Already has tokens.css"

    # Chercher la ligne avec shared-theme.css
    pattern = r'(<link\s+rel="stylesheet"\s+href="[^"]*shared-theme\.css"[^>]*>)'

    if not re.search(pattern, content):
        return False, "No shared-theme.css found"

    # Ajouter tokens.css AVANT shared-theme.css
    tokens_line = '<link rel="stylesheet" href="css/tokens.css">'
    new_content = re.sub(
        pattern,
        f'{tokens_line}\n    \\1',
        content
    )

    # Sauvegarder
    html_path.write_text(new_content, encoding='utf-8')
    return True, "Added tokens.css"

def main():
    static_dir = Path("static")
    html_files = sorted(static_dir.glob("*.html"))

    # Pages principales (priorité)
    priority_pages = [
        "ai-dashboard.html",
        "analytics-unified.html",
        "dashboard.html",
        "execution.html",
        "execution_history.html",
        "rebalance.html",
        "risk-dashboard.html",
        "settings.html",
        "simulations.html",
        "wealth-dashboard.html"
    ]

    updated = []
    skipped = []
    errors = []

    # Traiter pages principales d'abord
    for filename in priority_pages:
        html_path = static_dir / filename
        if not html_path.exists():
            continue

        success, message = add_tokens_css_to_file(html_path)
        if success:
            updated.append(filename)
            print(f"[OK] {filename} - {message}")
        else:
            skipped.append((filename, message))
            print(f"[SKIP] {filename} - {message}")

    # Traiter les autres pages
    for html_path in html_files:
        if html_path.name in priority_pages:
            continue
        if html_path.name in ['saxo-dashboard.html', 'ui-components-demo.html']:
            continue  # Déjà à jour

        success, message = add_tokens_css_to_file(html_path)
        if success:
            updated.append(html_path.name)
            print(f"[OK] {html_path.name} - {message}")
        else:
            skipped.append((html_path.name, message))
            print(f"[SKIP] {html_path.name} - {message}")

    print(f"\n[SUMMARY]")
    print(f"  Updated: {len(updated)} pages")
    print(f"  Skipped: {len(skipped)} pages")
    print(f"  Errors: {len(errors)}")

    if updated:
        print(f"\n[UPDATED PAGES]")
        for f in updated:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
