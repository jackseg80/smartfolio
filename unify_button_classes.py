#!/usr/bin/env python3
"""
Script pour unifier les classes de boutons dans rebalance.html
Remplace les classes non-standard par les classes standardisées de shared-theme.css
"""
import re
from pathlib import Path

def unify_button_classes():
    html_path = Path("static/rebalance.html")
    content = html_path.read_text(encoding='utf-8')

    # Compteurs
    replacements = 0

    # Pattern pour trouver toutes les classes avec "btn"
    # On doit gérer plusieurs cas :
    # 1. class="btn small secondary" -> class="btn btn-sm btn-secondary"
    # 2. class="btn secondary small" -> class="btn btn-sm btn-secondary"
    # 3. class="btn small" -> class="btn btn-sm"
    # 4. class="btn secondary" -> class="btn btn-secondary"
    # 5. class="btn ghost" -> class="btn btn-ghost"

    # Fonction pour normaliser une classe btn
    def normalize_btn_class(match):
        nonlocal replacements
        class_attr = match.group(1)

        # Si c'est un tab-btn, ne pas modifier
        if 'tab-btn' in class_attr:
            return match.group(0)

        # Extraire les mots
        classes = class_attr.split()

        # Construire la nouvelle classe
        new_classes = ['btn']

        # Ajouter les modifiers standardisés
        if 'small' in classes:
            new_classes.append('btn-sm')
        if 'secondary' in classes:
            new_classes.append('btn-secondary')
        if 'ghost' in classes:
            new_classes.append('btn-ghost')
        if 'primary' in classes:
            new_classes.append('btn-primary')

        # Si la nouvelle classe est différente, c'est un remplacement
        new_class = ' '.join(new_classes)
        if new_class != class_attr:
            replacements += 1

        return f'class="{new_class}"'

    # Pattern pour capturer class="..." contenant "btn"
    pattern = r'class="([^"]*\bbtn\b[^"]*)"'
    new_content = re.sub(pattern, normalize_btn_class, content)

    # Sauvegarder
    html_path.write_text(new_content, encoding='utf-8')

    print(f"[OK] Replaced {replacements} button classes in rebalance.html")

    return replacements

if __name__ == "__main__":
    count = unify_button_classes()
    print(f"[DONE] Total replacements: {count}")
