#!/usr/bin/env python3
"""
Script pour vérifier quelles pages HTML ont tokens.css
"""
import os
from pathlib import Path

def check_tokens_in_html():
    static_dir = Path("static")
    html_files = sorted(static_dir.glob("*.html"))

    with_tokens = []
    without_tokens = []

    for html_file in html_files:
        content = html_file.read_text(encoding='utf-8')

        # Vérifier si tokens.css est présent
        has_tokens = 'tokens.css' in content or 'css/tokens.css' in content

        if has_tokens:
            with_tokens.append(html_file.name)
        else:
            without_tokens.append(html_file.name)

    print(f"[OK] Pages avec tokens.css ({len(with_tokens)}):")
    for f in with_tokens:
        print(f"  - {f}")

    print(f"\n[TODO] Pages sans tokens.css ({len(without_tokens)}):")
    for f in without_tokens:
        print(f"  - {f}")

    return without_tokens

if __name__ == "__main__":
    without_tokens = check_tokens_in_html()
    print(f"\n[TOTAL] A mettre a jour: {len(without_tokens)}")
