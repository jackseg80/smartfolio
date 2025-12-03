#!/usr/bin/env python3
"""
Consolidate user configuration: remove API keys from config.json (keep in secrets.json only)
"""
import json
from pathlib import Path

def consolidate_user_config(user_id: str = "jack"):
    """
    Remove sensitive API keys from config.json (keep only in secrets.json)
    """
    user_dir = Path(f"data/users/{user_id}")
    config_path = user_dir / "config.json"
    secrets_path = user_dir / "secrets.json"

    if not config_path.exists():
        print(f"[ERROR] {config_path} not found")
        return

    if not secrets_path.exists():
        print(f"[ERROR] {secrets_path} not found")
        return

    # Load both files
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(secrets_path, 'r', encoding='utf-8') as f:
        secrets = json.load(f)

    print(f"\nConsolidating config for user: {user_id}")
    print(f"   Config: {config_path}")
    print(f"   Secrets: {secrets_path}")

    # Sensitive keys to remove from config.json
    sensitive_keys = [
        "cointracking_api_key",
        "cointracking_api_secret",
        "coingecko_api_key",
        "fred_api_key"
    ]

    # Backup original
    backup_path = config_path.with_suffix(".json.bak")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"   [OK] Backup created: {backup_path}")

    # Remove sensitive keys from config
    removed_keys = []
    for key in sensitive_keys:
        if key in config:
            removed_keys.append(key)
            del config[key]

    if removed_keys:
        print(f"   [REMOVED] From config.json: {', '.join(removed_keys)}")

        # Save cleaned config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"   [OK] config.json cleaned")
    else:
        print(f"   [INFO] No sensitive keys found in config.json")

    # Verify secrets.json has the keys
    print(f"\n   Verifying secrets.json:")
    print(f"      - CoinGecko: {'OK' if secrets.get('coingecko', {}).get('api_key') else 'MISSING'}")
    print(f"      - CoinTracking: {'OK' if secrets.get('cointracking', {}).get('api_key') else 'MISSING'}")

    print(f"\n[DONE] Keys now only in secrets.json (more secure)")

if __name__ == "__main__":
    import sys
    user = sys.argv[1] if len(sys.argv) > 1 else "jack"
    consolidate_user_config(user)
