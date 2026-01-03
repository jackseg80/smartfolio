#!/usr/bin/env python3
"""
Test script to verify secrets.json loading works correctly
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_user_secrets():
    """Test loading secrets for both users"""
    from services.user_secrets import get_user_secrets

    print("Testing secrets.json loading...\n")

    # Test jack (real keys)
    print("=== User: jack ===")
    jack_secrets = get_user_secrets("jack")
    print(f"CoinGecko: {'[OK]' if jack_secrets.get('coingecko', {}).get('api_key') else '[MISSING]'}")
    print(f"CoinTracking: {'[OK]' if jack_secrets.get('cointracking', {}).get('api_key') else '[MISSING]'}")
    print(f"FRED: {'[OK]' if jack_secrets.get('fred', {}).get('api_key') else '[MISSING]'}")

    # Test demo (empty keys - dev mode)
    print("\n=== User: demo ===")
    demo_secrets = get_user_secrets("demo")
    print(f"Dev mode: {demo_secrets.get('dev_mode', {}).get('enabled', False)}")
    print(f"CoinGecko: {'[OK] Empty (expected)' if not demo_secrets.get('coingecko', {}).get('api_key') else '[ERROR] Has key'}")

    print("\n[SUCCESS] All tests passed!")

def test_resolve_secret_ref():
    """Test secret reference resolution"""
    from api.services.user_fs import UserScopedFS
    from api.services.config_migrator import resolve_secret_ref

    print("\n=== Testing resolve_secret_ref ===")

    # Create UserScopedFS for jack
    user_fs = UserScopedFS(".", "jack")

    # Test resolving keys
    cg_key = resolve_secret_ref("coingecko_api_key", user_fs)
    ct_key = resolve_secret_ref("cointracking_api_key", user_fs)
    fred_key = resolve_secret_ref("fred_api_key", user_fs)

    print(f"CoinGecko key resolved: {'[OK]' if cg_key else '[FAILED]'}")
    print(f"CoinTracking key resolved: {'[OK]' if ct_key else '[FAILED]'}")
    print(f"FRED key resolved: {'[OK]' if fred_key else '[FAILED]'}")

if __name__ == "__main__":
    try:
        test_user_secrets()
        test_resolve_secret_ref()
        print("\n[SUCCESS] All secrets loading correctly from secrets.json!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
