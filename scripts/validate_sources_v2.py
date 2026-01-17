#!/usr/bin/env python3
"""
Sources V2 Integration Validator

Quick validation script to check that Sources V2 system is properly integrated.
Runs basic checks on the backend to ensure everything is configured correctly.

Usage:
    python scripts/validate_sources_v2.py
"""
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_check(name: str, passed: bool, details: str = ""):
    """Print check result with color."""
    icon = "✅" if passed else "❌"
    print(f"{icon} {name}")
    if details:
        print(f"   {details}")
    return passed


def validate_sources_v2():
    """Run validation checks."""
    print("=" * 60)
    print("🔍 Sources V2 Integration Validation")
    print("=" * 60)
    print()

    all_passed = True

    # Check 1: Source registry imports
    print("📦 Checking imports...")
    try:
        from services.sources import (
            source_registry,
            SourceCategory,
            SourceMode,
            SourceStatus,
            SourceBase,
            BalanceItem
        )
        all_passed &= print_check("Import services.sources", True)
    except ImportError as e:
        all_passed &= print_check("Import services.sources", False, str(e))
        return False

    # Check 2: Source registry initialization
    print("\n🏗️ Checking source registry...")
    try:
        source_ids = source_registry.source_ids
        all_passed &= print_check(
            "Source registry initialized",
            len(source_ids) > 0,
            f"Registered: {', '.join(source_ids)}"
        )
    except Exception as e:
        all_passed &= print_check("Source registry initialized", False, str(e))

    # Check 3: Expected sources registered
    print("\n📋 Checking expected sources...")
    expected_sources = [
        "manual_crypto",
        "manual_bourse",
        "cointracking_csv",
        "cointracking_api",
        "saxobank_csv"
    ]

    for source_id in expected_sources:
        is_registered = source_registry.is_registered(source_id)
        all_passed &= print_check(f"Source '{source_id}'", is_registered)

    # Check 4: Source categories
    print("\n🏷️ Checking categories...")
    crypto_sources = source_registry.list_sources(SourceCategory.CRYPTO)
    bourse_sources = source_registry.list_sources(SourceCategory.BOURSE)

    all_passed &= print_check(
        "Crypto sources",
        len(crypto_sources) >= 3,
        f"Found {len(crypto_sources)} sources"
    )
    all_passed &= print_check(
        "Bourse sources",
        len(bourse_sources) >= 2,
        f"Found {len(bourse_sources)} sources"
    )

    # Check 5: Manual sources can instantiate
    print("\n🔨 Checking source instantiation...")
    try:
        crypto_source = source_registry.get_source(
            "manual_crypto",
            user_id="test_validation",
            project_root=str(PROJECT_ROOT)
        )
        all_passed &= print_check(
            "Instantiate manual_crypto",
            crypto_source is not None
        )

        bourse_source = source_registry.get_source(
            "manual_bourse",
            user_id="test_validation",
            project_root=str(PROJECT_ROOT)
        )
        all_passed &= print_check(
            "Instantiate manual_bourse",
            bourse_source is not None
        )
    except Exception as e:
        all_passed &= print_check("Source instantiation", False, str(e))

    # Check 6: API endpoints registered
    print("\n🌐 Checking API integration...")
    try:
        from api.main import app

        # Check if sources_v2 router is registered
        routes = [route.path for route in app.routes]
        v2_routes = [r for r in routes if "/api/sources/v2" in r]

        all_passed &= print_check(
            "Sources V2 routes registered",
            len(v2_routes) > 0,
            f"Found {len(v2_routes)} V2 routes"
        )
    except Exception as e:
        all_passed &= print_check("API integration", False, str(e))

    # Check 7: balance_service integration
    print("\n⚙️ Checking balance_service integration...")
    try:
        from services.balance_service import balance_service, SOURCES_V2_ENABLED

        all_passed &= print_check(
            "SOURCES_V2_ENABLED flag",
            SOURCES_V2_ENABLED is True,
            f"Value: {SOURCES_V2_ENABLED}"
        )

        # Check that methods exist
        has_method = hasattr(balance_service, '_resolve_via_registry')
        all_passed &= print_check(
            "balance_service._resolve_via_registry()",
            has_method
        )

        has_method = hasattr(balance_service, '_is_category_based_user')
        all_passed &= print_check(
            "balance_service._is_category_based_user()",
            has_method
        )
    except Exception as e:
        all_passed &= print_check("balance_service integration", False, str(e))

    # Check 8: Migration module
    print("\n🔄 Checking migration module...")
    try:
        from services.sources.migration import SourceMigration, ensure_user_migrated

        migration = SourceMigration(str(PROJECT_ROOT))
        all_passed &= print_check("SourceMigration class", True)

        # Check if demo user needs migration (if exists)
        demo_config = PROJECT_ROOT / "data" / "users" / "demo" / "config.json"
        if demo_config.exists():
            needs_migration = migration.needs_migration("demo")
            with open(demo_config, 'r') as f:
                config = json.load(f)
                is_v2 = config.get("data_source") == "category_based"

            all_passed &= print_check(
                "Demo user config",
                not needs_migration or not is_v2,
                f"V2: {is_v2}, Needs migration: {needs_migration}"
            )
    except Exception as e:
        all_passed &= print_check("Migration module", False, str(e))

    # Check 9: Frontend components
    print("\n🎨 Checking frontend components...")
    components = [
        "static/components/manual-source-editor.js",
        "static/sources-manager-v2.js",
    ]

    for component in components:
        path = PROJECT_ROOT / component
        all_passed &= print_check(
            f"Component {path.name}",
            path.exists()
        )

    # Check 10: Documentation
    print("\n📚 Checking documentation...")
    docs = [
        "docs/SOURCES_V2.md",
        "docs/SOURCES_V2_INTEGRATION_CHECKLIST.md",
    ]

    for doc in docs:
        path = PROJECT_ROOT / doc
        all_passed &= print_check(
            f"Doc {path.name}",
            path.exists()
        )

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Sources V2 is properly integrated!")
        print("\nNext steps:")
        print("  1. Run integration tests: pytest tests/integration/test_sources_v2_integration.py")
        print("  2. Test manually following: docs/SOURCES_V2_INTEGRATION_CHECKLIST.md")
        print("  3. Monitor logs when users access the system")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review errors above")
        print("\nTroubleshooting:")
        print("  1. Check that all files are present")
        print("  2. Review import errors")
        print("  3. Check api/main.py includes sources_v2_router")
        return 1


def main():
    """Main entry point."""
    try:
        exit_code = validate_sources_v2()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
