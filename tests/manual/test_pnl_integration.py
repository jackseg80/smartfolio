"""
Test manuel d'intégration P&L - À exécuter manuellement pour validation

Ce script démontre le workflow complet P&L Today:
1. Création d'un snapshot initial (valeur de base)
2. Attente/modification du portfolio
3. Récupération du P&L via l'endpoint

Usage:
    python tests/manual/test_pnl_integration.py
"""

import asyncio
import httpx
import pytest
from datetime import datetime
from tests.manual.conftest import requires_server

BASE_URL = "http://localhost:8080"

async def create_snapshot(user_id: str = "demo", source: str = "cointracking"):
    """Crée un snapshot portfolio pour établir la baseline"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/portfolio/snapshot",
            params={"user_id": user_id, "source": source}
        )
        print(f"[OK] Snapshot créé: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.json()

async def get_pnl_summary(user_id: str = "demo", source: str = "cointracking", anchor: str = "midnight"):
    """Récupère le P&L summary depuis l'anchor point"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/performance/summary",
            params={"user_id": user_id, "source": source, "anchor": anchor}
        )
        print(f"\n P&L Summary (anchor={anchor}):")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            perf = data.get("performance", {})
            total = perf.get("total", {})

            print(f"   Current Value: ${total.get('current_value_usd', 0):,.2f}")
            print(f"   Absolute Change: ${total.get('absolute_change_usd', 0):,.2f}")
            print(f"   Percent Change: {total.get('percent_change', 0):.2%}")
            print(f"   As of: {perf.get('as_of')}")
            print(f"   Base snapshot: {perf.get('base_snapshot_at')}")

            # Headers de cache
            print(f"\n Cache Headers:")
            print(f"   ETag: {response.headers.get('etag')}")
            print(f"   Cache-Control: {response.headers.get('cache-control')}")

            return data
        else:
            print(f"   Error: {response.text}")
            return None

@requires_server
async def test_etag_caching():
    """Test du mécanisme ETag pour cache validation"""
    async with httpx.AsyncClient() as client:
        # Premier appel
        response1 = await client.get(f"{BASE_URL}/api/performance/summary")
        etag = response1.headers.get("etag")

        print(f"\n Test ETag Caching:")
        print(f"   Premier appel: {response1.status_code}, ETag: {etag}")

        # Deuxième appel avec If-None-Match
        response2 = await client.get(
            f"{BASE_URL}/api/performance/summary",
            headers={"if-none-match": etag}
        )
        print(f"   Deuxième appel (avec ETag): {response2.status_code} (attendu: 304)")

        if response2.status_code == 304:
            print("   [OK] Cache ETag fonctionne correctement!")
        else:
            print("   [Warning]  Cache ETag n'a pas retourné 304")

@requires_server
async def test_anchor_points():
    """Test des différents anchor points"""
    anchors = ["prev_close", "midnight", "session"]

    print(f"\n Test des anchor points:")
    for anchor in anchors:
        await get_pnl_summary(anchor=anchor)
        print("-" * 60)

async def main():
    """Test complet du workflow P&L"""
    print("=" * 80)
    print("TEST MANUEL - P&L Integration")
    print("=" * 80)

    # Étape 1: Créer un snapshot baseline
    print("\n Étape 1: Création du snapshot baseline")
    await create_snapshot()

    # Étape 2: Récupérer le P&L actuel
    print("\n" + "=" * 80)
    print(" Étape 2: Calcul du P&L")
    await get_pnl_summary(anchor="midnight")

    # Étape 3: Test des anchor points
    await test_anchor_points()

    # Étape 4: Test du cache ETag
    await test_etag_caching()

    print("\n" + "=" * 80)
    print("[OK] Test terminé!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

