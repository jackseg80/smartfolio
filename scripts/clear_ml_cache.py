"""
Script pour nettoyer le cache ML et forcer le re-téléchargement de données complètes

Usage:
    python scripts/clear_ml_cache.py --all          # Nettoie tout le cache
    python scripts/clear_ml_cache.py --benchmarks   # Nettoie seulement les benchmarks (SPY, QQQ, etc.)
    python scripts/clear_ml_cache.py --crypto       # Nettoie seulement les cryptos
"""

import os
import argparse
from pathlib import Path
import shutil

# Répertoires de cache
PARQUET_CACHE_DIR = Path("data/cache/bourse/ml")
BOURSE_CACHE_DIR = Path("data/cache/bourse")

def clear_all_cache():
    """Nettoie tout le cache ML"""
    print("🗑️  Nettoyage complet du cache ML...")

    if PARQUET_CACHE_DIR.exists():
        count = len(list(PARQUET_CACHE_DIR.glob("*.parquet")))
        shutil.rmtree(PARQUET_CACHE_DIR)
        PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {count} fichiers Parquet supprimés de {PARQUET_CACHE_DIR}")
    else:
        print(f"   ℹ️  Répertoire {PARQUET_CACHE_DIR} n'existe pas")

    if BOURSE_CACHE_DIR.exists():
        parquet_files = list(BOURSE_CACHE_DIR.glob("*.parquet"))
        for f in parquet_files:
            f.unlink()
        print(f"   ✅ {len(parquet_files)} fichiers Parquet supprimés de {BOURSE_CACHE_DIR}")

    print("✅ Cache nettoyé avec succès!")
    print("ℹ️  Les données seront re-téléchargées au prochain training (60-90s pour 20 ans)")

def clear_benchmarks_cache():
    """Nettoie seulement le cache des benchmarks (SPY, QQQ, IWM, DIA)"""
    print("🗑️  Nettoyage du cache des benchmarks...")

    benchmarks = ["SPY", "QQQ", "IWM", "DIA"]
    count = 0

    if PARQUET_CACHE_DIR.exists():
        for benchmark in benchmarks:
            files = list(PARQUET_CACHE_DIR.glob(f"{benchmark}_*.parquet"))
            for f in files:
                f.unlink()
                count += 1
                print(f"   🗑️  Supprimé: {f.name}")

    print(f"✅ {count} fichiers benchmark supprimés")
    print("ℹ️  Les benchmarks seront re-téléchargés au prochain training")

def clear_crypto_cache():
    """Nettoie seulement le cache crypto"""
    print("🗑️  Nettoyage du cache crypto...")

    cryptos = ["BTC", "ETH", "SOL"]
    count = 0

    if PARQUET_CACHE_DIR.exists():
        for crypto in cryptos:
            files = list(PARQUET_CACHE_DIR.glob(f"{crypto}_*.parquet"))
            for f in files:
                f.unlink()
                count += 1
                print(f"   🗑️  Supprimé: {f.name}")

    print(f"✅ {count} fichiers crypto supprimés")

def main():
    parser = argparse.ArgumentParser(description="Nettoie le cache ML pour forcer le re-téléchargement")
    parser.add_argument("--all", action="store_true", help="Nettoie tout le cache")
    parser.add_argument("--benchmarks", action="store_true", help="Nettoie seulement les benchmarks (SPY, QQQ, IWM, DIA)")
    parser.add_argument("--crypto", action="store_true", help="Nettoie seulement les cryptos")

    args = parser.parse_args()

    if args.all:
        clear_all_cache()
    elif args.benchmarks:
        clear_benchmarks_cache()
    elif args.crypto:
        clear_crypto_cache()
    else:
        print("❌ Aucune option spécifiée. Usage:")
        print("   python scripts/clear_ml_cache.py --all          # Nettoie tout")
        print("   python scripts/clear_ml_cache.py --benchmarks   # Nettoie benchmarks")
        print("   python scripts/clear_ml_cache.py --crypto       # Nettoie cryptos")

if __name__ == "__main__":
    main()
