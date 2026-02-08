"""
Diagnostic Phase Engine — Collecteur prospectif

Le phase_engine ne conserve que 10 observations en memoire.
Ce script collecte les predictions de phase + prix BTC dans un CSV
pour permettre un diagnostic de biais apres accumulation de donnees (30+ jours).

Usage:
    # Collecter un snapshot (a lancer periodiquement, ex: cron 1h)
    python scripts/analysis/diagnose_phase_engine.py --collect

    # Analyser les donnees collectees (apres 30+ jours)
    python scripts/analysis/diagnose_phase_engine.py --analyze

    # Afficher les dernieres collectes
    python scripts/analysis/diagnose_phase_engine.py --tail 20
"""
import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

# Ajouter le root du projet au path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "data" / "diagnostics" / "phase_engine_history.csv"
CSV_FIELDS = [
    "timestamp", "phase", "confidence", "phase_prob_btc", "phase_prob_eth",
    "phase_prob_large", "phase_prob_alt", "persistence_count",
    "cycle_score_simulated", "btc_price_usd",
]

# Mapping phase → cycle_score simule (identique a strategy_registry._simulate_cycle_score)
PHASE_TO_CYCLE = {"btc": 40.0, "eth": 60.0, "large": 75.0, "alt": 85.0}


async def collect_snapshot():
    """Collecte un snapshot phase_engine + prix BTC et l'ajoute au CSV."""
    from services.execution.phase_engine import get_phase_engine

    engine = get_phase_engine()
    state = await engine.get_current_phase(force_refresh=True)

    # Prix BTC depuis le cache local (pas d'API call externe)
    btc_price = None
    try:
        from services.di_backtest.data_sources import DIBacktestDataSources
        ds = DIBacktestDataSources()
        prices = await ds.get_btc_prices(days=7, force_refresh=False)
        if prices is not None and len(prices) > 0:
            btc_price = float(prices.iloc[-1])
    except Exception as e:
        print(f"Warning: could not fetch BTC price: {e}")

    row = {
        "timestamp": datetime.now().isoformat(),
        "phase": state.phase_now.value,
        "confidence": f"{state.confidence:.4f}",
        "phase_prob_btc": f"{state.phase_probs.get('btc', 0):.4f}",
        "phase_prob_eth": f"{state.phase_probs.get('eth', 0):.4f}",
        "phase_prob_large": f"{state.phase_probs.get('large', 0):.4f}",
        "phase_prob_alt": f"{state.phase_probs.get('alt', 0):.4f}",
        "persistence_count": state.persistence_count,
        "cycle_score_simulated": PHASE_TO_CYCLE.get(state.phase_now.value, 50.0),
        "btc_price_usd": f"{btc_price:.2f}" if btc_price else "",
    }

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[{row['timestamp']}] Phase={row['phase']} conf={row['confidence']} "
          f"cycle_sim={row['cycle_score_simulated']} BTC=${row['btc_price_usd']}")


def analyze():
    """Analyse les donnees collectees pour detecter un biais de phase."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("pandas et numpy requis: pip install pandas numpy")
        return

    if not CSV_PATH.exists():
        print(f"Pas de donnees collectees. Lancer d'abord: python {__file__} --collect")
        return

    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    print(f"\n=== Phase Engine Diagnostic ===")
    print(f"Observations: {len(df)}")
    print(f"Periode: {df['timestamp'].min()} → {df['timestamp'].max()}")
    duration_days = (df["timestamp"].max() - df["timestamp"].min()).days
    print(f"Duree: {duration_days} jours")

    if duration_days < 7:
        print(f"\nInsuffisant pour analyse (minimum 7 jours, idealement 30+)")
        return

    # Distribution des phases
    print(f"\n--- Distribution des phases ---")
    phase_counts = df["phase"].value_counts()
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} ({count/len(df)*100:.1f}%)")

    # Correlation cycle_score_simulated vs BTC returns
    if "btc_price_usd" in df.columns and df["btc_price_usd"].notna().sum() > 10:
        df["btc_price"] = pd.to_numeric(df["btc_price_usd"], errors="coerce")
        df = df.dropna(subset=["btc_price"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Returns forward 7j et 30j (approxime par index — les observations ne sont pas forcement journalieres)
        for offset_label, offset in [("7obs", 7), ("30obs", 30)]:
            if len(df) > offset:
                df[f"return_{offset_label}"] = df["btc_price"].shift(-offset) / df["btc_price"] - 1
                valid = df.dropna(subset=[f"return_{offset_label}"])
                if len(valid) > 5:
                    corr = valid["cycle_score_simulated"].corr(valid[f"return_{offset_label}"])
                    print(f"\n--- Correlation cycle_score_simule vs BTC return ({offset_label}) ---")
                    print(f"  r = {corr:.4f} (n={len(valid)})")
                    if corr > 0.05:
                        print(f"  → Positif: le cycle_score est predictif (bon signe)")
                    elif corr < -0.05:
                        print(f"  → NEGATIF: le cycle_score est contre-productif (biais potentiel!)")
                    else:
                        print(f"  → Quasi-nul: pas de signal significatif")

        # Matrice de confusion: phase vs direction prix
        if "return_7obs" in df.columns:
            valid = df.dropna(subset=["return_7obs"])
            valid["direction"] = np.where(valid["return_7obs"] > 0, "up", "down")
            print(f"\n--- Matrice Phase vs Direction prix (7 obs ahead) ---")
            ct = pd.crosstab(valid["phase"], valid["direction"], normalize="index")
            print(ct.to_string())
    else:
        print("\nPas assez de donnees prix BTC pour l'analyse de correlation.")

    print(f"\n--- Conclusion ---")
    print(f"Accumuler 30+ jours de donnees pour un diagnostic fiable.")
    print(f"Le backtest montre que la sigmoide (cycle) a une correlation qui s'inverse OOS.")
    print(f"Le phase_engine ML devrait avoir une correlation positive si non-biaise.")


def tail(n: int = 20):
    """Affiche les N dernieres collectes."""
    if not CSV_PATH.exists():
        print("Pas de donnees collectees.")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Dernieres {min(n, len(lines)-1)} observations (sur {len(lines)-1} total):")
    header = lines[0].strip()
    print(header)
    for line in lines[-(n):]:
        print(line.strip())


def main():
    parser = argparse.ArgumentParser(description="Phase Engine Diagnostic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect", action="store_true", help="Collecter un snapshot")
    group.add_argument("--analyze", action="store_true", help="Analyser les donnees collectees")
    group.add_argument("--tail", type=int, nargs="?", const=20, help="Afficher les N dernieres collectes")

    args = parser.parse_args()

    if args.collect:
        asyncio.run(collect_snapshot())
    elif args.analyze:
        analyze()
    elif args.tail is not None:
        tail(args.tail)


if __name__ == "__main__":
    main()
