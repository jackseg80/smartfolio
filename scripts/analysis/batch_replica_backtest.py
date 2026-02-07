"""
Batch Backtest for SmartFolio Replica Layer Combinations

Teste automatiquement toutes les combinaisons de layers (2^4 = 16)
et variations de paramètres sur différentes périodes historiques.

Usage:
    cd d:/Python/smartfolio
    .venv/Scripts/Activate.ps1
    python scripts/analysis/batch_replica_backtest.py

Résultat: data/analysis/batch_replica_results_YYYYMMDD_HHMMSS.csv
"""

import asyncio
import csv
import itertools
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Ajouter le root du projet au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from services.di_backtest.historical_di_calculator import HistoricalDICalculator
from services.di_backtest.di_backtest_engine import DIBacktestEngine
from services.di_backtest.trading_strategies import (
    DISmartfolioReplicaStrategy,
    ReplicaParams,
    DIStrategyConfig,
)

logging.basicConfig(
    level=logging.WARNING,  # Réduire le bruit des logs
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────
# Périodes de test
# ──────────────────────────────────────────────

PERIODS = [
    ("Bull 2017",          "2017-01-01", "2017-12-31"),
    ("Crash 2018",         "2018-01-01", "2018-12-31"),
    ("COVID Crash",        "2020-02-01", "2020-05-01"),
    ("Bull 2020-2021",     "2020-10-01", "2021-11-15"),
    ("Bear 2022",          "2021-11-15", "2022-11-15"),
    ("Recovery 2023-2024", "2022-11-15", "2024-04-01"),
    ("Full History",       "2017-01-01", "2025-12-31"),
]


# ──────────────────────────────────────────────
# Configurations à tester
# ──────────────────────────────────────────────

def generate_layer_combinations():
    """Génère les 16 combinaisons de layer toggles."""
    bools = [True, False]
    combos = list(itertools.product(bools, bools, bools, bools))
    configs = []
    for rb, mo, ec, gp in combos:
        label_parts = []
        if rb:
            label_parts.append("L1")
        if mo:
            label_parts.append("L2")
        if ec:
            label_parts.append("L3")
        if gp:
            label_parts.append("L4")
        label = "+".join(label_parts) if label_parts else "NoLayers"

        configs.append({
            "label": label,
            "params": ReplicaParams(
                enable_risk_budget=rb,
                enable_market_overrides=mo,
                enable_exposure_cap=ec,
                enable_governance_penalty=gp,
            ),
        })
    return configs


def generate_param_variations():
    """Génère des variations de paramètres pour les configs intéressantes."""
    variations = []

    # Variation: Exposure confidence (Layer 3)
    for conf in [0.55, 0.75, 0.85, 0.95]:
        variations.append({
            "label": f"AllLayers_conf{conf:.2f}",
            "params": ReplicaParams(exposure_confidence=conf),
        })

    # Variation: Risk budget bounds
    for rmin, rmax in [(0.10, 0.95), (0.15, 0.90), (0.30, 0.85), (0.20, 0.70)]:
        variations.append({
            "label": f"AllLayers_bounds{int(rmin*100)}-{int(rmax*100)}",
            "params": ReplicaParams(risk_budget_min=rmin, risk_budget_max=rmax),
        })

    # Variation: Max governance penalty
    for mgp in [0.0, 0.10, 0.15, 0.30]:
        variations.append({
            "label": f"AllLayers_govmax{int(mgp*100)}",
            "params": ReplicaParams(max_governance_penalty=mgp),
        })

    # Variation: L1+L2 only (no cap, no governance) with different bounds
    for rmin, rmax in [(0.10, 0.95), (0.20, 0.85), (0.30, 0.80)]:
        variations.append({
            "label": f"L1+L2_bounds{int(rmin*100)}-{int(rmax*100)}",
            "params": ReplicaParams(
                enable_exposure_cap=False,
                enable_governance_penalty=False,
                risk_budget_min=rmin,
                risk_budget_max=rmax,
            ),
        })

    # Variation: L1 only (risk budget seul) with wider bounds
    for rmin, rmax in [(0.10, 0.95), (0.15, 0.90)]:
        variations.append({
            "label": f"L1only_bounds{int(rmin*100)}-{int(rmax*100)}",
            "params": ReplicaParams(
                enable_market_overrides=False,
                enable_exposure_cap=False,
                enable_governance_penalty=False,
                risk_budget_min=rmin,
                risk_budget_max=rmax,
            ),
        })

    return variations


# ──────────────────────────────────────────────
# Exécution
# ──────────────────────────────────────────────

async def run_single_backtest(
    di_data,
    params: ReplicaParams,
    initial_capital: float = 10000.0,
    rebalance_frequency: str = "weekly",
):
    """Exécute un seul backtest avec les params donnés."""
    strategy = DISmartfolioReplicaStrategy(
        config=DIStrategyConfig(),
        replica_params=params,
    )
    strategy.set_di_series(di_data.df["decision_index"])

    if "cycle_score" in di_data.df.columns:
        strategy.set_cycle_series(di_data.df["cycle_score"])

    engine = DIBacktestEngine(
        transaction_cost=0.001,
        rebalance_threshold=0.05,
        risk_free_rate=0.02,
    )

    result = engine.run_backtest(
        di_history=di_data.di_history,
        strategy=strategy,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
    )
    return result


async def main():
    """Point d'entrée principal."""
    logger.info("=" * 70)
    logger.info("BATCH BACKTEST - SmartFolio Replica Layer Combinations")
    logger.info("=" * 70)

    # Générer toutes les configurations
    layer_combos = generate_layer_combinations()
    param_variations = generate_param_variations()
    all_configs = layer_combos + param_variations

    logger.info(f"Configurations: {len(layer_combos)} layer combos + {len(param_variations)} param variations = {len(all_configs)} total")
    logger.info(f"Périodes: {len(PERIODS)}")
    logger.info(f"Total backtests: {len(all_configs) * len(PERIODS)}")
    logger.info("")

    # Préparer le fichier de sortie
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"batch_replica_results_{timestamp}.csv"

    # En-têtes CSV
    fieldnames = [
        "config_label",
        "period_name",
        "period_start",
        "period_end",
        # Layer toggles
        "L1_risk_budget",
        "L2_market_overrides",
        "L3_exposure_cap",
        "L4_governance_penalty",
        # Params
        "risk_budget_min",
        "risk_budget_max",
        "exposure_confidence",
        "max_governance_penalty",
        # Performance
        "total_return_pct",
        "annualized_return_pct",
        "benchmark_return_pct",
        "excess_return_pct",
        # Risk
        "max_drawdown_pct",
        "volatility_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        # Allocation
        "avg_risky_allocation_pct",
        "rebalance_count",
        "turnover_annual",
        "upside_capture",
        "downside_capture",
        # Meta
        "final_value",
        "total_days",
    ]

    results_rows = []
    calculator = HistoricalDICalculator()

    # Cache le DI par période pour éviter de recalculer
    di_cache = {}

    for period_idx, (period_name, start, end) in enumerate(PERIODS):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[{period_idx+1}/{len(PERIODS)}] Période: {period_name} ({start} → {end})")
        logger.info(f"{'─' * 50}")

        # Calculer le DI historique (une seule fois par période)
        cache_key = f"{start}_{end}"
        if cache_key not in di_cache:
            try:
                di_data = await calculator.calculate_historical_di(
                    user_id="jack",
                    start_date=start,
                    end_date=end,
                    include_macro=True,
                )
                di_cache[cache_key] = di_data
                logger.info(f"  DI calculé: {len(di_data.di_history)} points")
            except Exception as e:
                logger.error(f"  ERREUR calcul DI pour {period_name}: {e}")
                continue
        else:
            di_data = di_cache[cache_key]
            logger.info(f"  DI depuis cache: {len(di_data.di_history)} points")

        if di_data.df.empty:
            logger.warning(f"  DI vide pour {period_name}, skip")
            continue

        # Exécuter chaque configuration
        for cfg_idx, config in enumerate(all_configs):
            label = config["label"]
            params = config["params"]

            try:
                result = await run_single_backtest(di_data, params)

                row = {
                    "config_label": label,
                    "period_name": period_name,
                    "period_start": start,
                    "period_end": end,
                    # Layer toggles
                    "L1_risk_budget": params.enable_risk_budget,
                    "L2_market_overrides": params.enable_market_overrides,
                    "L3_exposure_cap": params.enable_exposure_cap,
                    "L4_governance_penalty": params.enable_governance_penalty,
                    # Params
                    "risk_budget_min": params.risk_budget_min,
                    "risk_budget_max": params.risk_budget_max,
                    "exposure_confidence": params.exposure_confidence,
                    "max_governance_penalty": params.max_governance_penalty,
                    # Performance
                    "total_return_pct": round(result.total_return * 100, 2),
                    "annualized_return_pct": round(result.annualized_return * 100, 2),
                    "benchmark_return_pct": round(result.benchmark_return * 100, 2),
                    "excess_return_pct": round(result.excess_return * 100, 2),
                    # Risk
                    "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                    "volatility_pct": round(result.volatility * 100, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 3),
                    "sortino_ratio": round(result.sortino_ratio, 3),
                    "calmar_ratio": round(result.calmar_ratio, 3),
                    # Allocation
                    "avg_risky_allocation_pct": round(result.avg_risky_allocation * 100, 1),
                    "rebalance_count": result.rebalance_count,
                    "turnover_annual": round(result.turnover_annual, 3),
                    "upside_capture": round(result.upside_capture, 3),
                    "downside_capture": round(result.downside_capture, 3),
                    # Meta
                    "final_value": round(result.final_value, 2),
                    "total_days": (result.end_date - result.start_date).days,
                }
                results_rows.append(row)

                if (cfg_idx + 1) % 10 == 0:
                    logger.info(f"  ... {cfg_idx+1}/{len(all_configs)} configs done")

            except Exception as e:
                logger.warning(f"  ERREUR config '{label}' sur {period_name}: {e}")
                continue

        logger.info(f"  ✓ {period_name} terminée ({len(all_configs)} configs)")

    # Écrire le CSV
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Écriture de {len(results_rows)} résultats dans {output_file}")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    logger.info(f"✓ Fichier créé: {output_file}")

    # Résumé rapide
    print_summary(results_rows)

    return str(output_file)


def print_summary(rows):
    """Affiche un résumé des meilleurs résultats."""
    if not rows:
        logger.info("Aucun résultat.")
        return

    logger.info(f"\n{'=' * 70}")
    logger.info("RÉSUMÉ - TOP 10 par Sharpe Ratio (toutes périodes confondues)")
    logger.info(f"{'=' * 70}")

    # Trier par Sharpe
    sorted_rows = sorted(rows, key=lambda r: r["sharpe_ratio"], reverse=True)

    header = f"{'Config':<35} {'Période':<20} {'Return%':>8} {'Bench%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'AvgRisky':>9}"
    logger.info(header)
    logger.info("─" * len(header))

    for row in sorted_rows[:10]:
        logger.info(
            f"{row['config_label']:<35} {row['period_name']:<20} "
            f"{row['total_return_pct']:>7.1f}% {row['benchmark_return_pct']:>7.1f}% "
            f"{row['sharpe_ratio']:>7.3f} {row['max_drawdown_pct']:>7.1f}% "
            f"{row['avg_risky_allocation_pct']:>7.1f}%"
        )

    # Top par période (Sharpe)
    logger.info(f"\n{'=' * 70}")
    logger.info("MEILLEURE CONFIG PAR PÉRIODE (par Sharpe)")
    logger.info(f"{'=' * 70}")

    periods_seen = set()
    for row in rows:
        periods_seen.add(row["period_name"])

    for period in sorted(periods_seen):
        period_rows = [r for r in rows if r["period_name"] == period]
        if not period_rows:
            continue
        best = max(period_rows, key=lambda r: r["sharpe_ratio"])
        logger.info(
            f"  {period:<25} → {best['config_label']:<30} "
            f"Sharpe={best['sharpe_ratio']:.3f}  Return={best['total_return_pct']:.1f}%  "
            f"MaxDD={best['max_drawdown_pct']:.1f}%  AvgRisky={best['avg_risky_allocation_pct']:.1f}%"
        )

    # Analyse: quelle couche cause le plus de drag?
    logger.info(f"\n{'=' * 70}")
    logger.info("ANALYSE IMPACT LAYERS (Full History uniquement)")
    logger.info(f"{'=' * 70}")

    full_rows = [r for r in rows if r["period_name"] == "Full History"]
    if full_rows:
        # Trouver la baseline (no layers)
        no_layers = [r for r in full_rows if r["config_label"] == "NoLayers"]
        all_layers = [r for r in full_rows if r["config_label"] == "L1+L2+L3+L4"]

        if no_layers and all_layers:
            nl = no_layers[0]
            al = all_layers[0]
            logger.info(f"  NoLayers:     Sharpe={nl['sharpe_ratio']:.3f}  Return={nl['total_return_pct']:.1f}%  MaxDD={nl['max_drawdown_pct']:.1f}%  AvgRisky={nl['avg_risky_allocation_pct']:.1f}%")
            logger.info(f"  AllLayers:    Sharpe={al['sharpe_ratio']:.3f}  Return={al['total_return_pct']:.1f}%  MaxDD={al['max_drawdown_pct']:.1f}%  AvgRisky={al['avg_risky_allocation_pct']:.1f}%")
            logger.info("")

        # Impact individuel de chaque layer
        layer_names = {
            "L1": "Risk Budget",
            "L2": "Market Overrides",
            "L3": "Exposure Cap",
            "L4": "Governance Penalty",
        }
        for layer_key, layer_name in layer_names.items():
            # Trouver config avec SEUL ce layer activé
            only_this = [r for r in full_rows if r["config_label"] == layer_key]
            # Trouver config sans ce layer (tous les autres activés)
            without = {
                "L1": "L2+L3+L4",
                "L2": "L1+L3+L4",
                "L3": "L1+L2+L4",
                "L4": "L1+L2+L3",
            }
            without_this = [r for r in full_rows if r["config_label"] == without[layer_key]]

            if only_this:
                r = only_this[0]
                logger.info(f"  Only {layer_key} ({layer_name}): Sharpe={r['sharpe_ratio']:.3f}  Return={r['total_return_pct']:.1f}%  AvgRisky={r['avg_risky_allocation_pct']:.1f}%")
            if without_this:
                r = without_this[0]
                logger.info(f"  All except {layer_key}:            Sharpe={r['sharpe_ratio']:.3f}  Return={r['total_return_pct']:.1f}%  AvgRisky={r['avg_risky_allocation_pct']:.1f}%")
            if only_this or without_this:
                logger.info("")


if __name__ == "__main__":
    output = asyncio.run(main())
    print(f"\nRésultats sauvegardés dans: {output}")
