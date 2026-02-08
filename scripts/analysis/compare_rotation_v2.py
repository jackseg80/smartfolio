"""
Compare Rotation V2 — Conservative + Fast Smooth + Drawdown Breaker combos

Purpose: Find optimal production candidate by combining the two best findings:
  - Rot_conservative: best Sharpe (1.042), best MaxDD (-32.6%)
  - Fast smooth (alpha=0.30): sweet spot between reactivity and stability

Configs tested:

References (from V1):
1. Replica V2.1        — Production baseline (Sharpe 0.759)
2. Rot_conservative    — V1 winner (alpha=0.15, no DD breaker)
3. Rot_fast_smooth     — V1 fast smooth (default allocs, alpha=0.30)

New combos:
4. Cons+Fast           — Conservative allocs + alpha=0.30
5. Cons+DD_10_20       — Conservative + DD breaker (-10%/-20%)
6. Cons+DD_15_25       — Conservative + DD breaker (-15%/-25%)
7. Cons+Fast+DD_10_20  — Conservative + fast smooth + DD breaker (-10%/-20%)
8. Cons+Fast+DD_15_25  — Conservative + fast smooth + DD breaker (-15%/-25%)
9. Cons+Fast+DD_10_20_noRamp — Same but no ramp-up (hard cut/restore)
10. Cons+DD_10_20_mult70 — Conservative + DD breaker with 0.70 multiplier (less aggressive cut)

Usage:
    cd d:/Python/smartfolio
    .venv/Scripts/Activate.ps1
    python scripts/analysis/compare_rotation_v2.py

Output: data/analysis/rotation_v2_comparison_YYYYMMDD_HHMMSS.csv
"""

import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.di_backtest.historical_di_calculator import HistoricalDICalculator
from services.di_backtest.di_backtest_engine import DIBacktestEngine
from services.di_backtest.trading_strategies import (
    DISmartfolioReplicaStrategy,
    DICycleRotationStrategy,
    ReplicaParams,
    RotationParams,
    DIStrategyConfig,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PERIODS = [
    ("Full History",       "2017-01-01", "2025-12-31"),
    ("Bull 2020-2021",     "2020-10-01", "2021-11-15"),
    ("Bear 2022",          "2021-11-15", "2022-11-15"),
    ("Recovery 2023-2024", "2022-11-15", "2024-04-01"),
    ("COVID Crash",        "2020-02-01", "2020-05-01"),
]

# Conservative allocation preset (V1 winner)
CONSERVATIVE_ALLOCS = dict(
    alloc_bear=(0.10, 0.03, 0.87),
    alloc_peak=(0.15, 0.15, 0.70),
    alloc_distribution=(0.15, 0.05, 0.80),
)


def build_configs():
    """Build all configs for V2 comparison."""
    configs = []

    # ── References ──

    # Ref 1: Replica V2.1 (production baseline)
    configs.append({
        "label": "Replica_V2.1",
        "type": "replica",
        "multi_asset": False,
        "params": ReplicaParams(enable_direction_penalty=True),
    })

    # Ref 2: Rot_conservative (V1 winner, alpha=0.15)
    configs.append({
        "label": "Rot_conservative",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(**CONSERVATIVE_ALLOCS),
    })

    # Ref 3: Rot_fast_smooth (default allocs, alpha=0.30)
    configs.append({
        "label": "Rot_fast_smooth",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(smoothing_alpha=0.30),
    })

    # ── New combos ──

    # 4. Conservative + fast smooth (alpha=0.30) — THE main candidate
    configs.append({
        "label": "Cons+Fast",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONSERVATIVE_ALLOCS,
            smoothing_alpha=0.30,
        ),
    })

    # 5. Conservative + DD breaker (-10%/-20%)
    configs.append({
        "label": "Cons+DD_10_20",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONSERVATIVE_ALLOCS,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.10,
            dd_threshold_2=-0.20,
            dd_multiplier=0.50,
            dd_ramp_up=True,
        ),
    })

    # 6. Conservative + DD breaker (-15%/-25%)
    configs.append({
        "label": "Cons+DD_15_25",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONSERVATIVE_ALLOCS,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.15,
            dd_threshold_2=-0.25,
            dd_multiplier=0.50,
            dd_ramp_up=True,
        ),
    })

    # 7. Conservative + fast smooth + DD breaker (-10%/-20%) — full combo
    configs.append({
        "label": "Cons+Fast+DD_10_20",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONSERVATIVE_ALLOCS,
            smoothing_alpha=0.30,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.10,
            dd_threshold_2=-0.20,
            dd_multiplier=0.50,
            dd_ramp_up=True,
        ),
    })

    # 8. Conservative + fast smooth + DD breaker (-15%/-25%)
    configs.append({
        "label": "Cons+Fast+DD_15_25",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONSERVATIVE_ALLOCS,
            smoothing_alpha=0.30,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.15,
            dd_threshold_2=-0.25,
            dd_multiplier=0.50,
            dd_ramp_up=True,
        ),
    })

    # 9. Conservative + fast + DD (-10%/-20%) WITHOUT ramp-up (hard cut/restore)
    configs.append({
        "label": "Cons+Fast+DD_noRamp",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONSERVATIVE_ALLOCS,
            smoothing_alpha=0.30,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.10,
            dd_threshold_2=-0.20,
            dd_multiplier=0.50,
            dd_ramp_up=False,
        ),
    })

    # 10. Conservative + DD breaker with softer cut (0.70 multiplier)
    configs.append({
        "label": "Cons+DD_10_20_soft",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONSERVATIVE_ALLOCS,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.10,
            dd_threshold_2=-0.20,
            dd_multiplier=0.70,
            dd_ramp_up=True,
        ),
    })

    return configs


async def run_backtest(di_data, config, initial_capital=10000.0):
    """Run a single backtest with given config."""
    cfg = DIStrategyConfig()

    if config["type"] == "replica":
        strategy = DISmartfolioReplicaStrategy(config=cfg, replica_params=config["params"])
    elif config["type"] == "rotation":
        strategy = DICycleRotationStrategy(config=cfg, rotation_params=config["params"])
    else:
        raise ValueError(f"Unknown strategy type: {config['type']}")

    strategy.set_di_series(di_data.df["decision_index"])
    if hasattr(strategy, 'set_cycle_series') and "cycle_score" in di_data.df.columns:
        strategy.set_cycle_series(di_data.df["cycle_score"])

    engine = DIBacktestEngine(
        transaction_cost=0.001,
        rebalance_threshold=0.05,
        risk_free_rate=0.02,
    )

    return engine.run_backtest(
        di_history=di_data.di_history,
        strategy=strategy,
        initial_capital=initial_capital,
        rebalance_frequency="weekly",
        multi_asset=config.get("multi_asset", False),
    )


async def main():
    logger.info("=" * 70)
    logger.info("ROTATION V2 — Conservative + Fast Smooth + DD Breaker combos")
    logger.info("=" * 70)

    configs = build_configs()
    logger.info(f"Configs: {len(configs)} | Periods: {len(PERIODS)} | Total: {len(configs) * len(PERIODS)}")

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"rotation_v2_comparison_{timestamp}.csv"

    fieldnames = [
        "config_label", "strategy_type", "multi_asset", "period_name", "period_start", "period_end",
        "total_return_pct", "annualized_return_pct", "benchmark_return_pct", "excess_return_pct",
        "max_drawdown_pct", "volatility_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "avg_risky_allocation_pct", "rebalance_count", "turnover_annual",
        "upside_capture", "downside_capture", "capture_asymmetry",
        "final_value", "total_days",
    ]

    rows = []
    calculator = HistoricalDICalculator()
    di_cache = {}

    for pi, (pname, start, end) in enumerate(PERIODS):
        logger.info(f"\n[{pi+1}/{len(PERIODS)}] {pname} ({start} -> {end})")

        key = f"{start}_{end}"
        if key not in di_cache:
            try:
                di_data = await calculator.calculate_historical_di(
                    user_id="jack", start_date=start, end_date=end, include_macro=True,
                )
                di_cache[key] = di_data
                logger.info(f"  DI: {len(di_data.di_history)} points")
            except Exception as e:
                logger.error(f"  DI error: {e}")
                continue
        else:
            di_data = di_cache[key]

        if di_data.df.empty:
            continue

        for cfg in configs:
            try:
                result = await run_backtest(di_data, cfg)
                up = round(result.upside_capture, 3)
                down = round(result.downside_capture, 3)
                rows.append({
                    "config_label": cfg["label"],
                    "strategy_type": cfg["type"],
                    "multi_asset": cfg.get("multi_asset", False),
                    "period_name": pname,
                    "period_start": start,
                    "period_end": end,
                    "total_return_pct": round(result.total_return * 100, 2),
                    "annualized_return_pct": round(result.annualized_return * 100, 2),
                    "benchmark_return_pct": round(result.benchmark_return * 100, 2),
                    "excess_return_pct": round(result.excess_return * 100, 2),
                    "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                    "volatility_pct": round(result.volatility * 100, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 3),
                    "sortino_ratio": round(result.sortino_ratio, 3),
                    "calmar_ratio": round(result.calmar_ratio, 3),
                    "avg_risky_allocation_pct": round(result.avg_risky_allocation * 100, 1),
                    "rebalance_count": result.rebalance_count,
                    "turnover_annual": round(result.turnover_annual, 3),
                    "upside_capture": up,
                    "downside_capture": down,
                    "capture_asymmetry": round(up - down, 3),
                    "final_value": round(result.final_value, 2),
                    "total_days": (result.end_date - result.start_date).days,
                })
            except Exception as e:
                logger.warning(f"  ERROR {cfg['label']}: {e}")

        logger.info(f"  Done ({len(configs)} configs)")

    # Write CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Results: {len(rows)} rows -> {output_file}")

    print_comparison(rows)
    return str(output_file)


def print_comparison(rows):
    """Print focused V2 comparison."""
    if not rows:
        return

    # ── Full History comparison ──
    logger.info(f"\n{'=' * 70}")
    logger.info("FULL HISTORY — ALL CONFIGS (sorted by Sharpe)")
    logger.info(f"{'=' * 70}")

    full = [r for r in rows if r["period_name"] == "Full History"]
    if not full:
        logger.info("No Full History results")
        return

    header = f"{'Config':<25} {'Return%':>8} {'Ann%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'AvgRisky':>9} {'Asym':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in sorted(full, key=lambda x: x["sharpe_ratio"], reverse=True):
        logger.info(
            f"{r['config_label']:<25} {r['total_return_pct']:>7.1f}% "
            f"{r['annualized_return_pct']:>6.1f}% "
            f"{r['sharpe_ratio']:>7.3f} {r['sortino_ratio']:>8.3f} "
            f"{r['max_drawdown_pct']:>7.1f}% "
            f"{r['avg_risky_allocation_pct']:>7.1f}% "
            f"{r['capture_asymmetry']:>+6.3f}"
        )

    # ── Head-to-head: Cons vs Cons+Fast ──
    logger.info(f"\n{'=' * 70}")
    logger.info("HEAD-TO-HEAD: Conservative vs Conservative+FastSmooth")
    logger.info(f"{'=' * 70}")

    for pname in [p[0] for p in PERIODS]:
        pr = [r for r in rows if r["period_name"] == pname]
        cons = next((r for r in pr if r["config_label"] == "Rot_conservative"), None)
        cons_fast = next((r for r in pr if r["config_label"] == "Cons+Fast"), None)
        if cons and cons_fast:
            sharpe_delta = cons_fast["sharpe_ratio"] - cons["sharpe_ratio"]
            dd_delta = cons_fast["max_drawdown_pct"] - cons["max_drawdown_pct"]
            logger.info(
                f"  {pname:<25} Cons: Sharpe={cons['sharpe_ratio']:.3f} DD={cons['max_drawdown_pct']:.1f}%  |  "
                f"Cons+Fast: Sharpe={cons_fast['sharpe_ratio']:.3f} DD={cons_fast['max_drawdown_pct']:.1f}%  |  "
                f"Delta: Sharpe={sharpe_delta:+.3f} DD={dd_delta:+.1f}%"
            )

    # ── DD breaker impact ──
    logger.info(f"\n{'=' * 70}")
    logger.info("DRAWDOWN BREAKER IMPACT (Full History)")
    logger.info(f"{'=' * 70}")

    cons = next((r for r in full if r["config_label"] == "Rot_conservative"), None)
    dd_configs = [r for r in full if "DD" in r["config_label"]]
    if cons and dd_configs:
        header = f"{'Config':<25} {'Sharpe':>7} {'MaxDD%':>8} {'DeltaSharpe':>12} {'DeltaDD':>9}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in sorted(dd_configs, key=lambda x: x["sharpe_ratio"], reverse=True):
            logger.info(
                f"{r['config_label']:<25} {r['sharpe_ratio']:>7.3f} {r['max_drawdown_pct']:>7.1f}%  "
                f"{r['sharpe_ratio'] - cons['sharpe_ratio']:>+11.3f}  "
                f"{r['max_drawdown_pct'] - cons['max_drawdown_pct']:>+8.1f}%"
            )

    # ── COVID crash protection ──
    logger.info(f"\n{'=' * 70}")
    logger.info("COVID CRASH & BEAR 2022 PROTECTION")
    logger.info(f"{'=' * 70}")

    for bp in ["COVID Crash", "Bear 2022"]:
        br = [r for r in rows if r["period_name"] == bp]
        if not br:
            continue
        logger.info(f"\n  {bp} (sorted by MaxDD, least negative):")
        header = f"    {'Config':<25} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7}"
        logger.info(header)
        for r in sorted(br, key=lambda x: x["max_drawdown_pct"], reverse=True):
            logger.info(
                f"    {r['config_label']:<25} {r['total_return_pct']:>7.1f}% "
                f"{r['max_drawdown_pct']:>7.1f}% {r['sharpe_ratio']:>7.3f}"
            )

    # ── Best config per period ──
    logger.info(f"\n{'=' * 70}")
    logger.info("BEST CONFIG PER PERIOD (by Sharpe)")
    logger.info(f"{'=' * 70}")

    for pname in [p[0] for p in PERIODS]:
        pr = [r for r in rows if r["period_name"] == pname]
        if not pr:
            continue
        best = max(pr, key=lambda x: x["sharpe_ratio"])
        logger.info(
            f"  {pname:<25} -> {best['config_label']:<25} "
            f"Sharpe={best['sharpe_ratio']:.3f}  MaxDD={best['max_drawdown_pct']:.1f}%  "
            f"Return={best['total_return_pct']:.1f}%"
        )


if __name__ == "__main__":
    output = asyncio.run(main())
    print(f"\nResults saved to: {output}")
