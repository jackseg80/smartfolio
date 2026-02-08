"""
Compare Rotation V3 — SMA200 Safety Gate + Asymmetric Alpha

Purpose: Test two structural improvements to the Cons+Fast champion (Sharpe 1.069):
  1. SMA200 safety gate: force bear phase when BTC < SMA200 (crash protection)
  2. Asymmetric alpha: fast exit to bear, slow entry to bull (improve asymmetry)

Configs tested:

References:
1. Replica_V2.1        — Production baseline (Sharpe 0.759)
2. Cons+Fast           — V2 champion (Sharpe 1.069, MaxDD -31.1%)

Piste 1 — SMA200 Safety Gate:
3. Cons+Fast+SMA200    — + BTC < SMA200 → force bear
4. Cons+Fast+SMA150    — + shorter SMA (faster reaction)

Piste 2 — Asymmetric Alpha:
5. Cons+AsymA          — alpha_bull=0.15, alpha_bear=0.50 (slow in, fast out)
6. Cons+AsymB          — alpha_bull=0.20, alpha_bear=0.40 (moderate asymmetry)
7. Cons+AsymC          — alpha_bull=0.10, alpha_bear=0.60 (extreme asymmetry)

Combined:
8. Cons+SMA200+AsymA   — SMA200 gate + asymmetric (0.15/0.50)
9. Cons+SMA200+AsymB   — SMA200 gate + asymmetric (0.20/0.40)
10. Cons+SMA150+AsymA  — SMA150 gate + asymmetric (0.15/0.50)

Usage:
    cd d:/Python/smartfolio
    .venv/Scripts/Activate.ps1
    python scripts/analysis/compare_rotation_v3.py

Output: data/analysis/rotation_v3_comparison_YYYYMMDD_HHMMSS.csv
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
    ("Bull 2017",          "2017-01-01", "2017-12-31"),
    ("Crash 2018",         "2018-01-01", "2018-12-31"),
    ("Bull 2020-2021",     "2020-10-01", "2021-11-15"),
    ("Bear 2022",          "2021-11-15", "2022-11-15"),
    ("Recovery 2023-2024", "2022-11-15", "2024-04-01"),
    ("COVID Crash",        "2020-02-01", "2020-05-01"),
]

# Conservative allocation preset
CONS = dict(
    alloc_bear=(0.10, 0.03, 0.87),
    alloc_peak=(0.15, 0.15, 0.70),
    alloc_distribution=(0.15, 0.05, 0.80),
)


def build_configs():
    """Build all configs for V3 comparison."""
    configs = []

    # ── References ──

    configs.append({
        "label": "Replica_V2.1",
        "type": "replica",
        "multi_asset": False,
        "params": ReplicaParams(enable_direction_penalty=True),
    })

    configs.append({
        "label": "Cons+Fast",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(**CONS, smoothing_alpha=0.30),
    })

    # ── Piste 1: SMA Safety Gate ──

    configs.append({
        "label": "Cons+Fast+SMA200",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS, smoothing_alpha=0.30,
            enable_sma_gate=True, sma_period=200,
        ),
    })

    configs.append({
        "label": "Cons+Fast+SMA150",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS, smoothing_alpha=0.30,
            enable_sma_gate=True, sma_period=150,
        ),
    })

    # ── Piste 2: Asymmetric Alpha ──

    # A: slow bull (0.15) / fast bear (0.50)
    configs.append({
        "label": "Cons+AsymA",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            smoothing_alpha_bullish=0.15, smoothing_alpha_bearish=0.50,
        ),
    })

    # B: moderate asymmetry (0.20 / 0.40)
    configs.append({
        "label": "Cons+AsymB",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            smoothing_alpha_bullish=0.20, smoothing_alpha_bearish=0.40,
        ),
    })

    # C: extreme asymmetry (0.10 / 0.60)
    configs.append({
        "label": "Cons+AsymC",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            smoothing_alpha_bullish=0.10, smoothing_alpha_bearish=0.60,
        ),
    })

    # ── Combined: SMA + Asymmetric ──

    configs.append({
        "label": "Cons+SMA200+AsymA",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            enable_sma_gate=True, sma_period=200,
            smoothing_alpha_bullish=0.15, smoothing_alpha_bearish=0.50,
        ),
    })

    configs.append({
        "label": "Cons+SMA200+AsymB",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            enable_sma_gate=True, sma_period=200,
            smoothing_alpha_bullish=0.20, smoothing_alpha_bearish=0.40,
        ),
    })

    configs.append({
        "label": "Cons+SMA150+AsymA",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            enable_sma_gate=True, sma_period=150,
            smoothing_alpha_bullish=0.15, smoothing_alpha_bearish=0.50,
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
    logger.info("ROTATION V3 — SMA200 Safety Gate + Asymmetric Alpha")
    logger.info("=" * 70)

    configs = build_configs()
    logger.info(f"Configs: {len(configs)} | Periods: {len(PERIODS)} | Total: {len(configs) * len(PERIODS)}")

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"rotation_v3_comparison_{timestamp}.csv"

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
    """Print focused V3 comparison."""
    if not rows:
        return

    # ── Full History ──
    logger.info(f"\n{'=' * 70}")
    logger.info("FULL HISTORY — ALL CONFIGS (sorted by Sharpe)")
    logger.info(f"{'=' * 70}")

    full = [r for r in rows if r["period_name"] == "Full History"]
    if not full:
        logger.info("No Full History results")
        return

    header = f"{'Config':<25} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'AvgRisky':>9} {'Up':>6} {'Down':>6} {'Asym':>7}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in sorted(full, key=lambda x: x["sharpe_ratio"], reverse=True):
        logger.info(
            f"{r['config_label']:<25} {r['total_return_pct']:>7.1f}% "
            f"{r['sharpe_ratio']:>7.3f} {r['sortino_ratio']:>8.3f} "
            f"{r['max_drawdown_pct']:>7.1f}% "
            f"{r['avg_risky_allocation_pct']:>7.1f}% "
            f"{r['upside_capture']:>6.3f} {r['downside_capture']:>6.3f} "
            f"{r['capture_asymmetry']:>+6.3f}"
        )

    # ── Asymmetry analysis ──
    logger.info(f"\n{'=' * 70}")
    logger.info("ASYMMETRY ANALYSIS (Full History) — sorted by capture asymmetry")
    logger.info(f"{'=' * 70}")

    baseline = next((r for r in full if r["config_label"] == "Cons+Fast"), None)
    if baseline:
        header = f"{'Config':<25} {'Sharpe':>7} {'Up':>6} {'Down':>6} {'Asym':>7} {'dSharpe':>8} {'dAsym':>7}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in sorted(full, key=lambda x: x["capture_asymmetry"], reverse=True):
            logger.info(
                f"{r['config_label']:<25} {r['sharpe_ratio']:>7.3f} "
                f"{r['upside_capture']:>6.3f} {r['downside_capture']:>6.3f} "
                f"{r['capture_asymmetry']:>+6.3f}  "
                f"{r['sharpe_ratio'] - baseline['sharpe_ratio']:>+7.3f}  "
                f"{r['capture_asymmetry'] - baseline['capture_asymmetry']:>+6.3f}"
            )

    # ── Crash protection ──
    logger.info(f"\n{'=' * 70}")
    logger.info("CRASH PROTECTION (COVID + Bear 2022 + Crash 2018)")
    logger.info(f"{'=' * 70}")

    for bp in ["COVID Crash", "Bear 2022", "Crash 2018"]:
        br = [r for r in rows if r["period_name"] == bp]
        if not br:
            continue
        logger.info(f"\n  {bp} (sorted by MaxDD, least negative):")
        for r in sorted(br, key=lambda x: x["max_drawdown_pct"], reverse=True)[:5]:
            logger.info(
                f"    {r['config_label']:<25} Return={r['total_return_pct']:>7.1f}%  "
                f"MaxDD={r['max_drawdown_pct']:>7.1f}%  Sharpe={r['sharpe_ratio']:>7.3f}"
            )

    # ── Bull capture ──
    logger.info(f"\n{'=' * 70}")
    logger.info("BULL CAPTURE (Bull 2017 + Bull 2020-2021)")
    logger.info(f"{'=' * 70}")

    for bp in ["Bull 2017", "Bull 2020-2021"]:
        br = [r for r in rows if r["period_name"] == bp]
        if not br:
            continue
        logger.info(f"\n  {bp} (sorted by return):")
        for r in sorted(br, key=lambda x: x["total_return_pct"], reverse=True)[:5]:
            logger.info(
                f"    {r['config_label']:<25} Return={r['total_return_pct']:>7.1f}%  "
                f"MaxDD={r['max_drawdown_pct']:>7.1f}%  Sharpe={r['sharpe_ratio']:>7.3f}"
            )

    # ── Best per period ──
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
