"""
Walk-Forward Validation — Cycle Rotation strategies

Purpose: Measure overfitting by splitting data into in-sample (calibration)
and out-of-sample (validation) periods.

Split:
  - In-sample:  2017-01-01 → 2021-11-14 (Bull 2017, Crash 2018, COVID, Bull 2020-21)
  - Out-of-sample: 2021-11-15 → 2025-12-31 (Bear 2022, Recovery 2023-24, Halving 2024+)

Configs tested:
1. Replica V2.1           — Production baseline
2. Cons+Fast              — V2 champion (symmetric alpha=0.30)
3. Cons+AsymA             — Asymmetric alpha (0.15 bull / 0.50 bear)
4. Cons+Fast+SMA150       — SMA150 gate + fast alpha
5. Cons+SMA150+AsymA      — V3 champion (SMA150 + asymmetric)
6. Cons+SMA200+AsymA      — SMA200 variant for comparison

Key metric: if out-of-sample Sharpe holds ≥80% of in-sample Sharpe,
the strategy is robust. Below 60% = significant overfitting.

Usage:
    cd d:/Python/smartfolio
    .venv/Scripts/Activate.ps1
    python scripts/analysis/walk_forward_rotation.py

Output: data/analysis/walk_forward_rotation_YYYYMMDD_HHMMSS.csv
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

# Walk-forward split
PERIODS = [
    ("In-Sample (2017-2021)",  "2017-01-01", "2021-11-14"),
    ("Out-of-Sample (2022-2025)", "2021-11-15", "2025-12-31"),
    ("Full History",           "2017-01-01", "2025-12-31"),
]

# Conservative allocation preset
CONS = dict(
    alloc_bear=(0.10, 0.03, 0.87),
    alloc_peak=(0.15, 0.15, 0.70),
    alloc_distribution=(0.15, 0.05, 0.80),
)


def build_configs():
    configs = []

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

    configs.append({
        "label": "Cons+AsymA",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            smoothing_alpha_bullish=0.15, smoothing_alpha_bearish=0.50,
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

    return configs


async def run_backtest(di_data, config, initial_capital=10000.0):
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
    logger.info("WALK-FORWARD VALIDATION — Cycle Rotation strategies")
    logger.info("In-Sample: 2017-01-01 → 2021-11-14")
    logger.info("Out-of-Sample: 2021-11-15 → 2025-12-31")
    logger.info("=" * 70)

    configs = build_configs()
    logger.info(f"Configs: {len(configs)} | Periods: {len(PERIODS)} | Total: {len(configs) * len(PERIODS)}")

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"walk_forward_rotation_{timestamp}.csv"

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

    print_walk_forward(rows)
    return str(output_file)


def print_walk_forward(rows):
    if not rows:
        return

    # ── Walk-Forward Summary ──
    logger.info(f"\n{'=' * 70}")
    logger.info("WALK-FORWARD SUMMARY")
    logger.info(f"{'=' * 70}")

    header = (
        f"{'Config':<25} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'Retention':>10} "
        f"{'IS MaxDD':>9} {'OOS MaxDD':>10} {'Full Sharpe':>12}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    configs = sorted(
        set(r["config_label"] for r in rows),
        key=lambda c: next((r["sharpe_ratio"] for r in rows
                           if r["config_label"] == c and r["period_name"] == "Full History"), 0),
        reverse=True
    )

    for cfg_label in configs:
        is_row = next((r for r in rows if r["config_label"] == cfg_label
                       and "In-Sample" in r["period_name"]), None)
        oos_row = next((r for r in rows if r["config_label"] == cfg_label
                        and "Out-of-Sample" in r["period_name"]), None)
        full_row = next((r for r in rows if r["config_label"] == cfg_label
                         and r["period_name"] == "Full History"), None)

        if not all([is_row, oos_row, full_row]):
            continue

        is_sharpe = is_row["sharpe_ratio"]
        oos_sharpe = oos_row["sharpe_ratio"]
        retention = (oos_sharpe / is_sharpe * 100) if is_sharpe != 0 else 0

        logger.info(
            f"{cfg_label:<25} {is_sharpe:>10.3f} {oos_sharpe:>11.3f} "
            f"{retention:>9.0f}% "
            f"{is_row['max_drawdown_pct']:>8.1f}% {oos_row['max_drawdown_pct']:>9.1f}% "
            f"{full_row['sharpe_ratio']:>12.3f}"
        )

    # ── Detailed per-period ──
    logger.info(f"\n{'=' * 70}")
    logger.info("DETAILED RESULTS PER PERIOD")
    logger.info(f"{'=' * 70}")

    for pname in [p[0] for p in PERIODS]:
        pr = [r for r in rows if r["period_name"] == pname]
        if not pr:
            continue

        logger.info(f"\n  {pname}:")
        header = f"    {'Config':<25} {'Return%':>8} {'Ann%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'AvgRisky':>9} {'Asym':>7}"
        logger.info(header)
        logger.info("    " + "-" * (len(header) - 4))

        for r in sorted(pr, key=lambda x: x["sharpe_ratio"], reverse=True):
            logger.info(
                f"    {r['config_label']:<25} {r['total_return_pct']:>7.1f}% "
                f"{r['annualized_return_pct']:>6.1f}% "
                f"{r['sharpe_ratio']:>7.3f} {r['max_drawdown_pct']:>7.1f}% "
                f"{r['avg_risky_allocation_pct']:>7.1f}% "
                f"{r['capture_asymmetry']:>+6.3f}"
            )

    # ── Overfitting Assessment ──
    logger.info(f"\n{'=' * 70}")
    logger.info("OVERFITTING ASSESSMENT")
    logger.info(f"{'=' * 70}")
    logger.info("  Retention ≥ 80% = Robust")
    logger.info("  Retention 60-80% = Moderate overfitting")
    logger.info("  Retention < 60% = Significant overfitting")

    for cfg_label in configs:
        is_row = next((r for r in rows if r["config_label"] == cfg_label
                       and "In-Sample" in r["period_name"]), None)
        oos_row = next((r for r in rows if r["config_label"] == cfg_label
                        and "Out-of-Sample" in r["period_name"]), None)
        if not is_row or not oos_row:
            continue

        is_sharpe = is_row["sharpe_ratio"]
        oos_sharpe = oos_row["sharpe_ratio"]
        retention = (oos_sharpe / is_sharpe * 100) if is_sharpe != 0 else 0

        if retention >= 80:
            verdict = "ROBUST"
        elif retention >= 60:
            verdict = "MODERATE OVERFIT"
        else:
            verdict = "SIGNIFICANT OVERFIT"

        logger.info(
            f"  {cfg_label:<25} IS={is_sharpe:.3f} OOS={oos_sharpe:.3f} "
            f"Retention={retention:.0f}% → {verdict}"
        )


if __name__ == "__main__":
    output = asyncio.run(main())
    print(f"\nResults saved to: {output}")
