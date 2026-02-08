"""
Walk-Forward V2 — Expanding multi-window validation

Compares DI Calculator V1 (fixed normalization) vs V2 (adaptive normalization,
graduated macro, blended phase factor, conditional cycle correction).

Expanding windows (train grows, test = 1 year):
  Window 1: Train 2017-2019, Test 2020
  Window 2: Train 2017-2020, Test 2021
  Window 3: Train 2017-2021, Test 2022
  Window 4: Train 2017-2022, Test 2023
  Window 5: Train 2017-2023, Test 2024-2025

Plus fixed split (V1 compat): IS 2017-2021, OOS 2021-2025

Key metrics:
- Sharpe retention per window
- Average retention across all OOS windows
- KS distribution shift per DI component (V1 vs V2)
- Rank stability: does the IS champion remain top-3 OOS?

Success criterion: V2 average retention > 50% (vs ~18% V1)

Usage:
    cd d:/Python/smartfolio
    .venv/Scripts/Activate.ps1
    python scripts/analysis/walk_forward_rotation_v2.py

Output: data/analysis/walk_forward_v2_YYYYMMDD_HHMMSS.csv
"""

import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.di_backtest.historical_di_calculator import (
    HistoricalDICalculator,
    DICalculatorConfig,
)
from services.di_backtest.di_backtest_engine import DIBacktestEngine
from services.di_backtest.trading_strategies import (
    DISmartfolioReplicaStrategy,
    DICycleRotationStrategy,
    DIAdaptiveContinuousStrategy,
    ReplicaParams,
    RotationParams,
    ContinuousParams,
    DIStrategyConfig,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Expanding windows ──
EXPANDING_WINDOWS = [
    ("OOS-1 (2020)",    "2017-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("OOS-2 (2021)",    "2017-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("OOS-3 (2022)",    "2017-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("OOS-4 (2023)",    "2017-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("OOS-5 (2024-25)", "2017-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
]

# Fixed split for V1 comparison
FIXED_SPLIT = ("Fixed IS/OOS", "2017-01-01", "2021-11-14", "2021-11-15", "2025-12-31")

# Conservative allocation preset (same as V1 walk-forward)
CONS = dict(
    alloc_bear=(0.10, 0.03, 0.87),
    alloc_peak=(0.15, 0.15, 0.70),
    alloc_distribution=(0.15, 0.05, 0.80),
)

COMPONENTS = ["cycle_score", "onchain_score", "risk_score", "sentiment_score", "decision_index"]


def build_strategy_configs():
    """Build strategy configs — same as V1 for fair comparison."""
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
        "label": "Cons+SMA150+AsymA",
        "type": "rotation",
        "multi_asset": True,
        "params": RotationParams(
            **CONS,
            enable_sma_gate=True, sma_period=150,
            smoothing_alpha_bullish=0.15, smoothing_alpha_bearish=0.50,
        ),
    })

    # S9: Adaptive Continuous — parameter grid exploration
    # Baseline (default params)
    configs.append({
        "label": "AC-Default",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(),
    })

    # Lever 1: Reduce ceiling (cap max exposure)
    configs.append({
        "label": "AC-C70",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(alloc_ceiling=0.70),
    })
    configs.append({
        "label": "AC-C75",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(alloc_ceiling=0.75),
    })

    # Lever 2: Faster bear exit
    configs.append({
        "label": "AC-FB70",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(smoothing_alpha_bear=0.70),
    })

    # Combined: ceiling + fast bear
    configs.append({
        "label": "AC-C70-FB70",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(alloc_ceiling=0.70, smoothing_alpha_bear=0.70),
    })

    # Balanced 3 levers: mid ceiling + fast bear + stronger trend
    configs.append({
        "label": "AC-C75-FB65-T15",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(
            alloc_ceiling=0.75, smoothing_alpha_bear=0.65, trend_boost_pct=0.15,
        ),
    })

    # Aggressive protection: low ceiling + very fast bear + strong trend
    configs.append({
        "label": "AC-C70-FB80-T15",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(
            alloc_ceiling=0.70, smoothing_alpha_bear=0.80, trend_boost_pct=0.15,
        ),
    })

    # Mild adjustment: slightly lower ceiling + slightly faster bear
    configs.append({
        "label": "AC-C80-FB60",
        "type": "continuous",
        "multi_asset": True,
        "params": ContinuousParams(alloc_ceiling=0.80, smoothing_alpha_bear=0.60),
    })

    return configs


async def compute_di_data(calculator, user_id, start, end, version="v1"):
    """Compute DI data using V1 or V2 calculator."""
    if version == "v2":
        return await calculator.calculate_historical_di_v2(
            user_id=user_id,
            start_date=start,
            end_date=end,
            include_macro=True,
            config=DICalculatorConfig(),  # All V2 defaults
        )
    else:
        return await calculator.calculate_historical_di(
            user_id=user_id,
            start_date=start,
            end_date=end,
            include_macro=True,
        )


async def run_backtest(di_data, config, initial_capital=10000.0):
    """Run a single backtest."""
    cfg = DIStrategyConfig()

    if config["type"] == "replica":
        strategy = DISmartfolioReplicaStrategy(config=cfg, replica_params=config["params"])
    elif config["type"] == "rotation":
        strategy = DICycleRotationStrategy(config=cfg, rotation_params=config["params"])
    elif config["type"] == "continuous":
        strategy = DIAdaptiveContinuousStrategy(config=cfg, continuous_params=config["params"])
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


def compute_ks_stats(df_is, df_oos):
    """Compute KS test statistics for each component."""
    results = {}
    for comp in COMPONENTS:
        if comp in df_is.columns and comp in df_oos.columns:
            is_vals = df_is[comp].dropna()
            oos_vals = df_oos[comp].dropna()
            if len(is_vals) > 10 and len(oos_vals) > 10:
                stat, pvalue = stats.ks_2samp(is_vals, oos_vals)
                results[comp] = {"ks_stat": round(stat, 3), "p_value": pvalue}
            else:
                results[comp] = {"ks_stat": float("nan"), "p_value": float("nan")}
        else:
            results[comp] = {"ks_stat": float("nan"), "p_value": float("nan")}
    return results


async def run_window(calculator, user_id, window_name, train_start, train_end,
                     test_start, test_end, strategy_configs, di_version):
    """Run all strategies for one window (train + test)."""
    rows = []

    # Compute DI for train and test periods
    try:
        di_train = await compute_di_data(calculator, user_id, train_start, train_end, di_version)
        di_test = await compute_di_data(calculator, user_id, test_start, test_end, di_version)
    except Exception as e:
        logger.error(f"  DI error for {window_name} ({di_version}): {e}")
        return rows, {}

    if di_train.df.empty or di_test.df.empty:
        logger.warning(f"  Empty DI data for {window_name}")
        return rows, {}

    # KS tests between train and test distributions
    ks_results = compute_ks_stats(di_train.df, di_test.df)

    for period_label, di_data, period_start, period_end in [
        (f"Train ({train_start[:4]}-{train_end[:4]})", di_train, train_start, train_end),
        (f"Test ({test_start[:4]}-{test_end[:4]})", di_test, test_start, test_end),
    ]:
        for cfg in strategy_configs:
            try:
                result = await run_backtest(di_data, cfg)
                up = round(result.upside_capture, 3)
                down = round(result.downside_capture, 3)
                rows.append({
                    "di_version": di_version,
                    "window_name": window_name,
                    "config_label": cfg["label"],
                    "strategy_type": cfg["type"],
                    "multi_asset": cfg.get("multi_asset", False),
                    "period_type": "train" if "Train" in period_label else "test",
                    "period_label": period_label,
                    "period_start": period_start,
                    "period_end": period_end,
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
                logger.warning(f"  ERROR {cfg['label']} {period_label}: {e}")

    return rows, ks_results


def print_summary(all_rows, all_ks, versions=("v1", "v2")):
    """Print comprehensive walk-forward summary."""
    if not all_rows:
        logger.info("No results to display.")
        return

    logger.info(f"\n{'=' * 90}")
    logger.info("WALK-FORWARD V2 — EXPANDING MULTI-WINDOW RESULTS")
    logger.info(f"{'=' * 90}")

    for version in versions:
        version_rows = [r for r in all_rows if r["di_version"] == version]
        if not version_rows:
            continue

        logger.info(f"\n{'─' * 90}")
        logger.info(f"DI Calculator: {version.upper()}")
        logger.info(f"{'─' * 90}")

        # Get unique configs and windows
        config_labels = sorted(set(r["config_label"] for r in version_rows))
        windows = [w[0] for w in EXPANDING_WINDOWS] + [FIXED_SPLIT[0]]

        # Header
        header = f"{'Config':<22}"
        for w in windows:
            # Shorten window names
            short = w.replace("OOS-", "W").replace("Fixed IS/OOS", "Fixed")
            header += f" {short:>12}"
        header += f" {'Avg Ret%':>8}"
        logger.info(f"\n  Sharpe Retention per Window:")
        logger.info(f"  {header}")
        logger.info(f"  {'-' * len(header)}")

        for cfg_label in config_labels:
            line = f"  {cfg_label:<22}"
            retentions = []

            for window_name in windows:
                train_row = next(
                    (r for r in version_rows
                     if r["config_label"] == cfg_label
                     and r["window_name"] == window_name
                     and r["period_type"] == "train"), None)
                test_row = next(
                    (r for r in version_rows
                     if r["config_label"] == cfg_label
                     and r["window_name"] == window_name
                     and r["period_type"] == "test"), None)

                if train_row and test_row and train_row["sharpe_ratio"] != 0:
                    retention = test_row["sharpe_ratio"] / train_row["sharpe_ratio"] * 100
                    retentions.append(retention)
                    line += f" {retention:>11.0f}%"
                else:
                    line += f" {'N/A':>12}"

            avg_ret = np.mean(retentions) if retentions else float("nan")
            line += f" {avg_ret:>7.0f}%" if not np.isnan(avg_ret) else f" {'N/A':>8}"
            logger.info(line)

        # Detailed Train/Test Sharpe table
        logger.info(f"\n  Detailed Sharpe (Train → Test):")
        header2 = f"  {'Config':<22} {'Window':<18} {'Train Sharpe':>13} {'Test Sharpe':>12} {'Retention':>10}"
        logger.info(header2)
        logger.info(f"  {'-' * len(header2)}")

        for window_name in windows:
            for cfg_label in config_labels:
                train_row = next(
                    (r for r in version_rows
                     if r["config_label"] == cfg_label
                     and r["window_name"] == window_name
                     and r["period_type"] == "train"), None)
                test_row = next(
                    (r for r in version_rows
                     if r["config_label"] == cfg_label
                     and r["window_name"] == window_name
                     and r["period_type"] == "test"), None)

                if train_row and test_row:
                    ts = train_row["sharpe_ratio"]
                    os = test_row["sharpe_ratio"]
                    ret = (os / ts * 100) if ts != 0 else 0
                    logger.info(
                        f"  {cfg_label:<22} {window_name:<18} "
                        f"{ts:>13.3f} {os:>12.3f} {ret:>9.0f}%"
                    )

    # ── KS Distribution Shift comparison ──
    logger.info(f"\n{'=' * 90}")
    logger.info("KS DISTRIBUTION SHIFT — V1 vs V2")
    logger.info(f"{'=' * 90}")

    header3 = f"  {'Window':<18} {'Component':<18} {'V1 KS':>7} {'V2 KS':>7} {'Improvement':>12}"
    logger.info(header3)
    logger.info(f"  {'-' * len(header3)}")

    for window_name in [w[0] for w in EXPANDING_WINDOWS] + [FIXED_SPLIT[0]]:
        v1_ks = all_ks.get(("v1", window_name), {})
        v2_ks = all_ks.get(("v2", window_name), {})
        if not v1_ks and not v2_ks:
            continue

        for comp in COMPONENTS:
            v1_stat = v1_ks.get(comp, {}).get("ks_stat", float("nan"))
            v2_stat = v2_ks.get(comp, {}).get("ks_stat", float("nan"))
            if np.isnan(v1_stat) and np.isnan(v2_stat):
                continue
            improvement = ""
            if not np.isnan(v1_stat) and not np.isnan(v2_stat):
                diff = v1_stat - v2_stat
                improvement = f"{diff:>+.3f} {'OK' if diff > 0 else 'WORSE'}"
            logger.info(
                f"  {window_name:<18} {comp:<18} "
                f"{v1_stat:>7.3f} {v2_stat:>7.3f} {improvement:>12}"
            )

    # ── Overfitting Assessment ──
    logger.info(f"\n{'=' * 90}")
    logger.info("OVERFITTING ASSESSMENT")
    logger.info(f"{'=' * 90}")
    logger.info("  Retention ≥ 80% = Robust")
    logger.info("  Retention 60-80% = Moderate overfitting")
    logger.info("  Retention < 60% = Significant overfitting\n")

    for version in versions:
        version_rows = [r for r in all_rows if r["di_version"] == version]
        if not version_rows:
            continue

        config_labels = sorted(set(r["config_label"] for r in version_rows))
        logger.info(f"  {version.upper()}:")

        for cfg_label in config_labels:
            # Use expanding windows only (not fixed) for average
            retentions = []
            for window_name in [w[0] for w in EXPANDING_WINDOWS]:
                train_row = next(
                    (r for r in version_rows
                     if r["config_label"] == cfg_label
                     and r["window_name"] == window_name
                     and r["period_type"] == "train"), None)
                test_row = next(
                    (r for r in version_rows
                     if r["config_label"] == cfg_label
                     and r["window_name"] == window_name
                     and r["period_type"] == "test"), None)
                if train_row and test_row and train_row["sharpe_ratio"] != 0:
                    retentions.append(test_row["sharpe_ratio"] / train_row["sharpe_ratio"] * 100)

            avg_ret = np.mean(retentions) if retentions else 0
            if avg_ret >= 80:
                verdict = "ROBUST"
            elif avg_ret >= 60:
                verdict = "MODERATE OVERFIT"
            elif avg_ret >= 40:
                verdict = "MILD OVERFIT"
            else:
                verdict = "SIGNIFICANT OVERFIT"

            logger.info(f"    {cfg_label:<22} Avg Retention={avg_ret:.0f}% → {verdict}")

        logger.info("")

    # ── Rank Stability ──
    logger.info(f"{'=' * 90}")
    logger.info("RANK STABILITY — Does the IS champion stay top-3 OOS?")
    logger.info(f"{'=' * 90}")

    for version in versions:
        version_rows = [r for r in all_rows if r["di_version"] == version]
        if not version_rows:
            continue

        logger.info(f"\n  {version.upper()}:")
        config_labels = sorted(set(r["config_label"] for r in version_rows))

        for window_name in [w[0] for w in EXPANDING_WINDOWS]:
            train_sharpes = {}
            test_sharpes = {}
            for cfg_label in config_labels:
                tr = next((r for r in version_rows
                          if r["config_label"] == cfg_label
                          and r["window_name"] == window_name
                          and r["period_type"] == "train"), None)
                te = next((r for r in version_rows
                          if r["config_label"] == cfg_label
                          and r["window_name"] == window_name
                          and r["period_type"] == "test"), None)
                if tr:
                    train_sharpes[cfg_label] = tr["sharpe_ratio"]
                if te:
                    test_sharpes[cfg_label] = te["sharpe_ratio"]

            if not train_sharpes or not test_sharpes:
                continue

            train_rank = sorted(train_sharpes, key=train_sharpes.get, reverse=True)
            test_rank = sorted(test_sharpes, key=test_sharpes.get, reverse=True)
            champion = train_rank[0]
            oos_pos = test_rank.index(champion) + 1 if champion in test_rank else "?"
            stable = "YES" if isinstance(oos_pos, int) and oos_pos <= 3 else "NO"

            logger.info(
                f"    {window_name:<18} IS champion: {champion:<22} "
                f"OOS rank: #{oos_pos} → {stable}"
            )


async def main():
    logger.info("=" * 90)
    logger.info("WALK-FORWARD V2 — Expanding Multi-Window Validation")
    logger.info("Comparing DI Calculator V1 (fixed) vs V2 (adaptive)")
    logger.info("=" * 90)

    calculator = HistoricalDICalculator()
    strategy_configs = build_strategy_configs()
    user_id = "jack"

    all_windows = list(EXPANDING_WINDOWS) + [FIXED_SPLIT]
    total_tasks = len(all_windows) * 2 * len(strategy_configs) * 2  # windows × versions × strategies × train/test
    logger.info(f"Configs: {len(strategy_configs)} strategies × {len(all_windows)} windows × 2 versions")
    logger.info(f"Total backtests: ~{total_tasks}")

    all_rows = []
    all_ks = {}  # (version, window_name) -> ks_results

    for version in ("v1", "v2"):
        logger.info(f"\n{'─' * 70}")
        logger.info(f"DI Calculator: {version.upper()}")
        logger.info(f"{'─' * 70}")

        for i, window in enumerate(all_windows):
            window_name, train_start, train_end, test_start, test_end = window
            logger.info(f"\n  [{i+1}/{len(all_windows)}] {window_name} "
                        f"(Train: {train_start}→{train_end}, Test: {test_start}→{test_end})")

            rows, ks_results = await run_window(
                calculator, user_id, window_name,
                train_start, train_end, test_start, test_end,
                strategy_configs, version,
            )
            all_rows.extend(rows)
            if ks_results:
                all_ks[(version, window_name)] = ks_results
            logger.info(f"    → {len(rows)} results")

    # Write CSV
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"walk_forward_v2_{timestamp}.csv"

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"\nCSV: {len(all_rows)} rows → {output_file}")

    # Print summary
    print_summary(all_rows, all_ks)

    return str(output_file)


if __name__ == "__main__":
    output = asyncio.run(main())
    print(f"\nResults saved to: {output}")
