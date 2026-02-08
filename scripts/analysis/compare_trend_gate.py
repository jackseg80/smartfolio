"""
Compare SMA200 Trend Gate vs NoLayers vs SmartFolio Replica V2.1

Teste les variantes du Trend Gate:
1. SMA200 pure (80/20)
2. SMA200 + DI modulation (DI module dans le range risk-on)
3. SMA200 + drawdown circuit breaker
4. SMA200 + DI + drawdown (full V3 proposal)

Usage:
    cd d:/Python/smartfolio
    .venv/Scripts/Activate.ps1
    python scripts/analysis/compare_trend_gate.py

Résultat: data/analysis/trend_gate_comparison_YYYYMMDD_HHMMSS.csv
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
    DITrendGateStrategy,
    ReplicaParams,
    TrendGateParams,
    DIStrategyConfig,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PERIODS = [
    ("Bull 2017",          "2017-01-01", "2017-12-31"),
    ("Crash 2018",         "2018-01-01", "2018-12-31"),
    ("COVID Crash",        "2020-02-01", "2020-05-01"),
    ("Bull 2020-2021",     "2020-10-01", "2021-11-15"),
    ("Bear 2022",          "2021-11-15", "2022-11-15"),
    ("Recovery 2023-2024", "2022-11-15", "2024-04-01"),
    ("Full History",       "2017-01-01", "2025-12-31"),
]


def build_configs():
    """Build all strategy configs to compare."""
    configs = []

    # Baseline: NoLayers (fixed 60%)
    configs.append({
        "label": "NoLayers_60pct",
        "type": "replica",
        "params": ReplicaParams(
            enable_risk_budget=False,
            enable_market_overrides=False,
            enable_exposure_cap=False,
            enable_governance_penalty=False,
            enable_direction_penalty=False,
        ),
    })

    # V2.1: All layers + direction penalty
    configs.append({
        "label": "V2.1_AllLayers",
        "type": "replica",
        "params": ReplicaParams(enable_direction_penalty=True),
    })

    # V2.1 without DP
    configs.append({
        "label": "V2.1_noDP",
        "type": "replica",
        "params": ReplicaParams(enable_direction_penalty=False),
    })

    # ── Trend Gate variants ──

    # TG1: Pure SMA200 (80/20)
    configs.append({
        "label": "TG_pure_80_20",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
        ),
    })

    # TG2: SMA200 (70/20) — less aggressive
    configs.append({
        "label": "TG_pure_70_20",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.70,
            risk_off_alloc=0.20,
            whipsaw_days=5,
        ),
    })

    # TG3: SMA200 (80/30) — higher floor
    configs.append({
        "label": "TG_pure_80_30",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.30,
            whipsaw_days=5,
        ),
    })

    # TG4: No whipsaw filter
    configs.append({
        "label": "TG_no_whipsaw",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=1,
        ),
    })

    # TG5: Longer whipsaw (10 days)
    configs.append({
        "label": "TG_whipsaw_10d",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=10,
        ),
    })

    # TG6: SMA(100) faster
    configs.append({
        "label": "TG_SMA100",
        "type": "trend_gate",
        "params": TrendGateParams(
            sma_period=100,
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
        ),
    })

    # TG7: SMA(300) slower
    configs.append({
        "label": "TG_SMA300",
        "type": "trend_gate",
        "params": TrendGateParams(
            sma_period=300,
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
        ),
    })

    # TG8: DI modulation in risk-on mode
    configs.append({
        "label": "TG_DI_mod",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
            enable_di_modulation=True,
            di_mod_risk_on_min=0.40,
            di_mod_risk_on_max=0.85,
        ),
    })

    # ── Drawdown Breaker variants (fixed: uses real portfolio DD) ──

    # TG9: DD breaker default thresholds (-15%/-25%) + ramp-up
    configs.append({
        "label": "TG_DD_15_25",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.15,
            dd_threshold_2=-0.25,
            dd_ramp_up=True,
        ),
    })

    # TG10: DD breaker looser thresholds (-20%/-30%)
    configs.append({
        "label": "TG_DD_20_30",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.20,
            dd_threshold_2=-0.30,
            dd_ramp_up=True,
        ),
    })

    # TG11: DD breaker tight (-10%/-20%)
    configs.append({
        "label": "TG_DD_10_20",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.10,
            dd_threshold_2=-0.20,
            dd_ramp_up=True,
        ),
    })

    # TG12: DD breaker without ramp-up (hard snap-back)
    configs.append({
        "label": "TG_DD_no_ramp",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.15,
            dd_threshold_2=-0.25,
            dd_ramp_up=False,
        ),
    })

    # TG13: Full V3 — DI mod + DD breaker (default thresholds)
    configs.append({
        "label": "TG_V3_full",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
            enable_di_modulation=True,
            di_mod_risk_on_min=0.40,
            di_mod_risk_on_max=0.85,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.15,
            dd_threshold_2=-0.25,
            dd_ramp_up=True,
        ),
    })

    # TG14: V3 with looser DD (-20%/-30%)
    configs.append({
        "label": "TG_V3_DD_20_30",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=5,
            enable_di_modulation=True,
            di_mod_risk_on_min=0.40,
            di_mod_risk_on_max=0.85,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.20,
            dd_threshold_2=-0.30,
            dd_ramp_up=True,
        ),
    })

    # TG15: No whipsaw + DD breaker (best pure TG + DD protection)
    configs.append({
        "label": "TG_nowh_DD",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=1,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.15,
            dd_threshold_2=-0.25,
            dd_ramp_up=True,
        ),
    })

    # TG16: No whipsaw + DI mod + DD breaker (best combo candidate)
    configs.append({
        "label": "TG_nowh_V3",
        "type": "trend_gate",
        "params": TrendGateParams(
            risk_on_alloc=0.80,
            risk_off_alloc=0.20,
            whipsaw_days=1,
            enable_di_modulation=True,
            di_mod_risk_on_min=0.40,
            di_mod_risk_on_max=0.85,
            enable_drawdown_breaker=True,
            dd_threshold_1=-0.15,
            dd_threshold_2=-0.25,
            dd_ramp_up=True,
        ),
    })

    return configs


async def run_backtest(di_data, config, initial_capital=10000.0):
    """Run a single backtest with given config."""
    cfg = DIStrategyConfig()

    if config["type"] == "replica":
        strategy = DISmartfolioReplicaStrategy(config=cfg, replica_params=config["params"])
    else:
        strategy = DITrendGateStrategy(config=cfg, trend_params=config["params"])

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
    )


async def main():
    logger.info("=" * 70)
    logger.info("TREND GATE COMPARISON — SMA200 vs NoLayers vs V2.1")
    logger.info("=" * 70)

    configs = build_configs()
    logger.info(f"Configs: {len(configs)} | Periods: {len(PERIODS)} | Total: {len(configs) * len(PERIODS)}")

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"trend_gate_comparison_{timestamp}.csv"

    fieldnames = [
        "config_label", "strategy_type", "period_name", "period_start", "period_end",
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
        logger.info(f"\n[{pi+1}/{len(PERIODS)}] {pname} ({start} → {end})")

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
    logger.info(f"Results: {len(rows)} rows → {output_file}")

    # Summary
    print_comparison(rows)
    return str(output_file)


def print_comparison(rows):
    """Print focused comparison."""
    if not rows:
        return

    logger.info(f"\n{'=' * 70}")
    logger.info("COMPARISON — Full History")
    logger.info(f"{'=' * 70}")

    full = [r for r in rows if r["period_name"] == "Full History"]
    if not full:
        logger.info("No Full History results")
        return

    header = f"{'Config':<25} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'AvgRisky':>9} {'Up':>6} {'Down':>6} {'Asym':>7}"
    logger.info(header)
    logger.info("─" * len(header))

    for r in sorted(full, key=lambda x: x["sharpe_ratio"], reverse=True):
        logger.info(
            f"{r['config_label']:<25} {r['total_return_pct']:>7.1f}% "
            f"{r['sharpe_ratio']:>7.3f} {r['max_drawdown_pct']:>7.1f}% "
            f"{r['avg_risky_allocation_pct']:>7.1f}% "
            f"{r['upside_capture']:>6.3f} {r['downside_capture']:>6.3f} "
            f"{r['capture_asymmetry']:>+6.3f}"
        )

    # Best per period
    logger.info(f"\n{'=' * 70}")
    logger.info("BEST CONFIG PER PERIOD (by Sharpe)")
    logger.info(f"{'=' * 70}")

    periods = sorted(set(r["period_name"] for r in rows))
    for p in periods:
        pr = [r for r in rows if r["period_name"] == p]
        if not pr:
            continue
        best = max(pr, key=lambda x: x["sharpe_ratio"])
        logger.info(
            f"  {p:<25} → {best['config_label']:<25} "
            f"Sharpe={best['sharpe_ratio']:.3f}  Return={best['total_return_pct']:.1f}%  "
            f"MaxDD={best['max_drawdown_pct']:.1f}%  Asym={best['capture_asymmetry']:+.3f}"
        )

    # TrendGate vs baselines analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("TREND GATE vs BASELINES (Full History)")
    logger.info(f"{'=' * 70}")

    baseline = next((r for r in full if r["config_label"] == "NoLayers_60pct"), None)
    v21 = next((r for r in full if r["config_label"] == "V2.1_AllLayers"), None)
    tg_pure = next((r for r in full if r["config_label"] == "TG_pure_80_20"), None)

    if baseline and v21 and tg_pure:
        for label, r in [("NoLayers 60%", baseline), ("V2.1 AllLayers", v21), ("TG Pure 80/20", tg_pure)]:
            logger.info(
                f"  {label:<20} Sharpe={r['sharpe_ratio']:.3f}  Return={r['total_return_pct']:.1f}%  "
                f"MaxDD={r['max_drawdown_pct']:.1f}%  AvgRisky={r['avg_risky_allocation_pct']:.1f}%  "
                f"Up/Down={r['upside_capture']:.3f}/{r['downside_capture']:.3f}  "
                f"Asym={r['capture_asymmetry']:+.3f}"
            )

    # Bear market analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("BEAR MARKET PROTECTION (Bear 2022 + Crash 2018)")
    logger.info(f"{'=' * 70}")

    bear_periods = ["Bear 2022", "Crash 2018", "COVID Crash"]
    for bp in bear_periods:
        br = [r for r in rows if r["period_name"] == bp]
        if not br:
            continue
        logger.info(f"\n  {bp}:")
        for r in sorted(br, key=lambda x: x["max_drawdown_pct"], reverse=True)[:5]:
            logger.info(
                f"    {r['config_label']:<25} Return={r['total_return_pct']:>7.1f}%  "
                f"MaxDD={r['max_drawdown_pct']:>7.1f}%  AvgRisky={r['avg_risky_allocation_pct']:.1f}%"
            )


if __name__ == "__main__":
    output = asyncio.run(main())
    print(f"\nResults saved to: {output}")
