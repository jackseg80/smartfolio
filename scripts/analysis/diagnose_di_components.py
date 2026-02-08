"""
Diagnostic DI Components — IS vs OOS distribution analysis

Purpose: Quantify the distribution shift of each DI component between
In-Sample (2017-2021) and Out-of-Sample (2021-2025) periods.
This identifies which proxy degrades most, guiding the fix order.

Metrics per component:
- Mean, std, coefficient of variation (std/mean)
- % days stuck at floor (<10) or ceiling (>90)
- Correlation with 30-day forward BTC returns (predictive quality)
- KS test (Kolmogorov-Smirnov) between IS and OOS distributions
- Macro penalty frequency
- Phase factor distribution (% bearish / moderate / bullish)

Usage:
    cd d:/Python/smartfolio
    .venv/Scripts/Activate.ps1
    python scripts/analysis/diagnose_di_components.py

Output: console table + data/analysis/di_diagnostic_YYYYMMDD.png (histograms)
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.di_backtest.historical_di_calculator import HistoricalDICalculator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Walk-forward split boundary
SPLIT_DATE = "2021-11-14"

COMPONENTS = ["cycle_score", "onchain_score", "risk_score", "sentiment_score"]


def compute_diagnostics(df: pd.DataFrame, label: str) -> dict:
    """Compute diagnostic metrics for a period."""
    result = {"label": label, "days": len(df)}

    for comp in COMPONENTS:
        s = df[comp]
        mean = s.mean()
        std = s.std()
        cv = std / mean if mean != 0 else float("inf")
        floor_pct = (s < 10).sum() / len(s) * 100
        ceil_pct = (s > 90).sum() / len(s) * 100

        # Correlation with 30-day forward BTC returns
        fwd_returns = df["btc_price"].pct_change(30).shift(-30)
        valid = s.notna() & fwd_returns.notna()
        corr = s[valid].corr(fwd_returns[valid]) if valid.sum() > 30 else float("nan")

        result[comp] = {
            "mean": round(mean, 1),
            "std": round(std, 1),
            "cv": round(cv, 2),
            "floor_pct": round(floor_pct, 1),
            "ceil_pct": round(ceil_pct, 1),
            "corr_30d": round(corr, 3) if not np.isnan(corr) else "N/A",
        }

    # DI stats
    di = df["decision_index"]
    result["di"] = {
        "mean": round(di.mean(), 1),
        "std": round(di.std(), 1),
        "corr_30d": round(
            di.corr(df["btc_price"].pct_change(30).shift(-30).reindex(di.index))
            if len(di) > 30 else float("nan"),
            3,
        ),
    }

    # Macro penalty
    macro_days = (df["macro_penalty"] < 0).sum()
    result["macro"] = {
        "penalty_days": int(macro_days),
        "penalty_pct": round(macro_days / len(df) * 100, 1),
    }

    # Phase distribution
    phase_counts = df["phase"].value_counts(normalize=True) * 100
    result["phases"] = {
        "bearish": round(phase_counts.get("bearish", 0), 1),
        "moderate": round(phase_counts.get("moderate", 0), 1),
        "bullish": round(phase_counts.get("bullish", 0), 1),
    }

    return result


def run_ks_tests(df_is: pd.DataFrame, df_oos: pd.DataFrame) -> dict:
    """Run Kolmogorov-Smirnov tests between IS and OOS distributions."""
    ks_results = {}
    for comp in COMPONENTS + ["decision_index"]:
        stat, pvalue = stats.ks_2samp(df_is[comp].dropna(), df_oos[comp].dropna())
        ks_results[comp] = {
            "statistic": round(stat, 3),
            "p_value": f"{pvalue:.2e}",
            "significant": pvalue < 0.01,
        }
    return ks_results


def print_report(diag_is: dict, diag_oos: dict, ks: dict):
    """Print the diagnostic report."""
    logger.info("=" * 90)
    logger.info("DI COMPONENT DIAGNOSTIC — IS vs OOS")
    logger.info(f"IS: 2017-01-01 → {SPLIT_DATE} ({diag_is['days']} days)")
    logger.info(f"OOS: {SPLIT_DATE} → 2025-12-31 ({diag_oos['days']} days)")
    logger.info("=" * 90)

    # Component distributions
    header = (
        f"{'Component':<18} {'IS Mean':>8} {'OOS Mean':>9} {'Shift':>7} "
        f"{'IS Std':>7} {'OOS Std':>8} "
        f"{'IS CV':>6} {'OOS CV':>7} "
        f"{'IS Floor%':>10} {'OOS Floor%':>11} "
        f"{'IS Ceil%':>9} {'OOS Ceil%':>10}"
    )
    logger.info(f"\n{header}")
    logger.info("-" * len(header))

    for comp in COMPONENTS:
        is_c = diag_is[comp]
        oos_c = diag_oos[comp]
        shift = oos_c["mean"] - is_c["mean"]
        logger.info(
            f"{comp:<18} {is_c['mean']:>8.1f} {oos_c['mean']:>9.1f} {shift:>+7.1f} "
            f"{is_c['std']:>7.1f} {oos_c['std']:>8.1f} "
            f"{is_c['cv']:>6.2f} {oos_c['cv']:>7.2f} "
            f"{is_c['floor_pct']:>10.1f} {oos_c['floor_pct']:>11.1f} "
            f"{is_c['ceil_pct']:>9.1f} {oos_c['ceil_pct']:>10.1f}"
        )

    # DI summary
    logger.info(f"\n{'DI':<18} {diag_is['di']['mean']:>8.1f} {diag_oos['di']['mean']:>9.1f} "
                f"{diag_oos['di']['mean'] - diag_is['di']['mean']:>+7.1f} "
                f"{diag_is['di']['std']:>7.1f} {diag_oos['di']['std']:>8.1f}")

    # Predictive quality (correlation with 30d forward returns)
    logger.info(f"\n{'=' * 60}")
    logger.info("PREDICTIVE QUALITY (correlation with 30-day forward BTC returns)")
    logger.info(f"{'=' * 60}")
    header2 = f"{'Component':<18} {'IS Corr':>8} {'OOS Corr':>9} {'Delta':>7}"
    logger.info(header2)
    logger.info("-" * len(header2))

    for comp in COMPONENTS:
        is_corr = diag_is[comp]["corr_30d"]
        oos_corr = diag_oos[comp]["corr_30d"]
        if isinstance(is_corr, str) or isinstance(oos_corr, str):
            logger.info(f"{comp:<18} {str(is_corr):>8} {str(oos_corr):>9}     N/A")
        else:
            delta = oos_corr - is_corr
            logger.info(f"{comp:<18} {is_corr:>8.3f} {oos_corr:>9.3f} {delta:>+7.3f}")

    is_di_corr = diag_is["di"]["corr_30d"]
    oos_di_corr = diag_oos["di"]["corr_30d"]
    if not np.isnan(is_di_corr) and not np.isnan(oos_di_corr):
        logger.info(f"{'decision_index':<18} {is_di_corr:>8.3f} {oos_di_corr:>9.3f} "
                    f"{oos_di_corr - is_di_corr:>+7.3f}")

    # KS tests
    logger.info(f"\n{'=' * 60}")
    logger.info("KS TEST — Distribution shift significance")
    logger.info(f"{'=' * 60}")
    header3 = f"{'Component':<18} {'KS Stat':>8} {'p-value':>12} {'Significant':>12}"
    logger.info(header3)
    logger.info("-" * len(header3))

    for comp in COMPONENTS + ["decision_index"]:
        k = ks[comp]
        sig = "*** YES ***" if k["significant"] else "no"
        logger.info(f"{comp:<18} {k['statistic']:>8.3f} {k['p_value']:>12} {sig:>12}")

    # Macro penalty
    logger.info(f"\n{'=' * 60}")
    logger.info("MACRO PENALTY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  IS:  {diag_is['macro']['penalty_days']} days ({diag_is['macro']['penalty_pct']:.1f}%)")
    logger.info(f"  OOS: {diag_oos['macro']['penalty_days']} days ({diag_oos['macro']['penalty_pct']:.1f}%)")
    ratio = diag_oos["macro"]["penalty_pct"] / max(diag_is["macro"]["penalty_pct"], 0.1)
    logger.info(f"  Ratio OOS/IS: {ratio:.1f}x")

    # Phase distribution
    logger.info(f"\n{'=' * 60}")
    logger.info("PHASE FACTOR DISTRIBUTION")
    logger.info(f"{'=' * 60}")
    for phase_name in ["bearish", "moderate", "bullish"]:
        is_pct = diag_is["phases"][phase_name]
        oos_pct = diag_oos["phases"][phase_name]
        logger.info(f"  {phase_name:<10} IS: {is_pct:5.1f}%   OOS: {oos_pct:5.1f}%")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("DIAGNOSIS SUMMARY")
    logger.info(f"{'=' * 60}")

    # Rank components by KS statistic (worst first)
    ranked = sorted(
        [(c, ks[c]["statistic"]) for c in COMPONENTS],
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("  Components ranked by distribution shift (worst first):")
    for i, (comp, stat) in enumerate(ranked, 1):
        shift = diag_oos[comp]["mean"] - diag_is[comp]["mean"]
        logger.info(f"    {i}. {comp:<18} KS={stat:.3f}  shift={shift:+.1f}")


def save_histograms(df_is: pd.DataFrame, df_oos: pd.DataFrame, output_path: Path):
    """Save comparison histograms as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("DI Component Distributions — IS (blue) vs OOS (red)", fontsize=14)

        components_to_plot = COMPONENTS + ["decision_index"]
        for i, comp in enumerate(components_to_plot):
            ax = axes[i // 3][i % 3]
            ax.hist(df_is[comp].dropna(), bins=50, alpha=0.6, label="IS (2017-2021)", color="steelblue", density=True)
            ax.hist(df_oos[comp].dropna(), bins=50, alpha=0.6, label="OOS (2021-2025)", color="indianred", density=True)
            ax.set_title(comp)
            ax.legend(fontsize=8)
            ax.set_xlabel("Score (0-100)")

        # Empty subplot
        axes[1][2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"\nHistograms saved: {output_path}")
    except ImportError:
        logger.warning("matplotlib not available, skipping histogram export")


async def main():
    logger.info("Loading DI data for full period 2017-2025...")

    calculator = HistoricalDICalculator()
    di_data = await calculator.calculate_historical_di(
        user_id="jack",
        start_date="2017-01-01",
        end_date="2025-12-31",
        include_macro=True,
    )

    df = di_data.df
    logger.info(f"Total data points: {len(df)}")

    # Split
    split_dt = pd.Timestamp(SPLIT_DATE)
    df_is = df[df.index <= split_dt]
    df_oos = df[df.index > split_dt]

    logger.info(f"IS: {len(df_is)} days | OOS: {len(df_oos)} days")

    # Compute diagnostics
    diag_is = compute_diagnostics(df_is, "In-Sample")
    diag_oos = compute_diagnostics(df_oos, "Out-of-Sample")
    ks = run_ks_tests(df_is, df_oos)

    # Print report
    print_report(diag_is, diag_oos, ks)

    # Save histograms
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    save_histograms(df_is, df_oos, output_dir / f"di_diagnostic_{timestamp}.png")


if __name__ == "__main__":
    asyncio.run(main())
