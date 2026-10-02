"""
Opportunity Scanner for Market Opportunities System

Scans S&P 500 sectors vs current portfolio to detect gaps and opportunities.
Scores each gap using 3-pillar approach: Momentum 40%, Value 30%, Diversification 30%.

Author: Crypto Rebalancer Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Any, Optional
from datetime import datetime
from services.ml.bourse.horizons import OPPORTUNITY_HORIZONS
import json
import logging

from services.ml.bourse.sector_analyzer import SectorAnalyzer

logger = logging.getLogger(__name__)


# GICS Level 1 Sectors (11 Standard S&P 500 Sectors) + Geographic Sectors
STANDARD_SECTORS = {
    # === INDUSTRY SECTORS (GICS) ===
    "Technology": {
        "target_range": (15, 30),
        "etf": "XLK",
        "description": "Information Technology"
    },
    "Healthcare": {
        "target_range": (10, 18),
        "etf": "XLV",
        "description": "Healthcare"
    },
    "Financials": {
        "target_range": (10, 18),
        "etf": "XLF",
        "description": "Financial Services"
    },
    "Consumer Discretionary": {
        "target_range": (8, 15),
        "etf": "XLY",
        "description": "Consumer Cyclical"
    },
    "Communication Services": {
        "target_range": (8, 15),
        "etf": "XLC",
        "description": "Communication Services"
    },
    "Industrials": {
        "target_range": (8, 15),
        "etf": "XLI",
        "description": "Industrials"
    },
    "Consumer Staples": {
        "target_range": (5, 12),
        "etf": "XLP",
        "description": "Consumer Defensive"
    },
    "Energy": {
        "target_range": (3, 10),
        "etf": "XLE",
        "description": "Energy"
    },
    "Utilities": {
        "target_range": (2, 8),
        "etf": "XLU",
        "description": "Utilities"
    },
    "Real Estate": {
        "target_range": (2, 8),
        "etf": "XLRE",
        "description": "Real Estate"
    },
    "Materials": {
        "target_range": (2, 8),
        "etf": "XLB",
        "description": "Materials"
    },

    # === GEOGRAPHIC SECTORS (NEW - Oct 2025) ===
    "Europe": {
        "target_range": (10, 20),
        "etf": "VGK",  # Vanguard FTSE Europe ETF
        "description": "European developed markets"
    },
    "Asia Pacific": {
        "target_range": (5, 15),
        "etf": "VPL",  # Vanguard FTSE Pacific ETF
        "description": "Asia-Pacific ex-Japan"
    },
    "Emerging Markets": {
        "target_range": (5, 15),
        "etf": "VWO",  # Vanguard FTSE Emerging Markets ETF
        "description": "Emerging markets exposure"
    },
    "Japan": {
        "target_range": (3, 10),
        "etf": "EWJ",  # iShares MSCI Japan ETF
        "description": "Japanese equities"
    }
}

# Geographic ETF exposure is an independent dimension and must not be added
# to GICS industry allocations. Normalize the industry midpoints to 100%.
INDUSTRY_SECTORS = tuple(list(STANDARD_SECTORS)[:11])
INDUSTRY_TARGET_TOTAL = sum(
    sum(STANDARD_SECTORS[sector]["target_range"]) / 2
    for sector in INDUSTRY_SECTORS
)


def parse_sector_targets(raw):
    """Validate a complete user policy; unspecified sectors get a zero target."""
    if raw is None:
        return None
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(values, dict) or not values or set(values) - set(INDUSTRY_SECTORS):
            raise ValueError()
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values.values()):
            raise ValueError()
        targets = {sector: float(values.get(sector, 0)) for sector in INDUSTRY_SECTORS}
        if any(not math.isfinite(v) or v < 0 or v > 100 for v in targets.values()):
            raise ValueError()
        if not math.isclose(sum(targets.values()), 100.0, abs_tol=0.01):
            raise ValueError()
        return targets
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Sector targets must be industry percentages between 0 and 100 totaling 100%") from None


def default_sector_targets():
    return {sector: sum(STANDARD_SECTORS[sector]['target_range']) / 2 / INDUSTRY_TARGET_TOTAL * 100
            for sector in INDUSTRY_SECTORS}


# Sector mapping (Yahoo Finance → GICS)
SECTOR_MAPPING = {
    # Technology
    "Technology": "Technology",
    "Information Technology": "Technology",
    "Software": "Technology",
    "Hardware": "Technology",
    "Semiconductors": "Technology",

    # Healthcare
    "Healthcare": "Healthcare",
    "Biotechnology": "Healthcare",
    "Medical Devices": "Healthcare",
    "Pharmaceuticals": "Healthcare",

    # Financials
    "Financial Services": "Financials",
    "Financials": "Financials",
    "Banks": "Financials",
    "Insurance": "Financials",
    "Capital Markets": "Financials",

    # Consumer Discretionary
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Discretionary": "Consumer Discretionary",
    "Retail": "Consumer Discretionary",
    "Automotive": "Consumer Discretionary",

    # Communication Services
    "Communication Services": "Communication Services",
    "Telecommunications": "Communication Services",
    "Media": "Communication Services",
    "Entertainment": "Communication Services",

    # Industrials
    "Industrials": "Industrials",
    "Aerospace & Defense": "Industrials",
    "Construction": "Industrials",
    "Machinery": "Industrials",

    # Consumer Staples
    "Consumer Defensive": "Consumer Staples",
    "Consumer Staples": "Consumer Staples",
    "Food & Beverage": "Consumer Staples",
    "Household Products": "Consumer Staples",

    # Energy
    "Energy": "Energy",
    "Oil & Gas": "Energy",
    "Renewable Energy": "Energy",

    # Utilities
    "Utilities": "Utilities",
    "Electric Utilities": "Utilities",
    "Water Utilities": "Utilities",

    # Real Estate
    "Real Estate": "Real Estate",
    "REITs": "Real Estate",
    "Real Estate Services": "Real Estate",

    # Materials
    "Basic Materials": "Materials",
    "Materials": "Materials",
    "Chemicals": "Materials",
    "Metals & Mining": "Materials"
}


# ETF Sector Mapping (Yahoo Finance doesn't return sector for ETFs)
# Maps base symbol (without exchange suffix) → sector classification
ETF_SECTOR_MAPPING = {
    # Diversified World ETFs
    "IWDA": "Diversified",      # iShares Core MSCI World UCITS ETF
    "ACWI": "Diversified",      # iShares MSCI ACWI ETF
    "WORLD": "Diversified",     # UBS MSCI World UCITS ETF
    "VT": "Diversified",        # Vanguard Total World Stock ETF
    "CSPX": "Diversified",      # iShares Core S&P 500 UCITS ETF (European)

    # Sector-Specific ETFs
    "ITEK": "Technology",       # HAN-GINS Tech Megatrend Equal Weight UCITS ETF
    "BTEC": "Healthcare",       # iShares NASDAQ US Biotechnology UCITS ETF
    "SMH": "Technology",        # iShares MSCI Global Semiconductors ETF (LSE)
    "XLK": "Technology",        # SPDR Technology Select Sector
    "XLV": "Healthcare",        # SPDR Healthcare Select Sector
    "XLF": "Financials",        # SPDR Financials Select Sector
    "XLY": "Consumer Discretionary",  # SPDR Consumer Discretionary
    "XLC": "Communication Services",  # SPDR Communication Services
    "XLI": "Industrials",       # SPDR Industrials Select Sector
    "XLP": "Consumer Staples",  # SPDR Consumer Staples
    "XLE": "Energy",            # SPDR Energy Select Sector
    "XLU": "Utilities",         # SPDR Utilities Select Sector
    "XLRE": "Real Estate",      # SPDR Real Estate Select Sector
    "XLB": "Materials",         # SPDR Materials Select Sector

    # Geographic ETFs (NEW - Oct 2025)
    "VGK": "Europe",            # Vanguard FTSE Europe ETF
    "VPL": "Asia Pacific",      # Vanguard FTSE Pacific ETF
    "VWO": "Emerging Markets",  # Vanguard FTSE Emerging Markets ETF
    "EIMI": "Emerging Markets", # iShares Core MSCI EM IMI UCITS ETF (Swiss)
    "EWJ": "Japan",             # iShares MSCI Japan ETF
    "FEZ": "Europe",            # SPDR Euro Stoxx 50 ETF
    "EWU": "Europe",            # iShares MSCI United Kingdom ETF
    "EWG": "Europe",            # iShares MSCI Germany ETF
    "EWQ": "Europe",            # iShares MSCI France ETF
    "EWI": "Europe",            # iShares MSCI Italy ETF
    "EWP": "Europe",            # iShares MSCI Spain ETF
    "ASHR": "Emerging Markets", # Xtrackers Harvest CSI 300 China A-Shares
    "INDA": "Emerging Markets", # iShares MSCI India ETF
    "EWZ": "Emerging Markets",  # iShares MSCI Brazil ETF

    # India / Emerging Markets ETFs
    "FLXI": "Emerging Markets",  # Franklin FTSE India UCITS ETF

    # Alternative Assets
    "AGGS": "Fixed Income",     # iShares Core Global Aggregate Bond UCITS ETF
    "XGDU": "Commodities",      # Xtrackers IE Physical Gold ETC
    "GLD": "Commodities",       # SPDR Gold Shares
    "SLV": "Commodities",       # iShares Silver Trust
    "DBC": "Commodities",       # Invesco DB Commodity Index
}


class OpportunityScanner:
    """
    Scans portfolio for sector gaps and scoring opportunities.

    Methodology:
    - Compare current sector allocation vs S&P 500 standard sectors
    - Detect gaps (0% or underweight sectors)
    - Score each gap: Momentum 40% + Value 30% + Diversification 30%
    """

    def __init__(self):
        """Initialize scanner with sector analyzer"""
        self.sector_analyzer = SectorAnalyzer()

    @staticmethod
    def _finite_number(value: Any, fallback: Optional[float]) -> Optional[float]:
        """Keep partial market-data responses from leaking NaN into the API."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if math.isfinite(number) else fallback

    async def scan_opportunities(
        self,
        positions: List[Dict[str, Any]],
        horizon: str = "medium",
        min_gap_pct: float = 5.0,
        target_allocations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Scan portfolio for sector gaps and opportunities.

        Args:
            positions: List of portfolio positions with sector info
            horizon: Time horizon (short/medium/long)
            min_gap_pct: Minimum gap percentage to consider (default 5%)

        Returns:
            Dict with gaps, scored opportunities, and recommendations
        """
        try:
            logger.info(f" Scanning opportunities for {len(positions)} positions (horizon: {horizon})")

            # 1. Extract current sector allocation
            classified_positions = self._classify_positions(positions)
            current_allocation = self._extract_sector_allocation(classified_positions)
            logger.debug(f"Current allocation: {current_allocation}")

            # 2. Detect gaps vs standard sectors
            targets = parse_sector_targets(target_allocations)
            unclassified_pct = max(0.0, 100.0 - sum(current_allocation.get(s, 0.0) for s in INDUSTRY_SECTORS))
            gaps = self._detect_gaps(current_allocation, min_gap_pct, targets, unclassified_pct)
            logger.info(f"Detected {len(gaps)} sector gaps")

            # 3. Score each gap
            scored_gaps = []
            for gap in gaps:
                score = await self._score_gap(gap, horizon)
                if score.get("score") is not None:
                    scored_gaps.append({**gap, **score})

            # Sort by score (descending)
            scored_gaps.sort(key=lambda x: x.get("score", 0), reverse=True)

            # 4. Get top opportunities (top 5 gaps)
            top_gaps = scored_gaps[:5]

            logger.info(f" Scan complete: {len(scored_gaps)} gaps scored, top {len(top_gaps)} selected")

            return {
                "all_gaps": scored_gaps,
                "top_gaps": top_gaps,
                "current_allocation": current_allocation,
                "_classified_positions": classified_positions,
                "sector_assessment": self._describe_sector_bounds(current_allocation, targets, unclassified_pct),
                "target_allocations": targets if targets is not None else default_sector_targets(),
                "target_source": "user" if targets is not None else "generic_reference",
                "classification_coverage": (100.0 - unclassified_pct) / 100.0,
                "unclassified_pct": unclassified_pct,
                "horizon_details": OPPORTUNITY_HORIZONS[horizon].metadata(),
                "scan_time": datetime.now().isoformat(),
                "horizon": horizon
            }

        except Exception as e:
            logger.error(f" Error scanning opportunities: {e}", exc_info=True)
            raise

    def _enrich_position_with_sector(self, symbol: str) -> str:
        """
        Enrich position with sector from Yahoo Finance.
        Handles European stock symbols from Saxo Bank format (SYMBOL:xexchange).

        Args:
            symbol: Stock ticker (may be in Saxo format like "SLHn:xvtx")

        Returns:
            Sector name or "Unknown"
        """
        try:
            import yfinance as yf
            from services.risk.bourse.data_fetcher import BourseDataFetcher
            from services.ml.bourse.currency_detector import CurrencyExchangeDetector
            base_symbol, _, mic = symbol.partition(':')
            if base_symbol.upper() in ETF_SECTOR_MAPPING:
                return ETF_SECTOR_MAPPING[base_symbol.upper()]
            hint = BourseDataFetcher.MIC_TO_EXCHANGE_HINT.get(mic.lower()) if mic else None
            if mic and hint is None:
                return "Unknown"
            yahoo_symbol, _, _ = CurrencyExchangeDetector().detect_currency_and_exchange(
                base_symbol, exchange_hint=hint)

            # Try fetching with converted symbol
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            # Try different sector fields
            sector = info.get('sector') or info.get('sectorKey') or info.get('industry')

            if sector:
                logger.info(f" {symbol} → {sector}")
                return sector
            else:
                logger.info(f" {yahoo_symbol} → No sector found in Yahoo Finance")
                return "Unknown"

        except Exception as e:
            logger.info(f" {symbol} → Error fetching sector: {e}")
            return "Unknown"

    def _classify_positions(self, positions):
        """One classified copy for scan and impact; preserve the source snapshot."""
        classified = []
        for position in positions:
            row = dict(position)
            raw = row.get("sector")
            if not raw or raw == "Unknown":
                symbol = row.get("symbol") or row.get("instrument_id")
                raw = self._enrich_position_with_sector(symbol) if symbol else None
            raw = raw or "Unknown"
            sector = SECTOR_MAPPING.get(raw, raw)
            if sector not in STANDARD_SECTORS:
                sector = next((mapped for label, mapped in SECTOR_MAPPING.items()
                               if label.lower() in raw.lower()), "Other")
            row["sector"] = sector
            classified.append(row)
        return classified

    def _extract_sector_allocation(self, positions):
        classified = self._classify_positions(positions)
        values = {}
        for row in classified:
            value = row.get("market_value", 0) or row.get("market_value_usd", 0)
            values[row["sector"]] = values.get(row["sector"], 0) + value
        total = sum(values.values())
        return {sector: value / total * 100 for sector, value in values.items()} if total > 0 else {}

    def _describe_sector_bounds(self, allocation, targets, unclassified_pct):
        """Bounds express unknown fund exposure, never a ranked buy signal."""
        targets = targets if targets is not None else default_sector_targets()
        rows = []
        for sector in INDUSTRY_SECTORS:
            known = allocation.get(sector, 0.0)
            upper = min(100.0, known + unclassified_pct)
            minimum_gap = max(0.0, targets[sector] - upper)
            possible_gap = max(0.0, targets[sector] - known)
            rows.append({"sector": sector, "known_pct": known, "possible_total_pct": upper,
                         "target_pct": targets[sector], "minimum_gap_pct": minimum_gap,
                         "possible_gap_pct": possible_gap,
                         "status": "Verified underweight" if minimum_gap > 0 else
                                   "Indeterminate" if possible_gap > 0 else "No underweight"})
        return rows

    def _detect_gaps(
        self,
        current_allocation: Dict[str, float],
        min_gap_pct: float,
        target_allocations: Optional[Dict[str, float]] = None,
        unclassified_pct: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect sector gaps (missing or underweight sectors).

        Args:
            current_allocation: Current sector allocation
            min_gap_pct: Minimum gap to consider

        Returns:
            List of gaps with sector info
        """
        gaps = []
        targets = target_allocations if target_allocations is not None else default_sector_targets()

        for sector in INDUSTRY_SECTORS:
            info = STANDARD_SECTORS[sector]
            current = current_allocation.get(sector, 0.0)
            target = targets[sector]

            # Unknown ETF/sector exposure may already fill the apparent gap.
            gap_pct = target - current - unclassified_pct

            # Only consider gaps above threshold
            if gap_pct > 0 and gap_pct >= min_gap_pct:
                gaps.append({
                    "sector": sector,
                    "current_pct": round(current, 2),
                    "target_pct": round(target, 2),
                    "gap_pct": round(gap_pct, 2),
                    "gap_kind": "minimum_verified_gap",
                    "etf": info["etf"],
                    "description": info["description"]
                })

        return gaps

    async def _score_gap(
        self,
        gap: Dict[str, Any],
        horizon: str
    ) -> Dict[str, Any]:
        """
        Score a sector gap using 3-pillar approach.

        Pillars:
        - Momentum (40%): Price momentum, relative strength
        - Value (30%): Valuation metrics
        - Diversification (30%): Correlation with existing portfolio

        Args:
            gap: Gap info (sector, gap_pct, etc.)
            horizon: Time horizon

        Returns:
            Dict with score breakdown
        """
        try:
            sector = gap["sector"]
            etf = gap["etf"]

            # Analyze sector ETF to get metrics
            analysis = await self.sector_analyzer.analyze_sector(etf, horizon)

            if not analysis:
                logger.warning(f"No analysis available for {sector} ({etf})")
                return {
                    "momentum_score": None,
                    "value_score": None,
                    "diversification_score": None,
                    "score": None,
                    "confidence": 0.0
                }

            # Extract scores
            momentum_score = self._finite_number(analysis.get("momentum_score"), None)
            value_score = self._finite_number(analysis.get("value_score"), None)
            diversification_score = self._finite_number(analysis.get("diversification_score"), None)
            if momentum_score is None:
                return {"momentum_score": None, "value_score": value_score,
                        "diversification_score": diversification_score, "score": None, "confidence": 0.0}

            # Weighted average (Momentum 40%, Value 30%, Diversification 30%)
            components = [(momentum_score, 0.40), (value_score, 0.30), (diversification_score, 0.30)]
            available = [(value, weight) for value, weight in components if value is not None]
            score = sum(value * weight for value, weight in available) / sum(weight for _, weight in available)

            # Confidence based on data quality
            confidence = min(1.0, max(0.0, self._finite_number(analysis.get("confidence"), 0.7)))

            return {
                "momentum_score": round(momentum_score, 1),
                "value_score": round(value_score, 1) if value_score is not None else None,
                "diversification_score": round(diversification_score, 1) if diversification_score is not None else None,
                "score": round(score, 1),
                "confidence": round(confidence, 2),
                "score_components_available": [name for name, value in (("momentum", momentum_score), ("value", value_score), ("diversification", diversification_score)) if value is not None],
                "analysis": analysis
            }

        except Exception as e:
            logger.error(f"Error scoring gap {gap}: {e}", exc_info=True)
            # Unverified prices must not create a ranked opportunity.
            return {
                "momentum_score": None,
                "value_score": None,
                "diversification_score": None,
                "score": None,
                "confidence": 0.0
            }
