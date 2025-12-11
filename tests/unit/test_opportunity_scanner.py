"""
Unit tests for Market Opportunities Scanner

Tests the OpportunityScanner module responsible for:
- Sector allocation extraction (with Yahoo Finance enrichment)
- Gap detection (industry + geographic sectors)
- 3-pillar scoring (Momentum 40%, Value 30%, Diversification 30%)
- Complete scan workflow

Author: SmartFolio Team
Date: December 2025 - Sprint 5
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from services.ml.bourse.opportunity_scanner import (
    OpportunityScanner,
    STANDARD_SECTORS,
    SECTOR_MAPPING,
    ETF_SECTOR_MAPPING
)


@pytest.fixture
def scanner():
    """Create OpportunityScanner instance"""
    return OpportunityScanner()


@pytest.fixture
def sample_positions():
    """Sample portfolio positions for testing"""
    return [
        {
            "symbol": "AAPL",
            "market_value": 10000,
            "sector": "Technology"
        },
        {
            "symbol": "MSFT",
            "market_value": 8000,
            "sector": "Technology"
        },
        {
            "symbol": "JNJ",
            "market_value": 5000,
            "sector": "Healthcare"
        },
        {
            "symbol": "JPM",
            "market_value": 3000,
            "sector": "Financials"
        }
    ]


@pytest.fixture
def empty_portfolio():
    """Empty portfolio for edge case testing"""
    return []


@pytest.fixture
def portfolio_with_unknown_sectors():
    """Portfolio with unknown sectors requiring enrichment"""
    return [
        {
            "symbol": "AAPL",
            "market_value": 10000,
            "sector": "Unknown"
        },
        {
            "symbol": "MSFT",
            "market_value": 8000,
            "sector": None
        }
    ]


class TestSectorAllocationExtraction:
    """Tests for _extract_sector_allocation method"""

    def test_basic_sector_allocation(self, scanner, sample_positions):
        """Test basic sector allocation calculation"""
        allocation = scanner._extract_sector_allocation(sample_positions)

        # Total portfolio = 26,000
        # Technology: 18,000 / 26,000 = 69.23%
        # Healthcare: 5,000 / 26,000 = 19.23%
        # Financials: 3,000 / 26,000 = 11.54%
        assert allocation["Technology"] == pytest.approx(69.23, abs=0.01)
        assert allocation["Healthcare"] == pytest.approx(19.23, abs=0.01)
        assert allocation["Financials"] == pytest.approx(11.54, abs=0.01)

    def test_empty_portfolio_returns_empty_dict(self, scanner, empty_portfolio):
        """Test that empty portfolio returns empty allocation"""
        allocation = scanner._extract_sector_allocation(empty_portfolio)
        assert allocation == {}

    def test_zero_value_portfolio_returns_empty_dict(self, scanner):
        """Test that portfolio with zero total value returns empty allocation"""
        positions = [
            {"symbol": "AAPL", "market_value": 0, "sector": "Technology"}
        ]
        allocation = scanner._extract_sector_allocation(positions)
        assert allocation == {}

    def test_sector_mapping_normalization(self, scanner):
        """Test that Yahoo Finance sectors are mapped to GICS sectors"""
        positions = [
            {"symbol": "AAPL", "market_value": 10000, "sector": "Information Technology"},
            {"symbol": "JNJ", "market_value": 5000, "sector": "Biotechnology"}
        ]
        allocation = scanner._extract_sector_allocation(positions)

        # Should be mapped to standard GICS sectors
        assert "Technology" in allocation
        assert "Healthcare" in allocation
        assert allocation["Technology"] == pytest.approx(66.67, abs=0.01)
        assert allocation["Healthcare"] == pytest.approx(33.33, abs=0.01)

    def test_etf_sector_mapping(self, scanner):
        """Test that ETFs are correctly mapped to sectors"""
        positions = [
            {"symbol": "XLK", "market_value": 10000, "sector": "Unknown"},
            {"symbol": "VGK", "market_value": 5000, "sector": "Unknown"}
        ]

        with patch.object(scanner, '_enrich_position_with_sector') as mock_enrich:
            # Mock returns ETF sector from mapping
            mock_enrich.side_effect = lambda sym: ETF_SECTOR_MAPPING.get(sym, "Unknown")

            allocation = scanner._extract_sector_allocation(positions)

            # XLK → Technology, VGK → Europe
            assert allocation.get("Technology", 0) == pytest.approx(66.67, abs=0.01)
            assert allocation.get("Europe", 0) == pytest.approx(33.33, abs=0.01)

    def test_unknown_sectors_grouped_as_other(self, scanner):
        """Test that unmappable sectors are grouped as 'Other'"""
        positions = [
            {"symbol": "AAPL", "market_value": 10000, "sector": "Technology"},
            {"symbol": "WEIRD", "market_value": 2000, "sector": "Cryptocurrency Mining"}
        ]

        allocation = scanner._extract_sector_allocation(positions)

        assert "Technology" in allocation
        assert allocation.get("Other", 0) == pytest.approx(16.67, abs=0.01)

    @patch('yfinance.Ticker')
    def test_enrich_position_with_sector_yahoo_finance(self, mock_ticker_class, scanner):
        """Test Yahoo Finance enrichment for missing sectors"""
        # Mock Yahoo Finance response
        mock_ticker = Mock()
        mock_ticker.info = {"sector": "Technology"}
        mock_ticker_class.return_value = mock_ticker

        sector = scanner._enrich_position_with_sector("AAPL")
        assert sector == "Technology"
        mock_ticker_class.assert_called_once_with("AAPL")

    @patch('yfinance.Ticker')
    def test_enrich_position_saxo_format_conversion(self, mock_ticker_class, scanner):
        """Test conversion of Saxo format symbols to Yahoo Finance format"""
        # Mock Yahoo Finance response
        mock_ticker = Mock()
        mock_ticker.info = {"sector": "Financials"}
        mock_ticker_class.return_value = mock_ticker

        # Test Saxo format: "SYMBOL:xexchange" → "SYMBOL.SUFFIX"
        sector = scanner._enrich_position_with_sector("UBS:xvtx")

        # Should convert to UBS.SW (Swiss exchange)
        mock_ticker_class.assert_called_once_with("UBS.SW")
        assert sector == "Financials"

    @patch('yfinance.Ticker')
    def test_enrich_position_handles_errors_gracefully(self, mock_ticker_class, scanner):
        """Test that enrichment errors return 'Unknown' instead of crashing"""
        # Mock Yahoo Finance raising exception
        mock_ticker_class.side_effect = Exception("API Error")

        sector = scanner._enrich_position_with_sector("INVALID")
        assert sector == "Unknown"


class TestGapDetection:
    """Tests for _detect_gaps method"""

    def test_detect_missing_sectors(self, scanner):
        """Test detection of completely missing sectors (0% allocation)"""
        current_allocation = {
            "Technology": 50.0,
            "Healthcare": 30.0
        }

        gaps = scanner._detect_gaps(current_allocation, min_gap_pct=5.0)

        # Should detect gaps for all missing sectors with target > 5%
        gap_sectors = [g["sector"] for g in gaps]
        assert "Financials" in gap_sectors
        assert "Consumer Discretionary" in gap_sectors
        assert "Europe" in gap_sectors  # Geographic sector

    def test_detect_underweight_sectors(self, scanner):
        """Test detection of underweight sectors"""
        current_allocation = {
            "Technology": 10.0,  # Target: (15+30)/2 = 22.5% → Gap: 12.5%
            "Healthcare": 5.0,   # Target: (10+18)/2 = 14% → Gap: 9%
        }

        gaps = scanner._detect_gaps(current_allocation, min_gap_pct=5.0)

        # Find Technology and Healthcare gaps
        tech_gap = next((g for g in gaps if g["sector"] == "Technology"), None)
        health_gap = next((g for g in gaps if g["sector"] == "Healthcare"), None)

        assert tech_gap is not None
        assert tech_gap["gap_pct"] == pytest.approx(12.5, abs=0.1)
        assert tech_gap["current_pct"] == 10.0
        assert tech_gap["target_pct"] == 22.5

        assert health_gap is not None
        assert health_gap["gap_pct"] == pytest.approx(9.0, abs=0.1)

    def test_min_gap_threshold_filtering(self, scanner):
        """Test that min_gap_pct filters out small gaps"""
        current_allocation = {
            "Technology": 20.0,  # Target: 22.5% → Gap: 2.5% (below 5% threshold)
        }

        gaps = scanner._detect_gaps(current_allocation, min_gap_pct=5.0)

        # Technology gap should NOT be included (2.5% < 5%)
        tech_gaps = [g for g in gaps if g["sector"] == "Technology"]
        assert len(tech_gaps) == 0

    def test_gap_includes_etf_and_description(self, scanner):
        """Test that detected gaps include ETF and description"""
        current_allocation = {}

        gaps = scanner._detect_gaps(current_allocation, min_gap_pct=5.0)

        tech_gap = next((g for g in gaps if g["sector"] == "Technology"), None)
        assert tech_gap["etf"] == "XLK"
        assert tech_gap["description"] == "Information Technology"

        europe_gap = next((g for g in gaps if g["sector"] == "Europe"), None)
        assert europe_gap["etf"] == "VGK"
        assert europe_gap["description"] == "European developed markets"

    def test_geographic_sectors_detected(self, scanner):
        """Test that geographic sectors are detected correctly"""
        current_allocation = {
            "Technology": 30.0,
            "Healthcare": 20.0
        }

        gaps = scanner._detect_gaps(current_allocation, min_gap_pct=5.0)

        geographic_gaps = [g for g in gaps if g["sector"] in ["Europe", "Asia Pacific", "Emerging Markets", "Japan"]]
        assert len(geographic_gaps) == 4  # All 4 geographic sectors should be detected


class TestGapScoring:
    """Tests for _score_gap method"""

    @pytest.mark.asyncio
    async def test_score_gap_basic(self, scanner):
        """Test basic gap scoring with mock sector analyzer"""
        gap = {
            "sector": "Technology",
            "gap_pct": 15.0,
            "etf": "XLK"
        }

        # Mock sector analyzer
        mock_analysis = {
            "momentum_score": 70.0,
            "value_score": 60.0,
            "diversification_score": 50.0,
            "confidence": 0.85
        }
        scanner.sector_analyzer.analyze_sector = AsyncMock(return_value=mock_analysis)

        result = await scanner._score_gap(gap, horizon="medium")

        # Expected score: 0.40×70 + 0.30×60 + 0.30×50 = 28 + 18 + 15 = 61
        assert result["score"] == pytest.approx(61.0, abs=0.1)
        assert result["momentum_score"] == 70.0
        assert result["value_score"] == 60.0
        assert result["diversification_score"] == 50.0
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_score_gap_handles_missing_analysis(self, scanner):
        """Test that scoring handles missing sector analysis gracefully"""
        gap = {"sector": "Unknown", "gap_pct": 10.0, "etf": "UNKNOWN"}

        # Mock sector analyzer returning None
        scanner.sector_analyzer.analyze_sector = AsyncMock(return_value=None)

        result = await scanner._score_gap(gap, horizon="medium")

        # Should return neutral scores (50)
        assert result["score"] == 50.0
        assert result["momentum_score"] == 50.0
        assert result["value_score"] == 50.0
        assert result["diversification_score"] == 50.0
        assert result["confidence"] == 0.3  # Low confidence

    @pytest.mark.asyncio
    async def test_score_gap_handles_analyzer_exception(self, scanner):
        """Test that scoring handles analyzer exceptions gracefully"""
        gap = {"sector": "Technology", "gap_pct": 15.0, "etf": "XLK"}

        # Mock sector analyzer raising exception
        scanner.sector_analyzer.analyze_sector = AsyncMock(side_effect=Exception("API Error"))

        result = await scanner._score_gap(gap, horizon="medium")

        # Should return neutral scores without crashing
        assert result["score"] == 50.0
        assert result["confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_score_gap_weights_correct(self, scanner):
        """Test that scoring weights are applied correctly"""
        gap = {"sector": "Healthcare", "gap_pct": 10.0, "etf": "XLV"}

        mock_analysis = {
            "momentum_score": 100.0,
            "value_score": 0.0,
            "diversification_score": 0.0,
            "confidence": 1.0
        }
        scanner.sector_analyzer.analyze_sector = AsyncMock(return_value=mock_analysis)

        result = await scanner._score_gap(gap, horizon="short")

        # Expected: 0.40×100 + 0.30×0 + 0.30×0 = 40
        assert result["score"] == pytest.approx(40.0, abs=0.1)


class TestScanOpportunities:
    """Tests for scan_opportunities main workflow"""

    @pytest.mark.asyncio
    async def test_scan_opportunities_complete_workflow(self, scanner, sample_positions):
        """Test complete scan workflow"""
        # Mock sector analyzer for all gaps
        mock_analysis = {
            "momentum_score": 65.0,
            "value_score": 55.0,
            "diversification_score": 60.0,
            "confidence": 0.8
        }
        scanner.sector_analyzer.analyze_sector = AsyncMock(return_value=mock_analysis)

        result = await scanner.scan_opportunities(
            positions=sample_positions,
            horizon="medium",
            min_gap_pct=5.0
        )

        # Check result structure
        assert "all_gaps" in result
        assert "top_gaps" in result
        assert "current_allocation" in result
        assert "scan_time" in result
        assert "horizon" in result

        # Verify gaps are sorted by score
        all_gaps = result["all_gaps"]
        if len(all_gaps) > 1:
            for i in range(len(all_gaps) - 1):
                assert all_gaps[i]["score"] >= all_gaps[i + 1]["score"]

        # Verify top_gaps is limited to 5
        assert len(result["top_gaps"]) <= 5

    @pytest.mark.asyncio
    async def test_scan_opportunities_empty_portfolio(self, scanner, empty_portfolio):
        """Test scan with empty portfolio"""
        scanner.sector_analyzer.analyze_sector = AsyncMock(return_value={
            "momentum_score": 50.0,
            "value_score": 50.0,
            "diversification_score": 50.0,
            "confidence": 0.5
        })

        result = await scanner.scan_opportunities(
            positions=empty_portfolio,
            horizon="medium",
            min_gap_pct=5.0
        )

        # Should detect all standard sectors as gaps
        assert len(result["all_gaps"]) > 0
        assert result["current_allocation"] == {}

    @pytest.mark.asyncio
    async def test_scan_opportunities_different_horizons(self, scanner, sample_positions):
        """Test that different horizons are passed to sector analyzer"""
        calls = []

        async def mock_analyze(etf, horizon):
            calls.append(horizon)
            return {
                "momentum_score": 50.0,
                "value_score": 50.0,
                "diversification_score": 50.0,
                "confidence": 0.5
            }

        scanner.sector_analyzer.analyze_sector = mock_analyze

        await scanner.scan_opportunities(sample_positions, horizon="short", min_gap_pct=5.0)
        assert all(h == "short" for h in calls)

        calls.clear()
        await scanner.scan_opportunities(sample_positions, horizon="long", min_gap_pct=5.0)
        assert all(h == "long" for h in calls)

    @pytest.mark.asyncio
    async def test_scan_opportunities_min_gap_filtering(self, scanner, sample_positions):
        """Test that min_gap_pct parameter filters results correctly"""
        scanner.sector_analyzer.analyze_sector = AsyncMock(return_value={
            "momentum_score": 50.0,
            "value_score": 50.0,
            "diversification_score": 50.0,
            "confidence": 0.5
        })

        # Scan with high threshold
        result_high = await scanner.scan_opportunities(
            positions=sample_positions,
            horizon="medium",
            min_gap_pct=20.0
        )

        # Scan with low threshold
        result_low = await scanner.scan_opportunities(
            positions=sample_positions,
            horizon="medium",
            min_gap_pct=1.0
        )

        # Low threshold should find more gaps
        assert len(result_low["all_gaps"]) >= len(result_high["all_gaps"])


class TestConstants:
    """Tests for module constants and mappings"""

    def test_standard_sectors_completeness(self):
        """Test that STANDARD_SECTORS includes all required sectors"""
        # 11 GICS sectors
        industry_sectors = [
            "Technology", "Healthcare", "Financials", "Consumer Discretionary",
            "Communication Services", "Industrials", "Consumer Staples",
            "Energy", "Utilities", "Real Estate", "Materials"
        ]

        # 4 Geographic sectors
        geographic_sectors = ["Europe", "Asia Pacific", "Emerging Markets", "Japan"]

        all_sectors = industry_sectors + geographic_sectors

        for sector in all_sectors:
            assert sector in STANDARD_SECTORS
            assert "target_range" in STANDARD_SECTORS[sector]
            assert "etf" in STANDARD_SECTORS[sector]
            assert "description" in STANDARD_SECTORS[sector]

    def test_sector_mapping_consistency(self):
        """Test that SECTOR_MAPPING maps to valid STANDARD_SECTORS"""
        for yahoo_sector, gics_sector in SECTOR_MAPPING.items():
            # All mapped sectors should exist in STANDARD_SECTORS (or be "Other")
            if gics_sector != "Other":
                assert gics_sector in STANDARD_SECTORS, f"{yahoo_sector} maps to invalid sector {gics_sector}"

    def test_etf_sector_mapping_consistency(self):
        """Test that ETF_SECTOR_MAPPING maps to valid sectors"""
        valid_sectors = list(STANDARD_SECTORS.keys()) + ["Diversified", "Fixed Income", "Commodities"]

        for etf, sector in ETF_SECTOR_MAPPING.items():
            assert sector in valid_sectors, f"ETF {etf} maps to invalid sector {sector}"

    def test_geographic_sectors_have_correct_etfs(self):
        """Test that geographic sectors have correct ETF mappings"""
        assert STANDARD_SECTORS["Europe"]["etf"] == "VGK"
        assert STANDARD_SECTORS["Asia Pacific"]["etf"] == "VPL"
        assert STANDARD_SECTORS["Emerging Markets"]["etf"] == "VWO"
        assert STANDARD_SECTORS["Japan"]["etf"] == "EWJ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
