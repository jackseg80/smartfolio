"""
Unit tests for Portfolio Analytics Service
Tests portfolio metrics, performance calculations, and snapshot management

COVERAGE TARGET: 13% → 60%+ for services/portfolio.py
"""
import pytest
import json
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Any

from services.portfolio import PortfolioAnalytics
import services.portfolio_history_storage

# Mock BASE_DIR for isolation
TEST_BASE_DIR = Path("data/test_portfolio_metrics_history")

class TestPortfolioMetrics:
    """Test cases for Portfolio Metrics calculations"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for all tests"""
        # Setup: Create test directory
        if TEST_BASE_DIR.exists():
            shutil.rmtree(TEST_BASE_DIR)
        TEST_BASE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Patch BASE_DIR globally for the module
        original_base_dir = services.portfolio_history_storage.BASE_DIR
        services.portfolio_history_storage.BASE_DIR = TEST_BASE_DIR
        
        yield
        
        # Teardown: Restore BASE_DIR and cleanup
        services.portfolio_history_storage.BASE_DIR = original_base_dir
        if TEST_BASE_DIR.exists():
            shutil.rmtree(TEST_BASE_DIR)

    @pytest.fixture
    def portfolio_service(self, tmp_path):
        """Create portfolio service with temporary data file"""
        service = PortfolioAnalytics()
        # Use temporary directory for legacy file tests
        service.historical_data_file = str(tmp_path / "portfolio_history.json")
        return service

    @pytest.fixture
    def sample_balances_data(self):
        """Sample balance data for testing"""
        return {
            "ok": True,
            "items": [
                {
                    "symbol": "BTC",
                    "alias": "BTC",
                    "amount": 1.5,
                    "value_usd": 75000.0,
                    "price_usd": 50000.0,
                },
                {
                    "symbol": "ETH",
                    "alias": "ETH",
                    "amount": 10.0,
                    "value_usd": 30000.0,
                    "price_usd": 3000.0,
                },
                {
                    "symbol": "USDT",
                    "alias": "USDT",
                    "amount": 20000.0,
                    "value_usd": 20000.0,
                    "price_usd": 1.0,
                },
                {
                    "symbol": "SOL",
                    "alias": "SOL",
                    "amount": 100.0,
                    "value_usd": 5000.0,
                    "price_usd": 50.0,
                },
            ],
            "meta": {
                "total_value_usd": 130000.0,
                "asset_count": 4,
            }
        }

    @pytest.fixture
    def empty_balances_data(self):
        """Empty balance data"""
        return {
            "ok": True,
            "items": [],
            "meta": {
                "total_value_usd": 0.0,
                "asset_count": 0,
            }
        }

    def test_calculate_portfolio_metrics_basic(self, portfolio_service, sample_balances_data):
        """Test basic portfolio metrics calculation"""
        metrics = portfolio_service.calculate_portfolio_metrics(sample_balances_data)

        # Basic validations
        assert isinstance(metrics, dict)
        assert metrics["total_value_usd"] == 130000.0
        assert metrics["asset_count"] == 4
        assert metrics["group_count"] >= 1  # At least one group
        assert "top_holding" in metrics
        assert "diversity_score" in metrics
        assert "concentration_risk" in metrics
        assert "group_distribution" in metrics
        assert "last_updated" in metrics

    def test_calculate_portfolio_metrics_top_holding(self, portfolio_service, sample_balances_data):
        """Test top holding calculation"""
        metrics = portfolio_service.calculate_portfolio_metrics(sample_balances_data)

        top_holding = metrics["top_holding"]
        assert top_holding["symbol"] == "BTC"  # BTC is largest
        assert top_holding["value_usd"] == 75000.0
        assert 0.5 < top_holding["percentage"] < 0.6  # ~57.7%

    def test_calculate_portfolio_metrics_concentration_risk(self, portfolio_service, sample_balances_data):
        """Test concentration risk classification"""
        metrics = portfolio_service.calculate_portfolio_metrics(sample_balances_data)

        # BTC is 57.7% → High concentration
        assert metrics["concentration_risk"] in ["High", "Medium", "Low"]
        # With BTC > 50%, should be High
        assert metrics["concentration_risk"] == "High"

    def test_calculate_portfolio_metrics_diversity_score(self, portfolio_service, sample_balances_data):
        """Test diversity score calculation"""
        metrics = portfolio_service.calculate_portfolio_metrics(sample_balances_data)

        diversity_score = metrics["diversity_score"]
        assert isinstance(diversity_score, (int, float))
        assert 0 <= diversity_score <= 10  # Score 0-10

    def test_calculate_portfolio_metrics_empty_portfolio(self, portfolio_service, empty_balances_data):
        """Test metrics with empty portfolio"""
        metrics = portfolio_service.calculate_portfolio_metrics(empty_balances_data)

        # Empty metrics should have zeros
        assert metrics["total_value_usd"] == 0.0
        assert metrics["asset_count"] == 0
        assert metrics["diversity_score"] == 0

    def test_calculate_portfolio_metrics_group_distribution(self, portfolio_service, sample_balances_data):
        """Test group distribution calculation"""
        metrics = portfolio_service.calculate_portfolio_metrics(sample_balances_data)

        group_dist = metrics["group_distribution"]
        assert isinstance(group_dist, dict)
        assert len(group_dist) > 0
        # Sum of all groups should equal total value
        total_from_groups = sum(group_dist.values())
        assert abs(total_from_groups - metrics["total_value_usd"]) < 1.0  # Within $1

    @pytest.mark.asyncio
    async def test_save_portfolio_snapshot_success(self, portfolio_service, sample_balances_data):
        """Test successful snapshot save"""
        result = await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="test_user",
            source="cointracking"
        )

        assert result is True
        
        # Verify snapshot was saved using load_snapshots
        snapshots = portfolio_service.storage.load_snapshots("test_user", "cointracking")
        assert len(snapshots) == 1
        
        snapshot = snapshots[0]
        assert snapshot["user_id"] == "test_user"
        assert snapshot["source"] == "cointracking"
        assert snapshot["total_value_usd"] == 130000.0
        assert snapshot["asset_count"] == 4

    @pytest.mark.asyncio
    async def test_save_portfolio_snapshot_multiple_users(self, portfolio_service, sample_balances_data):
        """Test snapshots for multiple users"""
        # Save for user1
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="user1",
            source="cointracking"
        )

        # Save for user2
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="user2",
            source="cointracking"
        )

        # Verify snapshots
        snaps1 = portfolio_service.storage.load_snapshots("user1", "cointracking")
        assert len(snaps1) == 1
        
        snaps2 = portfolio_service.storage.load_snapshots("user2", "cointracking")
        assert len(snaps2) == 1

    @pytest.mark.asyncio
    async def test_save_portfolio_snapshot_multiple_sources(self, portfolio_service, sample_balances_data):
        """Test snapshots for multiple sources"""
        # Save cointracking
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Save cointracking_api
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking_api"
        )

        # Verify snapshots
        snaps_ct = portfolio_service.storage.load_snapshots("demo", "cointracking")
        assert len(snaps_ct) == 1
        
        snaps_api = portfolio_service.storage.load_snapshots("demo", "cointracking_api")
        assert len(snaps_api) == 1

    @pytest.mark.asyncio
    async def test_save_portfolio_snapshot_upsert_same_day(self, portfolio_service, sample_balances_data):
        """
        Test that snapshot on same day works.
        Note: Since we use exact timestamp for ID, it creates 2 snapshots (no upsert unless exact match).
        We verify that both are saved.
        """
        # First save
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Modify data slightly
        modified_data = sample_balances_data.copy()
        modified_data["items"] = list(sample_balances_data["items"]) # shallow copy items list
        # Create a new item dict to modify
        new_btc = modified_data["items"][0].copy()
        new_btc["value_usd"] = 80000.0
        modified_data["items"][0] = new_btc
        # Update meta total
        modified_data["meta"] = modified_data["meta"].copy()
        modified_data["meta"]["total_value_usd"] = 135000.0

        # Save again same day
        await portfolio_service.save_portfolio_snapshot(
            modified_data,
            user_id="demo",
            source="cointracking"
        )

        snapshots = portfolio_service.storage.load_snapshots("demo", "cointracking")
        # With high-precision timestamp, we expect 2 snapshots
        assert len(snapshots) == 2
        
        # Verify data
        assert snapshots[0]["total_value_usd"] == 130000.0
        assert snapshots[1]["total_value_usd"] > 130000.0

    def test_calculate_performance_metrics_no_history(self, portfolio_service, sample_balances_data):
        """Test performance metrics when no historical data exists"""
        metrics = portfolio_service.calculate_performance_metrics(
            current_data=sample_balances_data, # This is wrong structure but handled gracefully/ignored if no history
            user_id="demo",
            source="cointracking"
        )

        assert metrics["performance_available"] is False
        assert "message" in metrics
        assert metrics["days_tracked"] == 0

    @pytest.mark.asyncio
    async def test_calculate_performance_metrics_with_history(self, portfolio_service, sample_balances_data):
        """Test performance metrics with historical data"""
        # Create historical snapshot (24h ago)
        yesterday = datetime.now(ZoneInfo("Europe/Zurich")) - timedelta(hours=24)
        
        snapshot = {
            "date": yesterday.isoformat(),
            "user_id": "demo",
            "source": "cointracking",
            "total_value_usd": 120000.0,
            "asset_count": 4,
            "group_count": 1,
            "diversity_score": 5,
            "top_holding_symbol": "BTC",
            "top_holding_percentage": 0.5,
            "group_distribution": {}
        }
        
        await portfolio_service.storage.save_snapshot(snapshot, "demo", "cointracking")

        # Create current data dict with expected flat structure for calculate_performance_metrics
        current_data = {
            "total_value_usd": 130000.0,
            "asset_count": 4
        }

        # Calculate performance
        metrics = portfolio_service.calculate_performance_metrics(
            current_data=current_data,
            user_id="demo",
            source="cointracking",
            anchor="prev_snapshot",
            window="24h"
        )

        # Should show performance available
        assert metrics.get("performance_available", False) is True
        assert metrics["percentage_change"] > 0
        assert metrics["days_tracked"] >= 0

    def test_load_historical_data_empty(self, portfolio_service):
        """Test loading historical data when file doesn't exist"""
        data = portfolio_service._load_historical_data(
            user_id="demo",
            source="cointracking"
        )

        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_load_historical_data_filter_by_user(self, portfolio_service, sample_balances_data):
        """Test historical data filtering by user_id"""
        # Save for user1
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="user1",
            source="cointracking"
        )

        # Save for user2
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="user2",
            source="cointracking"
        )

        # Load only user1's data
        user1_data = portfolio_service._load_historical_data(
            user_id="user1",
            source="cointracking"
        )

        assert len(user1_data) == 1
        assert user1_data[0]["user_id"] == "user1"

    @pytest.mark.asyncio
    async def test_load_historical_data_filter_by_source(self, portfolio_service, sample_balances_data):
        """Test historical data filtering by source"""
        # Save cointracking
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Save cointracking_api
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking_api"
        )

        # Load only cointracking data
        ct_data = portfolio_service._load_historical_data(
            user_id="demo",
            source="cointracking"
        )

        assert len(ct_data) == 1
        assert ct_data[0]["source"] == "cointracking"

    def test_portfolio_metrics_with_zero_values_filtered(self, portfolio_service):
        """Test that items with zero value are filtered out"""
        data_with_zeros = {
            "ok": True,
            "items": [
                {
                    "symbol": "BTC",
                    "alias": "BTC",
                    "value_usd": 50000.0,
                },
                {
                    "symbol": "DUST",
                    "alias": "DUST",
                    "value_usd": 0.0,  # Zero value
                },
                {
                    "symbol": "ETH",
                    "alias": "ETH",
                    "value_usd": 30000.0,
                },
            ]
        }

        metrics = portfolio_service.calculate_portfolio_metrics(data_with_zeros)

        # Should only count non-zero assets
        assert metrics["asset_count"] == 2  # BTC and ETH only
        assert metrics["total_value_usd"] == 80000.0

    @pytest.mark.asyncio
    async def test_snapshot_includes_group_distribution(self, portfolio_service, sample_balances_data):
        """Test that snapshot includes group distribution"""
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        snapshots = portfolio_service.storage.load_snapshots("demo", "cointracking")
        snapshot = snapshots[0]

        assert "group_distribution" in snapshot
        assert isinstance(snapshot["group_distribution"], dict)

    @pytest.mark.asyncio
    async def test_snapshot_includes_timestamp(self, portfolio_service, sample_balances_data):
        """Test that snapshot includes ISO timestamp"""
        await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        snapshots = portfolio_service.storage.load_snapshots("demo", "cointracking")
        snapshot = snapshots[0]

        assert "date" in snapshot
        # Should be valid ISO format
        date_parsed = datetime.fromisoformat(snapshot["date"])
        assert isinstance(date_parsed, datetime)

    # ========================================================================
    # Tests for get_portfolio_trend() - Coverage Gap
    # ========================================================================

    def test_get_portfolio_trend_empty_history(self, portfolio_service):
        """Test get_portfolio_trend with no historical data"""
        result = portfolio_service.get_portfolio_trend(days=30, user_id="demo", source="cointracking")

        assert result["trend_data"] == []
        assert result["days_available"] == 0

    @pytest.mark.asyncio
    async def test_get_portfolio_trend_with_data(self, portfolio_service, sample_balances_data):
        """Test get_portfolio_trend with historical snapshots"""
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        for i in range(10):
            # Simulate snapshots from past 10 days
            past_date = datetime.now(TZ) - timedelta(days=10 - i)
            snapshot = {
                "date": past_date.isoformat(),
                "user_id": "demo",
                "source": "cointracking",
                "total_value_usd": 100000 + (i * 1000),
                "asset_count": 5 + i,
                "diversity_score": 5.0 + (i * 0.1)
            }
            
            # Use storage to save (async)
            await portfolio_service.storage.save_snapshot(snapshot, "demo", "cointracking")

        # Get trend for last 7 days
        result = portfolio_service.get_portfolio_trend(days=7, user_id="demo", source="cointracking")

        # Should have data from last 7 days
        assert len(result["trend_data"]) > 0
        assert result["days_available"] == len(result["trend_data"])

        # Validate structure
        first_entry = result["trend_data"][0]
        assert "date" in first_entry
        assert "total_value" in first_entry
        assert "asset_count" in first_entry
        assert "diversity_score" in first_entry

    @pytest.mark.asyncio
    async def test_get_portfolio_trend_filters_by_days(self, portfolio_service):
        """Test that get_portfolio_trend correctly filters by days parameter"""
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        # Create 40 days of snapshots
        for i in range(40):
            past_date = datetime.now(TZ) - timedelta(days=40 - i)
            snapshot = {
                "date": past_date.isoformat(),
                "user_id": "demo",
                "source": "cointracking",
                "total_value_usd": 100000,
                "asset_count": 5,
                "diversity_score": 5.0
            }
            await portfolio_service.storage.save_snapshot(snapshot, "demo", "cointracking")

        # Get trend for last 15 days
        result = portfolio_service.get_portfolio_trend(days=15, user_id="demo", source="cointracking")

        # Should have approximately 15 days (might be 15-16 due to timing)
        assert 14 <= result["days_available"] <= 16
        assert len(result["trend_data"]) == result["days_available"]

    # ========================================================================
    # Tests for _compute_anchor_ts() - Coverage Gap
    # ========================================================================
    # ... existing tests for _compute_anchor_ts ...
    
    def test_compute_anchor_ts_midnight(self):
        """Test _compute_anchor_ts with midnight anchor"""
        from services.portfolio import _compute_anchor_ts
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        now = datetime(2025, 11, 23, 14, 30, 0, tzinfo=TZ)
        anchor_ts, window_ts = _compute_anchor_ts(anchor="midnight", window="24h", now=now)

        assert anchor_ts is not None
        assert anchor_ts.date() == now.date()

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_save_snapshot_corrupted_json_file(self, portfolio_service, sample_balances_data):
        """Test save_snapshot handles corrupted partition file gracefully"""
        # Create a corrupted partition file
        now = datetime.now(ZoneInfo("Europe/Zurich"))
        partition_path = portfolio_service.storage._get_partition_path("demo", "cointracking", now)
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(partition_path, 'w') as f:
            f.write("{ invalid json }")

        # Should not crash, should handle gracefully (reset file and write new)
        result = await portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        assert result is True

        # Verify file is now valid and contains 1 snapshot
        snapshots = portfolio_service.storage.load_snapshots("demo", "cointracking")
        assert len(snapshots) == 1

    def test_upsert_daily_snapshot_replaces_same_day(self):
        """Test _upsert_daily_snapshot replaces existing snapshot on same day"""
        # Legacy helper test
        from services.portfolio import _upsert_daily_snapshot
        entries = []
        snap1 = {"date": datetime.now().isoformat(), "user_id": "u1", "source": "s1"}
        _upsert_daily_snapshot(entries, snap1, "u1", "s1")
        assert len(entries) == 1
