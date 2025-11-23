"""
Unit tests for Portfolio Analytics Service
Tests portfolio metrics, performance calculations, and snapshot management

COVERAGE TARGET: 13% → 60%+ for services/portfolio.py
"""
import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Any

from services.portfolio import PortfolioAnalytics


class TestPortfolioMetrics:
    """Test cases for Portfolio Metrics calculations"""

    @pytest.fixture
    def portfolio_service(self, tmp_path):
        """Create portfolio service with temporary data file"""
        service = PortfolioAnalytics()
        # Use temporary directory for tests
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

    def test_save_portfolio_snapshot_success(self, portfolio_service, sample_balances_data):
        """Test successful snapshot save"""
        result = portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="test_user",
            source="cointracking"
        )

        assert result is True
        assert os.path.exists(portfolio_service.historical_data_file)

        # Verify snapshot was saved
        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            snapshot = data[0]
            assert snapshot["user_id"] == "test_user"
            assert snapshot["source"] == "cointracking"
            assert snapshot["total_value_usd"] == 130000.0
            assert snapshot["asset_count"] == 4

    def test_save_portfolio_snapshot_multiple_users(self, portfolio_service, sample_balances_data):
        """Test snapshots for multiple users"""
        # Save for user1
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="user1",
            source="cointracking"
        )

        # Save for user2
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="user2",
            source="cointracking"
        )

        # Verify both snapshots saved
        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)
            assert len(data) == 2
            user_ids = [snap["user_id"] for snap in data]
            assert "user1" in user_ids
            assert "user2" in user_ids

    def test_save_portfolio_snapshot_multiple_sources(self, portfolio_service, sample_balances_data):
        """Test snapshots for multiple sources"""
        # Save cointracking
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Save cointracking_api
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking_api"
        )

        # Verify both snapshots saved
        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)
            assert len(data) == 2
            sources = [snap["source"] for snap in data]
            assert "cointracking" in sources
            assert "cointracking_api" in sources

    def test_save_portfolio_snapshot_upsert_same_day(self, portfolio_service, sample_balances_data):
        """Test that snapshot on same day is updated (not duplicated)"""
        # First save
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Modify data slightly
        modified_data = sample_balances_data.copy()
        modified_data["items"][0]["value_usd"] = 80000.0  # BTC value increased

        # Save again same day
        portfolio_service.save_portfolio_snapshot(
            modified_data,
            user_id="demo",
            source="cointracking"
        )

        # Should have only 1 snapshot (updated, not duplicated)
        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)
            assert len(data) == 1
            # Should have the updated value (not exact total but close)
            assert data[0]["total_value_usd"] > 130000.0  # BTC increase reflected

    def test_calculate_performance_metrics_no_history(self, portfolio_service, sample_balances_data):
        """Test performance metrics when no historical data exists"""
        metrics = portfolio_service.calculate_performance_metrics(
            current_data=sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        assert metrics["performance_available"] is False
        assert "message" in metrics
        assert metrics["days_tracked"] == 0

    def test_calculate_performance_metrics_with_history(self, portfolio_service, sample_balances_data):
        """Test performance metrics with historical data"""
        # Create historical snapshot (24h ago)
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Modify file to make snapshot 24h old
        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)

        # Backdate the snapshot to 24h ago
        yesterday = datetime.now(ZoneInfo("Europe/Zurich")) - timedelta(hours=24)
        data[0]["date"] = yesterday.isoformat()
        data[0]["total_value_usd"] = 120000.0  # Previous value

        with open(portfolio_service.historical_data_file, 'w') as f:
            json.dump(data, f)

        # Create current data with higher value
        current_data = sample_balances_data.copy()
        current_data["items"][0]["value_usd"] = 85000.0  # BTC increased

        # Calculate performance
        metrics = portfolio_service.calculate_performance_metrics(
            current_data=current_data,
            user_id="demo",
            source="cointracking",
            anchor="prev_snapshot",
            window="24h"
        )

        # Should show performance available
        assert metrics.get("performance_available", False) or "comparison" in metrics
        assert "days_tracked" in metrics

    def test_load_historical_data_empty(self, portfolio_service):
        """Test loading historical data when file doesn't exist"""
        data = portfolio_service._load_historical_data(
            user_id="demo",
            source="cointracking"
        )

        assert isinstance(data, list)
        assert len(data) == 0

    def test_load_historical_data_filter_by_user(self, portfolio_service, sample_balances_data):
        """Test historical data filtering by user_id"""
        # Save for user1
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="user1",
            source="cointracking"
        )

        # Save for user2
        portfolio_service.save_portfolio_snapshot(
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

    def test_load_historical_data_filter_by_source(self, portfolio_service, sample_balances_data):
        """Test historical data filtering by source"""
        # Save cointracking
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Save cointracking_api
        portfolio_service.save_portfolio_snapshot(
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

    def test_snapshot_includes_group_distribution(self, portfolio_service, sample_balances_data):
        """Test that snapshot includes group distribution"""
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)
            snapshot = data[0]

        assert "group_distribution" in snapshot
        assert isinstance(snapshot["group_distribution"], dict)

    def test_snapshot_includes_timestamp(self, portfolio_service, sample_balances_data):
        """Test that snapshot includes ISO timestamp"""
        portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)
            snapshot = data[0]

        assert "date" in snapshot
        # Should be valid ISO format
        date_parsed = datetime.fromisoformat(snapshot["date"])
        assert isinstance(date_parsed, datetime)

    # ========================================================================
    # Tests for get_portfolio_trend() - Coverage Gap
    # ========================================================================

    def test_get_portfolio_trend_empty_history(self, portfolio_service):
        """Test get_portfolio_trend with no historical data"""
        result = portfolio_service.get_portfolio_trend(days=30)

        assert result["trend_data"] == []
        assert result["days_available"] == 0

    def test_get_portfolio_trend_with_data(self, portfolio_service, sample_balances_data):
        """Test get_portfolio_trend with historical snapshots"""
        # Create some snapshots over time
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

            # Manually write to historical file
            if i == 0:
                data = [snapshot]
            else:
                with open(portfolio_service.historical_data_file, 'r') as f:
                    data = json.load(f)
                data.append(snapshot)

            with open(portfolio_service.historical_data_file, 'w') as f:
                json.dump(data, f)

        # Get trend for last 7 days
        result = portfolio_service.get_portfolio_trend(days=7)

        # Should have data from last 7 days
        assert len(result["trend_data"]) > 0
        assert result["days_available"] == len(result["trend_data"])

        # Validate structure
        first_entry = result["trend_data"][0]
        assert "date" in first_entry
        assert "total_value" in first_entry
        assert "asset_count" in first_entry
        assert "diversity_score" in first_entry

    def test_get_portfolio_trend_filters_by_days(self, portfolio_service):
        """Test that get_portfolio_trend correctly filters by days parameter"""
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        # Create 40 days of snapshots
        data = []
        for i in range(40):
            past_date = datetime.now(TZ) - timedelta(days=40 - i)
            data.append({
                "date": past_date.isoformat(),
                "user_id": "demo",
                "source": "cointracking",
                "total_value_usd": 100000,
                "asset_count": 5,
                "diversity_score": 5.0
            })

        with open(portfolio_service.historical_data_file, 'w') as f:
            json.dump(data, f)

        # Get trend for last 15 days
        result = portfolio_service.get_portfolio_trend(days=15)

        # Should have approximately 15 days (might be 15-16 due to timing)
        assert 14 <= result["days_available"] <= 16
        assert len(result["trend_data"]) == result["days_available"]

    # ========================================================================
    # Tests for _compute_anchor_ts() - Coverage Gap
    # ========================================================================

    def test_compute_anchor_ts_midnight(self):
        """Test _compute_anchor_ts with midnight anchor"""
        from services.portfolio import _compute_anchor_ts
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        now = datetime(2025, 11, 23, 14, 30, 0, tzinfo=TZ)
        anchor_ts, window_ts = _compute_anchor_ts(anchor="midnight", window="24h", now=now)

        # Anchor should be midnight today
        assert anchor_ts is not None
        assert anchor_ts.hour == 0
        assert anchor_ts.minute == 0
        assert anchor_ts.second == 0
        assert anchor_ts.date() == now.date()

        # Window should be 24h ago
        assert window_ts == now - timedelta(days=1)

    def test_compute_anchor_ts_prev_close(self):
        """Test _compute_anchor_ts with prev_close anchor"""
        from services.portfolio import _compute_anchor_ts
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        now = datetime(2025, 11, 23, 14, 30, 0, tzinfo=TZ)
        anchor_ts, window_ts = _compute_anchor_ts(anchor="prev_close", window="7d", now=now)

        # For crypto (24/7), prev_close falls back to midnight
        assert anchor_ts is not None
        assert anchor_ts.hour == 0
        assert anchor_ts.minute == 0

        # Window should be 7 days ago
        assert window_ts == now - timedelta(days=7)

    def test_compute_anchor_ts_prev_snapshot(self):
        """Test _compute_anchor_ts with prev_snapshot anchor (default)"""
        from services.portfolio import _compute_anchor_ts
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        now = datetime(2025, 11, 23, 14, 30, 0, tzinfo=TZ)
        anchor_ts, window_ts = _compute_anchor_ts(anchor="prev_snapshot", window="30d", now=now)

        # prev_snapshot: anchor_ts should be None (will use last snapshot < now)
        assert anchor_ts is None

        # Window should be 30 days ago
        assert window_ts == now - timedelta(days=30)

    def test_compute_anchor_ts_window_ytd(self):
        """Test _compute_anchor_ts with YTD window"""
        from services.portfolio import _compute_anchor_ts
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        now = datetime(2025, 11, 23, 14, 30, 0, tzinfo=TZ)
        anchor_ts, window_ts = _compute_anchor_ts(anchor="midnight", window="ytd", now=now)

        # Window should be start of year
        assert window_ts is not None
        assert window_ts.year == 2025
        assert window_ts.month == 1
        assert window_ts.day == 1
        assert window_ts.hour == 0
        assert window_ts.minute == 0

    # ========================================================================
    # Error Handling Tests - Coverage Gap
    # ========================================================================

    def test_save_snapshot_corrupted_json_file(self, portfolio_service, sample_balances_data):
        """Test save_snapshot handles corrupted JSON file gracefully"""
        # Write invalid JSON to file
        with open(portfolio_service.historical_data_file, 'w') as f:
            f.write("{ invalid json }")

        # Should not crash, should handle gracefully
        result = portfolio_service.save_portfolio_snapshot(
            sample_balances_data,
            user_id="demo",
            source="cointracking"
        )

        # Should succeed by treating corrupted file as empty
        assert result is True

        # Verify file is now valid
        with open(portfolio_service.historical_data_file, 'r') as f:
            data = json.load(f)
            assert len(data) == 1

    def test_atomic_json_dump_creates_parent_directories(self, tmp_path):
        """Test _atomic_json_dump creates parent directories if missing"""
        from services.portfolio import _atomic_json_dump

        # Path with non-existent parent
        nested_path = tmp_path / "level1" / "level2" / "data.json"

        data = {"test": "value"}
        _atomic_json_dump(data, nested_path)

        # Should have created all directories and file
        assert nested_path.exists()

        with open(nested_path, 'r') as f:
            loaded = json.load(f)
            assert loaded == data

    def test_upsert_daily_snapshot_replaces_same_day(self):
        """Test _upsert_daily_snapshot replaces existing snapshot on same day"""
        from services.portfolio import _upsert_daily_snapshot
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        # Create initial snapshot
        now = datetime.now(TZ)
        snap1 = {
            "date": now.isoformat(),
            "user_id": "demo",
            "source": "cointracking",
            "total_value_usd": 100000
        }

        # Create second snapshot same day, different time
        snap2 = {
            "date": (now + timedelta(hours=2)).isoformat(),
            "user_id": "demo",
            "source": "cointracking",
            "total_value_usd": 105000  # Different value
        }

        entries = []

        # Insert first
        _upsert_daily_snapshot(entries, snap1, "demo", "cointracking")
        assert len(entries) == 1
        assert entries[0]["total_value_usd"] == 100000

        # Upsert second (same day) - should replace
        _upsert_daily_snapshot(entries, snap2, "demo", "cointracking")
        assert len(entries) == 1  # Still only 1 entry
        assert entries[0]["total_value_usd"] == 105000  # Updated value

    def test_upsert_daily_snapshot_different_days(self):
        """Test _upsert_daily_snapshot adds new entry for different days"""
        from services.portfolio import _upsert_daily_snapshot
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        now = datetime.now(TZ)

        snap1 = {
            "date": (now - timedelta(days=1)).isoformat(),
            "user_id": "demo",
            "source": "cointracking",
            "total_value_usd": 100000
        }

        snap2 = {
            "date": now.isoformat(),
            "user_id": "demo",
            "source": "cointracking",
            "total_value_usd": 105000
        }

        entries = []

        _upsert_daily_snapshot(entries, snap1, "demo", "cointracking")
        assert len(entries) == 1

        # Add snapshot from different day - should append
        _upsert_daily_snapshot(entries, snap2, "demo", "cointracking")
        assert len(entries) == 2  # Now 2 entries

    def test_upsert_daily_snapshot_different_users(self):
        """Test _upsert_daily_snapshot isolates by user_id"""
        from services.portfolio import _upsert_daily_snapshot
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo("Europe/Zurich")

        now = datetime.now(TZ)

        snap_demo = {
            "date": now.isoformat(),
            "user_id": "demo",
            "source": "cointracking",
            "total_value_usd": 100000
        }

        snap_jack = {
            "date": now.isoformat(),
            "user_id": "jack",
            "source": "cointracking",
            "total_value_usd": 200000
        }

        entries = []

        _upsert_daily_snapshot(entries, snap_demo, "demo", "cointracking")
        _upsert_daily_snapshot(entries, snap_jack, "jack", "cointracking")

        # Should have 2 entries (different users, same day)
        assert len(entries) == 2
