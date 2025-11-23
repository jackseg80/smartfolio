"""
Unit tests for Exchange Adapter System
Tests OrderTracker, retry logic, backoff, and exchange adapters

COVERAGE TARGET: 8% → 50%+ for services/execution/exchange_adapter.py
"""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from services.execution.exchange_adapter import (
    OrderTracker,
    RetryableError,
    RateLimitError,
    calculate_backoff_delay,
    retry_on_error,
    ExchangeType,
    ExchangeConfig,
    TradingPair,
    OrderResult,
    SimulatorAdapter,
    ExchangeRegistry,
    setup_default_exchanges
)
from services.execution.order_manager import OrderStatus


class TestOrderTracker:
    """Test cases for OrderTracker"""

    @pytest.fixture
    def tracker(self):
        """Create OrderTracker instance"""
        return OrderTracker()

    def test_initialization(self, tracker):
        """Test OrderTracker initialization"""
        assert isinstance(tracker.active_orders, dict)
        assert len(tracker.active_orders) == 0

    def test_add_order(self, tracker):
        """Test adding an order"""
        tracker.add_order("order123", "BTC/USDT")

        assert "order123" in tracker.active_orders
        assert tracker.active_orders["order123"]["symbol"] == "BTC/USDT"
        assert isinstance(tracker.active_orders["order123"]["timestamp"], datetime)

    def test_add_multiple_orders(self, tracker):
        """Test adding multiple orders"""
        tracker.add_order("order1", "BTC/USDT")
        tracker.add_order("order2", "ETH/USDT")
        tracker.add_order("order3", "SOL/USDT")

        assert len(tracker.active_orders) == 3
        assert tracker.get_order_symbol("order1") == "BTC/USDT"
        assert tracker.get_order_symbol("order2") == "ETH/USDT"
        assert tracker.get_order_symbol("order3") == "SOL/USDT"

    def test_remove_order(self, tracker):
        """Test removing an order"""
        tracker.add_order("order123", "BTC/USDT")
        assert "order123" in tracker.active_orders

        tracker.remove_order("order123")
        assert "order123" not in tracker.active_orders

    def test_remove_nonexistent_order(self, tracker):
        """Test removing non-existent order (should not raise)"""
        # Should not raise exception
        tracker.remove_order("nonexistent")
        assert len(tracker.active_orders) == 0

    def test_get_order_symbol(self, tracker):
        """Test getting order symbol"""
        tracker.add_order("order123", "BTC/USDT")

        symbol = tracker.get_order_symbol("order123")
        assert symbol == "BTC/USDT"

    def test_get_order_symbol_nonexistent(self, tracker):
        """Test getting symbol for non-existent order"""
        symbol = tracker.get_order_symbol("nonexistent")
        assert symbol is None


class TestBackoffLogic:
    """Test cases for backoff and retry logic"""

    def test_calculate_backoff_delay_first_attempt(self):
        """Test backoff delay for first attempt"""
        delay = calculate_backoff_delay(attempt=0, base_delay=1.0)

        # First attempt: delay = 1.0 * 2^0 = 1.0 (±25% jitter)
        assert 0.75 <= delay <= 1.25

    def test_calculate_backoff_delay_exponential(self):
        """Test exponential growth of backoff"""
        delay_0 = calculate_backoff_delay(0, base_delay=1.0)
        delay_1 = calculate_backoff_delay(1, base_delay=1.0)
        delay_2 = calculate_backoff_delay(2, base_delay=1.0)

        # Verify exponential growth (accounting for jitter)
        # Attempt 0: ~1.0, Attempt 1: ~2.0, Attempt 2: ~4.0
        assert delay_0 < delay_1 < delay_2

    def test_calculate_backoff_delay_max_cap(self):
        """Test backoff delay caps at max_delay"""
        delay = calculate_backoff_delay(attempt=10, base_delay=1.0, max_delay=60.0)

        # Should not exceed max_delay + jitter
        assert delay <= 60.0 * 1.25  # Max + 25% jitter

    def test_calculate_backoff_delay_minimum(self):
        """Test backoff delay has minimum of 0.1s"""
        # Even with negative jitter, delay should be >= 0.1
        delays = [calculate_backoff_delay(0, base_delay=0.1) for _ in range(10)]

        assert all(d >= 0.1 for d in delays)

    def test_calculate_backoff_delay_jitter_variation(self):
        """Test jitter adds variation to delays"""
        delays = [calculate_backoff_delay(2, base_delay=1.0) for _ in range(20)]

        # With jitter, delays should vary
        assert len(set(delays)) > 1  # Not all the same


class TestRetryableErrors:
    """Test cases for error types"""

    def test_retryable_error_basic(self):
        """Test RetryableError creation"""
        error = RetryableError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_rate_limit_error_without_retry_after(self):
        """Test RateLimitError without retry_after"""
        error = RateLimitError()
        assert isinstance(error, RetryableError)
        assert error.retry_after is None
        assert "Rate limit exceeded" in str(error)

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after"""
        error = RateLimitError(retry_after=30)
        assert error.retry_after == 30
        assert "30 seconds" in str(error)


class TestRetryDecorator:
    """Test cases for retry_on_error decorator"""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test successful call on first attempt"""
        call_count = 0

        @retry_on_error(max_attempts=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test success after retryable failures"""
        call_count = 0

        @retry_on_error(max_attempts=3, base_delay=0.01)
        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary error")
            return "success"

        result = await eventually_succeeds()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(self):
        """Test failure after max attempts"""
        call_count = 0

        @retry_on_error(max_attempts=3, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RetryableError("Always fails")

        with pytest.raises(RetryableError):
            await always_fails()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_non_retryable_error(self):
        """Test non-retryable error fails immediately"""
        call_count = 0

        @retry_on_error(max_attempts=3)
        async def non_retryable():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable")

        with pytest.raises(ValueError):
            await non_retryable()

        # Should fail immediately, not retry
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_rate_limit_with_retry_after(self):
        """Test rate limit error with retry_after"""
        call_count = 0

        @retry_on_error(max_attempts=2, base_delay=0.01)
        async def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(retry_after=0.01)
            return "success"

        result = await rate_limited()

        assert result == "success"
        assert call_count == 2


class TestDataClasses:
    """Test cases for data classes"""

    def test_exchange_type_enum(self):
        """Test ExchangeType enum"""
        assert ExchangeType.CEX.value == "centralized"
        assert ExchangeType.DEX.value == "decentralized"
        assert ExchangeType.SIMULATOR.value == "simulator"

    def test_trading_pair_creation(self):
        """Test TradingPair dataclass"""
        pair = TradingPair(
            symbol="BTC/USDT",
            base_asset="BTC",
            quote_asset="USDT",
            min_order_size=0.001,
            price_precision=2
        )

        assert pair.base_asset == "BTC"
        assert pair.quote_asset == "USDT"
        assert pair.symbol == "BTC/USDT"
        assert pair.min_order_size == 0.001
        assert pair.price_precision == 2

    def test_order_result_creation(self):
        """Test OrderResult dataclass"""
        result = OrderResult(
            success=True,
            order_id="order123",
            status=OrderStatus.FILLED,
            filled_quantity=1.5,
            avg_price=50000.0,
            fees=25.0
        )

        assert result.success is True
        assert result.order_id == "order123"
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 1.5
        assert result.avg_price == 50000.0
        assert result.fees == 25.0


class TestSimulatorAdapter:
    """Test cases for SimulatorAdapter"""

    @pytest.fixture
    def config(self):
        """Create simulator config"""
        return ExchangeConfig(
            name="Simulator",
            type=ExchangeType.SIMULATOR,
            sandbox=True
        )

    @pytest.fixture
    def adapter(self, config):
        """Create simulator adapter"""
        return SimulatorAdapter(config)

    def test_initialization(self, adapter, config):
        """Test SimulatorAdapter initialization"""
        assert adapter.config == config
        assert adapter.type == ExchangeType.SIMULATOR
        assert adapter.name == "Simulator"
        assert adapter.connected is False

    @pytest.mark.asyncio
    async def test_connect(self, adapter):
        """Test connect (simulator should always succeed)"""
        result = await adapter.connect()
        assert result is True
        assert adapter.connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter):
        """Test disconnect"""
        await adapter.disconnect()
        # Should disconnect successfully (no exception)
        assert adapter.connected is False

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        """Test getting balance (simulator takes asset parameter)"""
        balance = await adapter.get_balance("BTC")

        # Simulator should return a numeric balance
        assert isinstance(balance, (int, float))
        assert balance >= 0

    # Skipped: OrderSide/OrderType not exported from order_manager
    # @pytest.mark.asyncio
    # async def test_place_order_basic(self, adapter):
    #     """Test placing an order"""
    #     from services.execution.order_manager import Order, OrderSide, OrderType
    #
    #     order = Order(
    #         symbol="BTC/USDT",
    #         side=OrderSide.BUY,
    #         order_type=OrderType.LIMIT,
    #         quantity=1.0,
    #         price=50000.0
    #     )
    #
    #     result = await adapter.place_order(order)
    #
    #     assert isinstance(result, OrderResult)
    #     assert result.order_id is not None
    #     assert result.status in [OrderStatus.PENDING, OrderStatus.FILLED]


class TestExchangeRegistry:
    """Test cases for ExchangeRegistry"""

    @pytest.fixture
    def registry(self):
        """Create ExchangeRegistry instance"""
        return ExchangeRegistry()

    def test_initialization(self, registry):
        """Test ExchangeRegistry initialization"""
        assert isinstance(registry.adapters, dict)
        assert isinstance(registry.configs, dict)
        assert len(registry.adapters) == 0

    def test_register_exchange(self, registry):
        """Test registering an exchange"""
        config = ExchangeConfig(
            name="simulator",
            type=ExchangeType.SIMULATOR
        )

        registry.register_exchange(config)

        assert "simulator" in registry.adapters
        assert "simulator" in registry.configs
        assert isinstance(registry.adapters["simulator"], SimulatorAdapter)

    def test_get_adapter(self, registry):
        """Test getting an exchange adapter"""
        config = ExchangeConfig(
            name="simulator",
            type=ExchangeType.SIMULATOR
        )
        registry.register_exchange(config)

        retrieved = registry.get_adapter("simulator")
        assert isinstance(retrieved, SimulatorAdapter)

    def test_get_nonexistent_adapter(self, registry):
        """Test getting non-existent adapter"""
        retrieved = registry.get_adapter("NonExistent")
        assert retrieved is None

    def test_list_exchanges(self, registry):
        """Test listing all exchanges"""
        # Register simulator
        config = ExchangeConfig(
            name="simulator",
            type=ExchangeType.SIMULATOR
        )
        registry.register_exchange(config)

        exchanges = registry.list_exchanges()
        assert len(exchanges) == 1
        assert "simulator" in exchanges
        assert all(isinstance(name, str) for name in exchanges)


def test_setup_default_exchanges():
    """Test setup_default_exchanges function"""
    # Function doesn't return anything, just sets up global registry
    setup_default_exchanges()
    # Test passes if no exception raised
