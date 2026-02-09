"""Tests for shared/error_handling.py - comprehensive coverage (40+ tests)"""

import pytest
from shared.error_handling import (
    safe_import, safe_call, safe_call_async, safe_get_data,
    safe_api_call, safe_pricing_operation, error_handler,
    safe_price_fetch, safe_exchange_operation,
)
from shared.exceptions import (
    NetworkException, TimeoutException, APIException,
    PricingException, ExchangeException, CryptoRebalancerException,
)


class TestSafeImport:
    def test_import_builtin(self):
        r = safe_import("json")
        assert r is not None and hasattr(r, "dumps")

    def test_import_with_package(self):
        assert safe_import("path", package="os") is not None

    def test_import_nonexistent(self):
        assert safe_import("nonexistent_xyz_123") is None

    def test_import_with_fallback_success(self):
        r = safe_import("nonexistent_xyz", fallback_module="json")
        assert r is not None and hasattr(r, "dumps")

    def test_import_with_fallback_also_fails(self):
        assert safe_import("nonexist_a", fallback_module="nonexist_b") is None

    def test_import_no_fallback(self):
        assert safe_import("nonexistent_xyz") is None


class TestSafeCall:
    def test_success(self):
        assert safe_call(lambda: 42) == 42

    def test_default_on_exception(self):
        assert safe_call(lambda: 1 / 0, default=-1) == -1

    def test_default_is_none(self):
        assert safe_call(lambda: 1 / 0) is None

    def test_expected_exceptions(self):
        def f():
            raise ValueError("x")
        r = safe_call(f, default="fb", context="ctx",
                      expected_exceptions=(ValueError,), log_level="info")
        assert r == "fb"

    def test_unexpected_exception_converted(self):
        assert safe_call(lambda: 1 / 0, default="s", context="div") == "s"

    def test_context_logging(self):
        assert safe_call(lambda: 1 / 0, default=0, context="ctx") == 0

    def test_returns_none_from_func(self):
        assert safe_call(lambda: None, default="x") is None


class TestSafeCallAsync:
    async def test_success(self):
        async def ok():
            return 99
        assert await safe_call_async(ok) == 99

    async def test_default_on_exception(self):
        async def f():
            raise RuntimeError("boom")
        assert await safe_call_async(f, default="safe") == "safe"

    async def test_default_none(self):
        async def f():
            raise ValueError("e")
        assert await safe_call_async(f) is None

    async def test_expected_exception(self):
        async def f():
            raise KeyError("m")
        r = await safe_call_async(f, default={},
            expected_exceptions=(KeyError,), context="at", log_level="debug")
        assert r == {}

    async def test_unexpected_converted(self):
        async def f():
            raise ConnectionError("net")
        assert await safe_call_async(f, default="off", context="n") == "off"


class TestSafeGetData:
    def test_success(self):
        assert safe_get_data(lambda: {"a": 1}) == {"a": 1}

    def test_fallback_on_none(self):
        assert safe_get_data(lambda: None, fallback="d") == "d"

    def test_fallback_on_key_error(self):
        def f():
            return {}["missing"]
        assert safe_get_data(f, fallback="fb", context="k") == "fb"

    def test_fallback_on_value_error(self):
        def f():
            raise ValueError("v")
        assert safe_get_data(f, fallback=0, context="v") == 0

    def test_fallback_on_type_error(self):
        def f():
            raise TypeError("t")
        assert safe_get_data(f, fallback=[], context="t") == []

    def test_fallback_on_unexpected(self):
        def f():
            raise RuntimeError("u")
        assert safe_get_data(f, fallback="s", context="r") == "s"

    def test_empty_string_valid(self):
        assert safe_get_data(lambda: "", fallback="x") == ""

    def test_zero_valid(self):
        assert safe_get_data(lambda: 0, fallback=999) == 0

    def test_false_valid(self):
        assert safe_get_data(lambda: False, fallback=True) is False


class TestSafeApiCall:
    def test_success(self):
        assert safe_api_call(lambda: {"d": "ok"}, "api") == {"d": "ok"}

    def test_default_on_none(self):
        assert safe_api_call(lambda: None, "api", default="e") == "e"

    def test_connection_error(self):
        def f():
            raise ConnectionError("r")
        with pytest.raises(NetworkException):
            safe_api_call(f, "api")

    def test_os_error(self):
        def f():
            raise OSError("s")
        with pytest.raises(NetworkException):
            safe_api_call(f, "api")

    def test_timeout_error(self):
        # TimeoutError is a subclass of OSError, so (ConnectionError, OSError)
        # branch catches it first, raising NetworkException instead of TimeoutException
        def f():
            raise TimeoutError("t")
        with pytest.raises(NetworkException):
            safe_api_call(f, "api")

    def test_generic_error(self):
        def f():
            raise RuntimeError("s")
        with pytest.raises(APIException):
            safe_api_call(f, "api")

    def test_retries_before_raising(self):
        c = 0
        def f():
            nonlocal c
            c += 1
            raise RuntimeError("f")
        with pytest.raises(APIException):
            safe_api_call(f, "a", retry_count=2)
        assert c == 3

    def test_retry_succeeds(self):
        c = 0
        def f():
            nonlocal c
            c += 1
            if c < 2:
                raise ConnectionError("t")
            return "ok"
        assert safe_api_call(f, "a", retry_count=2) == "ok"
        assert c == 2

    def test_retries_on_none(self):
        c = 0
        def f():
            nonlocal c
            c += 1
            return None
        r = safe_api_call(f, "a", default="fb", retry_count=1)
        assert r == "fb" and c == 2


class TestSafePricingOperation:
    def test_success(self):
        assert safe_pricing_operation(lambda: 50000.0, symbol="BTC") == 50000.0

    def test_value_error(self):
        def f():
            raise ValueError("b")
        with pytest.raises(PricingException) as e:
            safe_pricing_operation(f, symbol="ETH", source="cg")
        assert "ETH" in str(e.value)

    def test_key_error(self):
        def f():
            raise KeyError("m")
        with pytest.raises(PricingException):
            safe_pricing_operation(f, symbol="SOL")

    def test_connection_error(self):
        def f():
            raise ConnectionError("n")
        with pytest.raises(NetworkException):
            safe_pricing_operation(f, symbol="BTC")

    def test_os_error(self):
        def f():
            raise OSError("s")
        with pytest.raises(NetworkException):
            safe_pricing_operation(f, symbol="BTC")

    def test_generic_error(self):
        def f():
            raise RuntimeError("u")
        with pytest.raises(PricingException):
            safe_pricing_operation(f, symbol="ADA")

    def test_unknown_symbol(self):
        def f():
            raise ValueError("n")
        with pytest.raises(PricingException) as e:
            safe_pricing_operation(f)
        assert "unknown symbol" in str(e.value)


class TestErrorHandler:
    def test_sync_success(self):
        @error_handler()
        def add(a, b):
            return a + b
        assert add(2, 3) == 5

    def test_sync_reraise(self):
        @error_handler(context="st", reraise=True)
        def f():
            raise ValueError("b")
        with pytest.raises(CryptoRebalancerException):
            f()

    def test_sync_no_reraise(self):
        @error_handler(reraise=False, default_return="s")
        def f():
            raise RuntimeError("b")
        assert f() == "s"

    def test_sync_custom_exc_passes(self):
        @error_handler(reraise=True)
        def f():
            raise PricingException("bp", symbol="BTC")
        with pytest.raises(PricingException):
            f()

    def test_sync_custom_exc_no_reraise(self):
        @error_handler(reraise=False, default_return=0)
        def f():
            raise NetworkException("ne")
        assert f() == 0

    async def test_async_reraise(self):
        @error_handler(context="at", reraise=True)
        async def f():
            raise ConnectionError("off")
        with pytest.raises(CryptoRebalancerException):
            await f()

    async def test_async_no_reraise(self):
        @error_handler(reraise=False, default_return={})
        async def f():
            raise RuntimeError("ab")
        assert await f() == {}

    async def test_async_custom_exc_passes(self):
        @error_handler(reraise=True)
        async def f():
            raise ExchangeException("ed", "binance")
        with pytest.raises(ExchangeException):
            await f()

    async def test_async_custom_exc_no_reraise(self):
        @error_handler(reraise=False, default_return=None)
        async def f():
            raise APIException("ae", "cg")
        assert await f() is None

    def test_preserves_sync_name(self):
        @error_handler()
        def my_function():
            pass
        assert my_function.__name__ == "my_function"

    def test_preserves_async_name(self):
        @error_handler()
        async def my_async_function():
            pass
        assert my_async_function.__name__ == "my_async_function"

    async def test_async_success(self):
        @error_handler()
        async def async_add(a, b):
            return a + b
        assert await async_add(3, 4) == 7


class TestSafePriceFetch:
    def test_valid_prices(self):
        prices = {"BTC": 50000, "ETH": 3000}
        r = safe_price_fetch(lambda: prices, ["BTC", "ETH"], source="t")
        assert r == {"BTC": 50000.0, "ETH": 3000.0}

    def test_filters_none(self):
        r = safe_price_fetch(lambda: {"BTC": 50000, "ETH": None}, ["BTC"])
        assert "ETH" not in r and r["BTC"] == 50000.0

    def test_filters_zero(self):
        r = safe_price_fetch(lambda: {"BTC": 50000, "S": 0}, ["BTC", "S"])
        assert "S" not in r

    def test_filters_negative(self):
        r = safe_price_fetch(lambda: {"BTC": 50000, "B": -100}, ["BTC"])
        assert "B" not in r

    def test_filters_non_numeric(self):
        r = safe_price_fetch(lambda: {"BTC": 50000, "B": "nan"}, ["BTC"])
        assert "B" not in r and r["BTC"] == 50000.0

    def test_none_result(self):
        assert safe_price_fetch(lambda: None, ["BTC"]) == {}

    def test_empty_result(self):
        assert safe_price_fetch(lambda: {}, ["BTC"]) == {}

    def test_non_dict_result(self):
        assert safe_price_fetch(lambda: [1, 2], ["BTC"]) == {}

    def test_raises_on_error(self):
        def f():
            raise RuntimeError("ff")
        with pytest.raises(PricingException) as e:
            safe_price_fetch(f, ["BTC"], source="coingecko")
        assert "coingecko" in str(e.value)

    def test_string_number_converted(self):
        r = safe_price_fetch(lambda: {"BTC": "50000.5"}, ["BTC"])
        assert r["BTC"] == 50000.5


class TestSafeExchangeOperation:
    def test_success(self):
        r = safe_exchange_operation(lambda: {"id": "123"}, "binance", "place")
        assert r == {"id": "123"}

    def test_connection_error(self):
        def f():
            raise ConnectionError("r")
        with pytest.raises(ExchangeException) as e:
            safe_exchange_operation(f, "kraken", "fetch")
        assert "kraken" in str(e.value)

    def test_os_error(self):
        def f():
            raise OSError("s")
        with pytest.raises(ExchangeException):
            safe_exchange_operation(f, "binance")

    def test_timeout(self):
        # TimeoutError is a subclass of OSError, so (ConnectionError, OSError)
        # branch catches it first, raising ExchangeException
        def f():
            raise TimeoutError("sl")
        with pytest.raises(ExchangeException) as e:
            safe_exchange_operation(f, "coinbase", "fetch_orders")
        assert "coinbase" in str(e.value)

    def test_generic_error(self):
        def f():
            raise RuntimeError("u")
        with pytest.raises(ExchangeException) as e:
            safe_exchange_operation(f, "ftx", "withdraw")
        assert "ftx" in str(e.value)
        assert "withdraw" in str(e.value)

