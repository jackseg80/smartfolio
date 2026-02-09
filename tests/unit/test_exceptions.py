"""
Tests for shared/exceptions.py - Custom exception hierarchy and helpers.
"""
from __future__ import annotations
import pytest

from shared.exceptions import (
    ErrorCode, CryptoRebalancerException, ConfigurationError, APIException,
    RateLimitException, DataException, DataNotFoundException, ExchangeException,
    InsufficientBalanceException, PricingException, ConfigurationException,
    ValidationException, TradingException, StorageException, GovernanceException,
    MonitoringException, NetworkException, TimeoutException,
    convert_standard_exception, handle_exceptions,
)


class TestErrorCode:
    def test_all_codes_are_strings(self):
        for code in ErrorCode:
            assert isinstance(code.value, str)

    def test_known_codes_exist(self):
        expected = [
            "CONFIG_INVALID", "CONFIG_MISSING", "API_KEY_INVALID",
            "API_RATE_LIMITED", "API_UNAVAILABLE", "API_TIMEOUT",
            "DATA_INVALID", "DATA_NOT_FOUND", "DATA_STALE",
            "EXCHANGE_NOT_CONNECTED", "INSUFFICIENT_BALANCE",
            "ORDER_FAILED", "SYMBOL_NOT_SUPPORTED",
            "PRICE_NOT_AVAILABLE", "PRICE_SOURCE_ERROR",
            "NETWORK_ERROR", "TIMEOUT_ERROR", "PERMISSION_DENIED",
        ]
        actual_values = {c.value for c in ErrorCode}
        for code in expected:
            assert code in actual_values


class TestCryptoRebalancerException:
    def test_basic_instantiation(self):
        exc = CryptoRebalancerException("something failed")
        assert str(exc) == "something failed"
        assert exc.message == "something failed"
        assert exc.error_code is None
        assert exc.details == {}
        assert exc.cause is None

    def test_with_error_code(self):
        exc = CryptoRebalancerException("fail", error_code=ErrorCode.NETWORK_ERROR)
        assert exc.error_code == ErrorCode.NETWORK_ERROR

    def test_with_details(self):
        details = {"key": "value", "count": 42}
        exc = CryptoRebalancerException("fail", details=details)
        assert exc.details == details

    def test_with_cause(self):
        cause = ValueError("root cause")
        exc = CryptoRebalancerException("wrapper", cause=cause)
        assert exc.cause is cause

    def test_is_exception(self):
        assert isinstance(CryptoRebalancerException("test"), Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(CryptoRebalancerException, match="boom"):
            raise CryptoRebalancerException("boom")

class TestConfigurationError:
    def test_inheritance(self):
        assert isinstance(ConfigurationError("bad"), CryptoRebalancerException)

    def test_error_code_is_config_invalid(self):
        assert ConfigurationError("bad").error_code == ErrorCode.CONFIG_INVALID

    def test_config_key_in_details(self):
        exc = ConfigurationError("bad", config_key="db.host")
        assert exc.details["config_key"] == "db.host"

    def test_with_cause_kwarg(self):
        cause = KeyError("missing")
        assert ConfigurationError("bad", cause=cause).cause is cause


class TestAPIException:
    def test_inheritance(self):
        assert isinstance(APIException("down", api_name="coingecko"), CryptoRebalancerException)

    def test_error_code(self):
        assert APIException("down", api_name="cg").error_code == ErrorCode.API_UNAVAILABLE

    def test_details_contain_api_info(self):
        exc = APIException("down", api_name="coingecko", status_code=503)
        assert exc.details["api_name"] == "coingecko"
        assert exc.details["status_code"] == 503

    def test_str_representation(self):
        assert "Service unavailable" in str(APIException("Service unavailable", api_name="fred"))


class TestRateLimitException:
    def test_inheritance_chain(self):
        exc = RateLimitException("coingecko")
        assert isinstance(exc, APIException)
        assert isinstance(exc, CryptoRebalancerException)

    def test_error_code_overridden(self):
        assert RateLimitException("cg").error_code == ErrorCode.API_RATE_LIMITED

    def test_message_includes_api_name(self):
        assert "coingecko" in str(RateLimitException("coingecko"))

    def test_retry_after(self):
        exc = RateLimitException("cg", retry_after=60)
        assert exc.retry_after == 60
        assert "60 seconds" in str(exc)

    def test_no_retry_after(self):
        assert RateLimitException("binance").retry_after is None


class TestDataExceptions:
    def test_data_exception_basic(self):
        assert DataException("corrupt").error_code == ErrorCode.DATA_INVALID

    def test_data_exception_with_source(self):
        assert DataException("bad", data_source="csv").details["data_source"] == "csv"

    def test_not_found_inheritance(self):
        exc = DataNotFoundException("Portfolio")
        assert isinstance(exc, DataException)
        assert isinstance(exc, CryptoRebalancerException)

    def test_not_found_error_code(self):
        assert DataNotFoundException("P").error_code == ErrorCode.DATA_NOT_FOUND

    def test_not_found_message_with_identifier(self):
        assert "User not found: jack" in str(DataNotFoundException("User", identifier="jack"))

    def test_not_found_message_without_identifier(self):
        assert str(DataNotFoundException("Config")) == "Config not found"

class TestExchangeExceptions:
    def test_exchange_exception(self):
        exc = ExchangeException("disconnected", exchange="binance")
        assert exc.error_code == ErrorCode.EXCHANGE_NOT_CONNECTED
        assert exc.details["exchange"] == "binance"

    def test_insufficient_balance_inheritance(self):
        assert isinstance(InsufficientBalanceException("BTC", 1.0, 0.5, "binance"), ExchangeException)

    def test_insufficient_balance_error_code(self):
        assert InsufficientBalanceException("ETH", 2.0, 0.1, "kraken").error_code == ErrorCode.INSUFFICIENT_BALANCE

    def test_insufficient_balance_details(self):
        exc = InsufficientBalanceException("BTC", 1.0, 0.5, "binance")
        assert exc.details["symbol"] == "BTC"
        assert exc.details["required"] == 1.0
        assert exc.details["available"] == 0.5

    def test_insufficient_balance_message(self):
        exc = InsufficientBalanceException("SOL", 10.0, 3.0, "binance")
        assert "SOL" in str(exc) and "10.0" in str(exc)


class TestPricingException:
    def test_basic(self):
        assert PricingException("no price").error_code == ErrorCode.PRICE_NOT_AVAILABLE

    def test_with_symbol_and_source(self):
        exc = PricingException("no price", symbol="BTC", source="coingecko")
        assert exc.details["symbol"] == "BTC"
        assert exc.details["source"] == "coingecko"


class TestConfigurationException:
    def test_is_separate_from_configuration_error(self):
        assert type(ConfigurationException("m")) is not type(ConfigurationError("m"))

    def test_error_code(self):
        assert ConfigurationException("bad").error_code == ErrorCode.CONFIG_INVALID

    def test_with_details(self):
        exc = ConfigurationException("bad", details={"file": "config.json"})
        assert exc.details["file"] == "config.json"


class TestValidationException:
    def test_message_format(self):
        exc = ValidationException("email", "invalid format")
        assert "Validation error on email" in str(exc)

    def test_field_and_value(self):
        exc = ValidationException("age", "must be positive", value=-5)
        assert exc.field == "age"
        assert exc.value == -5

    def test_error_code(self):
        assert ValidationException("x", "bad").error_code == ErrorCode.DATA_INVALID

    def test_details_contain_field_and_value(self):
        exc = ValidationException("port", "out of range", value=99999)
        assert exc.details["field"] == "port"
        assert exc.details["value"] == 99999


class TestTradingException:
    def test_message_format(self):
        assert "Trading error in rebalance" in str(TradingException("rebalance", "order rejected"))

    def test_operation_attribute(self):
        assert TradingException("sell", "too low").operation == "sell"

    def test_error_code(self):
        assert TradingException("buy", "failed").error_code == ErrorCode.ORDER_FAILED

class TestStorageException:
    def test_message_format(self):
        assert "redis storage error during get" in str(StorageException("redis", "get", "connection refused"))

    def test_attributes(self):
        exc = StorageException("file", "write", "disk full")
        assert exc.storage_type == "file"
        assert exc.operation == "write"

    def test_error_code_is_none(self):
        assert StorageException("redis", "set", "fail").error_code is None

    def test_details_merged(self):
        exc = StorageException("redis", "get", "fail", details={"key": "cache:abc"})
        assert exc.details["storage_type"] == "redis"
        assert exc.details["key"] == "cache:abc"


class TestGovernanceException:
    def test_message_format(self):
        exc = GovernanceException("cooldown", "last trade too recent")
        assert "Governance rule" in str(exc) and "cooldown" in str(exc)

    def test_rule_attribute(self):
        assert GovernanceException("freeze", "frozen").rule == "freeze"


class TestMonitoringException:
    def test_message_format(self):
        assert "Monitoring error in health_check" in str(MonitoringException("health_check", "redis down"))

    def test_component_attribute(self):
        assert MonitoringException("api", "latency spike").component == "api"


class TestNetworkException:
    def test_error_code(self):
        assert NetworkException("refused").error_code == ErrorCode.NETWORK_ERROR

    def test_url_in_details(self):
        assert NetworkException("timeout", url="https://api.example.com").details["url"] == "https://api.example.com"


class TestTimeoutException:
    def test_error_code(self):
        assert TimeoutException("fetch_prices", 30).error_code == ErrorCode.TIMEOUT_ERROR

    def test_message_format(self):
        exc = TimeoutException("fetch_prices", 30)
        assert "fetch_prices" in str(exc) and "30 seconds" in str(exc)

    def test_details(self):
        exc = TimeoutException("download", 60)
        assert exc.details["operation"] == "download"
        assert exc.details["timeout"] == 60

class TestConvertStandardException:
    def test_passthrough_custom_exception(self):
        orig = DataException("already custom")
        assert convert_standard_exception(orig) is orig

    def test_connection_error_to_network(self):
        result = convert_standard_exception(ConnectionError("refused"))
        assert isinstance(result, NetworkException)
        assert result.cause is not None

    def test_os_error_to_network(self):
        assert isinstance(convert_standard_exception(OSError("socket")), NetworkException)

    def test_timeout_error_to_network(self):
        # TimeoutError is a subclass of OSError, matched before TimeoutException
        result = convert_standard_exception(TimeoutError("timed out"))
        assert isinstance(result, NetworkException)

    def test_import_error_to_configuration(self):
        assert isinstance(convert_standard_exception(ImportError("no mod")), ConfigurationError)

    def test_permission_error_to_network(self):
        # PermissionError is a subclass of OSError, matched before PERMISSION_DENIED check
        result = convert_standard_exception(PermissionError("denied"))
        assert isinstance(result, NetworkException)
        assert result.error_code == ErrorCode.NETWORK_ERROR

    def test_value_error_to_data(self):
        assert isinstance(convert_standard_exception(ValueError("bad")), DataException)

    def test_key_error_to_data(self):
        assert isinstance(convert_standard_exception(KeyError("missing")), DataException)

    def test_type_error_to_data(self):
        assert isinstance(convert_standard_exception(TypeError("wrong")), DataException)

    def test_generic_exception(self):
        result = convert_standard_exception(RuntimeError("something"))
        assert isinstance(result, CryptoRebalancerException)
        assert "Unexpected error" in result.message

    def test_context_included_in_message(self):
        result = convert_standard_exception(ValueError("bad"), context="parsing CSV")
        assert "parsing CSV" in result.message

    def test_cause_is_set(self):
        original = ValueError("original")
        assert convert_standard_exception(original, context="test").cause is original


class TestHandleExceptionsDecorator:
    def test_no_exception_passthrough(self):
        @handle_exceptions(context="test")
        def good_func():
            return 42
        assert good_func() == 42

    def test_custom_exception_reraised(self):
        @handle_exceptions(context="test")
        def bad_func():
            raise DataException("already typed")
        with pytest.raises(DataException):
            bad_func()

    def test_standard_exception_converted_and_reraised(self):
        @handle_exceptions(context="parsing")
        def bad_func():
            raise ValueError("bad input")
        with pytest.raises(DataException):
            bad_func()

    def test_reraise_false_returns_exception(self):
        @handle_exceptions(context="test", reraise=False)
        def bad_func():
            raise RuntimeError("boom")
        assert isinstance(bad_func(), CryptoRebalancerException)

    def test_custom_exception_reraise_false_does_not_return(self):
        @handle_exceptions(context="test", reraise=False)
        def bad_func():
            raise DataException("typed")
        assert bad_func() is None


class TestExceptionChaining:
    def test_cause_propagates_through_hierarchy(self):
        root = ValueError("root")
        mid = DataException("mid", cause=root)
        top = GovernanceException("rule1", "wrapped", cause=mid)
        assert top.cause is mid
        assert mid.cause is root

    def test_all_classes_accept_cause(self):
        root = RuntimeError("root")
        classes_and_args = [
            (CryptoRebalancerException, ("msg",)),
            (ConfigurationError, ("msg",)),
            (APIException, ("msg", "api")),
            (RateLimitException, ("api",)),
            (DataException, ("msg",)),
            (DataNotFoundException, ("Resource",)),
            (ExchangeException, ("msg", "binance")),
            (InsufficientBalanceException, ("BTC", 1.0, 0.5, "binance")),
            (PricingException, ("msg",)),
            (ConfigurationException, ("msg",)),
            (ValidationException, ("field", "msg")),
            (TradingException, ("op", "msg")),
            (StorageException, ("redis", "get", "msg")),
            (GovernanceException, ("rule", "msg")),
            (MonitoringException, ("comp", "msg")),
            (NetworkException, ("msg",)),
            (TimeoutException, ("op", 30)),
        ]
        for cls, args in classes_and_args:
            exc = cls(*args, cause=root)
            assert exc.cause is root, f"{cls.__name__} did not preserve cause"
