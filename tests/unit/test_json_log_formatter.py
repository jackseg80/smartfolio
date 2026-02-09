"""Tests unitaires pour shared/json_log_formatter.py."""
import json
import logging
import pytest

from shared.json_log_formatter import JsonLogFormatter


@pytest.fixture
def formatter():
    return JsonLogFormatter()


@pytest.fixture
def make_record():
    """Create a LogRecord factory."""
    def _make(msg="test message", level=logging.INFO, exc_info=None):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test.py",
            lineno=42,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        return record
    return _make


class TestJsonLogFormatter:
    def test_basic_format(self, formatter, make_record):
        record = make_record("Hello world")
        output = formatter.format(record)
        data = json.loads(output)
        assert data["msg"] == "Hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert "ts" in data

    def test_debug_level(self, formatter, make_record):
        record = make_record("debug msg", level=logging.DEBUG)
        data = json.loads(formatter.format(record))
        assert data["level"] == "DEBUG"

    def test_error_level(self, formatter, make_record):
        record = make_record("error msg", level=logging.ERROR)
        data = json.loads(formatter.format(record))
        assert data["level"] == "ERROR"

    def test_exception_info_included(self, formatter, make_record):
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = make_record("with exception", exc_info=exc_info)
            data = json.loads(formatter.format(record))
            assert "exception" in data
            assert data["exception"]["type"] == "ValueError"
            assert "test error" in data["exception"]["msg"]
            assert isinstance(data["exception"]["traceback"], list)

    def test_no_exception_no_key(self, formatter, make_record):
        record = make_record("no exception")
        data = json.loads(formatter.format(record))
        assert "exception" not in data

    def test_timestamp_is_numeric(self, formatter, make_record):
        record = make_record("timestamp test")
        data = json.loads(formatter.format(record))
        assert isinstance(data["ts"], (int, float))

    def test_output_is_valid_json(self, formatter, make_record):
        record = make_record("json validity check with special chars: é à ü")
        output = formatter.format(record)
        data = json.loads(output)
        assert "é à ü" in data["msg"]
