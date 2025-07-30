# python_modules/dagster/dagster_tests/logging_tests/test_bridge.py

import logging
import os
import sys
import threading
import time
import tempfile

import pytest
from loguru import logger

# Try to load environment variables from .env file for local testing convenience
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure the bridge module can be found when running tests locally.
# In a proper package installation, this would not be necessary.
root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import bridge components from their new core location
from dagster._core.loguru_bridge import LoguruConfigurator, dagster_context_sink, with_loguru_logger


# --- Mocks and Test Helpers ---

class MockLogHandler:
    """Mock log handler that prints (for capfd tests) and stores log messages."""
    def __init__(self):
        self.history = []

    def debug(self, msg):
        print(f"[dagster.debug] {msg}")  # noqa: T201 - Required for capfd
        self.history.append({"level": "debug", "message": msg})

    def info(self, msg):
        print(f"[dagster.info] {msg}")  # noqa: T201 - Required for capfd
        self.history.append({"level": "info", "message": msg})

    def warning(self, msg):
        print(f"[dagster.warning] {msg}")  # noqa: T201 - Required for capfd
        self.history.append({"level": "warning", "message": msg})

    def error(self, msg):
        print(f"[dagster.error] {msg}")  # noqa: T201 - Required for capfd
        self.history.append({"level": "error", "message": msg})

    def critical(self, msg):
        print(f"[dagster.critical] {msg}")  # noqa: T201 - Required for capfd
        self.history.append({"level": "critical", "message": msg})


class MockDagsterContext:
    """Simulates Dagster's context.log interface using the MockLogHandler."""
    def __init__(self):
        self.log = MockLogHandler()


class DagsterOperations:
    """A collection of mock operations to test the decorator."""
    def __init__(self, context):
        self._context = context

    @with_loguru_logger
    def successful_op(self, context=None):
        logger.info("Operation completed successfully!")
        return True

    @with_loguru_logger
    def failing_op(self, context=None):
        logger.error("Operation failed!")
        raise ValueError("Operation failed")

    @with_loguru_logger
    def complex_op(self, context=None):
        logger.debug("Starting complex operation...")
        logger.info("Processing data")
        logger.warning("Resource usage high")
        logger.success("Data processing complete")
        return "Success"


class DagsterTestContext:
    """Helper class to bundle a mock context and operations for tests."""
    def __init__(self):
        self.context = MockDagsterContext()
        self.test_ops = DagsterOperations(self.context)


# --- Pytest Fixtures and Hooks ---

@pytest.fixture
def setup_logger():
    """Fixture to setup and cleanup a standard logger for each test."""
    logger.remove()
    
    def test_formatter(record):
        level_name = record["level"].name.lower()
        if level_name == "success":
            level_name = "info"
        return f"[dagster.{level_name}] {record['message']}\n"

    # A printing lambda sink is required for tests that use the `capfd` fixture.
    handler_id = logger.add(
        lambda msg: print(test_formatter(msg.record), end=""),  # noqa: T201
        level="TRACE",
    )
    yield
    try:
        logger.remove(handler_id)
    except ValueError:
        pass  # Handler might have been removed by the test itself.


def pytest_itemcollected(item):
    """Add execution tags to all tests for analytics platforms like Buildkite."""
    item.add_marker(pytest.mark.execution_tag("test.framework.name", "pytest"))
    item.add_marker(pytest.mark.execution_tag("test.framework.version", pytest.__version__))
    item.add_marker(pytest.mark.execution_tag("cloud.provider", "aws"))
    item.add_marker(pytest.mark.execution_tag("language.version", sys.version))


# --- Basic Sink and Decorator Tests ---

def test_dagster_context_sink_basic_logging(capfd, setup_logger):
    """Tests that the sink correctly forwards messages to the mock context log."""
    context = MockDagsterContext()
    sink = dagster_context_sink(context)
    logger.remove()
    logger.add(sink, level="DEBUG")
    logger.debug("Debug message")
    logger.info("Info message")
    captured = capfd.readouterr()
    assert "[dagster.debug] Debug message" in captured.out
    assert "[dagster.info] Info message" in captured.out


def test_dagster_context_sink_with_structured_logging(capfd, setup_logger):
    """Tests that bound data from Loguru doesn't break the sink."""
    context = MockDagsterContext()
    sink = dagster_context_sink(context)
    logger.remove()
    logger.add(sink, level="DEBUG")
    logger.bind(user="test_user").info("User login attempt")
    captured = capfd.readouterr()
    assert "[dagster.info] User login attempt" in captured.out


def test_dagster_context_sink_different_log_levels(capfd, setup_logger):
    """Verifies all log levels are correctly mapped and logged."""
    context = MockDagsterContext()
    sink = dagster_context_sink(context)
    logger.remove()
    logger.add(sink, level="TRACE")
    test_messages = [
        (logger.trace, "Trace level message", "debug"),
        (logger.debug, "Debug level message", "debug"),
        (logger.info, "Info level message", "info"),
        (logger.success, "Success message", "info"),
        (logger.warning, "Warning level message", "warning"),
        (logger.error, "Error level message", "error"),
        (logger.critical, "Critical level message", "critical"),
    ]
    for log_func, message, expected_level in test_messages:
        log_func(message)
    captured = capfd.readouterr()
    for _, message, expected_level in test_messages:
        assert f"[dagster.{expected_level}] {message}" in captured.out


def test_with_loguru_logger_decorator_success(capfd, setup_logger):
    context = MockDagsterContext()
    test_ops = DagsterOperations(context)
    result = test_ops.successful_op()
    assert result is True
    captured = capfd.readouterr()
    assert "[dagster.info] Operation completed successfully!" in captured.out


def test_with_loguru_logger_decorator_failure(capfd, setup_logger):
    context = MockDagsterContext()
    test_ops = DagsterOperations(context)
    with pytest.raises(ValueError, match="Operation failed"):
        test_ops.failing_op()
    captured = capfd.readouterr()
    assert "[dagster.error] Operation failed!" in captured.out


def test_with_loguru_logger_decorator_complex(capfd, setup_logger):
    context = MockDagsterContext()
    test_ops = DagsterOperations(context)
    result = test_ops.complex_op()
    assert result == "Success"
    captured = capfd.readouterr()
    assert "[dagster.debug] Starting complex operation..." in captured.out
    assert "[dagster.info] Processing data" in captured.out
    assert "[dagster.warning] Resource usage high" in captured.out
    assert "[dagster.info] Data processing complete" in captured.out

def test_mixed_logging_systems(capfd, setup_logger):
    """Tests that both `context.log` and `logger` calls work together under the decorator."""
    test_ctx = DagsterTestContext()
    class MixedLogger:
        def __init__(self, context):
            self._context = context
        @with_loguru_logger
        def mixed_logging_op(self, context=None):
            self._context.log.info("Direct Dagster log")
            logger.info("Loguru log")
    
    mixed_logger = MixedLogger(test_ctx.context)
    mixed_logger.mixed_logging_op()
    captured = capfd.readouterr()
    assert "[dagster.info] Direct Dagster log" in captured.out
    assert "[dagster.info] Loguru log" in captured.out


def test_nested_operations_logging(capfd, setup_logger):
    """Tests that the decorator works correctly on nested function calls."""
    test_ctx = DagsterTestContext()
    class NestedLogger:
        def __init__(self, test_ctx):
            self._context = test_ctx.context
            self.test_ops = test_ctx.test_ops
        @with_loguru_logger
        def nested_op(self, context=None):
            logger.info("Starting nested operation")
            self.test_ops.complex_op()
            logger.info("Nested operation complete")

    nested_logger = NestedLogger(test_ctx)
    nested_logger.nested_op()
    captured = capfd.readouterr()
    assert "[dagster.info] Starting nested operation" in captured.out
    assert "[dagster.debug] Starting complex operation..." in captured.out
    assert "[dagster.info] Nested operation complete" in captured.out


def test_exception_handling_with_logging(capfd, setup_logger):
    """Ensures logs inside try/except blocks are captured correctly."""
    test_ctx = DagsterTestContext()
    @with_loguru_logger
    def exception_op(context=None):
        try:
            raise ValueError("Simulated error")
        except ValueError as e:
            logger.error(f"Caught error: {e!s}")
            context.log.error(f"Dagster also logged: {e!s}")

    exception_op(context=test_ctx.context)
    captured = capfd.readouterr()
    assert "[dagster.error] Caught error: Simulated error" in captured.out
    assert "[dagster.error] Dagster also logged: Simulated error" in captured.out


def test_log_level_filtering(capfd):
    """Tests that setting a higher log level correctly filters out messages."""
    logger.remove()
    # Use a direct sink to stdout for this test to avoid fixture interference
    logger.add(sys.stdout, level="INFO", format="{message}")
    logger.debug("This should not appear")
    logger.info("Info message should appear")
    captured = capfd.readouterr()
    assert "This should not appear" not in captured.out
    assert "Info message should appear" in captured.out


def test_concurrent_operations_logging(setup_logger):
    """Tests that the bridge is thread-safe."""
    log_messages = []
    logger.remove()
    logger.add(log_messages.append, level="INFO")

    @with_loguru_logger
    def concurrent_op(op_id, context=None):
        logger.info(f"Operation {op_id} started")
        time.sleep(0.01)
        logger.info(f"Operation {op_id} completed")

    threads = []
    for i in range(3):
        thread = threading.Thread(target=concurrent_op, args=(f"thread-{i}", MockDagsterContext()))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    # Verify that logs from all threads were captured
    full_log_string = "".join(log_messages)
    for i in range(3):
        assert f"Operation thread-{i} started" in full_log_string
        assert f"Operation thread-{i} completed" in full_log_string


def test_loguru_configurator_initialization(monkeypatch):
    """Tests that the configurator correctly reads from environment variables."""
    monkeypatch.setenv("DAGSTER_LOGURU_ENABLED", "false")
    monkeypatch.setenv("DAGSTER_LOGURU_LOG_LEVEL", "WARNING")
    
    # Reset the singleton flag to allow re-initialization for this test
    LoguruConfigurator._initialized = False  # noqa: SLF001
    configurator = LoguruConfigurator(enable_terminal_sink=False)
    
    assert not configurator.config["enabled"]
    assert configurator.config["log_level"] == "WARNING"


def test_loguru_bridge_with_stdout_integration(capfd, setup_logger):
    """Tests that direct prints to stdout and stderr are not lost."""
    @with_loguru_logger
    def log_with_stdout(context=None):
        print("Direct stdout print")  # noqa: T201
        logger.info("Loguru info message")
        print("Error message", file=sys.stderr)  # noqa: T201

    log_with_stdout(context=MockDagsterContext())
    captured = capfd.readouterr()
    assert "Direct stdout print" in captured.out
    # The fixture logs to stdout, and the mock handler also prints to stdout
    assert "[dagster.info] Loguru info message" in captured.out
    assert "Error message" in captured.err


def test_direct_loguru_usage(capfd, setup_logger):
    """Tests direct logger calls without any Dagster context or decorator."""
    logger.debug("==Debug== Loguru is working!")
    captured = capfd.readouterr()
    assert "[dagster.debug] ==Debug== Loguru is working!" in captured.out


def test_context_log_with_loguru_decorator(capfd, setup_logger):
    """Tests a simple context.log call under the decorator."""
    context = MockDagsterContext()
    @with_loguru_logger
    def simulate_op(context=None):
        context.log.info("This is a context.log.info message")
    simulate_op(context=context)
    captured = capfd.readouterr()
    assert "[dagster.info] This is a context.log.info message" in captured.out

