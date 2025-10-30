"""
Test suite for benchmarking utility modules

Validates errors.py, retry.py, and logging.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.errors import (
    BenchmarkError, ConfigurationError, DatasetError, APIError,
    IntelComplianceError, EmbeddingDimensionError, handle_error
)
from utils.retry import (
    retry_with_backoff, RetryContext, calculate_delay,
    should_retry, retry_api_call
)
from utils.logging import (
    get_logger, log_dataset_loading, log_embedding_progress,
    log_retrieval_result, log_evaluation_summary, log_operation
)


def test_errors():
    """Test error handling module"""
    print("=" * 60)
    print("🧪 Testing Error Handling")
    print("=" * 60)
    
    # Test 1: Basic BenchmarkError
    try:
        raise BenchmarkError(
            "Test error",
            error_code="TEST_001",
            context={"key": "value"},
            suggestions=["Try this", "Or that"]
        )
    except BenchmarkError as e:
        assert e.error_code == "TEST_001"
        assert "key" in e.context
        assert len(e.suggestions) == 2
        print("✅ Test 1 passed: Basic BenchmarkError")
    
    # Test 2: Specific error types
    try:
        raise IntelComplianceError("gpt-4", "chat")
    except ConfigurationError as e:
        assert "INTEL_COMPLIANCE" in e.error_code
        assert "gpt-4" in e.context.values()
        print("✅ Test 2 passed: IntelComplianceError")
    
    # Test 3: Embedding dimension error
    try:
        raise EmbeddingDimensionError(1536, 768)
    except BenchmarkError as e:
        assert e.context["expected_dimension"] == 1536
        assert e.context["actual_dimension"] == 768
        print("✅ Test 3 passed: EmbeddingDimensionError")
    
    # Test 4: Error handling
    try:
        raise FileNotFoundError("test.json")
    except Exception as e:
        handled = handle_error(e, {"operation": "load_data"})
        assert isinstance(handled, DatasetError)
        assert "FILE_001" in handled.error_code
        print("✅ Test 4 passed: File error handling")
    
    # Test 5: Connection error handling
    try:
        raise ConnectionError("Network timeout")
    except Exception as e:
        handled = handle_error(e)
        assert isinstance(handled, APIError)
        assert "CONNECTION_001" in handled.error_code
        print("✅ Test 5 passed: Connection error handling")
    
    # Test 6: Full error message
    error = BenchmarkError(
        "Test message",
        error_code="TEST_002",
        context={"a": 1, "b": 2},
        suggestions=["Suggestion 1", "Suggestion 2"]
    )
    full_msg = error.get_full_message()
    assert "Test message" in full_msg
    assert "TEST_002" in full_msg
    assert "Suggestion 1" in full_msg
    print("✅ Test 6 passed: Full error message generation")
    
    print("\n✅ All Error Handling Tests Passed!\n")


def test_retry():
    """Test retry logic module"""
    print("=" * 60)
    print("🧪 Testing Retry Logic")
    print("=" * 60)
    
    # Test 1: Successful function (no retry needed)
    call_count = [0]
    
    @retry_with_backoff(max_retries=3)
    def successful_function():
        call_count[0] += 1
        return "success"
    
    result = successful_function()
    assert result == "success"
    assert call_count[0] == 1
    print("✅ Test 1 passed: Successful function (no retry)")
    
    # Test 2: Transient failure then success
    attempt_count = [0]
    
    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert attempt_count[0] == 3
    print("✅ Test 2 passed: Retry on transient failure")
    
    # Test 3: Permanent failure
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def failing_function():
        raise ValueError("Permanent failure")
    
    try:
        failing_function()
        assert False, "Should have raised exception"
    except BenchmarkError as e:
        assert "Permanent failure" in str(e)
        print("✅ Test 3 passed: Permanent failure handled")
    
    # Test 4: Calculate delay
    delay1 = calculate_delay(1, 1.0, 60.0, 2.0, jitter=False)
    delay2 = calculate_delay(2, 1.0, 60.0, 2.0, jitter=False)
    delay3 = calculate_delay(3, 1.0, 60.0, 2.0, jitter=False)
    
    assert delay1 == 1.0
    assert delay2 == 2.0
    assert delay3 == 4.0
    print("✅ Test 4 passed: Exponential backoff calculation")
    
    # Test 5: Should retry logic
    connection_error = ConnectionError("Test")
    assert should_retry(connection_error) == True
    
    config_error = BenchmarkError("Config error", error_code="CONFIG_001")
    assert should_retry(config_error) == False
    
    value_error = ValueError("Test")
    assert should_retry(value_error) == False
    print("✅ Test 5 passed: Should retry logic")
    
    # Test 6: RetryContext
    print("\nTest 6: RetryContext")
    success_count = [0]
    
    with RetryContext("test_operation", max_retries=2) as ctx:
        def test_func():
            success_count[0] += 1
            if success_count[0] < 2:
                raise ConnectionError("Retry me")
            return "context_success"
        
        result = ctx.retry(test_func)
        assert result == "context_success"
        assert success_count[0] == 2
    
    print("✅ Test 6 passed: RetryContext with retry")
    
    # Test 7: Retry with callback
    callback_calls = [0]
    
    def on_retry_callback(attempt, error):
        callback_calls[0] += 1
    
    retry_count = [0]
    
    @retry_with_backoff(max_retries=2, base_delay=0.01, on_retry=on_retry_callback)
    def callback_test():
        retry_count[0] += 1
        if retry_count[0] < 2:
            raise ConnectionError("Test")
        return "done"
    
    result = callback_test()
    assert result == "done"
    assert callback_calls[0] == 1  # Called once for the retry
    print("✅ Test 7 passed: Retry with callback")
    
    print("\n✅ All Retry Logic Tests Passed!\n")


def test_logging():
    """Test logging module"""
    print("=" * 60)
    print("🧪 Testing Logging")
    print("=" * 60)
    
    # Test 1: Create logger
    logger = get_logger("test_logger", debug_mode=True)
    assert logger is not None
    print("✅ Test 1 passed: Logger creation")
    
    # Test 2: Basic logging (visual confirmation)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print("✅ Test 2 passed: Basic logging levels")
    
    # Test 3: Operation tracking with timing
    with logger.operation("test_operation", dataset="test"):
        time.sleep(0.05)
    
    assert "test_operation" in logger.performance_data
    assert logger.performance_data["test_operation"] >= 0.05
    print("✅ Test 3 passed: Operation tracking with timing")
    
    # Test 4: Performance logging
    logger.log_performance({"metric1": 0.95, "metric2": 0.87, "count": 100})
    print("✅ Test 4 passed: Performance logging")
    
    # Test 5: Dataset logging
    logger.log_dataset_info("TestDataset", 100, source="huggingface", format="json")
    print("✅ Test 5 passed: Dataset logging")
    
    # Test 6: Evaluation result logging
    logger.log_evaluation_result("query_001", True, {"rouge": 0.85, "bleu": 0.78})
    logger.log_evaluation_result("query_002", False, {})
    print("✅ Test 6 passed: Evaluation result logging")
    
    # Test 7: Convenience functions
    log_dataset_loading("TestDS", 50, splits=["train", "test"])
    log_embedding_progress(25, 100, batch_size=5)
    log_retrieval_result("What is LightRAG?", 10, top_score=0.92)
    log_evaluation_summary("TestDS", 100, 95, {"accuracy": 0.95})
    print("✅ Test 7 passed: Convenience logging functions")
    
    # Test 8: Log operation context manager
    with log_operation("context_test", logger_name="test_ops"):
        time.sleep(0.02)
    print("✅ Test 8 passed: Log operation context manager")
    
    # Test 9: Nested operations
    with logger.operation("outer_operation"):
        logger.info("In outer operation")
        with logger.operation("inner_operation"):
            logger.info("In inner operation")
            time.sleep(0.01)
    
    assert "outer_operation" in logger.performance_data
    assert "inner_operation" in logger.performance_data
    print("✅ Test 9 passed: Nested operations")
    
    # Test 10: Logger singleton behavior
    logger2 = get_logger("test_logger")
    assert logger2 is logger
    print("✅ Test 10 passed: Logger singleton behavior")
    
    print("\n✅ All Logging Tests Passed!\n")


def test_integration():
    """Test integration between utility modules"""
    print("=" * 60)
    print("🧪 Testing Utility Integration")
    print("=" * 60)
    
    logger = get_logger("integration_test", debug_mode=False, verbose=True)
    
    # Test 1: Retry with logging
    attempt_num = [0]
    
    @retry_with_backoff(
        max_retries=2,
        base_delay=0.01,
        on_retry=lambda attempt, error: logger.warning(f"Retry {attempt}: {error}")
    )
    def integrated_function():
        attempt_num[0] += 1
        if attempt_num[0] < 2:
            raise APIError("API temporarily unavailable", error_code="API_TEMP")
        return "success"
    
    with logger.operation("integrated_retry_operation"):
        result = integrated_function()
    
    assert result == "success"
    assert attempt_num[0] == 2
    print("✅ Test 1 passed: Retry with logging integration")
    
    # Test 2: Error handling with logging
    try:
        with logger.operation("error_test_operation"):
            raise DatasetError("Dataset not found", error_code="DS_404")
    except BenchmarkError as e:
        logger.error(f"Caught error: {e.get_full_message()}")
        assert e.error_code == "DS_404"
    
    print("✅ Test 2 passed: Error handling with logging")
    
    # Test 3: Full workflow simulation
    with logger.operation("full_workflow_simulation"):
        # Simulate data loading with retry
        @retry_with_backoff(max_retries=1, base_delay=0.01)
        def load_data():
            logger.info("Loading dataset...")
            return {"samples": 100}
        
        data = load_data()
        log_dataset_loading("SimulatedDataset", data["samples"])
        
        # Simulate processing
        logger.info(f"Processing {data['samples']} samples")
        
        # Simulate metrics
        metrics = {"accuracy": 0.95, "f1": 0.92}
        logger.log_performance(metrics)
    
    print("✅ Test 3 passed: Full workflow simulation")
    
    print("\n✅ All Integration Tests Passed!\n")


def run_all_tests():
    """Run all utility tests"""
    print("\n" + "=" * 60)
    print("🚀 LightRAG Benchmarking Utilities Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_errors()
        test_retry()
        test_logging()
        test_integration()
        
        print("=" * 60)
        print("🎉 ALL UTILITY TESTS PASSED!")
        print("=" * 60)
        print("\n✅ All utility modules are working correctly")
        print("✅ errors.py - Error handling ✓")
        print("✅ retry.py - Retry logic ✓")
        print("✅ logging.py - Structured logging ✓")
        print("✅ Integration between modules ✓")
        print("\n✅ Ready for Phase 1.6: Foundation integration tests")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
