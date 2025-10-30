"""
Retry Logic for Benchmark System

Provides robust retry mechanisms with exponential backoff,
jitter, and intelligent error handling.

Adapted from Unified_RAG benchmarking system for LightRAG integration.
Config dependencies removed for standalone operation.
"""

import time
import random
import functools
from typing import Callable, Any, Optional, Type, Union, List
from dataclasses import dataclass

from .errors import BenchmarkError, APIError, handle_error


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


# Default retry configuration
_DEFAULT_RETRY_CONFIG = RetryConfig()


def get_retry_config() -> RetryConfig:
    """Get retry configuration (can be overridden later with config module)"""
    return _DEFAULT_RETRY_CONFIG


def calculate_delay(attempt: int, base_delay: float, max_delay: float, 
                   exponential_base: float, jitter: bool = True) -> float:
    """Calculate delay for retry attempt with exponential backoff"""
    
    # Calculate exponential delay
    delay = base_delay * (exponential_base ** (attempt - 1))
    
    # Cap at max delay
    delay = min(delay, max_delay)
    
    # Add jitter to prevent thundering herd
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


def should_retry(error: Exception, retryable_errors: List[Type[Exception]] = None) -> bool:
    """Determine if an error should trigger a retry"""
    
    if retryable_errors is None:
        # Default retryable errors
        retryable_errors = [
            ConnectionError,
            TimeoutError,
            APIError,
            BenchmarkError  # Will check specific error codes
        ]
    
    # Check if error type is retryable
    if not any(isinstance(error, err_type) for err_type in retryable_errors):
        return False
    
    # Special handling for BenchmarkError
    if isinstance(error, BenchmarkError):
        # Don't retry configuration errors
        if error.error_code and error.error_code.startswith('CONFIG'):
            return False
        
        # Don't retry validation errors
        if error.error_code and error.error_code.startswith('VALIDATION'):
            return False
        
        # Don't retry Intel compliance errors
        if error.error_code and error.error_code.startswith('INTEL_COMPLIANCE'):
            return False
    
    return True


def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exponential_base: Optional[float] = None,
    retryable_errors: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts (uses config default if None)
        base_delay: Base delay between retries (uses config default if None)
        max_delay: Maximum delay cap (uses config default if None)
        exponential_base: Exponential backoff base (uses config default if None)
        retryable_errors: List of exception types that should trigger retries
        on_retry: Callback function called on each retry attempt
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get configuration
            retry_config = get_retry_config()
            
            # Use provided values or fall back to config
            _max_retries = max_retries if max_retries is not None else retry_config.max_retries
            _base_delay = base_delay if base_delay is not None else retry_config.base_delay
            _max_delay = max_delay if max_delay is not None else retry_config.max_delay
            _exponential_base = exponential_base if exponential_base is not None else retry_config.exponential_base
            
            last_error = None
            
            for attempt in range(_max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as error:
                    last_error = error
                    
                    # Convert to structured error if needed
                    structured_error = handle_error(error, {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_retries": _max_retries
                    })
                    
                    # Check if we should retry
                    if attempt >= _max_retries or not should_retry(structured_error, retryable_errors):
                        raise structured_error
                    
                    # Calculate delay
                    delay = calculate_delay(attempt + 1, _base_delay, _max_delay, _exponential_base)
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, structured_error)
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_error
        
        return wrapper
    return decorator


class RetryContext:
    """Context manager for retry operations with detailed logging"""
    
    def __init__(self, operation_name: str, max_retries: Optional[int] = None):
        self.operation_name = operation_name
        self.max_retries = max_retries or get_retry_config().max_retries
        self.attempt = 0
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.attempt = 1
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            elapsed = time.time() - self.start_time
            print(f"✅ {self.operation_name} succeeded on attempt {self.attempt} ({elapsed:.2f}s)")
        else:
            # Failure
            elapsed = time.time() - self.start_time
            print(f"❌ {self.operation_name} failed after {self.attempt} attempts ({elapsed:.2f}s)")
        
        return False  # Don't suppress exceptions
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic within context"""
        
        retry_config = get_retry_config()
        
        for attempt in range(self.max_retries + 1):
            self.attempt = attempt + 1
            
            try:
                if attempt > 0:
                    delay = calculate_delay(attempt, retry_config.base_delay, 
                                          retry_config.max_delay, retry_config.exponential_base)
                    print(f"🔄 Retrying {self.operation_name} (attempt {self.attempt}/{self.max_retries + 1}) after {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"🚀 Starting {self.operation_name}...")
                
                return func(*args, **kwargs)
                
            except Exception as error:
                structured_error = handle_error(error, {
                    "operation": self.operation_name,
                    "attempt": self.attempt,
                    "max_retries": self.max_retries
                })
                
                if attempt >= self.max_retries or not should_retry(structured_error):
                    print(f"💥 {self.operation_name} failed permanently: {structured_error}")
                    raise structured_error
                
                print(f"⚠️  {self.operation_name} failed on attempt {self.attempt}: {error}")
        
        # Should never reach here
        raise RuntimeError(f"Retry logic error for {self.operation_name}")


# Convenience functions for common retry patterns
def retry_api_call(func: Callable, *args, **kwargs) -> Any:
    """Retry API calls with appropriate settings"""
    
    @retry_with_backoff(
        retryable_errors=[ConnectionError, TimeoutError, APIError],
        on_retry=lambda attempt, error: print(f"🔄 API call retry {attempt}: {error}")
    )
    def wrapper():
        return func(*args, **kwargs)
    
    return wrapper()


def retry_embedding_generation(func: Callable, *args, **kwargs) -> Any:
    """Retry embedding generation with appropriate settings"""
    
    @retry_with_backoff(
        max_retries=5,  # More retries for embeddings
        retryable_errors=[ConnectionError, APIError],
        on_retry=lambda attempt, error: print(f"🔄 Embedding retry {attempt}: {error}")
    )
    def wrapper():
        return func(*args, **kwargs)
    
    return wrapper()


def retry_vector_operation(func: Callable, *args, **kwargs) -> Any:
    """Retry vector database operations with appropriate settings"""
    
    @retry_with_backoff(
        retryable_errors=[ConnectionError, Exception],  # Broader for DB issues
        on_retry=lambda attempt, error: print(f"🔄 Vector DB retry {attempt}: {error}")
    )
    def wrapper():
        return func(*args, **kwargs)
    
    return wrapper()


# Testing and example usage
if __name__ == "__main__":
    print("🧪 Testing Retry Logic")
    
    # Test 1: Successful function
    @retry_with_backoff(max_retries=3)
    def successful_function():
        print("  Executing successful function")
        return "success"
    
    result = successful_function()
    assert result == "success"
    print("✅ Test 1 passed: Successful function")
    
    # Test 2: Function that fails then succeeds
    attempt_count = [0]
    
    @retry_with_backoff(max_retries=3, base_delay=0.1)
    def flaky_function():
        attempt_count[0] += 1
        print(f"  Attempt {attempt_count[0]}")
        if attempt_count[0] < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert attempt_count[0] == 3
    print("✅ Test 2 passed: Retry on transient failure")
    
    # Test 3: Function that always fails
    @retry_with_backoff(max_retries=2, base_delay=0.1)
    def failing_function():
        print("  Executing failing function")
        raise ValueError("Permanent failure")
    
    try:
        failing_function()
        assert False, "Should have raised exception"
    except BenchmarkError as e:
        print(f"  Caught expected error: {e.error_code}")
        print("✅ Test 3 passed: Permanent failure handled")
    
    # Test 4: RetryContext
    print("\nTest 4: RetryContext")
    with RetryContext("test_operation", max_retries=2) as ctx:
        result = ctx.retry(lambda: "context_success")
        assert result == "context_success"
    
    print("\n✅ All retry tests passed!")
