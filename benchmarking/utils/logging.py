"""
Structured Logging for Benchmark System

Provides comprehensive logging with context, performance tracking,
and easy debugging capabilities.

Adapted from Unified_RAG benchmarking system for LightRAG integration.
Simplified to work standalone without config dependencies.
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager


class BenchmarkFormatter(logging.Formatter):
    """Custom formatter for benchmark logs with colors and structure"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to level name
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Format timestamp
        record.timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Add operation context if available
        operation = getattr(record, 'operation', None)
        if operation:
            record.operation_info = f"[{operation}] "
        else:
            record.operation_info = ""
        
        # Format the message
        fmt = "%(timestamp)s %(levelname)-8s %(operation_info)s%(message)s"
        formatter = logging.Formatter(fmt)
        return formatter.format(record)


class BenchmarkLogger:
    """Enhanced logger for benchmark operations"""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None, 
                 debug_mode: bool = False, verbose: bool = True):
        self.logger = logging.getLogger(name)
        self.operation_stack = []
        self.performance_data = {}
        self.log_dir = log_dir or Path("benchmarks/results/logs")
        
        if not self.logger.handlers:
            self._setup_logger(debug_mode, verbose)
    
    def _setup_logger(self, debug_mode: bool, verbose: bool):
        """Set up logger with appropriate handlers and formatters"""
        
        # Set level based on configuration
        if debug_mode:
            level = logging.DEBUG
        elif verbose:
            level = logging.INFO
        else:
            level = logging.WARNING
        
        self.logger.setLevel(level)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(BenchmarkFormatter())
        
        # Ensure UTF-8 encoding for Windows compatibility
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except:
                pass  # Fallback gracefully if reconfigure not available
        
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging (optional)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = self.log_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _add_context(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add current operation context to log record"""
        context = extra or {}
        
        if self.operation_stack:
            context['operation'] = self.operation_stack[-1]
        
        return context
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(message, extra=self._add_context(kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, extra=self._add_context(kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, extra=self._add_context(kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, extra=self._add_context(kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self.logger.critical(message, extra=self._add_context(kwargs))
    
    @contextmanager
    def operation(self, operation_name: str, **context):
        """Context manager for tracking operations with timing"""
        self.operation_stack.append(operation_name)
        start_time = time.time()
        
        # Log operation start
        context_str = " ".join(f"{k}={v}" for k, v in context.items())
        self.info(f">> Starting {operation_name}" + (f" ({context_str})" if context_str else ""))
        
        try:
            yield self
            
            # Log successful completion
            elapsed = time.time() - start_time
            self.performance_data[operation_name] = elapsed
            self.info(f"<< Completed {operation_name} in {elapsed:.3f}s")
            
        except Exception as error:
            # Log operation failure
            elapsed = time.time() - start_time
            self.error(f"XX Failed {operation_name} after {elapsed:.3f}s: {error}")
            raise
            
        finally:
            self.operation_stack.pop()
    
    def log_performance(self, metrics: Dict[str, float]):
        """Log performance metrics"""
        self.info("Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"   {metric}: {value:.3f}")
            else:
                self.info(f"   {metric}: {value}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration details"""
        self.debug("⚙️  Configuration:")
        for key, value in config.items():
            self.debug(f"   {key}: {value}")
    
    def log_dataset_info(self, dataset_name: str, sample_count: int, **info):
        """Log dataset information"""
        self.info(f"📚 Dataset {dataset_name}: {sample_count} samples")
        for key, value in info.items():
            self.info(f"   {key}: {value}")
    
    def log_evaluation_result(self, query_id: str, success: bool, metrics: Dict[str, float]):
        """Log evaluation result"""
        status = "✅" if success else "❌"
        self.info(f"{status} Query {query_id}: {success}")
        
        if success and metrics:
            for metric, value in metrics.items():
                if isinstance(value, float):
                    self.debug(f"   {metric}: {value:.4f}")


# Global logger instances
_loggers: Dict[str, BenchmarkLogger] = {}


def get_logger(name: str = "benchmark", **kwargs) -> BenchmarkLogger:
    """Get logger instance for given name"""
    if name not in _loggers:
        _loggers[name] = BenchmarkLogger(name, **kwargs)
    return _loggers[name]


# Convenience functions for common logging patterns
def log_dataset_loading(dataset_name: str, sample_count: int, **kwargs):
    """Log dataset loading information"""
    logger = get_logger("dataset")
    logger.log_dataset_info(dataset_name, sample_count, **kwargs)


def log_embedding_progress(current: int, total: int, batch_size: int = None):
    """Log embedding progress"""
    logger = get_logger("embedding")
    
    percent = (current / total) * 100 if total > 0 else 0
    batch_info = f" (batch size: {batch_size})" if batch_size else ""
    
    logger.info(f"📊 Embedding progress: {current}/{total} ({percent:.1f}%){batch_info}")


def log_retrieval_result(query: str, num_retrieved: int, top_score: float = None):
    """Log retrieval result"""
    logger = get_logger("retrieval")
    
    score_info = f" (top score: {top_score:.3f})" if top_score else ""
    logger.info(f"🔍 Retrieved {num_retrieved} passages for query: {query[:50]}...{score_info}")


def log_evaluation_summary(dataset_name: str, total_queries: int, successful: int, 
                          avg_metrics: Dict[str, float]):
    """Log evaluation summary"""
    logger = get_logger("evaluation")
    
    success_rate = (successful / total_queries) * 100 if total_queries > 0 else 0
    
    logger.info(f"📈 Evaluation Summary for {dataset_name}:")
    logger.info(f"   Queries: {successful}/{total_queries} ({success_rate:.1f}% success)")
    
    for metric, value in avg_metrics.items():
        logger.info(f"   {metric}: {value:.4f}")


# Context managers for common operations
@contextmanager
def log_operation(operation_name: str, logger_name: str = "benchmark", **context):
    """Context manager for logging operations"""
    logger = get_logger(logger_name)
    with logger.operation(operation_name, **context) as log:
        yield log


# Testing and example usage
if __name__ == "__main__":
    print("🧪 Testing Benchmark Logger")
    
    # Create logger
    logger = get_logger("test", debug_mode=True)
    
    # Test basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test operation tracking
    with logger.operation("test_operation", dataset="test_ds"):
        logger.info("Doing some work...")
        time.sleep(0.1)
    
    # Test performance logging
    logger.log_performance({"metric1": 0.95, "metric2": 0.87})
    
    # Test dataset logging
    logger.log_dataset_info("TestDataset", 100, source="huggingface")
    
    # Test convenience functions
    log_embedding_progress(50, 100, batch_size=10)
    log_retrieval_result("What is LightRAG?", 5, top_score=0.92)
    log_evaluation_summary("TestDataset", 100, 98, {"accuracy": 0.95, "f1": 0.93})
    
    print("\n✅ All logging tests passed!")
