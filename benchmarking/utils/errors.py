"""
Custom Exceptions for Benchmark System

Provides structured error handling with context and recovery suggestions.

Copied from Unified_RAG benchmarking system for LightRAG integration.
"""

from typing import Optional, Dict, Any, List


class BenchmarkError(Exception):
    """Base exception for benchmark system"""
    
    def __init__(self, message: str, error_code: str = None, 
                 context: Dict[str, Any] = None, suggestions: List[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.suggestions = suggestions or []
    
    def get_full_message(self) -> str:
        """Get detailed error message with context and suggestions"""
        lines = [str(self)]
        
        if self.error_code:
            lines.append(f"Error Code: {self.error_code}")
        
        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
        
        if self.suggestions:
            lines.append("Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")
        
        return "\n".join(lines)


class ConfigurationError(BenchmarkError):
    """Configuration-related errors"""
    pass


class DatasetError(BenchmarkError):
    """Dataset loading and processing errors"""
    pass


class EmbeddingError(BenchmarkError):
    """Embedding generation and storage errors"""
    pass


class RetrievalError(BenchmarkError):
    """Vector retrieval and search errors"""
    pass


class GenerationError(BenchmarkError):
    """Answer generation errors"""
    pass


class MetricCalculationError(BenchmarkError):
    """Metric calculation errors"""
    pass


class VectorStoreError(BenchmarkError):
    """Vector database errors"""
    pass


class APIError(BenchmarkError):
    """API communication errors"""
    pass


class EvaluationError(BenchmarkError):
    """Evaluation and benchmarking execution errors"""
    pass


class ValidationError(BenchmarkError):
    """Data validation errors"""
    pass


# Specific error instances for common issues
class IntelComplianceError(ConfigurationError):
    """Error when non-Intel-approved models are used"""
    
    def __init__(self, model_name: str, model_type: str):
        super().__init__(
            f"Model '{model_name}' is not Intel-approved for {model_type}",
            error_code="INTEL_COMPLIANCE_001",
            context={"model_name": model_name, "model_type": model_type},
            suggestions=[
                "Use only Intel-approved models",
                "Check the configuration documentation",
                "Contact IT for approved model list"
            ]
        )


class EmbeddingDimensionError(EmbeddingError):
    """Error when embedding dimensions don't match"""
    
    def __init__(self, expected_dim: int, actual_dim: int):
        super().__init__(
            f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}",
            error_code="EMBEDDING_001",
            context={"expected_dimension": expected_dim, "actual_dimension": actual_dim},
            suggestions=[
                "Check embedding model configuration",
                "Verify vector store compatibility",
                "Clear and rebuild vector store if needed"
            ]
        )


class DatasetFormatError(DatasetError):
    """Error when dataset format is unexpected"""
    
    def __init__(self, dataset_name: str, expected_format: str, actual_format: str):
        super().__init__(
            f"Dataset '{dataset_name}' format error: expected {expected_format}, got {actual_format}",
            error_code="DATASET_001",
            context={
                "dataset_name": dataset_name,
                "expected_format": expected_format,
                "actual_format": actual_format
            },
            suggestions=[
                "Check dataset version compatibility",
                "Update dataset loaders",
                "Verify HuggingFace dataset structure"
            ]
        )


class APIQuotaError(APIError):
    """Error when API quota is exceeded"""
    
    def __init__(self, api_name: str, quota_type: str):
        super().__init__(
            f"API quota exceeded for {api_name}: {quota_type}",
            error_code="API_001",
            context={"api_name": api_name, "quota_type": quota_type},
            suggestions=[
                "Wait for quota reset",
                "Use smaller batch sizes",
                "Check API usage limits",
                "Consider using different model"
            ]
        )


class VectorStoreCorruptionError(VectorStoreError):
    """Error when vector store data is corrupted"""
    
    def __init__(self, store_path: str, corruption_type: str):
        super().__init__(
            f"Vector store corruption detected at '{store_path}': {corruption_type}",
            error_code="VECTOR_001",
            context={"store_path": store_path, "corruption_type": corruption_type},
            suggestions=[
                "Clear and rebuild vector store",
                "Check disk space and permissions",
                "Verify ChromaDB installation",
                "Restore from backup if available"
            ]
        )


def handle_error(error: Exception, context: Dict[str, Any] = None) -> BenchmarkError:
    """Convert generic exceptions to structured benchmark errors"""
    
    error_context = context or {}
    
    # Map common exceptions to benchmark errors
    if isinstance(error, FileNotFoundError):
        return DatasetError(
            f"Required file not found: {error}",
            error_code="FILE_001",
            context=error_context,
            suggestions=[
                "Check file path",
                "Verify file permissions", 
                "Ensure required data is downloaded"
            ]
        )
    
    elif isinstance(error, MemoryError):
        return BenchmarkError(
            f"Out of memory: {error}",
            error_code="MEMORY_001",
            context=error_context,
            suggestions=[
                "Reduce batch size",
                "Use smaller dataset samples",
                "Close other applications",
                "Consider using a machine with more RAM"
            ]
        )
    
    elif isinstance(error, ConnectionError):
        return APIError(
            f"Connection error: {error}",
            error_code="CONNECTION_001",
            context=error_context,
            suggestions=[
                "Check internet connection",
                "Verify API endpoints",
                "Check firewall settings",
                "Try again later"
            ]
        )
    
    elif isinstance(error, KeyError):
        return ValidationError(
            f"Missing required key: {error}",
            error_code="VALIDATION_001",
            context=error_context,
            suggestions=[
                "Check data structure",
                "Verify configuration completeness",
                "Update data format handling"
            ]
        )
    
    # If it's already a BenchmarkError, return as-is
    elif isinstance(error, BenchmarkError):
        return error
    
    # Generic error wrapper
    else:
        return BenchmarkError(
            f"Unexpected error: {error}",
            error_code="GENERIC_001",
            context=error_context,
            suggestions=[
                "Check logs for more details",
                "Verify system requirements",
                "Contact support if issue persists"
            ]
        )


# Testing and example usage
if __name__ == "__main__":
    print("🧪 Testing Benchmark Errors")
    
    # Test basic error
    try:
        raise BenchmarkError("Test error", error_code="TEST_001", 
                           context={"key": "value"},
                           suggestions=["Try this", "Or this"])
    except BenchmarkError as e:
        print(f"\n{e.get_full_message()}")
    
    # Test specific error
    try:
        raise IntelComplianceError("gpt-4", "chat")
    except BenchmarkError as e:
        print(f"\n{e.get_full_message()}")
    
    # Test error handling
    try:
        raise FileNotFoundError("test.json")
    except Exception as e:
        handled = handle_error(e, {"operation": "load_data"})
        print(f"\n{handled.get_full_message()}")
    
    print("\n✅ All error tests passed!")
