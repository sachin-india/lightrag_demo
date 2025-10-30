"""
Efficiency and performance metrics

System performance tracking for RAG evaluation.
Measures latency, throughput, memory usage, and API costs.

Copied from Unified_RAG benchmarking system for LightRAG integration.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot"""
    timestamp: float
    memory_mb: float
    cpu_percent: float


@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""
    query_id: str
    total_latency: float = 0.0
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    embedding_latency: float = 0.0
    
    api_calls: Dict[str, int] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    num_chunks_retrieved: int = 0
    num_tokens_generated: int = 0
    
    # Cost estimation (if available)
    estimated_cost: float = 0.0


class EfficiencyMetrics:
    """Performance and efficiency tracking for RAG systems"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring_active = False
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self._monitor_thread = None
        self._api_call_counts = {'embeddings': 0, 'chat': 0}
        
    def start_monitoring(self, interval: float = 0.1):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.performance_snapshots.clear()
        
        def monitor():
            while self.monitoring_active:
                try:
                    snapshot = PerformanceSnapshot(
                        timestamp=time.time(),
                        memory_mb=self.process.memory_info().rss / 1024 / 1024,
                        cpu_percent=self.process.cpu_percent()
                    )
                    self.performance_snapshots.append(snapshot)
                    time.sleep(interval)
                except Exception:
                    break
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    @contextmanager
    def measure_query(self, query_id: str):
        """Context manager for measuring query performance"""
        metrics = QueryMetrics(query_id=query_id)
        
        # Start monitoring
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        initial_api_counts = self._api_call_counts.copy()
        
        try:
            yield metrics
        finally:
            # Calculate totals
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            metrics.total_latency = end_time - start_time
            metrics.memory_usage = {
                'start_mb': start_memory,
                'end_mb': end_memory,
                'peak_mb': max([s.memory_mb for s in self.performance_snapshots[-50:]] + [end_memory]),
                'delta_mb': end_memory - start_memory
            }
            
            # API call counts
            for api_type in self._api_call_counts:
                metrics.api_calls[api_type] = (
                    self._api_call_counts[api_type] - initial_api_counts.get(api_type, 0)
                )
    
    @contextmanager
    def measure_component(self, component_name: str):
        """Context manager for measuring individual component performance"""
        start_time = time.time()
        try:
            yield
        finally:
            latency = time.time() - start_time
            # This would be used within measure_query context
            # The calling code would set the appropriate latency field
            pass
    
    def record_api_call(self, api_type: str, count: int = 1):
        """Record API call for cost tracking"""
        if api_type in self._api_call_counts:
            self._api_call_counts[api_type] += count
    
    def calculate_throughput(self, num_queries: int, total_time: float) -> float:
        """Calculate queries per second"""
        return num_queries / total_time if total_time > 0 else 0.0
    
    def calculate_latency_percentiles(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not latencies:
            return {}
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            'p50': sorted_latencies[int(0.5 * n)],
            'p75': sorted_latencies[int(0.75 * n)],
            'p90': sorted_latencies[int(0.90 * n)],
            'p95': sorted_latencies[int(0.95 * n)],
            'p99': sorted_latencies[int(0.99 * n)] if n >= 100 else sorted_latencies[-1],
            'mean': sum(latencies) / len(latencies),
            'min': min(latencies),
            'max': max(latencies)
        }
    
    def estimate_cost(self, metrics: QueryMetrics, 
                     embedding_cost_per_1k: float = 0.0001,
                     chat_cost_per_1k_tokens: float = 0.002) -> float:
        """
        Estimate cost per query (if pricing available)
        
        Args:
            metrics: Query metrics
            embedding_cost_per_1k: Cost per 1K embedding tokens
            chat_cost_per_1k_tokens: Cost per 1K chat tokens
            
        Returns:
            Estimated cost in dollars
        """
        cost = 0.0
        
        # Embedding costs (rough estimate)
        if metrics.num_chunks_retrieved > 0:
            # Assume ~100 tokens per chunk for embedding
            embedding_tokens = metrics.num_chunks_retrieved * 100
            cost += (embedding_tokens / 1000) * embedding_cost_per_1k
        
        # Generation costs
        if metrics.num_tokens_generated > 0:
            cost += (metrics.num_tokens_generated / 1000) * chat_cost_per_1k_tokens
        
        return cost
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_process_id': self.process.pid,
            'system_platform': psutil.disk_usage('/').total if hasattr(psutil, 'disk_usage') else 'unknown'
        }
    
    def aggregate_query_metrics(self, query_metrics_list: List[QueryMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across multiple queries"""
        if not query_metrics_list:
            return {}
        
        # Extract latency lists
        total_latencies = [m.total_latency for m in query_metrics_list]
        retrieval_latencies = [m.retrieval_latency for m in query_metrics_list if m.retrieval_latency > 0]
        generation_latencies = [m.generation_latency for m in query_metrics_list if m.generation_latency > 0]
        
        # Calculate aggregated statistics
        results = {
            'total_queries': len(query_metrics_list),
            'total_runtime': sum(total_latencies),
            'throughput_qps': len(query_metrics_list) / sum(total_latencies) if sum(total_latencies) > 0 else 0,
            
            # Latency statistics
            'latency_stats': {
                'total': self.calculate_latency_percentiles(total_latencies),
                'retrieval': self.calculate_latency_percentiles(retrieval_latencies),
                'generation': self.calculate_latency_percentiles(generation_latencies)
            },
            
            # API usage
            'total_api_calls': {
                'embeddings': sum(m.api_calls.get('embeddings', 0) for m in query_metrics_list),
                'chat': sum(m.api_calls.get('chat', 0) for m in query_metrics_list)
            },
            
            # Resource usage
            'memory_stats': {
                'peak_mb': max(m.memory_usage.get('peak_mb', 0) for m in query_metrics_list),
                'avg_delta_mb': sum(m.memory_usage.get('delta_mb', 0) for m in query_metrics_list) / len(query_metrics_list)
            },
            
            # Cost estimation
            'total_estimated_cost': sum(m.estimated_cost for m in query_metrics_list),
            'avg_cost_per_query': sum(m.estimated_cost for m in query_metrics_list) / len(query_metrics_list)
        }
        
        return results


# Integration helper for AlloyLLM API tracking
class AlloyAPITracker:
    """Helper to track Alloy LLM API calls for cost calculation"""
    
    def __init__(self, efficiency_metrics: EfficiencyMetrics):
        self.efficiency_metrics = efficiency_metrics
    
    def track_embedding_call(self, texts: List[str]):
        """Track embedding API call"""
        self.efficiency_metrics.record_api_call('embeddings', len(texts))
    
    def track_chat_call(self, response_length: int = 0):
        """Track chat API call"""
        self.efficiency_metrics.record_api_call('chat', 1)


# Testing and example usage
if __name__ == "__main__":
    # Test efficiency metrics
    print("🧪 Testing Efficiency Metrics")
    
    metrics = EfficiencyMetrics()
    
    # Test query measurement
    with metrics.measure_query("test_query_1") as query_metrics:
        # Simulate work
        time.sleep(0.1)
        
        # Simulate component measurements
        with metrics.measure_component("retrieval"):
            time.sleep(0.05)
            query_metrics.retrieval_latency = 0.05
        
        with metrics.measure_component("generation"):
            time.sleep(0.03)
            query_metrics.generation_latency = 0.03
        
        # Simulate API calls
        metrics.record_api_call('embeddings', 5)
        metrics.record_api_call('chat', 1)
        
        query_metrics.num_chunks_retrieved = 5
        query_metrics.num_tokens_generated = 150
        query_metrics.estimated_cost = metrics.estimate_cost(query_metrics)
    
    print(f"Query latency: {query_metrics.total_latency:.3f}s")
    print(f"Memory usage: {query_metrics.memory_usage}")
    print(f"API calls: {query_metrics.api_calls}")
    print(f"Estimated cost: ${query_metrics.estimated_cost:.6f}")
    
    # Test aggregation
    query_list = [query_metrics]
    aggregated = metrics.aggregate_query_metrics(query_list)
    print(f"\n📊 Aggregated Metrics:")
    print(f"Throughput: {aggregated['throughput_qps']:.2f} QPS")
    print(f"Total cost: ${aggregated['total_estimated_cost']:.6f}")
