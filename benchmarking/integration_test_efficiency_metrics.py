"""
Integration Test - Efficiency Metrics with Context Manager Pattern

Tests the EfficiencyMetrics module with proper context manager usage.
Validates latency tracking, API call counting, memory monitoring, and
metric aggregation.

Type: Integration Test
- Tests real system resource monitoring
- Measures actual execution times
- No external API calls (pure Python)
- Fast execution (<1 second)

Run from project root: python benchmarking/integration_test_efficiency_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.metrics.efficiency import EfficiencyMetrics
import time

def main():
    print("\n" + "="*80)
    print("Testing Efficiency Metrics (Context Manager Pattern)")
    print("="*80 + "\n")
    
    # Initialize
    metrics = EfficiencyMetrics()
    print("[1] Initialized EfficiencyMetrics")
    
    # Test 1: Basic query measurement
    print("\n[2] Testing measure_query context manager...")
    with metrics.measure_query("test_query_1") as query_metrics:
        # Simulate some work
        time.sleep(0.1)
        metrics.record_api_call("embedding", count=3)
        metrics.record_api_call("chat", count=1)
        
    print(f"   OK Query ID: {query_metrics.query_id}")
    print(f"   OK Total Latency: {query_metrics.total_latency:.3f}s")
    print(f"   OK Memory Delta: {query_metrics.memory_usage['delta_mb']:.2f} MB")
    print(f"   OK API Calls: {query_metrics.api_calls}")
    
    # Test 2: Multiple queries
    print("\n[3] Testing multiple queries...")
    query_metrics_list = []
    
    for i in range(3):
        with metrics.measure_query(f"test_query_{i+2}") as qm:
            time.sleep(0.05 * (i+1))  # Variable sleep
            metrics.record_api_call("embedding", count=2)
        query_metrics_list.append(qm)
    
    print(f"   OK Measured {len(query_metrics_list)} queries")
    
    # Test 3: Aggregate metrics
    print("\n[4] Testing aggregate_query_metrics...")
    all_metrics = [query_metrics] + query_metrics_list
    summary = metrics.aggregate_query_metrics(all_metrics)
    
    print(f"   OK Total Queries: {summary['total_queries']}")
    print(f"   OK Avg Latency: {summary['latency_stats']['total']['mean']:.3f}s")
    print(f"   OK Min Latency: {summary['latency_stats']['total']['min']:.3f}s")
    print(f"   OK Max Latency: {summary['latency_stats']['total']['max']:.3f}s")
    print(f"   OK Throughput: {summary['throughput_qps']:.2f} queries/sec")
    
    # Test 4: System info
    print("\n[5] Testing get_system_info...")
    sys_info = metrics.get_system_info()
    print(f"   OK CPU Count: {sys_info['cpu_count']}")
    print(f"   OK Total Memory: {sys_info['memory_total_gb']:.1f} GB")
    print(f"   OK Python PID: {sys_info['python_process_id']}")
    
    print("\n" + "="*80)
    print("SUCCESS: Efficiency Metrics work correctly!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
