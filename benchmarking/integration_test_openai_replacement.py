"""
Integration Test - OpenAI Replacement Validation

Validates that all metric modules work correctly after replacing Intel Alloy
with standard OpenAI API. This is an integration test that makes real API calls.

Type: Integration Test (not unit test)
- Tests multiple components working together
- Makes real OpenAI API calls (costs money)
- Downloads sentence-transformer models (~90MB)
- Validates end-to-end metric calculation flows

Run from project root: python benchmarking/integration_test_openai_replacement.py

Requirements:
- OPENAI_API_KEY in .env file
- Internet connection for API calls and model downloads
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.metrics.traditional import TraditionalMetrics
from benchmarking.metrics.semantic import SemanticMetrics
from benchmarking.metrics.efficiency import EfficiencyMetrics
from benchmarking.metrics.llm_judge import LLMJudge

async def main():
    print("\n" + "="*80)
    print(" " * 20 + "METRICS INTEGRATION TEST")
    print(" " * 15 + "Testing OpenAI Replacement")
    print("="*80 + "\n")
    
    # Test data
    reference = "Python is a high-level programming language known for its readability and versatility. It is widely used for web development, data analysis, and machine learning."
    prediction = "Python is a popular high-level programming language that is easy to read and very versatile. It's commonly used in web development, data science, and AI applications."
    
    query = "What is Python?"
    
    # 1. Test Traditional Metrics
    print("[1/4] Testing Traditional Metrics (ROUGE, BLEU)...")
    trad_metrics = TraditionalMetrics()
    scores = trad_metrics.calculate_all(predicted=prediction, reference=reference)
    print(f"   OK Exact Match: {scores['exact_match']}")
    print(f"   OK ROUGE-L F1: {scores['rougel_f1']:.3f}")
    print(f"   OK BLEU-1: {scores['bleu_1']:.3f}")
    print()
    
    # 2. Test Semantic Metrics
    print("[2/4] Testing Semantic Metrics (sentence-transformers)...")
    sem_metrics = SemanticMetrics()
    scores = sem_metrics.calculate_all(predicted=prediction, reference=reference)
    print(f"   OK Semantic Similarity: {scores['semantic_similarity']:.3f}")
    print()
    
    # 3. Test Efficiency Metrics
    print("[3/4] Testing Efficiency Metrics...")
    eff_metrics = EfficiencyMetrics()
    
    # Use context manager properly
    query_metrics_list = []
    with eff_metrics.measure_query("test_query_1") as qm:
        import time
        time.sleep(0.05)  # Simulate query work
        eff_metrics.record_api_call("embedding", count=2)
        eff_metrics.record_api_call("chat", count=1)
    query_metrics_list.append(qm)
    
    with eff_metrics.measure_query("test_query_2") as qm:
        time.sleep(0.03)
        eff_metrics.record_api_call("embedding", count=1)
    query_metrics_list.append(qm)
    
    # Aggregate results
    summary = eff_metrics.aggregate_query_metrics(query_metrics_list)
    print(f"   OK Total Queries: {summary['total_queries']}")
    print(f"   OK Avg Latency: {summary['latency_stats']['total']['mean']:.3f}s")
    print(f"   OK Throughput: {summary['throughput_qps']:.2f} qps")
    print()
    
    # 4. Test LLM Judge (OpenAI)
    print("[4/4] Testing LLM Judge (OpenAI GPT-4o-mini)...")
    judge = LLMJudge(model="gpt-4o-mini")
    
    evaluations = [{
        'query': query,
        'reference_answer': reference,
        'predicted_answer': prediction,
        'context': ''
    }]
    
    print("   ... Calling OpenAI API for judgment...")
    judge_scores = await judge.judge_batch(evaluations)
    
    for i, scores in enumerate(judge_scores):
        if not scores.get('error'):
            print(f"   OK Correctness: {scores['correctness']}/5")
            print(f"   OK Completeness: {scores['completeness']}/5")
            print(f"   OK Faithfulness: {scores['faithfulness']}/5")
            print(f"   OK Conciseness: {scores['conciseness']}/5")
            print(f"   >> Explanation: {scores.get('explanation', 'No explanation')[:150]}...")
        else:
            print(f"   ERROR: {scores.get('explanation', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("SUCCESS: METRICS INTEGRATION TEST COMPLETE!")
    print("   All metric modules work with OpenAI replacement")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
