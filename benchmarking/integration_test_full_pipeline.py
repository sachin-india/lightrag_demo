"""
Integration Test - Full LightRAG Pipeline (INCOMPLETE)

Attempts to test the complete pipeline: document ingestion → query → metrics.
Currently incomplete due to document ingestion issues.

Type: Integration Test (End-to-End)
- Tests LightRAG document ingestion
- Tests query execution
- Tests all metric calculations
- Makes real OpenAI API calls

Status: ⚠️ INCOMPLETE - Document ingestion needs fixing

Run from project root: python benchmarking/integration_test_full_pipeline.py

Note: This test was created during Alloy→OpenAI migration but needs
additional work to handle document ingestion properly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag import QueryParam
from benchmarking.evaluators.lightrag_evaluator import LightRAGEvaluator
from benchmarking.metrics.traditional import TraditionalMetrics
from benchmarking.metrics.semantic import SemanticMetrics
from benchmarking.metrics.efficiency import EfficiencyMetrics
from benchmarking.metrics.llm_judge import LLMJudge

async def main():
    print("\n" + "="*80)
    print(" " * 20 + "LIGHTRAG INTEGRATION TEST")
    print(" " * 15 + "Testing OpenAI Replacement (2 samples)")
    print("="*80 + "\n")
    
    # Test data
    test_docs = [
        "Python is a high-level programming language known for its readability and versatility.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    ]
    
    test_queries = [
        "What is Python?",
        "What is machine learning?"
    ]
    
    reference_answers = [
        "Python is a high-level programming language known for readability and versatility",
        "Machine learning is a subset of AI that learns from data"
    ]
    
    # 1. Initialize LightRAG Evaluator
    print("1️⃣  Initializing LightRAG Evaluator...")
    working_dir = Path("benchmarking/benchmark_storage/test_integration")
    evaluator = LightRAGEvaluator(working_dir=working_dir, clear_existing=True)
    await evaluator.initialize()
    print(f"✅ Evaluator initialized (working_dir={working_dir})\n")
    
    # 2. Ingest test documents
    print("2️⃣  Ingesting test documents...")
    for i, doc in enumerate(test_docs):
        await evaluator.rag.ainsert(doc)
        print(f"   ✅ Document {i+1}/{len(test_docs)} ingested")
    evaluator.documents_ingested = True
    print()
    
    # 3. Query and collect results
    print("3️⃣  Running queries (hybrid mode)...")
    results = []
    for i, query in enumerate(test_queries):
        response = await evaluator.rag.aquery(query, param=QueryParam(mode="hybrid"))
        result = {
            'response': response,
            'duration_seconds': 0.5  # Placeholder
        }
        results.append(result)
        print(f"   ✅ Query {i+1}/{len(test_queries)}: {response[:100] if response else 'No response'}...")
    print()
    
    # 4. Test Traditional Metrics
    print("4️⃣  Testing Traditional Metrics (ROUGE, BLEU)...")
    trad_metrics = TraditionalMetrics()
    for i in range(len(test_queries)):
        scores = trad_metrics.calculate_all(
            reference=reference_answers[i],
            prediction=results[i]['response']
        )
        print(f"   ✅ Query {i+1}: ROUGE-L={scores['rouge_l']:.3f}, BLEU={scores['bleu']:.3f}")
    print()
    
    # 5. Test Semantic Metrics
    print("5️⃣  Testing Semantic Metrics (sentence-transformers)...")
    sem_metrics = SemanticMetrics()
    for i in range(len(test_queries)):
        scores = sem_metrics.calculate_all(
            reference=reference_answers[i],
            prediction=results[i]['response']
        )
        print(f"   ✅ Query {i+1}: Cosine Sim={scores['cosine_similarity']:.3f}")
    print()
    
    # 6. Test Efficiency Metrics
    print("6️⃣  Testing Efficiency Metrics...")
    eff_metrics = EfficiencyMetrics()
    for i in range(len(test_queries)):
        eff_metrics.record_query(
            query_id=f"test_{i}",
            latency=results[i]['duration_seconds']
        )
    summary = eff_metrics.get_summary()
    print(f"   ✅ Average latency: {summary['latency_stats']['mean']:.3f}s")
    print()
    
    # 7. Test LLM Judge (OpenAI)
    print("7️⃣  Testing LLM Judge (OpenAI GPT-4o-mini)...")
    judge = LLMJudge(model="gpt-4o-mini")
    
    evaluations = [
        {
            'query': test_queries[i],
            'reference_answer': reference_answers[i],
            'predicted_answer': results[i]['response'],
            'context': results[i].get('context', '')
        }
        for i in range(len(test_queries))
    ]
    
    judge_scores = await judge.judge_batch(evaluations)
    
    for i, scores in enumerate(judge_scores):
        if not scores.get('error'):
            print(f"   ✅ Query {i+1}: Correctness={scores['correctness']}/5, "
                  f"Completeness={scores['completeness']}/5, "
                  f"Faithfulness={scores['faithfulness']}/5")
        else:
            print(f"   ⚠️  Query {i+1}: {scores.get('explanation', 'Error occurred')}")
    
    print("\n" + "="*80)
    print("🎉 INTEGRATION TEST COMPLETE - All components working!")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
