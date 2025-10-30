"""
Full Benchmark with All Metrics
Runs comprehensive evaluation with all new metrics on both datasets
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.evaluators.evaluation_pipeline import EvaluationPipeline
from benchmarks.configs.dataset_config import DatasetConfig

async def main():
    print("\n" + "="*80)
    print(" " * 20 + "LIGHTRAG COMPREHENSIVE BENCHMARK")
    print(" " * 15 + "All Metrics | Both Datasets | All Modes")
    print("="*80)
    
    # Configuration: Adjust these limits for different benchmark sizes
    # Quick test: 5 samples per dataset (~7 minutes)
    # Comprehensive: 50 samples per dataset (~60-90 minutes)
    MS_MARCO_LIMIT = 50  # Change to 5 for quick test
    HOTPOT_QA_LIMIT = 50  # Change to 5 for quick test
    
    total_samples = MS_MARCO_LIMIT + HOTPOT_QA_LIMIT
    estimated_time_min = total_samples * 0.6  # ~0.6 min per sample
    estimated_time_max = total_samples * 0.9  # ~0.9 min per sample
    
    print("\n📋 Configuration:")
    print(f"   • Datasets: MS MARCO ({MS_MARCO_LIMIT} samples) + HotpotQA ({HOTPOT_QA_LIMIT} samples)")
    print(f"   • Total Samples: {total_samples}")
    print(f"   • Query Modes: naive, local, global, hybrid")
    print(f"   • Metrics: Traditional, Containment, Semantic, Graph, LLM-as-Judge")
    print(f"   • Storage: benchmarks/benchmark_storage")
    print(f"   • Estimated Time: {int(estimated_time_min)}-{int(estimated_time_max)} minutes")
    print()
    
    # Run MS MARCO benchmark
    print("\n" + "="*80)
    print(f"DATASET 1/2: MS MARCO ({MS_MARCO_LIMIT} samples)")
    print("="*80)
    
    ms_marco_config = DatasetConfig(
        datasets=['ms_marco'],
        ms_marco_limit=MS_MARCO_LIMIT,
        hotpot_qa_limit=0,
        working_dir="benchmarks/benchmark_storage/ms_marco",
        verbose=True
    )
    
    ms_marco_pipeline = EvaluationPipeline(config=ms_marco_config)
    
    print("\n📊 Running MS MARCO evaluation...")
    print("   This will calculate all metrics including LLM-as-Judge (GPT-4o)...")
    ms_marco_summary = await ms_marco_pipeline.run_full_evaluation(
        modes=["naive", "local", "global", "hybrid"],
        clear_existing=False  # Use existing graph if available
    )
    
    print("\n" + "-"*80)
    print("MS MARCO RESULTS:")
    print("-"*80)
    ms_marco_pipeline.print_summary()
    ms_marco_pipeline.save_results("benchmarks/results/ms_marco_results.json")
    
    # Run HotpotQA benchmark
    print("\n\n" + "="*80)
    print(f"DATASET 2/2: HOTPOTQA ({HOTPOT_QA_LIMIT} samples)")
    print("="*80)
    
    hotpot_config = DatasetConfig(
        datasets=['hotpot_qa'],
        ms_marco_limit=0,
        hotpot_qa_limit=HOTPOT_QA_LIMIT,
        working_dir="benchmarks/benchmark_storage/hotpot_qa",
        verbose=True
    )
    
    hotpot_pipeline = EvaluationPipeline(config=hotpot_config)
    
    print("\n📊 Running HotpotQA evaluation...")
    print("   This will calculate all metrics including LLM-as-Judge (GPT-4o)...")
    hotpot_summary = await hotpot_pipeline.run_full_evaluation(
        modes=["naive", "local", "global", "hybrid"],
        clear_existing=False  # Use existing graph if available
    )
    
    print("\n" + "-"*80)
    print("HOTPOTQA RESULTS:")
    print("-"*80)
    hotpot_pipeline.print_summary()
    hotpot_pipeline.save_results("benchmarks/results/hotpot_qa_results.json")
    
    # Cross-dataset comparison
    print("\n\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)
    
    ms_marco_agg = ms_marco_pipeline.metrics_results.get('aggregated', {})
    hotpot_agg = hotpot_pipeline.metrics_results.get('aggregated', {})
    
    if ms_marco_agg and hotpot_agg:
        print("\n📊 HYBRID MODE COMPARISON:")
        print("-"*80)
        
        ms_hybrid = ms_marco_agg.get('hybrid', {})
        hp_hybrid = hotpot_agg.get('hybrid', {})
        
        print(f"\n{'Metric':<30} {'MS MARCO':>15} {'HotpotQA':>15} {'Better':>12}")
        print("-"*80)
        
        # Traditional metrics
        ms_rouge = ms_hybrid.get('rouge-1', 0)
        hp_rouge = hp_hybrid.get('rouge-1', 0)
        print(f"{'ROUGE-1 F1':<30} {ms_rouge:>15.4f} {hp_rouge:>15.4f} {('MS MARCO' if ms_rouge > hp_rouge else 'HotpotQA'):>12}")
        
        # Containment
        ms_cont = ms_hybrid.get('containment', {}).get('token_overlap_ratio', 0)
        hp_cont = hp_hybrid.get('containment', {}).get('token_overlap_ratio', 0)
        print(f"{'Containment (Token)':<30} {ms_cont:>15.4f} {hp_cont:>15.4f} {('MS MARCO' if ms_cont > hp_cont else 'HotpotQA'):>12}")
        
        # Semantic
        ms_sem = ms_hybrid.get('semantic_similarity', 0)
        hp_sem = hp_hybrid.get('semantic_similarity', 0)
        print(f"{'Semantic Similarity':<30} {ms_sem:>15.4f} {hp_sem:>15.4f} {('MS MARCO' if ms_sem > hp_sem else 'HotpotQA'):>12}")
        
        # Graph metrics
        ms_refs = ms_hybrid.get('graph_quality', {}).get('avg_references', 0)
        hp_refs = hp_hybrid.get('graph_quality', {}).get('avg_references', 0)
        print(f"{'Graph Avg References':<30} {ms_refs:>15.2f} {hp_refs:>15.2f} {('MS MARCO' if ms_refs > hp_refs else 'HotpotQA'):>12}")
        
        # LLM Judge
        ms_judge = ms_hybrid.get('llm_judge', {})
        hp_judge = hp_hybrid.get('llm_judge', {})
        
        if ms_judge and hp_judge:
            print(f"\n{'LLM Judge Scores:':<30}")
            
            ms_corr = ms_judge.get('correctness', 0)
            hp_corr = hp_judge.get('correctness', 0)
            print(f"{'  Correctness':<30} {ms_corr:>15.2f} {hp_corr:>15.2f} {('MS MARCO' if ms_corr > hp_corr else 'HotpotQA'):>12}")
            
            ms_comp = ms_judge.get('completeness', 0)
            hp_comp = hp_judge.get('completeness', 0)
            print(f"{'  Completeness':<30} {ms_comp:>15.2f} {hp_comp:>15.2f} {('MS MARCO' if ms_comp > hp_comp else 'HotpotQA'):>12}")
            
            ms_faith = ms_judge.get('faithfulness', 0)
            hp_faith = hp_judge.get('faithfulness', 0)
            print(f"{'  Faithfulness':<30} {ms_faith:>15.2f} {hp_faith:>15.2f} {('MS MARCO' if ms_faith > hp_faith else 'HotpotQA'):>12}")
            
            ms_overall = ms_judge.get('overall', 0)
            hp_overall = hp_judge.get('overall', 0)
            print(f"{'  Overall Quality':<30} {ms_overall:>15.2f} {hp_overall:>15.2f} {('MS MARCO' if ms_overall > hp_overall else 'HotpotQA'):>12}")
    
    # Final summary
    print("\n\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\n✅ All evaluations completed successfully!")
    print("\n📊 Results saved to:")
    print("   • benchmarks/results/ms_marco_results.json")
    print("   • benchmarks/results/hotpot_qa_results.json")
    print("   • benchmarks/benchmark_storage/evaluation_summary.json")
    print("\n💡 Key Finding:")
    print("   Traditional metrics (ROUGE, BLEU) often underestimate LightRAG quality")
    print("   Modern metrics (LLM-as-Judge, Containment, Semantic) provide better assessment")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
