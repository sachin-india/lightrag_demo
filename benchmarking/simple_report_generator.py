"""
Simple report generator that works with EvaluationPipeline output format
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def generate_simple_report(data: dict, output_file: Path):
    """Generate a simple markdown report from evaluation results"""
    
    lines = []
    
    # Header
    dataset_name = "MS MARCO" if "ms_marco" in str(output_file).lower() else "HotpotQA"
    lines.append(f"# LightRAG Benchmark Report: {dataset_name}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if 'timestamp' in data:
        lines.append(f"**Benchmark Run**: {data['timestamp']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Dataset Info
    dataset_samples = data.get('dataset_samples', [])
    query_results = data.get('query_results', [])
    
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- **Total Samples**: {len(dataset_samples)}")
    lines.append(f"- **Total Queries**: {len(query_results)}")
    lines.append("")
    
    # Knowledge Graph Stats
    metrics = data.get('metrics', {})
    graph_metrics = metrics.get('graph', {})
    
    if '_global_stats' in graph_metrics:
        stats = graph_metrics['_global_stats']
        lines.append("## Knowledge Graph")
        lines.append("")
        lines.append(f"- **Nodes**: {stats.get('num_nodes', 0)}")
        lines.append(f"- **Edges**: {stats.get('num_edges', 0)}")
        lines.append(f"- **Entities**: {stats.get('num_entities', 0)}")
        lines.append(f"- **Relations**: {stats.get('num_relations', 0)}")
        lines.append("")
    
    # Metrics by Mode
    aggregated = metrics.get('aggregated', {})
    
    if aggregated:
        lines.append("## Metrics by Query Mode")
        lines.append("")
        
        for mode in ['naive', 'local', 'global', 'hybrid']:
            if mode not in aggregated:
                continue
                
            mode_data = aggregated[mode]
            lines.append(f"### {mode.upper()} Mode")
            lines.append("")
            
            # Traditional Metrics
            lines.append("#### Traditional Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| ROUGE-1 | {mode_data.get('rouge-1', 0):.4f} |")
            lines.append(f"| ROUGE-2 | {mode_data.get('rouge-2', 0):.4f} |")
            lines.append(f"| ROUGE-L | {mode_data.get('rouge-l', 0):.4f} |")
            lines.append(f"| BLEU | {mode_data.get('bleu', 0):.4f} |")
            lines.append(f"| F1 Score | {mode_data.get('f1', 0):.4f} |")
            lines.append(f"| Exact Match | {mode_data.get('exact_match_rate', 0):.4f} |")
            lines.append("")
            
            # Containment Metrics
            containment = mode_data.get('containment', {})
            if containment:
                lines.append("#### Containment Metrics")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Exact Match | {containment.get('exact_match', 0):.4f} |")
                lines.append(f"| Normalized Match | {containment.get('normalized_match', 0):.4f} |")
                lines.append(f"| Token Overlap Ratio | {containment.get('token_overlap_ratio', 0):.4f} |")
                lines.append(f"| All Tokens Present | {containment.get('all_tokens_present', 0):.4f} |")
                lines.append("")
            
            # Semantic Similarity
            sem_sim = mode_data.get('semantic_similarity', 0)
            lines.append("#### Semantic Similarity")
            lines.append("")
            lines.append(f"- **Score**: {sem_sim:.4f}")
            lines.append("")
            
            # Graph Quality
            graph_quality = mode_data.get('graph_quality', {})
            if graph_quality:
                lines.append("#### Graph Quality")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Avg References | {graph_quality.get('avg_references', 0):.2f} |")
                lines.append(f"| Entity Density | {graph_quality.get('entity_density', 0):.3f} |")
                lines.append(f"| Token Diversity | {graph_quality.get('unique_token_ratio', 0):.3f} |")
                lines.append(f"| Reference Usage Rate | {graph_quality.get('reference_usage_rate', 0):.3f} |")
                lines.append("")
            
            # LLM Judge
            llm_judge = mode_data.get('llm_judge', {})
            if llm_judge:
                lines.append("#### LLM Judge Evaluation (GPT-4o-mini)")
                lines.append("")
                lines.append("| Dimension | Score |")
                lines.append("|-----------|-------|")
                lines.append(f"| Correctness | {llm_judge.get('correctness', 0):.2f}/5 |")
                lines.append(f"| Completeness | {llm_judge.get('completeness', 0):.2f}/5 |")
                lines.append(f"| Faithfulness | {llm_judge.get('faithfulness', 0):.2f}/5 |")
                lines.append(f"| Conciseness | {llm_judge.get('conciseness', 0):.2f}/5 |")
                lines.append(f"| **Overall** | **{llm_judge.get('overall', 0):.2f}/5** |")
                lines.append("")
            
            # Latency
            avg_latency = mode_data.get('avg_latency_seconds', 0)
            lines.append(f"#### Performance")
            lines.append("")
            lines.append(f"- **Average Latency**: {avg_latency:.2f}s")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # Sample Queries
    if query_results and len(query_results) > 0:
        lines.append("## Sample Query Results")
        lines.append("")
        
        for i, result in enumerate(query_results[:3], 1):  # Show first 3
            lines.append(f"### Query {i}")
            lines.append("")
            lines.append(f"**Question**: {result.get('query', 'N/A')}")
            lines.append("")
            lines.append(f"**Reference Answer**: {result.get('reference_answer', 'N/A')[:200]}...")
            lines.append("")
            
            mode_results = result.get('mode_results', {})
            for mode in ['naive', 'local', 'global', 'hybrid']:
                if mode in mode_results and 'response' in mode_results[mode]:
                    response = mode_results[mode]['response']
                    lines.append(f"**{mode.upper()} Response**: {response[:200]}...")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
    
    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_file

def find_latest_results(results_dir: Path, prefix: str) -> list:
    """Find all result files matching prefix, sorted by timestamp (newest first)"""
    pattern = f"{prefix}_*.json"
    timestamped_files = sorted(results_dir.glob(pattern), reverse=True)
    
    # Also check for non-timestamped version
    fallback = results_dir / f"{prefix}.json"
    if fallback.exists() and fallback not in timestamped_files:
        timestamped_files.append(fallback)
    
    return timestamped_files

def extract_timestamp(filepath: Path, data: dict) -> str:
    """Extract timestamp from filename or data, return descriptive name"""
    stem = filepath.stem
    
    # Check if filename has timestamp like ms_marco_results_20251030_123456
    parts = stem.split('_')
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit() and len(parts[-2]) == 8:
        # Has timestamp format YYYYMMDD_HHMMSS
        return f"{parts[-2]}_{parts[-1]}"
    
    # Check if data has sample count info
    dataset_samples = data.get('dataset_samples', [])
    if dataset_samples:
        sample_count = len(dataset_samples)
        return f"{sample_count}sample" + ("s" if sample_count != 1 else "")
    
    return "latest"

def main():
    results_dir = Path("benchmarking/results")
    reports_dir = Path("benchmarking/reports")
    
    print("\n📊 Finding latest benchmark results...")
    
    # Find all MS MARCO results
    ms_marco_files = find_latest_results(results_dir, "ms_marco_results")
    if ms_marco_files:
        for ms_marco_file in ms_marco_files[:1]:  # Process only the latest
            print(f"   📝 MS MARCO: {ms_marco_file.name}")
            
            with open(ms_marco_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            timestamp = extract_timestamp(ms_marco_file, data)
            output = reports_dir / f"ms_marco_report_{timestamp}.md"
            generate_simple_report(data, output)
            print(f"      ✅ Generated: {output.name}")
    else:
        print("   ⚠️  No MS MARCO results found")
    
    # Find all HotpotQA results
    hotpot_files = find_latest_results(results_dir, "hotpot_qa_results")
    if hotpot_files:
        for hotpot_file in hotpot_files[:1]:  # Process only the latest
            print(f"   📝 HotpotQA: {hotpot_file.name}")
            
            with open(hotpot_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            timestamp = extract_timestamp(hotpot_file, data)
            output = reports_dir / f"hotpot_qa_report_{timestamp}.md"
            generate_simple_report(data, output)
            print(f"      ✅ Generated: {output.name}")
    else:
        print("   ⚠️  No HotpotQA results found")
    
    print("\n✅ Reports generated successfully!")

if __name__ == "__main__":
    main()
