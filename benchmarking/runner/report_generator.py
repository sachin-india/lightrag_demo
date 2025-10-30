"""
Report Generator for LightRAG Benchmarking

Creates formatted reports with:
- Markdown and HTML output
- Metrics visualization tables
- Mode comparison charts
- Performance trends
- Summary statistics
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import time

from ..utils.logging import get_logger

logger = get_logger("report_generator")


class ReportGenerator:
    """
    Generate formatted benchmark reports.
    
    Supports Markdown and HTML output with tables,
    metrics visualization, and comparison views.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output reports (str or Path)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("benchmarks/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Report generator initialized (output={self.output_dir})")
    
    def generate_markdown_report(self,
                                 run_data: Dict[str, Any],
                                 output_file: Optional[Path] = None) -> Path:
        """
        Generate Markdown report for a benchmark run.
        
        Args:
            run_data: Benchmark run data
            output_file: Optional output file path
            
        Returns:
            Path to generated report
        """
        run_id = run_data.get('run_id', 'unknown')
        output_file = output_file or (self.output_dir / f"{run_id}_report.md")
        
        with logger.operation("generate_markdown_report", run_id=run_id):
            # Build report
            report_lines = []
            
            # Header
            report_lines.extend(self._generate_header(run_data))
            report_lines.append("")
            
            # Configuration
            report_lines.extend(self._generate_config_section(run_data))
            report_lines.append("")
            
            # Dataset Info
            report_lines.extend(self._generate_dataset_section(run_data))
            report_lines.append("")
            
            # Graph Stats
            report_lines.extend(self._generate_graph_section(run_data))
            report_lines.append("")
            
            # Metrics by Mode
            report_lines.extend(self._generate_metrics_section(run_data))
            report_lines.append("")
            
            # Performance
            report_lines.extend(self._generate_performance_section(run_data))
            
            # Write report
            report_text = "\n".join(report_lines)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            logger.info(f"✅ Generated Markdown report: {output_file}")
            return output_file
    
    def generate_comparison_report(self,
                                   comparison_data: Dict[str, Any],
                                   output_file: Optional[Path] = None) -> Path:
        """
        Generate comparison report for multiple runs.
        
        Args:
            comparison_data: Comparison results from BenchmarkRunner.compare_runs()
            output_file: Optional output file path
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file or (self.output_dir / f"comparison_{timestamp}.md")
        
        with logger.operation("generate_comparison_report", runs=len(comparison_data.get('runs', []))):
            report_lines = []
            
            # Header
            report_lines.append("# LightRAG Benchmark Comparison Report")
            report_lines.append("")
            report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**Runs Compared**: {len(comparison_data.get('runs', []))}")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
            
            # Run Overview
            report_lines.append("## Run Overview")
            report_lines.append("")
            report_lines.append("| Run ID | Timestamp | Duration (s) |")
            report_lines.append("|--------|-----------|--------------|")
            
            for i, run_id in enumerate(comparison_data.get('runs', [])):
                ts = comparison_data.get('timestamps', [])[i] if i < len(comparison_data.get('timestamps', [])) else 0
                duration = comparison_data.get('durations', [])[i] if i < len(comparison_data.get('durations', [])) else 0
                ts_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else 'N/A'
                report_lines.append(f"| {run_id} | {ts_str} | {duration:.2f} |")
            
            report_lines.append("")
            
            # Metrics Comparison
            report_lines.append("## Metrics Comparison by Mode")
            report_lines.append("")
            
            metrics_comp = comparison_data.get('metrics_comparison', {})
            for mode, mode_data in metrics_comp.items():
                report_lines.append(f"### {mode.upper()} Mode")
                report_lines.append("")
                
                if not mode_data:
                    report_lines.append("*No data available*")
                    report_lines.append("")
                    continue
                
                # Create comparison table
                report_lines.append("| Run ID | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | F1 | EM Rate | Latency (s) |")
                report_lines.append("|--------|---------|---------|---------|------|----|---------| ------------|")
                
                for run_metrics in mode_data:
                    run_id = run_metrics.get('run_id', 'N/A')
                    r1 = run_metrics.get('rouge-1', 0)
                    r2 = run_metrics.get('rouge-2', 0)
                    rl = run_metrics.get('rouge-l', 0)
                    bleu = run_metrics.get('bleu', 0)
                    f1 = run_metrics.get('f1', 0)
                    em = run_metrics.get('exact_match_rate', 0)
                    lat = run_metrics.get('avg_latency_seconds', 0)
                    
                    report_lines.append(
                        f"| {run_id} | {r1:.4f} | {r2:.4f} | {rl:.4f} | "
                        f"{bleu:.4f} | {f1:.4f} | {em:.4f} | {lat:.2f} |"
                    )
                
                report_lines.append("")
            
            # Write report
            report_text = "\n".join(report_lines)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            logger.info(f"✅ Generated comparison report: {output_file}")
            return output_file
    
    def _generate_header(self, run_data: Dict[str, Any]) -> List[str]:
        """Generate report header"""
        run_id = run_data.get('run_id', 'Unknown')
        status = run_data.get('status', 'unknown')
        timestamp = run_data.get('timestamp', time.time())
        duration = run_data.get('duration', 0)
        
        ts_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        status_emoji = {"completed": "✅", "failed": "❌", "running": "🔄"}
        emoji = status_emoji.get(status, "❓")
        
        return [
            f"# LightRAG Benchmark Report: {run_id}",
            "",
            f"**Status**: {emoji} {status.upper()}",
            f"**Timestamp**: {ts_str}",
            f"**Duration**: {duration:.2f} seconds",
            "",
            "---"
        ]
    
    def _generate_config_section(self, run_data: Dict[str, Any]) -> List[str]:
        """Generate configuration section"""
        config = run_data.get('config', {})
        modes = run_data.get('modes', [])
        
        lines = [
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Datasets | {', '.join(config.get('datasets', []))} |",
            f"| Query Modes | {', '.join(modes)} |",
            f"| MS MARCO Limit | {config.get('ms_marco_limit', 'N/A')} |",
            f"| HotpotQA Limit | {config.get('hotpot_qa_limit', 'N/A')} |",
            f"| Working Dir | {config.get('working_dir', 'N/A')} |"
        ]
        
        return lines
    
    def _generate_dataset_section(self, run_data: Dict[str, Any]) -> List[str]:
        """Generate dataset information section"""
        results = run_data.get('results', {})
        dataset = results.get('dataset', {})
        
        return [
            "## Dataset",
            "",
            f"- **Total Samples**: {dataset.get('total_samples', 0)}",
            f"- **Total Documents**: {dataset.get('total_documents', 0)}",
            f"- **Sources**: {', '.join(dataset.get('sources', []))}",
        ]
    
    def _generate_graph_section(self, run_data: Dict[str, Any]) -> List[str]:
        """Generate graph statistics section"""
        results = run_data.get('results', {})
        metrics = results.get('metrics', {})
        graph = metrics.get('graph', {})
        
        return [
            "## Knowledge Graph",
            "",
            f"- **Nodes**: {graph.get('num_nodes', 0):,}",
            f"- **Edges**: {graph.get('num_edges', 0):,}",
            f"- **Entities**: {graph.get('num_entities', 0):,}",
            f"- **Relations**: {graph.get('num_relations', 0):,}",
            f"- **Average Degree**: {graph.get('avg_degree', 0):.2f}",
            f"- **Density**: {graph.get('density', 0):.6f}",
        ]
    
    def _generate_metrics_section(self, run_data: Dict[str, Any]) -> List[str]:
        """Generate metrics by mode section"""
        results = run_data.get('results', {})
        metrics = results.get('metrics', {})
        aggregated = metrics.get('aggregated', {})
        
        lines = [
            "## Metrics by Query Mode",
            ""
        ]
        
        for mode in ["naive", "local", "global", "hybrid"]:
            if mode not in aggregated:
                continue
            
            mode_metrics = aggregated[mode]
            
            lines.extend([
                f"### {mode.upper()} Mode",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| ROUGE-1 | {mode_metrics.get('rouge-1', 0):.4f} |",
                f"| ROUGE-2 | {mode_metrics.get('rouge-2', 0):.4f} |",
                f"| ROUGE-L | {mode_metrics.get('rouge-l', 0):.4f} |",
                f"| BLEU | {mode_metrics.get('bleu', 0):.4f} |",
                f"| F1 Score | {mode_metrics.get('f1', 0):.4f} |",
                f"| Exact Match Rate | {mode_metrics.get('exact_match_rate', 0):.4f} |",
                f"| Avg Latency | {mode_metrics.get('avg_latency_seconds', 0):.2f}s |",
                f"| Queries | {mode_metrics.get('count', 0)} |",
                ""
            ])
        
        return lines
    
    def _generate_performance_section(self, run_data: Dict[str, Any]) -> List[str]:
        """Generate performance section"""
        results = run_data.get('results', {})
        ingestion = results.get('ingestion', {})
        
        return [
            "## Performance",
            "",
            "### Document Ingestion",
            "",
            f"- **Documents Ingested**: {ingestion.get('ingested', 0)}",
            f"- **Failed**: {ingestion.get('failed', 0)}",
            f"- **Duration**: {ingestion.get('duration_seconds', 0):.2f}s",
            f"- **Throughput**: {ingestion.get('docs_per_second', 0):.2f} docs/sec",
        ]


# Testing
if __name__ == "__main__":
    import time
    
    print("🧪 Testing Report Generator\n")
    
    # Create generator
    generator = ReportGenerator(output_dir=Path("test_reports"))
    
    # Sample run data
    sample_run = {
        'run_id': 'test_run_20251026',
        'status': 'completed',
        'timestamp': time.time(),
        'duration': 45.5,
        'config': {
            'datasets': ['ms_marco', 'hotpot_qa'],
            'ms_marco_limit': 50,
            'hotpot_qa_limit': 50,
            'working_dir': 'benchmarks/test'
        },
        'modes': ['hybrid', 'global'],
        'results': {
            'dataset': {
                'total_samples': 100,
                'total_documents': 400,
                'sources': ['ms_marco', 'hotpot_qa']
            },
            'metrics': {
                'graph': {
                    'num_nodes': 1250,
                    'num_edges': 3420,
                    'num_entities': 856,
                    'num_relations': 1147,
                    'avg_degree': 5.472,
                    'density': 0.00219
                },
                'aggregated': {
                    'hybrid': {
                        'rouge-1': 0.5987,
                        'rouge-2': 0.4123,
                        'rouge-l': 0.5434,
                        'bleu': 0.4567,
                        'f1': 0.6789,
                        'exact_match_rate': 0.26,
                        'avg_latency_seconds': 2.56,
                        'count': 100
                    },
                    'global': {
                        'rouge-1': 0.5678,
                        'rouge-2': 0.3891,
                        'rouge-l': 0.5123,
                        'bleu': 0.4234,
                        'f1': 0.6456,
                        'exact_match_rate': 0.22,
                        'avg_latency_seconds': 2.34,
                        'count': 100
                    }
                }
            },
            'ingestion': {
                'ingested': 400,
                'failed': 0,
                'duration_seconds': 12.3,
                'docs_per_second': 32.5
            }
        }
    }
    
    # Generate report
    report_file = generator.generate_markdown_report(sample_run)
    print(f"✅ Generated report: {report_file}")
    
    # Sample comparison data
    comparison_data = {
        'runs': ['run_001', 'run_002'],
        'timestamps': [time.time() - 3600, time.time()],
        'durations': [45.5, 43.2],
        'metrics_comparison': {
            'hybrid': [
                {
                    'run_id': 'run_001',
                    'rouge-1': 0.5987,
                    'rouge-2': 0.4123,
                    'rouge-l': 0.5434,
                    'bleu': 0.4567,
                    'f1': 0.6789,
                    'exact_match_rate': 0.26,
                    'avg_latency_seconds': 2.56
                },
                {
                    'run_id': 'run_002',
                    'rouge-1': 0.6121,
                    'rouge-2': 0.4289,
                    'rouge-l': 0.5598,
                    'bleu': 0.4712,
                    'f1': 0.6923,
                    'exact_match_rate': 0.28,
                    'avg_latency_seconds': 2.34
                }
            ]
        }
    }
    
    # Generate comparison report
    comp_file = generator.generate_comparison_report(comparison_data)
    print(f"✅ Generated comparison report: {comp_file}")
    
    print("\n✅ Report generator tests complete!")
