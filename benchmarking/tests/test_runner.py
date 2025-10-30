"""
Tests for Benchmark Runner and Report Generator

Tests the automated execution, result management,
and report generation capabilities.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.runner.benchmark_runner import BenchmarkRunner, BenchmarkRun
from benchmarks.runner.report_generator import ReportGenerator
from benchmarks.configs.dataset_config import DatasetConfig, QUICK_TEST_CONFIG


class TestBenchmarkRun:
    """Tests for BenchmarkRun"""
    
    @pytest.fixture
    def sample_run(self):
        """Create sample benchmark run"""
        config = DatasetConfig(datasets=["ms_marco"], ms_marco_limit=10)
        return BenchmarkRun(
            run_id="test_run_001",
            config=config,
            modes=["hybrid"]
        )
    
    def test_run_initialization(self, sample_run):
        """Test run initialization"""
        assert sample_run.run_id == "test_run_001"
        assert sample_run.modes == ["hybrid"]
        assert sample_run.status == "pending"
        assert sample_run.results is None
        assert sample_run.error is None
        
        print("✅ Run initialization test passed")
    
    def test_run_lifecycle(self, sample_run):
        """Test run lifecycle (start -> complete/fail)"""
        # Start
        sample_run.start()
        assert sample_run.status == "running"
        assert sample_run.start_time is not None
        
        # Add small delay to ensure duration > 0
        time.sleep(0.01)
        
        # Complete
        results = {'test': 'data'}
        sample_run.complete(results)
        assert sample_run.status == "completed"
        assert sample_run.results == results
        assert sample_run.end_time is not None
        assert sample_run.get_duration() > 0
        
        print("✅ Run lifecycle test passed")
    
    def test_run_failure(self, sample_run):
        """Test run failure handling"""
        sample_run.start()
        sample_run.fail("Test error")
        
        assert sample_run.status == "failed"
        assert sample_run.error == "Test error"
        assert sample_run.end_time is not None
        
        print("✅ Run failure test passed")
    
    def test_run_serialization(self, sample_run):
        """Test run to_dict and from_dict"""
        sample_run.start()
        sample_run.complete({'metric': 0.95})
        
        # Serialize
        data = sample_run.to_dict()
        assert data['run_id'] == "test_run_001"
        assert data['status'] == "completed"
        assert data['results'] == {'metric': 0.95}
        
        # Deserialize
        restored = BenchmarkRun.from_dict(data)
        assert restored.run_id == sample_run.run_id
        assert restored.status == sample_run.status
        assert restored.results == sample_run.results
        
        print("✅ Run serialization test passed")


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner"""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """Create runner with temporary storage"""
        return BenchmarkRunner(
            storage_dir=tmp_path / "runner_test",
            auto_save=True
        )
    
    def test_runner_initialization(self, runner):
        """Test runner initialization"""
        assert runner.storage_dir.exists()
        assert runner.auto_save == True
        assert isinstance(runner.runs, list)
        
        print("✅ Runner initialization test passed")
    
    def test_generate_run_id(self, runner):
        """Test run ID generation"""
        config = QUICK_TEST_CONFIG
        modes = ["hybrid", "global"]
        
        run_id1 = runner._generate_run_id(config, modes)
        run_id2 = runner._generate_run_id(config, modes)
        
        # Should be unique (different timestamps)
        assert run_id1.startswith("run_")
        assert run_id2.startswith("run_")
        
        print(f"✅ Run ID generation test passed: {run_id1}")
    
    def test_save_and_load_runs(self, runner):
        """Test saving and loading runs index"""
        # Create test run
        config = QUICK_TEST_CONFIG
        run = BenchmarkRun("test_save_001", config, ["hybrid"])
        run.start()
        run.complete({'test': 'data'})
        
        runner.runs.append(run)
        runner._save_runs()
        
        # Verify file exists
        runs_file = runner.storage_dir / "runs_index.json"
        assert runs_file.exists()
        
        # Load runs
        runner.runs = []
        runner._load_runs()
        
        assert len(runner.runs) == 1
        assert runner.runs[0].run_id == "test_save_001"
        
        print("✅ Save and load runs test passed")
    
    def test_save_run_results(self, runner):
        """Test saving individual run results"""
        config = QUICK_TEST_CONFIG
        run = BenchmarkRun("test_results_001", config, ["hybrid"])
        run.start()
        run.complete({'metrics': {'f1': 0.95}})
        
        runner._save_run_results(run)
        
        # Verify file exists
        run_file = runner.storage_dir / f"{run.run_id}.json"
        assert run_file.exists()
        
        # Verify content
        with open(run_file, 'r') as f:
            data = json.load(f)
        assert data['run_id'] == "test_results_001"
        assert data['results']['metrics']['f1'] == 0.95
        
        print("✅ Save run results test passed")
    
    def test_get_run(self, runner):
        """Test retrieving run by ID"""
        config = QUICK_TEST_CONFIG
        run = BenchmarkRun("test_get_001", config, ["hybrid"])
        runner.runs.append(run)
        
        # Get existing run
        found = runner.get_run("test_get_001")
        assert found is not None
        assert found.run_id == "test_get_001"
        
        # Get non-existent run
        not_found = runner.get_run("nonexistent")
        assert not_found is None
        
        print("✅ Get run test passed")
    
    def test_filter_runs_by_status(self, runner):
        """Test filtering runs by status"""
        config = QUICK_TEST_CONFIG
        
        # Create runs with different statuses
        run1 = BenchmarkRun("completed_001", config, ["hybrid"])
        run1.start()
        run1.complete({'data': 1})
        
        run2 = BenchmarkRun("failed_001", config, ["hybrid"])
        run2.start()
        run2.fail("Test error")
        
        run3 = BenchmarkRun("pending_001", config, ["hybrid"])
        
        runner.runs = [run1, run2, run3]
        
        # Test filters
        completed = runner.get_completed_runs()
        assert len(completed) == 1
        assert completed[0].run_id == "completed_001"
        
        failed = runner.get_failed_runs()
        assert len(failed) == 1
        assert failed[0].run_id == "failed_001"
        
        print("✅ Filter runs by status test passed")
    
    def test_compare_runs(self, runner):
        """Test comparing multiple runs"""
        config = QUICK_TEST_CONFIG
        
        # Create runs with metrics
        run1 = BenchmarkRun("compare_001", config, ["hybrid"])
        run1.start()
        run1.complete({
            'metrics': {
                'aggregated': {
                    'hybrid': {
                        'rouge-1': 0.50,
                        'bleu': 0.40,
                        'f1': 0.60
                    }
                }
            }
        })
        
        run2 = BenchmarkRun("compare_002", config, ["hybrid"])
        run2.start()
        run2.complete({
            'metrics': {
                'aggregated': {
                    'hybrid': {
                        'rouge-1': 0.55,
                        'bleu': 0.45,
                        'f1': 0.65
                    }
                }
            }
        })
        
        runner.runs = [run1, run2]
        
        # Compare
        comparison = runner.compare_runs(["compare_001", "compare_002"])
        
        assert len(comparison['runs']) == 2
        assert 'metrics_comparison' in comparison
        assert 'hybrid' in comparison['metrics_comparison']
        assert len(comparison['metrics_comparison']['hybrid']) == 2
        
        print("✅ Compare runs test passed")
    
    def test_get_statistics(self, runner):
        """Test runner statistics"""
        config = QUICK_TEST_CONFIG
        
        # Create various runs
        run1 = BenchmarkRun("stats_001", config, ["hybrid"])
        run1.start()
        run1.complete({'data': 1})
        
        run2 = BenchmarkRun("stats_002", config, ["hybrid"])
        run2.start()
        run2.fail("Error")
        
        run3 = BenchmarkRun("stats_003", config, ["hybrid"])
        
        runner.runs = [run1, run2, run3]
        
        stats = runner.get_statistics()
        
        assert stats['total_runs'] == 3
        assert stats['completed'] == 1
        assert stats['failed'] == 1
        assert stats['pending'] == 1
        
        print(f"✅ Get statistics test passed: {stats}")
    
    def test_print_summary(self, runner, capsys):
        """Test printing runner summary"""
        config = QUICK_TEST_CONFIG
        run = BenchmarkRun("summary_001", config, ["hybrid"])
        run.start()
        run.complete({'test': 'data'})
        
        runner.runs = [run]
        runner.print_summary()
        
        captured = capsys.readouterr()
        assert "BENCHMARK RUNNER SUMMARY" in captured.out
        assert "summary_001" in captured.out
        
        print("✅ Print summary test passed")


class TestReportGenerator:
    """Tests for ReportGenerator"""
    
    @pytest.fixture
    def generator(self, tmp_path):
        """Create report generator with temporary output"""
        return ReportGenerator(output_dir=tmp_path / "reports_test")
    
    @pytest.fixture
    def sample_run_data(self):
        """Sample run data for testing"""
        return {
            'run_id': 'test_report_001',
            'status': 'completed',
            'timestamp': time.time(),
            'duration': 45.5,
            'config': {
                'datasets': ['ms_marco'],
                'ms_marco_limit': 50,
                'hotpot_qa_limit': 0,
                'working_dir': 'test'
            },
            'modes': ['hybrid'],
            'results': {
                'dataset': {
                    'total_samples': 50,
                    'total_documents': 200,
                    'sources': ['ms_marco']
                },
                'metrics': {
                    'graph': {
                        'num_nodes': 1000,
                        'num_edges': 2500,
                        'num_entities': 600,
                        'num_relations': 800,
                        'avg_degree': 5.0,
                        'density': 0.0025
                    },
                    'aggregated': {
                        'hybrid': {
                            'rouge-1': 0.55,
                            'rouge-2': 0.35,
                            'rouge-l': 0.48,
                            'bleu': 0.40,
                            'f1': 0.62,
                            'exact_match_rate': 0.20,
                            'avg_latency_seconds': 1.5,
                            'count': 50
                        }
                    }
                },
                'ingestion': {
                    'ingested': 200,
                    'failed': 0,
                    'duration_seconds': 10.0,
                    'docs_per_second': 20.0
                }
            }
        }
    
    def test_generator_initialization(self, generator):
        """Test report generator initialization"""
        assert generator.output_dir.exists()
        
        print("✅ Report generator initialization test passed")
    
    def test_generate_markdown_report(self, generator, sample_run_data):
        """Test Markdown report generation"""
        report_file = generator.generate_markdown_report(sample_run_data)
        
        assert report_file.exists()
        assert report_file.suffix == '.md'
        
        # Verify content
        content = report_file.read_text(encoding='utf-8')
        assert "# LightRAG Benchmark Report" in content
        assert "test_report_001" in content
        assert "Configuration" in content
        assert "Knowledge Graph" in content
        assert "HYBRID Mode" in content
        
        print(f"✅ Markdown report generation test passed: {report_file}")
    
    def test_generate_comparison_report(self, generator):
        """Test comparison report generation"""
        comparison_data = {
            'runs': ['run_001', 'run_002'],
            'timestamps': [time.time() - 3600, time.time()],
            'durations': [45.0, 43.0],
            'metrics_comparison': {
                'hybrid': [
                    {
                        'run_id': 'run_001',
                        'rouge-1': 0.55,
                        'rouge-2': 0.35,
                        'bleu': 0.40,
                        'f1': 0.62,
                        'exact_match_rate': 0.20,
                        'avg_latency_seconds': 1.5
                    },
                    {
                        'run_id': 'run_002',
                        'rouge-1': 0.58,
                        'rouge-2': 0.38,
                        'bleu': 0.43,
                        'f1': 0.65,
                        'exact_match_rate': 0.22,
                        'avg_latency_seconds': 1.4
                    }
                ]
            }
        }
        
        report_file = generator.generate_comparison_report(comparison_data)
        
        assert report_file.exists()
        assert report_file.suffix == '.md'
        
        # Verify content
        content = report_file.read_text(encoding='utf-8')
        assert "Comparison Report" in content
        assert "run_001" in content
        assert "run_002" in content
        assert "HYBRID Mode" in content
        
        print(f"✅ Comparison report generation test passed: {report_file}")
    
    def test_report_sections(self, generator, sample_run_data):
        """Test individual report sections"""
        # Header
        header = generator._generate_header(sample_run_data)
        assert any("test_report_001" in line for line in header)
        
        # Config
        config_section = generator._generate_config_section(sample_run_data)
        assert any("ms_marco" in line for line in config_section)
        
        # Dataset
        dataset_section = generator._generate_dataset_section(sample_run_data)
        assert any("50" in line for line in dataset_section)  # total_samples
        
        # Graph
        graph_section = generator._generate_graph_section(sample_run_data)
        assert any("1,000" in line or "1000" in line for line in graph_section)  # nodes
        
        # Metrics
        metrics_section = generator._generate_metrics_section(sample_run_data)
        assert any("HYBRID" in line for line in metrics_section)
        
        # Performance
        perf_section = generator._generate_performance_section(sample_run_data)
        assert any("200" in line for line in perf_section)  # ingested docs
        
        print("✅ Report sections test passed")


class TestIntegration:
    """Integration tests for runner and reporting"""
    
    def test_runner_with_report_generation(self, tmp_path):
        """Test complete workflow: run -> save -> report"""
        # Create runner
        runner = BenchmarkRunner(storage_dir=tmp_path / "integration")
        
        # Create and save run
        config = QUICK_TEST_CONFIG
        run = BenchmarkRun("integration_001", config, ["hybrid"])
        run.start()
        run.complete({
            'dataset': {'total_samples': 10},
            'metrics': {
                'aggregated': {
                    'hybrid': {
                        'rouge-1': 0.50,
                        'bleu': 0.40,
                        'f1': 0.60
                    }
                }
            }
        })
        
        runner.runs.append(run)
        runner._save_run_results(run)
        runner._save_runs()
        
        # Generate report
        generator = ReportGenerator(output_dir=tmp_path / "integration_reports")
        report_file = generator.generate_markdown_report(run.to_dict())
        
        assert report_file.exists()
        content = report_file.read_text()
        assert "integration_001" in content
        
        print("✅ Runner with report generation integration test passed")


# Run tests
if __name__ == "__main__":
    print("🧪 Running Benchmark Runner Tests\n")
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
