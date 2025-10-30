"""
Benchmark Runner for LightRAG Evaluation

Automated orchestration of benchmark evaluations with:
- Configuration-driven execution
- Multiple run management
- Result persistence and comparison
- Progress tracking and reporting
- Error recovery and retry logic
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime
import hashlib

from ..evaluators.evaluation_pipeline import EvaluationPipeline
from ..configs.dataset_config import DatasetConfig, QUICK_TEST_CONFIG, FULL_BENCHMARK_CONFIG
from ..utils.logging import get_logger
from ..utils.errors import EvaluationError
from ..utils.retry import retry_with_backoff

logger = get_logger("benchmark_runner")


class BenchmarkRun:
    """Represents a single benchmark run with metadata and results"""
    
    def __init__(self, 
                 run_id: str,
                 config: DatasetConfig,
                 modes: List[str],
                 timestamp: Optional[float] = None):
        """
        Initialize benchmark run.
        
        Args:
            run_id: Unique identifier for this run
            config: Dataset configuration
            modes: Query modes to evaluate
            timestamp: Run start time (defaults to now)
        """
        self.run_id = run_id
        self.config = config
        self.modes = modes
        self.timestamp = timestamp or time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "pending"  # pending, running, completed, failed
        self.results: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        
    def start(self):
        """Mark run as started"""
        self.start_time = time.time()
        self.status = "running"
        logger.info(f"🚀 Starting benchmark run: {self.run_id}")
    
    def complete(self, results: Dict[str, Any]):
        """Mark run as completed with results"""
        self.end_time = time.time()
        self.status = "completed"
        self.results = results
        duration = self.end_time - self.start_time if self.start_time else 0
        logger.info(f"✅ Completed benchmark run: {self.run_id} ({duration:.1f}s)")
    
    def fail(self, error: str):
        """Mark run as failed"""
        self.end_time = time.time()
        self.status = "failed"
        self.error = error
        logger.error(f"❌ Failed benchmark run: {self.run_id} - {error}")
    
    def get_duration(self) -> float:
        """Get run duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'run_id': self.run_id,
            'config': self.config.to_dict(),
            'modes': self.modes,
            'timestamp': self.timestamp,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.get_duration(),
            'status': self.status,
            'results': self.results,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkRun':
        """Create from dictionary"""
        run = cls(
            run_id=data['run_id'],
            config=DatasetConfig.from_dict(data['config']),
            modes=data['modes'],
            timestamp=data['timestamp']
        )
        run.start_time = data.get('start_time')
        run.end_time = data.get('end_time')
        run.status = data['status']
        run.results = data.get('results')
        run.error = data.get('error')
        return run


class BenchmarkRunner:
    """
    Orchestrates automated benchmark evaluations.
    
    Manages multiple runs, result persistence, and provides
    comparison capabilities across runs.
    """
    
    def __init__(self, 
                 storage_dir: Optional[Union[str, Path]] = None,
                 auto_save: bool = True):
        """
        Initialize benchmark runner.
        
        Args:
            storage_dir: Directory for storing run results (str or Path)
            auto_save: Whether to automatically save after each run
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path("benchmarks/results")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.runs: List[BenchmarkRun] = []
        self.current_run: Optional[BenchmarkRun] = None
        
        # Load previous runs
        self._load_runs()
        
        logger.info(f"Benchmark runner initialized (storage={self.storage_dir})")
        logger.info(f"  Previous runs: {len(self.runs)}")
    
    def _generate_run_id(self, config: DatasetConfig, modes: List[str]) -> str:
        """Generate unique run ID based on configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create hash of config for uniqueness
        config_str = f"{config.datasets}_{modes}_{config.ms_marco_limit}_{config.hotpot_qa_limit}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"run_{timestamp}_{config_hash}"
    
    def _load_runs(self):
        """Load previous runs from storage"""
        runs_file = self.storage_dir / "runs_index.json"
        
        if runs_file.exists():
            try:
                with open(runs_file, 'r', encoding='utf-8') as f:
                    runs_data = json.load(f)
                
                self.runs = [BenchmarkRun.from_dict(r) for r in runs_data]
                logger.info(f"Loaded {len(self.runs)} previous runs")
            except Exception as e:
                logger.warning(f"Failed to load previous runs: {e}")
                self.runs = []
    
    def _save_runs(self):
        """Save runs index to storage"""
        runs_file = self.storage_dir / "runs_index.json"
        
        try:
            runs_data = [r.to_dict() for r in self.runs]
            with open(runs_file, 'w', encoding='utf-8') as f:
                json.dump(runs_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.runs)} runs to index")
        except Exception as e:
            logger.error(f"Failed to save runs index: {e}")
    
    def _save_run_results(self, run: BenchmarkRun):
        """Save individual run results"""
        run_file = self.storage_dir / f"{run.run_id}.json"
        
        try:
            with open(run_file, 'w', encoding='utf-8') as f:
                json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved run results to {run_file}")
        except Exception as e:
            logger.error(f"Failed to save run results: {e}")
    
    async def run_benchmark(self,
                          config: Optional[DatasetConfig] = None,
                          modes: Optional[List[str]] = None,
                          run_id: Optional[str] = None,
                          clear_existing: bool = False) -> BenchmarkRun:
        """
        Execute a benchmark evaluation.
        
        Args:
            config: Dataset configuration (uses QUICK_TEST_CONFIG if not provided)
            modes: Query modes to evaluate (defaults to all modes)
            run_id: Optional custom run ID
            clear_existing: Whether to clear existing LightRAG graph
            
        Returns:
            BenchmarkRun with results
        """
        config = config or QUICK_TEST_CONFIG
        modes = modes or ["naive", "local", "global", "hybrid"]
        run_id = run_id or self._generate_run_id(config, modes)
        
        # Create run
        run = BenchmarkRun(run_id=run_id, config=config, modes=modes)
        self.current_run = run
        run.start()
        
        try:
            # Create pipeline
            pipeline = EvaluationPipeline(config=config)
            
            # Run evaluation
            with logger.operation("run_benchmark", run_id=run_id, modes=modes):
                summary = await pipeline.run_full_evaluation(
                    modes=modes,
                    clear_existing=clear_existing
                )
                
                # Store results
                run.complete(summary)
                self.runs.append(run)
                
                # Save if auto_save enabled
                if self.auto_save:
                    self._save_run_results(run)
                    self._save_runs()
                
                return run
        
        except Exception as e:
            error_msg = f"Benchmark run failed: {e}"
            run.fail(error_msg)
            self.runs.append(run)
            
            if self.auto_save:
                self._save_run_results(run)
                self._save_runs()
            
            raise EvaluationError(
                error_msg,
                error_code="BENCHMARK_RUN_FAILED",
                context={"run_id": run_id, "modes": modes}
            )
    
    async def run_suite(self,
                       configs: List[DatasetConfig],
                       modes_list: Optional[List[List[str]]] = None) -> List[BenchmarkRun]:
        """
        Run multiple benchmark configurations.
        
        Args:
            configs: List of configurations to run
            modes_list: Optional list of mode lists (one per config)
            
        Returns:
            List of BenchmarkRun results
        """
        modes_list = modes_list or [["hybrid"] for _ in configs]
        
        if len(modes_list) != len(configs):
            raise ValueError("modes_list must match configs length")
        
        results = []
        
        with logger.operation("run_suite", count=len(configs)):
            for i, (config, modes) in enumerate(zip(configs, modes_list)):
                logger.info(f"\n{'='*70}")
                logger.info(f"Running benchmark {i+1}/{len(configs)}")
                logger.info(f"  Datasets: {config.datasets}")
                logger.info(f"  Modes: {modes}")
                logger.info(f"{'='*70}\n")
                
                try:
                    run = await self.run_benchmark(
                        config=config,
                        modes=modes,
                        clear_existing=(i == 0)  # Clear only on first run
                    )
                    results.append(run)
                    
                except Exception as e:
                    logger.error(f"Benchmark {i+1} failed: {e}")
                    # Continue with remaining benchmarks
        
        logger.info(f"\n✅ Completed {len(results)}/{len(configs)} benchmarks")
        return results
    
    def get_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Get run by ID"""
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None
    
    def get_completed_runs(self) -> List[BenchmarkRun]:
        """Get all completed runs"""
        return [r for r in self.runs if r.status == "completed"]
    
    def get_failed_runs(self) -> List[BenchmarkRun]:
        """Get all failed runs"""
        return [r for r in self.runs if r.status == "failed"]
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison results
        """
        runs = [self.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r and r.status == "completed"]
        
        if len(runs) < 2:
            raise ValueError("Need at least 2 completed runs to compare")
        
        comparison = {
            'runs': [r.run_id for r in runs],
            'timestamps': [r.timestamp for r in runs],
            'durations': [r.get_duration() for r in runs],
            'metrics_comparison': {}
        }
        
        # Compare aggregated metrics across modes
        for mode in ["naive", "local", "global", "hybrid"]:
            mode_metrics = []
            
            for run in runs:
                if not run.results or 'metrics' not in run.results:
                    continue
                
                aggregated = run.results['metrics'].get('aggregated', {})
                if mode in aggregated:
                    mode_metrics.append({
                        'run_id': run.run_id,
                        **aggregated[mode]
                    })
            
            if mode_metrics:
                comparison['metrics_comparison'][mode] = mode_metrics
        
        return comparison
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runner statistics"""
        return {
            'total_runs': len(self.runs),
            'completed': len(self.get_completed_runs()),
            'failed': len(self.get_failed_runs()),
            'pending': len([r for r in self.runs if r.status == "pending"]),
            'running': len([r for r in self.runs if r.status == "running"]),
            'storage_dir': str(self.storage_dir)
        }
    
    def print_summary(self):
        """Print runner summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("BENCHMARK RUNNER SUMMARY")
        print("="*70)
        print(f"\n📊 Run Statistics:")
        print(f"   Total Runs: {stats['total_runs']}")
        print(f"   ✅ Completed: {stats['completed']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"   ⏸️  Pending: {stats['pending']}")
        print(f"   🔄 Running: {stats['running']}")
        print(f"\n💾 Storage: {stats['storage_dir']}")
        
        # Show recent runs
        if self.runs:
            print(f"\n📝 Recent Runs:")
            for run in sorted(self.runs, key=lambda r: r.timestamp, reverse=True)[:5]:
                status_emoji = {"completed": "✅", "failed": "❌", "running": "🔄", "pending": "⏸️"}
                emoji = status_emoji.get(run.status, "❓")
                print(f"   {emoji} {run.run_id} - {run.status} ({run.get_duration():.1f}s)")
        
        print("="*70 + "\n")


# CLI and testing
if __name__ == "__main__":
    async def test_runner():
        print("🧪 Testing Benchmark Runner\n")
        
        # Create runner
        runner = BenchmarkRunner(storage_dir=Path("test_runner_results"))
        
        # Run single benchmark
        print("Running quick test benchmark...")
        run = await runner.run_benchmark(
            config=QUICK_TEST_CONFIG,
            modes=["hybrid"],
            clear_existing=True
        )
        
        print(f"\n✅ Run completed: {run.run_id}")
        print(f"   Status: {run.status}")
        print(f"   Duration: {run.get_duration():.1f}s")
        
        # Show summary
        runner.print_summary()
        
        print("\n✅ Runner test complete!")
    
    # Run test
    asyncio.run(test_runner())
