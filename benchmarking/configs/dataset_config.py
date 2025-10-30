"""
Dataset Configuration for LightRAG Benchmarking

Centralized configuration for dataset loading, processing, and evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import json


@dataclass
class DatasetConfig:
    """Configuration for benchmark datasets"""
    
    # Dataset selection
    datasets: List[str] = field(default_factory=lambda: ["ms_marco", "hotpot_qa"])
    
    # Sample limits (None = load all)
    ms_marco_limit: Optional[int] = 100
    hotpot_qa_limit: Optional[int] = 100
    mixed_samples_per_dataset: int = 50
    
    # Dataset splits
    ms_marco_split: str = "train"
    hotpot_qa_split: str = "validation"
    
    # Storage paths
    working_dir: Path = field(default_factory=lambda: Path("benchmarks/benchmark_storage"))
    cache_dir: Path = field(default_factory=lambda: Path("benchmarks/cache"))
    
    # Document conversion
    save_documents: bool = True
    document_format: str = "txt"  # txt or json
    preserve_metadata: bool = True
    
    # Validation settings
    validate_data: bool = True
    min_query_length: int = 10
    max_query_length: int = 500
    min_passage_length: int = 20
    max_passage_length: int = 5000
    
    # HuggingFace settings
    trust_remote_code: bool = True
    use_auth_token: Optional[str] = None
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Convert paths to Path objects"""
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'datasets': self.datasets,
            'ms_marco_limit': self.ms_marco_limit,
            'hotpot_qa_limit': self.hotpot_qa_limit,
            'mixed_samples_per_dataset': self.mixed_samples_per_dataset,
            'ms_marco_split': self.ms_marco_split,
            'hotpot_qa_split': self.hotpot_qa_split,
            'working_dir': str(self.working_dir),
            'cache_dir': str(self.cache_dir),
            'save_documents': self.save_documents,
            'document_format': self.document_format,
            'preserve_metadata': self.preserve_metadata,
            'validate_data': self.validate_data,
            'min_query_length': self.min_query_length,
            'max_query_length': self.max_query_length,
            'min_passage_length': self.min_passage_length,
            'max_passage_length': self.max_passage_length,
            'trust_remote_code': self.trust_remote_code,
            'verbose': self.verbose,
            'log_level': self.log_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON file"""
        if path is None:
            path = self.working_dir / "dataset_config.json"
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'DatasetConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configurations for different use cases
DEFAULT_CONFIG = DatasetConfig()

QUICK_TEST_CONFIG = DatasetConfig(
    ms_marco_limit=10,
    hotpot_qa_limit=10,
    mixed_samples_per_dataset=5,
    save_documents=False,
    validate_data=False,
    verbose=True
)

FULL_BENCHMARK_CONFIG = DatasetConfig(
    ms_marco_limit=None,  # Load all
    hotpot_qa_limit=None,
    mixed_samples_per_dataset=100,
    save_documents=True,
    validate_data=True,
    verbose=True
)


# Testing
if __name__ == "__main__":
    print("🧪 Testing Dataset Configuration")
    
    # Test default config
    config = DEFAULT_CONFIG
    print(f"\n📋 Default Config:")
    print(f"   Datasets: {config.datasets}")
    print(f"   MS MARCO limit: {config.ms_marco_limit}")
    print(f"   HotpotQA limit: {config.hotpot_qa_limit}")
    print(f"   Working dir: {config.working_dir}")
    
    # Test save/load
    test_path = Path("test_config.json")
    config.save(test_path)
    print(f"\n💾 Saved config to {test_path}")
    
    loaded_config = DatasetConfig.load(test_path)
    print(f"✅ Loaded config successfully")
    print(f"   Loaded datasets: {loaded_config.datasets}")
    
    # Clean up
    test_path.unlink()
    
    # Test quick config
    quick_config = QUICK_TEST_CONFIG
    print(f"\n⚡ Quick Test Config:")
    print(f"   MS MARCO limit: {quick_config.ms_marco_limit}")
    print(f"   Save documents: {quick_config.save_documents}")
    
    print("\n✅ Configuration tests complete!")
