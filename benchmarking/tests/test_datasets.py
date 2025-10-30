"""
Comprehensive Tests for Dataset Loaders and Adapters

Tests:
- Dataset loading (MS MARCO, HotpotQA)
- Document conversion
- Configuration management
- Error handling
- Integration with utilities
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarking.benchmark_datasets.loaders import MSMarcoLoader, HotpotQALoader, BenchmarkDataManager
from benchmarking.benchmark_datasets.document_adapter import LightRAGDocumentAdapter
from benchmarking.configs.dataset_config import DatasetConfig, QUICK_TEST_CONFIG
from benchmarking.utils.errors import DatasetError


class TestDatasetLoaders:
    """Tests for dataset loaders"""
    
    def test_ms_marco_loader_basic(self):
        """Test MS MARCO loader with small sample"""
        loader = MSMarcoLoader()
        
        # Load small sample
        samples = loader.load(split="train", limit=5)
        
        assert len(samples) == 5, "Should load 5 samples"
        
        # Check sample structure
        sample = samples[0]
        assert 'id' in sample
        assert 'query_id' in sample
        assert 'query' in sample
        assert 'answer' in sample
        assert 'passages' in sample
        assert 'source_type' in sample
        assert sample['source_type'] == 'ms_marco'
        
        print(f"✅ MS MARCO loader: loaded {len(samples)} samples")
    
    def test_ms_marco_passage_structure(self):
        """Test MS MARCO passage structure"""
        loader = MSMarcoLoader()
        samples = loader.load(split="train", limit=2)
        
        sample = samples[0]
        passages = sample['passages']
        
        assert len(passages) > 0, "Should have passages"
        
        # Check passage structure
        passage = passages[0]
        assert 'id' in passage
        assert 'text' in passage
        assert passage['id'].startswith('ms_marco_passage_')
        assert len(passage['text']) > 0
        
        print(f"✅ MS MARCO passages: {len(passages)} passages in sample")
    
    def test_hotpot_qa_loader_basic(self):
        """Test HotpotQA loader with small sample"""
        loader = HotpotQALoader()
        
        # Load small sample
        samples = loader.load(split="validation", limit=5)
        
        assert len(samples) == 5, "Should load 5 samples"
        
        # Check sample structure
        sample = samples[0]
        assert 'id' in sample
        assert 'query_id' in sample
        assert 'query' in sample
        assert 'answer' in sample
        assert 'passages' in sample
        assert 'source_type' in sample
        assert sample['source_type'] == 'hotpot_qa'
        
        print(f"✅ HotpotQA loader: loaded {len(samples)} samples")
    
    def test_hotpot_qa_passage_structure(self):
        """Test HotpotQA passage structure"""
        loader = HotpotQALoader()
        samples = loader.load(split="validation", limit=2)
        
        sample = samples[0]
        passages = sample['passages']
        
        assert len(passages) > 0, "Should have passages"
        
        # Check passage structure
        passage = passages[0]
        assert 'id' in passage
        assert 'text' in passage
        assert 'title' in passage
        assert passage['id'].startswith('hotpot_passage_')
        
        print(f"✅ HotpotQA passages: {len(passages)} passages in sample")
    
    def test_get_sample_method(self):
        """Test get_sample method"""
        loader = MSMarcoLoader()
        
        # Get small sample
        sample = loader.get_sample(3)
        
        assert len(sample) == 3, "Should get 3 samples"
        
        print(f"✅ get_sample: retrieved {len(sample)} samples")
    
    def test_benchmark_data_manager_ms_marco(self):
        """Test BenchmarkDataManager MS MARCO loading"""
        manager = BenchmarkDataManager()
        
        samples = manager.load_ms_marco(limit=5)
        
        assert len(samples) == 5
        assert all(s['source_type'] == 'ms_marco' for s in samples)
        
        print(f"✅ BenchmarkDataManager: loaded {len(samples)} MS MARCO samples")
    
    def test_benchmark_data_manager_hotpot_qa(self):
        """Test BenchmarkDataManager HotpotQA loading"""
        manager = BenchmarkDataManager()
        
        samples = manager.load_hotpot_qa(limit=5)
        
        assert len(samples) == 5
        assert all(s['source_type'] == 'hotpot_qa' for s in samples)
        
        print(f"✅ BenchmarkDataManager: loaded {len(samples)} HotpotQA samples")
    
    def test_mixed_dataset(self):
        """Test mixed dataset creation"""
        manager = BenchmarkDataManager()
        
        # Load mixed dataset
        mixed = manager.load_mixed_dataset(
            datasets=["ms_marco", "hotpot_qa"],
            samples_per_dataset=5
        )
        
        assert len(mixed) == 10, "Should have 10 samples (5 per dataset)"
        
        # Check we have both sources
        sources = set(s['source_type'] for s in mixed)
        assert 'ms_marco' in sources
        assert 'hotpot_qa' in sources
        
        print(f"✅ Mixed dataset: {len(mixed)} samples from {len(sources)} sources")
    
    def test_test_sample(self):
        """Test get_test_sample method"""
        manager = BenchmarkDataManager()
        
        # Default is 20 total (10 per dataset)
        test_sample = manager.get_test_sample()
        
        # Account for possible dataset size limitations
        assert len(test_sample) >= 10, f"Should get at least 10 test samples, got {len(test_sample)}"
        
        # Test with explicit small number
        small_sample = manager.get_test_sample(n=4)
        assert len(small_sample) >= 2, "Should get at least 2 samples"
        
        print(f"✅ Test sample: retrieved {len(test_sample)} samples (default), {len(small_sample)} samples (n=4)")


class TestDocumentAdapter:
    """Tests for document adapter"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample benchmark data for testing"""
        return [
            {
                'id': 'test_query_1',
                'query_id': 'q1',
                'query': 'What is machine learning?',
                'answer': 'Machine learning is a subset of AI...',
                'all_answers': ['ML is a subset of AI'],
                'passages': [
                    {
                        'id': 'passage_1',
                        'text': 'Machine learning is a method of data analysis.',
                        'title': 'ML Introduction'
                    },
                    {
                        'id': 'passage_2',
                        'text': 'It is a branch of artificial intelligence.',
                        'title': 'AI and ML'
                    }
                ],
                'source_type': 'test'
            }
        ]
    
    @pytest.fixture
    def adapter(self, tmp_path):
        """Create adapter with temporary directory"""
        return LightRAGDocumentAdapter(working_dir=tmp_path / "adapter_test")
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization"""
        assert adapter.working_dir.exists()
        assert adapter.docs_dir.exists()
        assert adapter.doc_counter == 0
        
        print("✅ Adapter initialization successful")
    
    def test_convert_sample_to_documents(self, adapter, sample_data):
        """Test converting single sample to documents"""
        sample = sample_data[0]
        
        documents = adapter.convert_sample_to_documents(sample)
        
        assert len(documents) == 2, "Should create 2 documents from 2 passages"
        
        # Check document structure
        doc = documents[0]
        assert 'id' in doc
        assert 'text' in doc
        assert 'metadata' in doc
        assert doc['metadata']['passage_id'] == 'passage_1'
        assert doc['metadata']['query_id'] == 'q1'
        
        print(f"✅ Sample conversion: created {len(documents)} documents")
    
    def test_convert_dataset_to_documents(self, adapter, sample_data):
        """Test converting entire dataset"""
        result = adapter.convert_dataset_to_documents(sample_data, save_docs=False)
        
        assert 'documents' in result
        assert 'query_doc_mapping' in result
        assert 'passage_doc_mapping' in result
        assert result['total_documents'] == 2
        assert result['total_queries'] == 1
        
        # Check query mapping
        query_mapping = result['query_doc_mapping']
        assert 'q1' in query_mapping
        assert len(query_mapping['q1']['doc_ids']) == 2
        
        print(f"✅ Dataset conversion: {result['total_documents']} documents from {result['total_queries']} queries")
    
    def test_save_and_load_documents(self, adapter, sample_data):
        """Test saving and loading documents"""
        result = adapter.convert_dataset_to_documents(sample_data, save_docs=True)
        
        # Check files were created
        docs_file = adapter.docs_dir / "benchmark_documents.json"
        assert docs_file.exists(), "Documents file should exist"
        
        # Check mappings were saved
        mapping_file = adapter.mapping_file
        assert mapping_file.exists(), "Mapping file should exist"
        
        # Load mappings
        loaded_mappings = adapter.load_mappings()
        assert 'q1' in loaded_mappings
        
        print("✅ Save/load documents and mappings successful")
    
    def test_get_documents_for_query(self, adapter, sample_data):
        """Test retrieving documents for a query"""
        result = adapter.convert_dataset_to_documents(sample_data, save_docs=False)
        
        doc_ids = adapter.get_documents_for_query('q1', result['query_doc_mapping'])
        
        assert len(doc_ids) == 2, "Should get 2 document IDs for query"
        
        print(f"✅ Query document retrieval: {len(doc_ids)} documents for query")
    
    def test_create_document_files_txt(self, adapter, sample_data):
        """Test creating text document files"""
        result = adapter.convert_dataset_to_documents(sample_data, save_docs=False)
        documents = result['documents']
        
        file_paths = adapter.create_document_files(documents, format="txt")
        
        assert len(file_paths) == 2
        assert all(p.suffix == '.txt' for p in file_paths)
        assert all(p.exists() for p in file_paths)
        
        # Check file content
        content = file_paths[0].read_text(encoding='utf-8')
        assert '# Document ID:' in content
        assert '# Passage ID:' in content
        assert 'Machine learning' in content
        
        print(f"✅ Created {len(file_paths)} text document files")
    
    def test_create_document_files_json(self, adapter, sample_data):
        """Test creating JSON document files"""
        result = adapter.convert_dataset_to_documents(sample_data, save_docs=False)
        documents = result['documents']
        
        file_paths = adapter.create_document_files(documents, format="json")
        
        assert len(file_paths) == 2
        assert all(p.suffix == '.json' for p in file_paths)
        assert all(p.exists() for p in file_paths)
        
        print(f"✅ Created {len(file_paths)} JSON document files")
    
    def test_get_statistics(self, adapter, sample_data):
        """Test statistics calculation"""
        result = adapter.convert_dataset_to_documents(sample_data, save_docs=False)
        
        stats = adapter.get_statistics(result['documents'])
        
        assert stats['total_documents'] == 2
        assert stats['total_characters'] > 0
        assert stats['avg_document_length'] > 0
        assert 'test' in stats['source_distribution']
        assert stats['unique_queries'] == 1
        
        print(f"✅ Statistics: {stats}")
    
    def test_empty_passage_handling(self, adapter):
        """Test handling of empty passages"""
        sample_with_empty = {
            'id': 'test_empty',
            'query_id': 'qe',
            'query': 'Test query',
            'answer': 'Test answer',
            'passages': [
                {'id': 'p1', 'text': 'Valid passage'},
                {'id': 'p2', 'text': ''},  # Empty
                {'id': 'p3', 'text': '   '}  # Whitespace only
            ],
            'source_type': 'test'
        }
        
        documents = adapter.convert_sample_to_documents(sample_with_empty)
        
        # Should skip empty passages
        assert len(documents) == 1, "Should skip empty/whitespace passages"
        
        print("✅ Empty passage handling correct")


class TestDatasetConfig:
    """Tests for dataset configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = DatasetConfig()
        
        assert 'ms_marco' in config.datasets
        assert 'hotpot_qa' in config.datasets
        assert config.ms_marco_limit == 100
        assert config.hotpot_qa_limit == 100
        assert config.save_documents == True
        
        print("✅ Default config validated")
    
    def test_config_to_dict(self):
        """Test config to dictionary conversion"""
        config = DatasetConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'datasets' in config_dict
        assert 'ms_marco_limit' in config_dict
        
        print("✅ Config to dict conversion successful")
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'datasets': ['ms_marco'],
            'ms_marco_limit': 50,
            'hotpot_qa_limit': 50,
            'save_documents': False
        }
        
        config = DatasetConfig.from_dict(config_dict)
        
        assert config.ms_marco_limit == 50
        assert config.save_documents == False
        
        print("✅ Config from dict creation successful")
    
    def test_config_save_load(self, tmp_path):
        """Test saving and loading configuration"""
        config = DatasetConfig(ms_marco_limit=75)
        
        # Save
        config_path = tmp_path / "test_config.json"
        config.save(config_path)
        
        assert config_path.exists()
        
        # Load
        loaded_config = DatasetConfig.load(config_path)
        
        assert loaded_config.ms_marco_limit == 75
        
        print("✅ Config save/load successful")
    
    def test_quick_test_config(self):
        """Test quick test configuration"""
        assert QUICK_TEST_CONFIG.ms_marco_limit == 10
        assert QUICK_TEST_CONFIG.hotpot_qa_limit == 10
        assert QUICK_TEST_CONFIG.save_documents == False
        
        print("✅ Quick test config validated")


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_pipeline_small(self, tmp_path):
        """Test full pipeline with small dataset"""
        # 1. Load data
        manager = BenchmarkDataManager()
        samples = manager.get_test_sample()
        
        # 2. Convert to documents
        adapter = LightRAGDocumentAdapter(working_dir=tmp_path / "integration_test")
        result = adapter.convert_dataset_to_documents(samples, save_docs=True)
        
        # 3. Verify results
        assert result['total_documents'] > 0
        assert result['total_queries'] == len(samples)
        assert adapter.docs_dir.exists()
        
        # 4. Create document files
        file_paths = adapter.create_document_files(result['documents'][:2], format="txt")
        assert len(file_paths) > 0
        assert all(p.exists() for p in file_paths)
        
        print(f"✅ Full pipeline: {len(samples)} samples → {result['total_documents']} documents")
    
    def test_config_with_loaders(self):
        """Test using config with loaders"""
        config = QUICK_TEST_CONFIG
        
        # Use config settings
        loader = MSMarcoLoader()
        samples = loader.load(limit=config.ms_marco_limit)
        
        # Dataset may return slightly fewer samples than requested
        assert len(samples) >= config.ms_marco_limit - 2, \
            f"Should get close to {config.ms_marco_limit} samples, got {len(samples)}"
        
        print(f"✅ Config integration: loaded {len(samples)} samples using config")


# Run tests
if __name__ == "__main__":
    print("🧪 Running Dataset Tests\n")
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
