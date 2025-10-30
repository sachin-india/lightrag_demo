"""
Tests for LightRAG Evaluator and Evaluation Pipeline

Tests the evaluation infrastructure with mocked LightRAG instances
to avoid dependencies on actual LLM services.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.evaluators.lightrag_evaluator import LightRAGEvaluator
from benchmarks.evaluators.evaluation_pipeline import EvaluationPipeline
from benchmarks.configs.dataset_config import DatasetConfig
from benchmarks.utils.errors import EvaluationError


class TestLightRAGEvaluator:
    """Tests for LightRAGEvaluator"""
    
    @pytest.fixture
    def mock_lightrag(self):
        """Create a mocked LightRAG instance"""
        mock_rag = AsyncMock()
        mock_rag.ainsert = AsyncMock(return_value=None)
        mock_rag.aquery = AsyncMock(return_value="This is a test response from LightRAG.")
        return mock_rag
    
    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temporary directory"""
        return LightRAGEvaluator(
            working_dir=tmp_path / "eval_test",
            clear_existing=True
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {'id': 'doc_1', 'text': 'Machine learning is a subset of AI.'},
            {'id': 'doc_2', 'text': 'Deep learning uses neural networks.'},
            {'id': 'doc_3', 'text': 'Natural language processing analyzes text.'}
        ]
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing"""
        return [
            {
                'query_id': 'q1',
                'query': 'What is machine learning?',
                'answer': 'Machine learning is a subset of AI.'
            },
            {
                'query_id': 'q2',
                'query': 'What is deep learning?',
                'answer': 'Deep learning uses neural networks.'
            }
        ]
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.working_dir.exists()
        assert evaluator.rag is None
        assert not evaluator.documents_ingested
        assert len(evaluator.query_history) == 0
        
        print("✅ Evaluator initialization test passed")
    
    def test_clear_storage(self, tmp_path):
        """Test clearing existing storage"""
        test_dir = tmp_path / "clear_test"
        test_dir.mkdir(parents=True)
        (test_dir / "test_file.txt").write_text("test")
        
        evaluator = LightRAGEvaluator(working_dir=test_dir, clear_existing=True)
        
        assert test_dir.exists()
        assert not (test_dir / "test_file.txt").exists()
        
        print("✅ Clear storage test passed")
    
    @pytest.mark.asyncio
    async def test_initialize_with_mock(self, evaluator, mock_lightrag):
        """Test initialization with mocked LightRAG"""
        with patch('benchmarks.evaluators.lightrag_evaluator.create_alloy_lightrag_async', 
                   return_value=mock_lightrag):
            await evaluator.initialize()
            
            assert evaluator.rag is not None
            assert isinstance(evaluator.graph_stats, dict)
        
        print("✅ Initialize with mock test passed")
    
    @pytest.mark.asyncio
    async def test_ingest_documents(self, evaluator, mock_lightrag, sample_documents):
        """Test document ingestion"""
        evaluator.rag = mock_lightrag
        
        documents = sample_documents
        
        # Add small delay to ensure timing > 0
        time.sleep(0.01)
        
        stats = evaluator.ingest_documents(documents)
        
        assert stats['total_documents'] == 3
        assert stats['ingested'] == 3
        assert stats['failed'] == 0
        assert stats['duration_seconds'] > 0
        assert evaluator.documents_ingested
        
        # Verify insert was called for each document
        assert mock_lightrag.ainsert.call_count == 3
        
        print(f"✅ Ingest documents test passed: {stats['ingested']} docs ingested")
    
    @pytest.mark.asyncio
    async def test_ingest_empty_documents(self, evaluator, mock_lightrag):
        """Test handling of empty documents"""
        evaluator.rag = mock_lightrag
        
        docs_with_empty = [
            {'id': 'doc_1', 'text': 'Valid content'},
            {'id': 'doc_2', 'text': ''},  # Empty
            {'id': 'doc_3', 'text': '   '}  # Whitespace only
        ]
        
        stats = await evaluator.ingest_documents(docs_with_empty)
        
        # Should only ingest the valid document
        assert stats['ingested'] == 1
        assert mock_lightrag.ainsert.call_count == 1
        
        print("✅ Empty documents handling test passed")
    
    @pytest.mark.asyncio
    async def test_query_single(self, evaluator, mock_lightrag):
        """Test single query execution"""
        evaluator.rag = mock_lightrag
        evaluator.documents_ingested = True
        
        result = await evaluator.query(
            query_text="What is AI?",
            mode="hybrid",
            top_k=10,
            query_id="test_q1"
        )
        
        assert result['query_id'] == "test_q1"
        assert result['query'] == "What is AI?"
        assert result['mode'] == "hybrid"
        assert result['top_k'] == 10
        assert 'response' in result
        assert result['duration_seconds'] > 0
        assert len(evaluator.query_history) == 1
        
        # Verify query was called
        mock_lightrag.aquery.assert_called_once()
        
        print("✅ Single query test passed")
    
    @pytest.mark.asyncio
    async def test_query_all_modes(self, evaluator, mock_lightrag):
        """Test querying in all modes"""
        evaluator.rag = mock_lightrag
        evaluator.documents_ingested = True
        
        results = await evaluator.query_all_modes(
            query_text="What is machine learning?",
            query_id="test_multi"
        )
        
        assert len(results) == 4  # naive, local, global, hybrid
        assert 'naive' in results
        assert 'local' in results
        assert 'global' in results
        assert 'hybrid' in results
        
        for mode, result in results.items():
            assert result['mode'] == mode
            assert 'response' in result
        
        # Should have 4 queries in history
        assert len(evaluator.query_history) == 4
        
        print("✅ Query all modes test passed")
    
    @pytest.mark.asyncio
    async def test_evaluate_queries(self, evaluator, mock_lightrag, sample_queries):
        """Test evaluating multiple queries"""
        evaluator.rag = mock_lightrag
        evaluator.documents_ingested = True
        
        results = await evaluator.evaluate_queries(
            sample_queries,
            modes=["hybrid", "global"]
        )
        
        assert len(results) == 2  # 2 queries
        
        for result in results:
            assert 'query_id' in result
            assert 'query' in result
            assert 'reference_answer' in result
            assert 'mode_results' in result
            assert len(result['mode_results']) == 2  # 2 modes
        
        print(f"✅ Evaluate queries test passed: {len(results)} queries evaluated")
    
    def test_get_statistics(self, evaluator):
        """Test statistics retrieval"""
        # Add some mock query history
        evaluator.query_history = [
            {'mode': 'hybrid', 'query': 'test1'},
            {'mode': 'hybrid', 'query': 'test2'},
            {'mode': 'global', 'query': 'test3'}
        ]
        evaluator.documents_ingested = True
        
        stats = evaluator.get_statistics()
        
        assert stats['documents_ingested'] == True
        assert stats['total_queries'] == 3
        assert stats['query_modes']['hybrid'] == 2
        assert stats['query_modes']['global'] == 1
        
        print("✅ Get statistics test passed")
    
    def test_save_results(self, evaluator, tmp_path):
        """Test saving results"""
        evaluator.query_history = [
            {'mode': 'hybrid', 'query': 'test', 'response': 'answer'}
        ]
        evaluator.documents_ingested = True
        
        output_path = tmp_path / "test_results.json"
        evaluator.save_results(output_path)
        
        assert output_path.exists()
        
        import json
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'statistics' in data
        assert 'query_history' in data
        
        print("✅ Save results test passed")
    
    @pytest.mark.asyncio
    async def test_query_without_initialization(self, evaluator):
        """Test that querying without initialization raises error"""
        with pytest.raises(EvaluationError) as exc_info:
            await evaluator.query("test query")
        
        assert "not initialized" in str(exc_info.value).lower()
        
        print("✅ Query without initialization error test passed")


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline"""
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration"""
        return DatasetConfig(
            datasets=["ms_marco"],
            ms_marco_limit=5,
            hotpot_qa_limit=5,
            working_dir=tmp_path / "pipeline_test",
            save_documents=False,
            validate_data=False
        )
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create evaluation pipeline"""
        return EvaluationPipeline(config=test_config)
    
    def test_pipeline_initialization(self, pipeline, test_config):
        """Test pipeline initialization"""
        assert pipeline.config == test_config
        assert pipeline.data_manager is not None
        assert pipeline.doc_adapter is not None
        assert pipeline.traditional_metrics is not None
        assert pipeline.retrieval_metrics is not None
        
        print("✅ Pipeline initialization test passed")
    
    def test_aggregate_metrics(self, pipeline):
        """Test metrics aggregation"""
        # Create sample metrics
        metrics = {
            'traditional': {
                'q1_hybrid': {
                    'rouge': {'rouge-1': {'f': 0.5}, 'rouge-2': {'f': 0.3}, 'rouge-l': {'f': 0.4}},
                    'bleu': 0.35,
                    'f1': 0.6,
                    'exact_match': 0
                },
                'q2_hybrid': {
                    'rouge': {'rouge-1': {'f': 0.7}, 'rouge-2': {'f': 0.5}, 'rouge-l': {'f': 0.6}},
                    'bleu': 0.45,
                    'f1': 0.8,
                    'exact_match': 1
                }
            },
            'efficiency': {
                'q1_hybrid': {'latency_seconds': 1.5, 'mode': 'hybrid'},
                'q2_hybrid': {'latency_seconds': 1.3, 'mode': 'hybrid'}
            }
        }
        
        aggregated = pipeline._aggregate_metrics(metrics)
        
        assert 'hybrid' in aggregated
        assert aggregated['hybrid']['rouge-1'] == 0.6  # (0.5 + 0.7) / 2
        assert aggregated['hybrid']['bleu'] == 0.4  # (0.35 + 0.45) / 2
        assert aggregated['hybrid']['f1'] == 0.7  # (0.6 + 0.8) / 2
        assert aggregated['hybrid']['exact_match_rate'] == 0.5  # (0 + 1) / 2
        assert aggregated['hybrid']['avg_latency_seconds'] == 1.4  # (1.5 + 1.3) / 2
        
        print("✅ Aggregate metrics test passed")
    
    def test_print_summary_empty(self, pipeline):
        """Test printing summary with no metrics"""
        # Should not raise error
        pipeline.print_summary()
        
        print("✅ Print empty summary test passed")
    
    def test_print_summary_with_metrics(self, pipeline, capsys):
        """Test printing summary with metrics"""
        pipeline.dataset_samples = [{'id': 1}, {'id': 2}]
        pipeline.documents = [{'id': 'doc1'}, {'id': 'doc2'}]
        pipeline.metrics_results = {
            'graph': {
                'num_nodes': 100,
                'num_edges': 200,
                'num_entities': 50,
                'num_relations': 75
            },
            'aggregated': {
                'hybrid': {
                    'rouge-1': 0.65,
                    'rouge-2': 0.45,
                    'rouge-l': 0.55,
                    'bleu': 0.40,
                    'f1': 0.70,
                    'exact_match_rate': 0.25,
                    'avg_latency_seconds': 1.5
                }
            }
        }
        
        pipeline.print_summary()
        
        captured = capsys.readouterr()
        assert "LIGHTRAG BENCHMARK EVALUATION SUMMARY" in captured.out
        assert "Nodes: 100" in captured.out
        assert "HYBRID:" in captured.out
        
        print("✅ Print summary with metrics test passed")


class TestIntegration:
    """Integration tests with mocked components"""
    
    @pytest.mark.asyncio
    async def test_small_pipeline_flow(self, tmp_path):
        """Test small pipeline with mocked dataset"""
        # Create config
        config = DatasetConfig(
            datasets=["ms_marco"],
            ms_marco_limit=2,
            working_dir=tmp_path / "integration_test",
            save_documents=False
        )
        
        # Create pipeline
        pipeline = EvaluationPipeline(config=config)
        
        # Mock dataset samples
        pipeline.dataset_samples = [
            {
                'id': 'test_1',
                'query_id': 'q1',
                'query': 'What is AI?',
                'answer': 'AI is artificial intelligence.',
                'passages': [
                    {'id': 'p1', 'text': 'AI stands for artificial intelligence.'}
                ],
                'source_type': 'test'
            }
        ]
        
        # Convert to documents
        await pipeline._convert_to_documents()
        
        assert len(pipeline.documents) > 0
        
        print("✅ Small pipeline flow test passed")


# Run tests
if __name__ == "__main__":
    print("🧪 Running Evaluator Tests\n")
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short", "-k", "not asyncio"])
