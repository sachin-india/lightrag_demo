"""
Evaluation Pipeline for LightRAG Benchmarking

Combines dataset loading, document ingestion, query execution,
and metrics calculation into a unified evaluation pipeline.

Workflow:
1. Load benchmark dataset (MS MARCO, HotpotQA)
2. Convert passages to documents
3. Ingest documents into LightRAG
4. Execute queries in multiple modes
5. Calculate metrics (ROUGE, BLEU, retrieval metrics)
6. Generate evaluation report
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from ..benchmark_datasets.loaders import BenchmarkDataManager
from ..benchmark_datasets.document_adapter import LightRAGDocumentAdapter
from ..evaluators.lightrag_evaluator import LightRAGEvaluator
from ..metrics.traditional import TraditionalMetrics
from ..metrics.retrieval import RetrievalMetrics
from ..metrics.efficiency import EfficiencyMetrics
from ..metrics.graph_metrics import GraphMetrics
from ..metrics.llm_judge import LLMJudge
from ..configs.dataset_config import DatasetConfig
from ..utils.logging import get_logger
from ..utils.errors import EvaluationError

logger = get_logger("evaluation_pipeline")


class EvaluationPipeline:
    """
    End-to-end evaluation pipeline for LightRAG benchmarking.
    
    Orchestrates the complete workflow from dataset loading to
    metrics calculation and reporting.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize evaluation pipeline.
        
        Args:
            config: Dataset configuration (uses defaults if not provided)
        """
        self.config = config or DatasetConfig()
        
        # Components
        self.data_manager = BenchmarkDataManager()
        self.doc_adapter = LightRAGDocumentAdapter(working_dir=self.config.working_dir)
        self.evaluator: Optional[LightRAGEvaluator] = None
        
        # Metrics calculators
        self.traditional_metrics = TraditionalMetrics()
        self.retrieval_metrics = RetrievalMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.graph_metrics = GraphMetrics()
        
        # Semantic metrics (load model lazy to avoid startup cost)
        self._semantic_metrics = None
        
        # LLM Judge (lazy load to avoid unnecessary API calls)
        self._llm_judge = None
        
        # Results
        self.dataset_samples: List[Dict[str, Any]] = []
        self.documents: List[Dict[str, Any]] = []
        self.query_results: List[Dict[str, Any]] = []
        self.metrics_results: Dict[str, Any] = {}
        
        logger.info(f"Evaluation pipeline initialized")
        logger.info(f"  Datasets: {self.config.datasets}")
        logger.info(f"  Limits: MS MARCO={self.config.ms_marco_limit}, HotpotQA={self.config.hotpot_qa_limit}")
    
    @property
    def semantic_metrics(self):
        """Lazy load semantic metrics (model is large)"""
        if self._semantic_metrics is None:
            from ..metrics.semantic import SemanticMetrics
            self._semantic_metrics = SemanticMetrics()
        return self._semantic_metrics
    
    @property
    def llm_judge(self):
        """Lazy load LLM judge (avoid unnecessary API calls)"""
        if self._llm_judge is None:
            self._llm_judge = LLMJudge(max_concurrent=3)
        return self._llm_judge
    
    async def run_full_evaluation(self, 
                                  modes: Optional[List[str]] = None,
                                  clear_existing: bool = False) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            modes: Query modes to evaluate (defaults to all)
            clear_existing: Whether to clear existing LightRAG graph
            
        Returns:
            Complete evaluation results
        """
        modes = modes or ["naive", "local", "global", "hybrid"]
        
        with logger.operation("run_full_evaluation", modes=modes):
            start_time = time.time()
            
            try:
                # Step 1: Load dataset
                logger.info("📊 Step 1/6: Loading dataset...")
                await self._load_dataset()
                
                # Step 2: Convert to documents
                logger.info("📝 Step 2/6: Converting to documents...")
                await self._convert_to_documents()
                
                # Step 3: Initialize evaluator
                logger.info("🔧 Step 3/6: Initializing LightRAG...")
                await self._initialize_evaluator(clear_existing=clear_existing)
                
                # Step 4: Ingest documents
                logger.info("📥 Step 4/6: Ingesting documents...")
                ingest_stats = await self._ingest_documents()
                
                # Step 5: Execute queries
                logger.info(f"🔍 Step 5/6: Executing queries in {len(modes)} modes...")
                self.query_results = await self._execute_queries(modes=modes)
                
                # Step 6: Calculate metrics
                logger.info("📏 Step 6/6: Calculating metrics...")
                self.metrics_results = await self._calculate_metrics()
                
                # Generate summary
                duration = time.time() - start_time
                summary = self._generate_summary(duration, ingest_stats)
                
                logger.info(f"✅ Evaluation complete in {duration:.1f}s")
                
                return summary
                
            except Exception as e:
                logger.error(f"Evaluation pipeline failed: {e}")
                raise EvaluationError(
                    f"Pipeline execution failed: {e}",
                    error_code="PIPELINE_ERROR"
                )
    
    async def _load_dataset(self):
        """Load benchmark dataset"""
        # Load based on configured datasets
        if "ms_marco" in self.config.datasets and "hotpot_qa" in self.config.datasets:
            self.dataset_samples = self.data_manager.load_mixed_dataset(
                datasets=self.config.datasets,
                samples_per_dataset=self.config.mixed_samples_per_dataset
            )
        elif "ms_marco" in self.config.datasets:
            self.dataset_samples = self.data_manager.load_ms_marco(
                limit=self.config.ms_marco_limit,
                split=self.config.ms_marco_split
            )
        elif "hotpot_qa" in self.config.datasets:
            self.dataset_samples = self.data_manager.load_hotpot_qa(
                limit=self.config.hotpot_qa_limit,
                split=self.config.hotpot_qa_split
            )
        else:
            raise EvaluationError(
                "No datasets configured",
                error_code="NO_DATASETS"
            )
        
        logger.info(f"  Loaded {len(self.dataset_samples)} samples")
    
    async def _convert_to_documents(self):
        """Convert benchmark passages to LightRAG documents"""
        result = self.doc_adapter.convert_dataset_to_documents(
            self.dataset_samples,
            save_docs=self.config.save_documents
        )
        
        self.documents = result['documents']
        logger.info(f"  Converted to {len(self.documents)} documents")
    
    async def _initialize_evaluator(self, clear_existing: bool = False):
        """Initialize LightRAG evaluator"""
        lightrag_dir = self.config.working_dir / "lightrag"
        self.evaluator = LightRAGEvaluator(
            working_dir=lightrag_dir,
            clear_existing=clear_existing
        )
        await self.evaluator.initialize()
    
    async def _ingest_documents(self) -> Dict[str, Any]:
        """Ingest documents into LightRAG"""
        return await self.evaluator.ingest_documents(self.documents)
    
    async def _execute_queries(self, modes: List[str]) -> List[Dict[str, Any]]:
        """Execute benchmark queries"""
        return await self.evaluator.evaluate_queries(
            self.dataset_samples,
            modes=modes
        )
    
    async def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate all evaluation metrics"""
        metrics = {
            'traditional': {},
            'retrieval': {},
            'efficiency': {},
            'graph': {},
            'semantic': {},  # Semantic similarity metrics
            'llm_judge': {}  # LLM-as-Judge scores
        }
        
        # Calculate traditional metrics (ROUGE, BLEU, F1, EM)
        for result in self.query_results:
            query_id = result['query_id']
            reference = result['reference_answer']
            
            for mode, mode_result in result['mode_results'].items():
                if 'error' in mode_result:
                    continue
                
                prediction = mode_result.get('response', '')
                
                # Traditional metrics - use calculate_all() for all metrics at once
                all_metrics = self.traditional_metrics.calculate_all(prediction, reference)
                
                # Format ROUGE scores to match expected structure
                rouge_scores = {
                    'rouge-1': {
                        'p': all_metrics.get('rouge1_precision', 0),
                        'r': all_metrics.get('rouge1_recall', 0),
                        'f': all_metrics.get('rouge1_f1', 0)
                    },
                    'rouge-2': {
                        'p': all_metrics.get('rouge2_precision', 0),
                        'r': all_metrics.get('rouge2_recall', 0),
                        'f': all_metrics.get('rouge2_f1', 0)
                    },
                    'rouge-l': {
                        'p': all_metrics.get('rougel_precision', 0),
                        'r': all_metrics.get('rougel_recall', 0),
                        'f': all_metrics.get('rougel_f1', 0)
                    }
                }
                
                key = f"{query_id}_{mode}"
                metrics['traditional'][key] = {
                    'rouge': rouge_scores,
                    'bleu': all_metrics.get('bleu_1', 0),  # Use BLEU-1 score
                    'f1': all_metrics.get('token_f1', 0),
                    'exact_match': all_metrics.get('exact_match', 0),
                    # Answer containment (NEW - handles verbose answers)
                    'containment': {
                        'exact_substring': all_metrics.get('containment_exact_substring', 0),
                        'normalized_substring': all_metrics.get('containment_normalized_substring', 0),
                        'token_overlap_ratio': all_metrics.get('containment_token_overlap_ratio', 0),
                        'all_tokens_present': all_metrics.get('containment_all_tokens_present', 0)
                    }
                }
                
                # Semantic similarity (NEW - handles paraphrasing and verbose answers)
                semantic_score = self.semantic_metrics.semantic_similarity(prediction, reference)
                metrics['semantic'][key] = {
                    'similarity': semantic_score
                }
                
                # Graph quality metrics (NEW - graph-based retrieval quality)
                # Extract basic graph-related features from the response
                graph_features = self._extract_graph_features(prediction)
                metrics['graph'][key] = graph_features
                
                # Efficiency metrics
                duration = mode_result.get('duration_seconds', 0)
                metrics['efficiency'][key] = {
                    'latency_seconds': duration,
                    'mode': mode
                }
        
        # LLM-as-Judge evaluation (NEW - GPT-4 quality assessment)
        # Batch evaluate all query results for efficiency
        logger.info("🤖 Running LLM-as-Judge evaluations...")
        judge_evaluations = []
        judge_keys = []
        
        for result in self.query_results:
            query_id = result['query_id']
            query = result['query']
            reference = result['reference_answer']
            
            for mode, mode_result in result['mode_results'].items():
                if 'error' in mode_result:
                    continue
                
                prediction = mode_result.get('response', '')
                key = f"{query_id}_{mode}"
                
                judge_evaluations.append({
                    'query': query,
                    'reference_answer': reference,
                    'predicted_answer': prediction,
                    'context': prediction[:1000]  # Use first 1000 chars as context
                })
                judge_keys.append(key)
        
        # Batch evaluate with LLM judge
        if judge_evaluations:
            judge_results = await self.llm_judge.judge_batch(judge_evaluations)
            
            # Store results
            for key, judge_result in zip(judge_keys, judge_results):
                metrics['llm_judge'][key] = judge_result
        
        # Calculate aggregate metrics
        metrics['aggregated'] = self._aggregate_metrics(metrics)
        
        # Add global graph statistics to the graph metrics
        if self.evaluator:
            graph_stats = self.evaluator.graph_stats
            # Add global stats to the graph metrics dict (alongside per-query stats)
            metrics['graph']['_global_stats'] = {
                'num_nodes': graph_stats.get('num_nodes', 0),
                'num_edges': graph_stats.get('num_edges', 0),
                'num_entities': graph_stats.get('num_entities', 0),
                'num_relations': graph_stats.get('num_relations', 0),
                'avg_degree': graph_stats.get('avg_degree', 0),
                'density': graph_stats.get('density', 0)
            }
        
        return metrics
    
    def _extract_graph_features(self, text: str) -> Dict[str, Any]:
        """
        Extract graph-related features from response text
        
        Simple heuristics to measure graph-based response characteristics:
        - Entity density: uppercase words (potential entities)
        - Structural markers: presence of references, citations
        - Information richness: unique tokens, diversity
        """
        import re
        
        if not text:
            return {
                'has_references': False,
                'reference_count': 0,
                'entity_density': 0.0,
                'unique_token_ratio': 0.0
            }
        
        # Check for references (LightRAG includes [1], [2], etc.)
        references = re.findall(r'\[\d+\]', text)
        has_references = len(references) > 0
        reference_count = len(set(references))  # Unique references
        
        # Tokenize
        tokens = text.lower().split()
        if not tokens:
            return {
                'has_references': has_references,
                'reference_count': reference_count,
                'entity_density': 0.0,
                'unique_token_ratio': 0.0
            }
        
        # Entity density: ratio of capitalized words (rough proxy for entities)
        words = text.split()
        capitalized = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
        entity_density = capitalized / len(words) if words else 0.0
        
        # Unique token ratio (vocabulary richness)
        unique_token_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
        
        return {
            'has_references': has_references,
            'reference_count': reference_count,
            'entity_density': entity_density,
            'unique_token_ratio': unique_token_ratio
        }
    
    def _aggregate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate metrics by mode"""
        aggregated = {}
        
        # Group by mode
        for mode in ["naive", "local", "global", "hybrid"]:
            mode_traditional = []
            mode_efficiency = []
            mode_semantic = []
            mode_graph = []
            mode_judge = []
            
            for key, values in metrics['traditional'].items():
                if key.endswith(f"_{mode}"):
                    mode_traditional.append(values)
            
            for key, values in metrics['efficiency'].items():
                if values.get('mode') == mode:
                    mode_efficiency.append(values)
            
            for key, values in metrics['semantic'].items():
                if key.endswith(f"_{mode}"):
                    mode_semantic.append(values)
            
            for key, values in metrics['graph'].items():
                if key.endswith(f"_{mode}"):
                    mode_graph.append(values)
            
            for key, values in metrics['llm_judge'].items():
                if key.endswith(f"_{mode}") and not values.get('error'):
                    mode_judge.append(values)
            
            if mode_traditional:
                # Average ROUGE, BLEU, F1
                avg_rouge_1 = sum(m['rouge']['rouge-1']['f'] for m in mode_traditional) / len(mode_traditional)
                avg_rouge_2 = sum(m['rouge']['rouge-2']['f'] for m in mode_traditional) / len(mode_traditional)
                avg_rouge_l = sum(m['rouge']['rouge-l']['f'] for m in mode_traditional) / len(mode_traditional)
                avg_bleu = sum(m['bleu'] for m in mode_traditional) / len(mode_traditional)
                avg_f1 = sum(m['f1'] for m in mode_traditional) / len(mode_traditional)
                em_rate = sum(m['exact_match'] for m in mode_traditional) / len(mode_traditional)
                
                # Average containment scores (NEW)
                avg_containment = {
                    'exact_substring': sum(m['containment']['exact_substring'] for m in mode_traditional) / len(mode_traditional),
                    'normalized_substring': sum(m['containment']['normalized_substring'] for m in mode_traditional) / len(mode_traditional),
                    'token_overlap_ratio': sum(m['containment']['token_overlap_ratio'] for m in mode_traditional) / len(mode_traditional),
                    'all_tokens_present': sum(m['containment']['all_tokens_present'] for m in mode_traditional) / len(mode_traditional)
                }
                
                aggregated[mode] = {
                    'rouge-1': avg_rouge_1,
                    'rouge-2': avg_rouge_2,
                    'rouge-l': avg_rouge_l,
                    'bleu': avg_bleu,
                    'f1': avg_f1,
                    'exact_match_rate': em_rate,
                    'containment': avg_containment,  # NEW
                    'count': len(mode_traditional)
                }
            
            # Add semantic similarity if available (NEW)
            if mode_semantic:
                avg_semantic = sum(m['similarity'] for m in mode_semantic) / len(mode_semantic)
                if mode in aggregated:
                    aggregated[mode]['semantic_similarity'] = avg_semantic
                else:
                    aggregated[mode] = {'semantic_similarity': avg_semantic, 'count': len(mode_semantic)}
            
            # Add graph quality metrics if available (NEW)
            if mode_graph:
                avg_reference_count = sum(m['reference_count'] for m in mode_graph) / len(mode_graph)
                avg_entity_density = sum(m['entity_density'] for m in mode_graph) / len(mode_graph)
                avg_unique_ratio = sum(m['unique_token_ratio'] for m in mode_graph) / len(mode_graph)
                has_refs_ratio = sum(1 for m in mode_graph if m['has_references']) / len(mode_graph)
                
                if mode in aggregated:
                    aggregated[mode]['graph_quality'] = {
                        'avg_references': avg_reference_count,
                        'entity_density': avg_entity_density,
                        'unique_token_ratio': avg_unique_ratio,
                        'reference_usage_rate': has_refs_ratio
                    }
                else:
                    aggregated[mode] = {
                        'graph_quality': {
                            'avg_references': avg_reference_count,
                            'entity_density': avg_entity_density,
                            'unique_token_ratio': avg_unique_ratio,
                            'reference_usage_rate': has_refs_ratio
                        },
                        'count': len(mode_graph)
                    }
            
            # Add LLM judge scores if available (NEW)
            if mode_judge:
                avg_correctness = sum(m['correctness'] for m in mode_judge) / len(mode_judge)
                avg_completeness = sum(m['completeness'] for m in mode_judge) / len(mode_judge)
                avg_faithfulness = sum(m['faithfulness'] for m in mode_judge) / len(mode_judge)
                avg_conciseness = sum(m['conciseness'] for m in mode_judge) / len(mode_judge)
                avg_overall = (avg_correctness + avg_completeness + avg_faithfulness + avg_conciseness) / 4
                
                if mode in aggregated:
                    aggregated[mode]['llm_judge'] = {
                        'correctness': avg_correctness,
                        'completeness': avg_completeness,
                        'faithfulness': avg_faithfulness,
                        'conciseness': avg_conciseness,
                        'overall': avg_overall
                    }
                else:
                    aggregated[mode] = {
                        'llm_judge': {
                            'correctness': avg_correctness,
                            'completeness': avg_completeness,
                            'faithfulness': avg_faithfulness,
                            'conciseness': avg_conciseness,
                            'overall': avg_overall
                        },
                        'count': len(mode_judge)
                    }
            
            if mode_efficiency:
                avg_latency = sum(m['latency_seconds'] for m in mode_efficiency) / len(mode_efficiency)
                aggregated[mode]['avg_latency_seconds'] = avg_latency
        
        return aggregated
    
    def _generate_summary(self, duration: float, ingest_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation summary"""
        return {
            'config': self.config.to_dict(),
            'dataset': {
                'total_samples': len(self.dataset_samples),
                'total_documents': len(self.documents),
                'sources': list(set(s['source_type'] for s in self.dataset_samples))
            },
            'ingestion': ingest_stats,
            'queries': {
                'total': len(self.query_results),
                'modes_evaluated': list(set(
                    mode for r in self.query_results 
                    for mode in r.get('mode_results', {}).keys()
                ))
            },
            'metrics': self.metrics_results,
            'duration_seconds': duration,
            'timestamp': time.time()
        }
    
    def save_results(self, output_path: Optional[Path] = None):
        """Save evaluation results to JSON"""
        output_path = output_path or (self.config.working_dir / "evaluation_summary.json")
        
        summary = {
            'dataset_samples': self.dataset_samples,
            'query_results': self.query_results,
            'metrics': self.metrics_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved evaluation results to {output_path}")
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.metrics_results:
            print("No metrics calculated yet")
            return
        
        print("\n" + "="*70)
        print("LIGHTRAG BENCHMARK EVALUATION SUMMARY")
        print("="*70)
        
        # Dataset info
        print(f"\n📊 Dataset:")
        print(f"   Samples: {len(self.dataset_samples)}")
        print(f"   Documents: {len(self.documents)}")
        
        # Graph stats
        if 'graph' in self.metrics_results:
            graph = self.metrics_results['graph']
            # Check for global stats
            if '_global_stats' in graph:
                global_stats = graph['_global_stats']
                print(f"\n🕸️  Knowledge Graph:")
                print(f"   Nodes: {global_stats.get('num_nodes', 0)}")
                print(f"   Edges: {global_stats.get('num_edges', 0)}")
                print(f"   Entities: {global_stats.get('num_entities', 0)}")
                print(f"   Relations: {global_stats.get('num_relations', 0)}")
            else:
                # Fallback to old structure
                print(f"\n🕸️  Knowledge Graph:")
                print(f"   Nodes: {graph.get('num_nodes', 0)}")
                print(f"   Edges: {graph.get('num_edges', 0)}")
                print(f"   Entities: {graph.get('num_entities', 0)}")
                print(f"   Relations: {graph.get('num_relations', 0)}")
        
        # Aggregated metrics by mode
        if 'aggregated' in self.metrics_results:
            print(f"\n📏 Metrics by Mode:")
            for mode, metrics in self.metrics_results['aggregated'].items():
                print(f"\n   {mode.upper()}:")
                print(f"      ROUGE-1: {metrics.get('rouge-1', 0):.4f}")
                print(f"      ROUGE-2: {metrics.get('rouge-2', 0):.4f}")
                print(f"      ROUGE-L: {metrics.get('rouge-l', 0):.4f}")
                print(f"      BLEU: {metrics.get('bleu', 0):.4f}")
                print(f"      F1: {metrics.get('f1', 0):.4f}")
                print(f"      EM Rate: {metrics.get('exact_match_rate', 0):.4f}")
                
                # Containment scores (NEW)
                if 'containment' in metrics:
                    containment = metrics['containment']
                    print(f"      Containment (Exact): {containment.get('exact_substring', 0):.4f}")
                    print(f"      Containment (Normalized): {containment.get('normalized_substring', 0):.4f}")
                    print(f"      Containment (Token Ratio): {containment.get('token_overlap_ratio', 0):.4f}")
                    print(f"      Containment (All Tokens): {containment.get('all_tokens_present', 0):.4f}")
                
                # Semantic similarity (NEW)
                if 'semantic_similarity' in metrics:
                    print(f"      Semantic Similarity: {metrics.get('semantic_similarity', 0):.4f}")
                
                # Graph quality (NEW)
                if 'graph_quality' in metrics:
                    gq = metrics['graph_quality']
                    print(f"      Graph Quality:")
                    print(f"        References Used: {gq.get('avg_references', 0):.2f} avg")
                    print(f"        Entity Density: {gq.get('entity_density', 0):.3f}")
                    print(f"        Token Diversity: {gq.get('unique_token_ratio', 0):.3f}")
                
                # LLM Judge scores (NEW)
                if 'llm_judge' in metrics:
                    judge = metrics['llm_judge']
                    print(f"      LLM Judge (GPT-4 Evaluation):")
                    print(f"        Correctness:  {judge.get('correctness', 0):.2f}/5")
                    print(f"        Completeness: {judge.get('completeness', 0):.2f}/5")
                    print(f"        Faithfulness: {judge.get('faithfulness', 0):.2f}/5")
                    print(f"        Conciseness:  {judge.get('conciseness', 0):.2f}/5")
                    print(f"        Overall:      {judge.get('overall', 0):.2f}/5")
                
                print(f"      Avg Latency: {metrics.get('avg_latency_seconds', 0):.2f}s")
        
        print("\n" + "="*70)


# Testing and example usage
if __name__ == "__main__":
    async def test_pipeline():
        print("🧪 Testing Evaluation Pipeline\n")
        
        from ..configs.dataset_config import QUICK_TEST_CONFIG
        
        # Create pipeline with quick test config
        pipeline = EvaluationPipeline(config=QUICK_TEST_CONFIG)
        
        # Run evaluation
        summary = await pipeline.run_full_evaluation(
            modes=["hybrid"],  # Test with one mode for speed
            clear_existing=True
        )
        
        # Print summary
        pipeline.print_summary()
        
        # Save results
        pipeline.save_results()
        
        print("\n✅ Pipeline test complete!")
    
    # Run test
    asyncio.run(test_pipeline())
