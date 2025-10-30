"""
LightRAG Evaluator for Benchmarking

Wraps LightRAG for systematic evaluation on benchmark datasets.
Handles graph building from benchmark documents and query execution
with separate storage to avoid impacting production systems.

Key Features:
- Separate storage for benchmark graphs
- Document ingestion from benchmark datasets
- Multi-mode query execution (Naive, Local, Global, Hybrid)
- Result tracking with timing and metadata
- Entity/relation mapping for evaluation
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import xml.etree.ElementTree as ET

from lightrag import LightRAG, QueryParam
from lightrag_intel.adapter import create_alloy_lightrag_async

from ..utils.logging import get_logger
from ..utils.errors import EvaluationError, handle_error
from ..metrics.efficiency import EfficiencyMetrics

logger = get_logger("lightrag_evaluator")


class LightRAGEvaluator:
    """
    Evaluator for LightRAG performance on benchmark datasets.
    
    Provides isolated environment for benchmarking without affecting
    production LightRAG instances.
    """
    
    def __init__(self, 
                 working_dir: Optional[Path] = None,
                 clear_existing: bool = False):
        """
        Initialize LightRAG evaluator.
        
        Args:
            working_dir: Directory for benchmark graph storage
            clear_existing: Whether to clear existing graph before evaluation
        """
        self.working_dir = working_dir or Path("benchmarks/benchmark_storage/lightrag")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.rag: Optional[LightRAG] = None
        self.documents_ingested = False
        self.query_history: List[Dict[str, Any]] = []
        self.graph_stats: Dict[str, Any] = {}
        
        # Clear existing if requested
        if clear_existing:
            self._clear_storage()
        
        logger.info(f"LightRAG evaluator initialized (working_dir={self.working_dir})")
    
    def _clear_storage(self):
        """Clear existing benchmark storage"""
        if self.working_dir.exists():
            import shutil
            shutil.rmtree(self.working_dir)
            self.working_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared existing benchmark storage")
    
    async def initialize(self):
        """Initialize LightRAG instance"""
        with logger.operation("initialize_lightrag"):
            try:
                self.rag = await create_alloy_lightrag_async(
                    working_dir=str(self.working_dir)
                )
                
                # Check if graph exists
                await self._load_graph_stats()
                
                logger.info(f"✅ LightRAG initialized: {self.graph_stats.get('num_nodes', 0)} nodes, "
                          f"{self.graph_stats.get('num_edges', 0)} edges")
                
            except Exception as e:
                raise EvaluationError(
                    f"Failed to initialize LightRAG: {e}",
                    error_code="LIGHTRAG_INIT_ERROR",
                    context={"working_dir": str(self.working_dir)}
                )
    
    async def _load_graph_stats(self):
        """Load statistics about the knowledge graph"""
        try:
            graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            
            if not graph_file.exists():
                self.graph_stats = {
                    'num_nodes': 0,
                    'num_edges': 0,
                    'avg_degree': 0.0,
                    'density': 0.0,
                    'num_entities': 0,
                    'num_relations': 0
                }
                return
            
            # Parse GraphML
            tree = ET.parse(graph_file)
            root = tree.getroot()
            
            # Count nodes and edges
            nodes = root.findall('.//{http://graphml.graphdrawing.org/xmlns}node')
            edges = root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge')
            
            num_nodes = len(nodes)
            num_edges = len(edges)
            
            # Calculate statistics
            avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
            density = (2 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0
            
            self.graph_stats = {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'avg_degree': avg_degree,
                'density': density
            }
            
            # Count entities and relations from KV stores
            entities_file = self.working_dir / "kv_store_full_entities.json"
            relations_file = self.working_dir / "kv_store_full_relations.json"
            
            if entities_file.exists():
                with open(entities_file, 'r', encoding='utf-8') as f:
                    entities = json.load(f)
                    self.graph_stats['num_entities'] = len(entities)
            
            if relations_file.exists():
                with open(relations_file, 'r', encoding='utf-8') as f:
                    relations = json.load(f)
                    self.graph_stats['num_relations'] = len(relations)
            
        except Exception as e:
            logger.warning(f"Failed to load graph stats: {e}")
            self.graph_stats = {}
    
    async def ingest_documents(self, documents: List[Dict[str, Any]], 
                              batch_size: int = 10,
                              parallel: bool = True,
                              max_concurrent: int = 10) -> Dict[str, Any]:
        """
        Ingest benchmark documents into LightRAG.
        
        Args:
            documents: List of document dictionaries with 'id' and 'text'
            batch_size: Number of documents to process in each batch
            parallel: Whether to process documents in parallel (default: True)
            max_concurrent: Maximum number of concurrent insertions (increased from 5 to 10 for faster ingestion)
            
        Returns:
            Ingestion statistics
        """
        with logger.operation("ingest_documents", count=len(documents)):
            if self.rag is None:
                raise EvaluationError(
                    "LightRAG not initialized. Call initialize() first.",
                    error_code="NOT_INITIALIZED"
                )
            
            start_time = time.time()
            ingested_count = 0
            failed_count = 0
            errors = []
            
            try:
                # Process in batches
                num_batches = (len(documents) + batch_size - 1) // batch_size
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    logger.info(f"Processing batch {batch_num}/{num_batches} "
                              f"({'parallel' if parallel else 'sequential'}, "
                              f"max_concurrent={max_concurrent if parallel else 1})")
                    
                    if parallel:
                        # Parallel processing with concurrency limit
                        async def ingest_doc(doc):
                            """Ingest single document with error handling"""
                            try:
                                doc_id = doc.get('id', f'doc_{doc.get("index", 0)}')
                                text = doc.get('text', '')
                                
                                if not text.strip():
                                    return {'status': 'skipped', 'id': doc_id, 'reason': 'empty'}
                                
                                await self.rag.ainsert(text)
                                return {'status': 'success', 'id': doc_id}
                                
                            except Exception as e:
                                return {'status': 'failed', 'id': doc_id, 'error': str(e)}
                        
                        # Process in smaller concurrent groups to avoid overwhelming the system
                        for j in range(0, len(batch), max_concurrent):
                            concurrent_batch = batch[j:j + max_concurrent]
                            results = await asyncio.gather(
                                *[ingest_doc(doc) for doc in concurrent_batch],
                                return_exceptions=True
                            )
                            
                            # Process results
                            for result in results:
                                if isinstance(result, dict):
                                    if result['status'] == 'success':
                                        ingested_count += 1
                                    elif result['status'] == 'failed':
                                        failed_count += 1
                                        errors.append(f"{result['id']}: {result.get('error', 'Unknown')}")
                                    # skipped doesn't count as failed
                                else:
                                    # Exception was raised
                                    failed_count += 1
                                    errors.append(f"Unexpected error: {result}")
                    else:
                        # Sequential processing (original behavior)
                        for doc in batch:
                            try:
                                doc_id = doc.get('id', f'doc_{i}')
                                text = doc.get('text', '')
                                
                                if not text.strip():
                                    logger.warning(f"Skipping empty document: {doc_id}")
                                    continue
                                
                                # Insert document into LightRAG
                                await self.rag.ainsert(text)
                                ingested_count += 1
                                
                            except Exception as e:
                                failed_count += 1
                                error_msg = f"Failed to ingest {doc_id}: {e}"
                                logger.error(error_msg)
                                errors.append(error_msg)
                
                # Reload graph stats after ingestion
                await self._load_graph_stats()
                
                duration = time.time() - start_time
                
                stats = {
                    'total_documents': len(documents),
                    'ingested': ingested_count,
                    'failed': failed_count,
                    'duration_seconds': duration,
                    'docs_per_second': ingested_count / duration if duration > 0 else 0,
                    'graph_stats': self.graph_stats,
                    'errors': errors[:10],  # Keep first 10 errors
                    'parallel_mode': parallel,
                    'max_concurrent': max_concurrent if parallel else 1
                }
                
                self.documents_ingested = True
                
                logger.info(f"✅ Ingested {ingested_count}/{len(documents)} documents in {duration:.1f}s "
                          f"({stats['docs_per_second']:.2f} docs/sec)")
                
                return stats
                
            except Exception as e:
                raise EvaluationError(
                    f"Document ingestion failed: {e}",
                    error_code="INGESTION_ERROR",
                    context={
                        "total_documents": len(documents),
                        "ingested": ingested_count,
                        "failed": failed_count
                    }
                )
    
    async def query(self, 
                   query_text: str,
                   mode: str = "hybrid",
                   top_k: int = 10,
                   query_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a query against LightRAG.
        
        Args:
            query_text: Query string
            mode: Query mode (naive, local, global, hybrid)
            top_k: Number of results to return
            query_id: Optional query identifier for tracking
            
        Returns:
            Query result with response, timing, and metadata
        """
        if self.rag is None:
            raise EvaluationError(
                "LightRAG not initialized. Call initialize() first.",
                error_code="NOT_INITIALIZED"
            )
        
        if not self.documents_ingested:
            logger.warning("Querying before documents ingested - results may be empty")
        
        start_time = time.time()
        
        try:
            # Create query parameters
            param = QueryParam(
                mode=mode,
                top_k=top_k
            )
            
            # Execute query
            response = await self.rag.aquery(query_text, param=param)
            
            duration = time.time() - start_time
            
            # Build result
            result = {
                'query_id': query_id or f"query_{len(self.query_history)}",
                'query': query_text,
                'mode': mode,
                'top_k': top_k,
                'response': response,
                'duration_seconds': duration,
                'timestamp': time.time(),
                'graph_stats': self.graph_stats.copy()
            }
            
            # Store in history
            self.query_history.append(result)
            
            logger.debug(f"Query executed in {duration:.2f}s (mode={mode})")
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise EvaluationError(
                f"Query failed: {e}",
                error_code="QUERY_ERROR",
                context={
                    "query": query_text,
                    "mode": mode,
                    "query_id": query_id
                }
            )
    
    async def query_all_modes(self, 
                             query_text: str,
                             top_k: int = 10,
                             query_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Execute query in all modes (naive, local, global, hybrid).
        
        Args:
            query_text: Query string
            top_k: Number of results per mode
            query_id: Optional query identifier
            
        Returns:
            Dictionary mapping mode to result
        """
        modes = ["naive", "local", "global", "hybrid"]
        results = {}
        
        with logger.operation("query_all_modes", query=query_text[:50]):
            for mode in modes:
                try:
                    result = await self.query(
                        query_text=query_text,
                        mode=mode,
                        top_k=top_k,
                        query_id=f"{query_id}_{mode}" if query_id else None
                    )
                    results[mode] = result
                    
                except Exception as e:
                    logger.error(f"Failed to query in {mode} mode: {e}")
                    results[mode] = {
                        'error': str(e),
                        'mode': mode,
                        'query': query_text
                    }
        
        return results
    
    async def evaluate_queries(self, 
                              queries: List[Dict[str, Any]],
                              modes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate a list of benchmark queries.
        
        Args:
            queries: List of query dictionaries with 'query_id', 'query', 'answer'
            modes: Query modes to evaluate (defaults to all)
            
        Returns:
            List of evaluation results
        """
        modes = modes or ["naive", "local", "global", "hybrid"]
        results = []
        
        with logger.operation("evaluate_queries", count=len(queries), modes=modes):
            for i, query_sample in enumerate(queries):
                query_id = query_sample.get('query_id', f'q_{i}')
                query_text = query_sample.get('query', '')
                reference_answer = query_sample.get('answer', '')
                
                logger.info(f"Evaluating query {i+1}/{len(queries)}: {query_id}")
                
                # Query in all requested modes
                mode_results = {}
                for mode in modes:
                    try:
                        result = await self.query(
                            query_text=query_text,
                            mode=mode,
                            query_id=f"{query_id}_{mode}"
                        )
                        mode_results[mode] = result
                        
                    except Exception as e:
                        logger.error(f"Query {query_id} failed in {mode} mode: {e}")
                        mode_results[mode] = {'error': str(e)}
                
                # Combine results
                eval_result = {
                    'query_id': query_id,
                    'query': query_text,
                    'reference_answer': reference_answer,
                    'mode_results': mode_results,
                    'metadata': query_sample.get('metadata', {})
                }
                
                results.append(eval_result)
        
        logger.info(f"✅ Evaluated {len(queries)} queries in {len(modes)} modes")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluator statistics"""
        return {
            'working_dir': str(self.working_dir),
            'documents_ingested': self.documents_ingested,
            'total_queries': len(self.query_history),
            'graph_stats': self.graph_stats,
            'query_modes': self._count_query_modes()
        }
    
    def _count_query_modes(self) -> Dict[str, int]:
        """Count queries by mode"""
        mode_counts = {}
        for query in self.query_history:
            mode = query.get('mode', 'unknown')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        return mode_counts
    
    def save_results(self, output_path: Optional[Path] = None):
        """Save evaluation results to JSON"""
        output_path = output_path or (self.working_dir / "evaluation_results.json")
        
        results = {
            'statistics': self.get_statistics(),
            'query_history': self.query_history
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved evaluation results to {output_path}")


# Testing and example usage
if __name__ == "__main__":
    async def test_evaluator():
        print("🧪 Testing LightRAG Evaluator\n")
        
        # Create evaluator
        evaluator = LightRAGEvaluator(
            working_dir=Path("test_benchmark_eval"),
            clear_existing=True
        )
        
        # Initialize
        await evaluator.initialize()
        print(f"✅ Evaluator initialized\n")
        
        # Test documents
        test_docs = [
            {
                'id': 'doc_1',
                'text': 'Machine learning is a method of data analysis that automates analytical model building.'
            },
            {
                'id': 'doc_2',
                'text': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks.'
            }
        ]
        
        # Ingest documents
        print("📝 Ingesting test documents...")
        ingest_stats = await evaluator.ingest_documents(test_docs)
        print(f"   Ingested: {ingest_stats['ingested']}/{ingest_stats['total_documents']}")
        print(f"   Duration: {ingest_stats['duration_seconds']:.2f}s\n")
        
        # Test single query
        print("🔍 Testing single query...")
        result = await evaluator.query("What is machine learning?", mode="hybrid")
        print(f"   Response: {result['response'][:100]}...")
        print(f"   Duration: {result['duration_seconds']:.2f}s\n")
        
        # Test all modes
        print("🔍 Testing all query modes...")
        all_results = await evaluator.query_all_modes("What is deep learning?")
        for mode, res in all_results.items():
            print(f"   {mode}: {res.get('response', 'ERROR')[:50]}...")
        
        # Get statistics
        stats = evaluator.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Graph nodes: {stats['graph_stats'].get('num_nodes', 0)}")
        
        # Save results
        evaluator.save_results()
        
        print("\n✅ Evaluator tests complete!")
    
    # Run test
    asyncio.run(test_evaluator())
