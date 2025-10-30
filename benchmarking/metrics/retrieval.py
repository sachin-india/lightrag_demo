"""
Retrieval metrics for RAG evaluation

Pure algorithmic implementations of retrieval quality metrics.
No external model dependencies - Intel compliant.

Copied from Unified_RAG benchmarking system for LightRAG integration.
"""

import math
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict


class RetrievalMetrics:
    """Retrieval quality metrics for RAG systems"""
    
    def __init__(self):
        pass
    
    def precision_at_k(self, retrieved_items: List[str], relevant_items: Set[str], k: int = 5) -> float:
        """
        Precision@K: Fraction of top-k retrieved items that are relevant
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevant_items: Set of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            Precision@K score
        """
        if not retrieved_items or k == 0:
            return 0.0
        
        top_k = retrieved_items[:k]
        relevant_retrieved = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_retrieved / min(k, len(top_k))
    
    def recall_at_k(self, retrieved_items: List[str], relevant_items: Set[str], k: int = 5) -> float:
        """
        Recall@K: Fraction of relevant items found in top-k results
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevant_items: Set of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            Recall@K score
        """
        if not relevant_items:
            return 0.0
        
        if not retrieved_items or k == 0:
            return 0.0
        
        top_k = retrieved_items[:k]
        relevant_retrieved = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_retrieved / len(relevant_items)
    
    def mean_reciprocal_rank(self, retrieved_items: List[str], relevant_items: Set[str]) -> float:
        """
        Mean Reciprocal Rank: 1/rank of first relevant item
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevant_items: Set of relevant item IDs
            
        Returns:
            MRR score
        """
        for rank, item in enumerate(retrieved_items, 1):
            if item in relevant_items:
                return 1.0 / rank
        
        return 0.0
    
    def average_precision(self, retrieved_items: List[str], relevant_items: Set[str]) -> float:
        """
        Average Precision: Mean of precision values after each relevant item
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevant_items: Set of relevant item IDs
            
        Returns:
            Average precision score
        """
        if not relevant_items:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for rank, item in enumerate(retrieved_items, 1):
            if item in relevant_items:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank
        
        return precision_sum / len(relevant_items) if relevant_items else 0.0
    
    def ndcg_at_k(self, retrieved_items: List[str], relevance_scores: Dict[str, float], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevance_scores: Dictionary mapping item IDs to relevance scores
            k: Number of top items to consider
            
        Returns:
            NDCG@K score
        """
        def dcg(items: List[str], scores: Dict[str, float], k: int) -> float:
            dcg_sum = 0.0
            for i, item in enumerate(items[:k]):
                if item in scores:
                    gain = scores[item]
                    discount = math.log2(i + 2)  # i+2 because rank starts from 1
                    dcg_sum += gain / discount
            return dcg_sum
        
        # Calculate DCG for retrieved items
        actual_dcg = dcg(retrieved_items, relevance_scores, k)
        
        # Calculate IDCG (ideal DCG) - sort by relevance
        ideal_items = sorted(relevance_scores.keys(), 
                           key=lambda x: relevance_scores[x], 
                           reverse=True)
        ideal_dcg = dcg(ideal_items, relevance_scores, k)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def hit_rate_at_k(self, retrieved_items: List[str], relevant_items: Set[str], k: int = 5) -> float:
        """
        Hit Rate@K: Whether at least one relevant item is in top-k
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevant_items: Set of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            Hit rate (1.0 if hit, 0.0 otherwise)
        """
        if not relevant_items or not retrieved_items:
            return 0.0
        
        top_k = retrieved_items[:k]
        return 1.0 if any(item in relevant_items for item in top_k) else 0.0
    
    def coverage_at_k(self, retrieved_items: List[str], relevant_items: Set[str], k: int = 5) -> float:
        """
        Coverage@K: Fraction of relevant items covered in top-k
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevant_items: Set of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            Coverage score
        """
        if not relevant_items:
            return 0.0
        
        if not retrieved_items:
            return 0.0
        
        top_k = set(retrieved_items[:k])
        covered_relevant = top_k.intersection(relevant_items)
        
        return len(covered_relevant) / len(relevant_items)
    
    def calculate_retrieval_metrics(self, 
                                  retrieved_items: List[str], 
                                  relevant_items: Set[str],
                                  relevance_scores: Optional[Dict[str, float]] = None,
                                  k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Calculate comprehensive retrieval metrics
        
        Args:
            retrieved_items: List of retrieved item IDs (in ranked order)
            relevant_items: Set of relevant item IDs
            relevance_scores: Optional relevance scores for NDCG
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with all retrieval metrics
        """
        results = {}
        
        # Calculate metrics for different k values
        for k in k_values:
            results[f'precision_at_{k}'] = self.precision_at_k(retrieved_items, relevant_items, k)
            results[f'recall_at_{k}'] = self.recall_at_k(retrieved_items, relevant_items, k)
            results[f'hit_rate_at_{k}'] = self.hit_rate_at_k(retrieved_items, relevant_items, k)
            results[f'coverage_at_{k}'] = self.coverage_at_k(retrieved_items, relevant_items, k)
            
            # NDCG if relevance scores provided
            if relevance_scores:
                results[f'ndcg_at_{k}'] = self.ndcg_at_k(retrieved_items, relevance_scores, k)
        
        # Single-value metrics
        results['mrr'] = self.mean_reciprocal_rank(retrieved_items, relevant_items)
        results['average_precision'] = self.average_precision(retrieved_items, relevant_items)
        
        return results


class RAGRetrievalEvaluator:
    """Specialized evaluator for RAG retrieval quality"""
    
    def __init__(self):
        self.metrics = RetrievalMetrics()
    
    def evaluate_rag_retrieval(self, 
                             query: str,
                             retrieved_chunks: List[Dict[str, Any]],
                             ground_truth_passages: List[Dict[str, Any]],
                             similarity_threshold: float = 0.7) -> Dict[str, float]:
        """
        Evaluate RAG retrieval quality using chunk metadata
        
        Args:
            query: Original query
            retrieved_chunks: Chunks returned by RAG system
            ground_truth_passages: Expected relevant passages
            similarity_threshold: Threshold for considering passages similar
            
        Returns:
            Retrieval quality metrics
        """
        # Extract retrieved chunk IDs or texts
        retrieved_items = []
        for chunk in retrieved_chunks:
            # Use document_id + chunk position as identifier
            doc_id = chunk.get('metadata', {}).get('document_id', 'unknown')
            chunk_id = chunk.get('id', f"{doc_id}_chunk")
            retrieved_items.append(chunk_id)
        
        # Create ground truth set (this would need to be adapted based on dataset format)
        relevant_items = set()
        for passage in ground_truth_passages:
            passage_id = passage.get('id', passage.get('title', f"passage_{len(relevant_items)}"))
            relevant_items.add(passage_id)
        
        # For RAG evaluation, we need to match on content similarity
        # This is a simplified version - in practice, you'd want more sophisticated matching
        content_based_relevant = self._find_content_matches(
            retrieved_chunks, ground_truth_passages, similarity_threshold
        )
        
        # Calculate metrics
        return self.metrics.calculate_retrieval_metrics(
            retrieved_items=retrieved_items,
            relevant_items=content_based_relevant
        )
    
    def _find_content_matches(self, 
                            retrieved_chunks: List[Dict[str, Any]], 
                            ground_truth_passages: List[Dict[str, Any]],
                            threshold: float) -> Set[str]:
        """Find content-based matches between retrieved and ground truth"""
        # Simplified content matching - would use more sophisticated methods in practice
        relevant_set = set()
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get('text', '').lower()
            chunk_id = chunk.get('id', f"chunk_{i}")
            
            for passage in ground_truth_passages:
                passage_text = passage.get('text', '').lower()
                
                # Simple word overlap check (would use better similarity in practice)
                chunk_words = set(chunk_text.split())
                passage_words = set(passage_text.split())
                
                if chunk_words and passage_words:
                    overlap = len(chunk_words.intersection(passage_words))
                    jaccard = overlap / len(chunk_words.union(passage_words))
                    
                    if jaccard >= threshold:
                        relevant_set.add(chunk_id)
                        break
        
        return relevant_set


# Testing and example usage
if __name__ == "__main__":
    # Test retrieval metrics
    metrics = RetrievalMetrics()
    
    retrieved = ["doc1", "doc3", "doc2", "doc5", "doc4"]
    relevant = {"doc1", "doc2", "doc6"}
    relevance_scores = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0, "doc4": 0.5, "doc5": 0.0, "doc6": 2.5}
    
    print("🧪 Testing Retrieval Metrics")
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    print()
    
    # Calculate all metrics
    results = metrics.calculate_retrieval_metrics(
        retrieved_items=retrieved,
        relevant_items=relevant,
        relevance_scores=relevance_scores
    )
    
    print("📊 Retrieval Metrics:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
