"""
Semantic Similarity Metrics

Embedding-based semantic similarity for comparing answers.
Uses SentenceTransformers to measure meaning similarity, not just token overlap.

This is crucial for evaluating verbose answers (like LightRAG) where the same
fact may be expressed in many words vs. a short reference answer.
"""

from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer, util
import numpy as np


class SemanticMetrics:
    """
    Semantic similarity metrics using sentence embeddings.
    
    Advantages over ROUGE/BLEU:
    - Measures meaning, not just token overlap
    - Handles paraphrasing well
    - Fair to both concise and verbose answers
    - Language-agnostic (with multilingual models)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic metrics with a sentence transformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Popular choices:
                       - 'all-MiniLM-L6-v2': Fast, 384-dim, good quality (default)
                       - 'all-mpnet-base-v2': Slower, 768-dim, better quality
                       - 'multi-qa-MiniLM-L6-cos-v1': Optimized for Q&A
        """
        # print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        # print(f"Model loaded: {model_name}")
    
    def semantic_similarity(self, predicted: str, reference: str) -> float:
        """
        Calculate semantic similarity between predicted and reference answers.
        
        Uses cosine similarity between sentence embeddings.
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Similarity score (0-1, where 1 is perfect similarity)
        """
        if not predicted.strip() or not reference.strip():
            return 0.0
        
        # Encode both texts
        pred_embedding = self.model.encode(predicted, convert_to_tensor=True)
        ref_embedding = self.model.encode(reference, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(pred_embedding, ref_embedding).item()
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def semantic_similarity_batch(self, predictions: List[str], references: List[str]) -> List[float]:
        """
        Calculate semantic similarity for multiple pairs efficiently.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers (same length as predictions)
            
        Returns:
            List of similarity scores
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        
        if not predictions:
            return []
        
        # Batch encode for efficiency
        pred_embeddings = self.model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
        ref_embeddings = self.model.encode(references, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate pairwise cosine similarities
        similarities = util.cos_sim(pred_embeddings, ref_embeddings)
        
        # Extract diagonal (pred[i] vs ref[i])
        scores = [similarities[i][i].item() for i in range(len(predictions))]
        
        # Ensure scores are in [0, 1] range
        return [max(0.0, min(1.0, score)) for score in scores]
    
    def semantic_search(self, query: str, candidates: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most semantically similar candidates to a query.
        
        Useful for finding the best matching reference or analyzing
        which answers are most similar to each other.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of dicts with 'text', 'score', and 'index' for top-k matches
        """
        if not query.strip() or not candidates:
            return []
        
        # Encode query and candidates
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate similarities
        similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
        # Get top-k indices
        top_k = min(top_k, len(candidates))
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'text': candidates[idx],
                'score': similarities[idx].item(),
                'index': idx.item()
            })
        
        return results
    
    def calculate_all(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        Calculate all semantic metrics for a prediction.
        
        Currently just semantic similarity, but structured for future expansion
        (e.g., BERTScore, semantic F1, etc.).
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Dictionary with semantic metrics
        """
        return {
            'semantic_similarity': self.semantic_similarity(predicted, reference)
        }


# Testing and example usage
if __name__ == "__main__":
    print("🧪 Testing Semantic Metrics\n")
    
    # Initialize metrics
    metrics = SemanticMetrics()
    
    # Test 1: Similar meaning, different words
    print("Test 1: Paraphrasing")
    predicted = "The capital of France is Paris."
    reference = "Paris is France's capital city."
    
    similarity = metrics.semantic_similarity(predicted, reference)
    print(f"  Predicted: {predicted}")
    print(f"  Reference: {reference}")
    print(f"  Semantic Similarity: {similarity:.3f}\n")
    
    # Test 2: Verbose answer vs short reference (LightRAG scenario)
    print("Test 2: Verbose vs Concise (LightRAG style)")
    verbose_pred = """The series you are referring to is **Animorphs**, a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives."""
    
    short_ref = "Animorphs"
    
    similarity = metrics.semantic_similarity(verbose_pred, short_ref)
    print(f"  Predicted (verbose): {verbose_pred[:80]}...")
    print(f"  Reference (concise): {short_ref}")
    print(f"  Semantic Similarity: {similarity:.3f}")
    print(f"  (Much higher than ROUGE-1: ~0.04!)\n")
    
    # Test 3: Completely different topics
    print("Test 3: Different Topics")
    pred_diff = "The theory of relativity was developed by Einstein."
    ref_diff = "Shakespeare wrote Hamlet."
    
    similarity = metrics.semantic_similarity(pred_diff, ref_diff)
    print(f"  Predicted: {pred_diff}")
    print(f"  Reference: {ref_diff}")
    print(f"  Semantic Similarity: {similarity:.3f}\n")
    
    # Test 4: Batch processing
    print("Test 4: Batch Processing")
    predictions = [
        "Paris is the capital of France",
        "London is the capital of England",
        "Berlin is the capital of Germany"
    ]
    references = [
        "The capital of France is Paris",
        "England's capital city is London",
        "Germany's capital is Berlin"
    ]
    
    batch_similarities = metrics.semantic_similarity_batch(predictions, references)
    for i, (pred, ref, sim) in enumerate(zip(predictions, references, batch_similarities)):
        print(f"  {i+1}. Similarity: {sim:.3f}")
    
    print("\n📊 All Metrics:")
    all_metrics = metrics.calculate_all(verbose_pred, short_ref)
    for metric, score in all_metrics.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n✅ Semantic metrics test complete!")
