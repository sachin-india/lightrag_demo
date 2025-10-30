"""
Traditional text generation metrics

Pure Python implementations of ROUGE, BLEU, Exact Match, and F1 scores.
No external model dependencies - Intel compliant.

Copied from Unified_RAG benchmarking system for LightRAG integration.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Set
import math


class TraditionalMetrics:
    """Traditional NLP metrics without external model dependencies"""
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize answer for comparison"""
        if s is None:
            return ""
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def get_tokens(s: str) -> List[str]:
        """Tokenize text into words"""
        if not s:
            return []
        return TraditionalMetrics.normalize_answer(s).split()
    
    def exact_match(self, predicted: str, reference: str) -> float:
        """
        Exact match score after normalization
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return float(self.normalize_answer(predicted) == self.normalize_answer(reference))
    
    def token_f1_score(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        Token-level F1 score
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        pred_tokens = self.get_tokens(predicted)
        ref_tokens = self.get_tokens(reference)
        
        if not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if not pred_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        # Calculate overlap
        overlap = sum((pred_counter & ref_counter).values())
        
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def rouge_1(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        ROUGE-1 (unigram) score
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        pred_tokens = self.get_tokens(predicted)
        ref_tokens = self.get_tokens(reference)
        
        if not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if not pred_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        pred_unigrams = set(pred_tokens)
        ref_unigrams = set(ref_tokens)
        
        overlap = len(pred_unigrams & ref_unigrams)
        
        precision = overlap / len(pred_unigrams) if pred_unigrams else 0.0
        recall = overlap / len(ref_unigrams) if ref_unigrams else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def rouge_2(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        ROUGE-2 (bigram) score
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        pred_tokens = self.get_tokens(predicted)
        ref_tokens = self.get_tokens(reference)
        
        if len(ref_tokens) < 2:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if len(pred_tokens) < 2:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Create bigrams
        pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        
        overlap = len(pred_bigrams & ref_bigrams)
        
        precision = overlap / len(pred_bigrams) if pred_bigrams else 0.0
        recall = overlap / len(ref_bigrams) if ref_bigrams else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def rouge_l(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        ROUGE-L (Longest Common Subsequence) score
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        pred_tokens = self.get_tokens(predicted)
        ref_tokens = self.get_tokens(reference)
        
        if not ref_tokens or not pred_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def bleu_score(self, predicted: str, reference: str, max_n: int = 4) -> Dict[str, float]:
        """
        BLEU score calculation
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            max_n: Maximum n-gram order
            
        Returns:
            Dictionary with BLEU scores for different n-gram orders
        """
        pred_tokens = self.get_tokens(predicted)
        ref_tokens = self.get_tokens(reference)
        
        if not pred_tokens:
            return {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)}
        
        scores = {}
        
        for n in range(1, max_n + 1):
            if len(pred_tokens) < n or len(ref_tokens) < n:
                scores[f'bleu_{n}'] = 0.0
                continue
            
            # Create n-grams
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            # Calculate precision
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            precision = overlap / total if total > 0 else 0.0
            
            # Brevity penalty
            bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))
            
            scores[f'bleu_{n}'] = bp * precision
        
        return scores
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)
    
    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """Calculate brevity penalty for BLEU"""
        if pred_len >= ref_len:
            return 1.0
        else:
            return math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
    
    def answer_containment(self, predicted: str, reference: str) -> Dict[str, Any]:
        """
        Check if reference answer is contained in the prediction.
        
        This metric is designed for verbose answers (like LightRAG) where
        the reference answer may be embedded within a longer explanation.
        
        Args:
            predicted: Predicted answer (may be verbose)
            reference: Reference answer (typically short)
            
        Returns:
            Dictionary with containment scores:
            - exact_substring: 1.0 if reference is substring of prediction
            - normalized_substring: 1.0 if normalized reference is in normalized prediction
            - token_overlap_ratio: Ratio of reference tokens found in prediction
            - all_tokens_present: 1.0 if all reference tokens appear in prediction
        """
        # Exact substring match (case-insensitive)
        pred_lower = predicted.lower()
        ref_lower = reference.lower()
        exact_substring = 1.0 if ref_lower in pred_lower else 0.0
        
        # Normalized substring match (removes punctuation, articles, etc.)
        pred_normalized = self.normalize_answer(predicted)
        ref_normalized = self.normalize_answer(reference)
        normalized_substring = 1.0 if ref_normalized in pred_normalized else 0.0
        
        # Token-level containment
        pred_tokens = set(self.get_tokens(predicted))
        ref_tokens = set(self.get_tokens(reference))
        
        if not ref_tokens:
            token_overlap_ratio = 0.0
            all_tokens_present = 0.0
        else:
            overlapping_tokens = pred_tokens & ref_tokens
            token_overlap_ratio = len(overlapping_tokens) / len(ref_tokens)
            all_tokens_present = 1.0 if len(overlapping_tokens) == len(ref_tokens) else 0.0
        
        return {
            'exact_substring': exact_substring,
            'normalized_substring': normalized_substring,
            'token_overlap_ratio': token_overlap_ratio,
            'all_tokens_present': all_tokens_present
        }
    
    def calculate_all(self, predicted: str, reference: str) -> Dict[str, Any]:
        """
        Calculate all traditional metrics
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Dictionary with all metric scores
        """
        results = {}
        
        # Exact match
        results['exact_match'] = self.exact_match(predicted, reference)
        
        # Token F1
        f1_scores = self.token_f1_score(predicted, reference)
        results.update({f'token_{k}': v for k, v in f1_scores.items()})
        
        # ROUGE scores
        rouge1 = self.rouge_1(predicted, reference)
        results.update({f'rouge1_{k}': v for k, v in rouge1.items()})
        
        rouge2 = self.rouge_2(predicted, reference)
        results.update({f'rouge2_{k}': v for k, v in rouge2.items()})
        
        rougel = self.rouge_l(predicted, reference)
        results.update({f'rougel_{k}': v for k, v in rougel.items()})
        
        # BLEU scores
        bleu_scores = self.bleu_score(predicted, reference)
        results.update(bleu_scores)
        
        # Answer containment
        containment = self.answer_containment(predicted, reference)
        results.update({f'containment_{k}': v for k, v in containment.items()})
        
        return results


# Testing and example usage
if __name__ == "__main__":
    # Test traditional metrics
    metrics = TraditionalMetrics()
    
    predicted = "The capital of France is Paris, a beautiful city."
    reference = "Paris is the capital of France."
    
    print("🧪 Testing Traditional Metrics")
    print(f"Predicted: {predicted}")
    print(f"Reference: {reference}")
    print()
    
    # Test individual metrics
    print(f"Exact Match: {metrics.exact_match(predicted, reference)}")
    print(f"Token F1: {metrics.token_f1_score(predicted, reference)}")
    print(f"ROUGE-1: {metrics.rouge_1(predicted, reference)}")
    print(f"ROUGE-2: {metrics.rouge_2(predicted, reference)}")
    print(f"ROUGE-L: {metrics.rouge_l(predicted, reference)}")
    print(f"BLEU: {metrics.bleu_score(predicted, reference)}")
    
    # Test answer containment (NEW)
    print(f"\nAnswer Containment: {metrics.answer_containment(predicted, reference)}")
    
    print("\n📊 All Metrics:")
    all_scores = metrics.calculate_all(predicted, reference)
    for metric, score in all_scores.items():
        print(f"  {metric}: {score:.3f}")
    
    # Test with verbose answer (LightRAG-style)
    print("\n\n🧪 Testing with Verbose Answer (LightRAG-style)")
    verbose_predicted = """The series you are referring to is **Animorphs**, a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives.

### References
[1] Animorphs is a science fantasy series..."""
    
    short_reference = "Animorphs"
    
    print(f"Predicted (187 words): {verbose_predicted[:100]}...")
    print(f"Reference: {short_reference}")
    print()
    
    containment = metrics.answer_containment(verbose_predicted, short_reference)
    print("Answer Containment Results:")
    for key, value in containment.items():
        print(f"  {key}: {value:.3f}")
    
    rouge1_scores = metrics.rouge_1(verbose_predicted, short_reference)
    print(f"\nROUGE-1 F1: {rouge1_scores['f1']:.3f}")
    print(f"Comparison: Containment shows reference IS in answer (1.0), ROUGE shows low overlap ({rouge1_scores['f1']:.3f})!")
