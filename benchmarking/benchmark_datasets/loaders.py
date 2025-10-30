"""
Dataset loaders for LightRAG benchmark evaluation

Loads MS MARCO and HotpotQA datasets using HuggingFace datasets library.
Adapted from Unified_RAG for LightRAG graph-based benchmarking.

Intel-compliant: Only loads structured data, no model dependencies.
"""

from typing import List, Dict, Any, Optional
import random
from pathlib import Path
import sys
import datasets as hf_datasets

from ..utils.logging import get_logger
from ..utils.errors import DatasetError

logger = get_logger("dataset")


class BaseDatasetLoader:
    """Base class for dataset loaders"""
    
    def __init__(self, dataset_name: str, config_name: str = None):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.dataset = None
    
    def load(self, split: str = "train", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load dataset split with optional limit"""
        raise NotImplementedError("Subclasses must implement load method")
    
    def get_sample(self, n: int = 10, split: str = "train") -> List[Dict[str, Any]]:
        """Get a small sample for testing"""
        return self.load(split=split, limit=n)


class MSMarcoLoader(BaseDatasetLoader):
    """MS MARCO dataset loader for question-answering evaluation"""
    
    def __init__(self):
        super().__init__("microsoft/ms_marco", "v1.1")
    
    def load(self, split: str = "train", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load MS MARCO dataset
        
        Args:
            split: Dataset split ("train", "validation", "test") 
            limit: Optional limit on number of samples
            
        Returns:
            List of standardized question-answer samples
        """
        try:
            logger.info(f"Loading MS MARCO dataset (split={split}, limit={limit})...")
            
            # Load dataset with limit
            if limit:
                dataset = hf_datasets.load_dataset(
                    self.dataset_name, 
                    self.config_name, 
                    split=f"{split}[:{limit}]",
                    trust_remote_code=True
                )
            else:
                dataset = hf_datasets.load_dataset(
                    self.dataset_name, 
                    self.config_name, 
                    split=split,
                    trust_remote_code=True
                )
            
            samples = []
            for item in dataset:
                # Extract query and answers
                query = item.get('query', '')
                answers = item.get('answers', [])
                raw_passages = item.get('passages', {})
                
                # Skip items without answers
                if not answers or not query:
                    continue
                
                # Process passages - MS MARCO format has 'passage_text' as list
                passages = []
                if isinstance(raw_passages, dict) and 'passage_text' in raw_passages:
                    passage_texts = raw_passages.get('passage_text', [])
                    urls = raw_passages.get('url', [])
                    is_selected = raw_passages.get('is_selected', [])
                    
                    for i, text in enumerate(passage_texts):
                        if text and text.strip():  # Skip empty passages
                            passages.append({
                                'id': f"ms_marco_passage_{len(samples)}_{i}",
                                'text': text.strip(),
                                'url': urls[i] if i < len(urls) else '',
                                'is_selected': is_selected[i] if i < len(is_selected) else 0,
                                'index': i
                            })
                elif isinstance(raw_passages, list):
                    # Handle as list of texts
                    for i, text in enumerate(raw_passages):
                        if text and text.strip():
                            passages.append({
                                'id': f"ms_marco_passage_{len(samples)}_{i}",
                                'text': text.strip(),
                                'index': i
                            })
                
                # Create standardized format
                sample = {
                    'id': f"ms_marco_{len(samples)}",
                    'query_id': item.get('query_id', f"ms_marco_query_{len(samples)}"),
                    'query': query,
                    'answer': answers[0] if answers else "",  # Use first answer as reference
                    'all_answers': answers,
                    'passages': passages,
                    'source_type': 'ms_marco',
                    'metadata': {
                        'original_id': item.get('id', ''),
                        'num_passages': len(passages),
                        'num_answers': len(answers)
                    }
                }
                
                samples.append(sample)
            
            logger.info(f"✅ Loaded {len(samples)} MS MARCO samples from {split} split")
            return samples
            
        except Exception as e:
            raise DatasetError(
                f"Failed to load MS MARCO dataset: {e}",
                error_code="MS_MARCO_LOAD_ERROR",
                context={"split": split, "limit": limit},
                suggestions=[
                    "Check internet connection",
                    "Verify HuggingFace datasets library is installed",
                    "Try a smaller limit value",
                    "Check dataset availability on HuggingFace"
                ]
            )


class HotpotQALoader(BaseDatasetLoader):
    """HotpotQA dataset loader for multi-hop reasoning evaluation"""
    
    def __init__(self):
        super().__init__("hotpot_qa", "fullwiki")
    
    def load(self, split: str = "validation", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load HotpotQA dataset
        
        Args:
            split: Dataset split ("train", "validation") 
            limit: Optional limit on number of samples
            
        Returns:
            List of standardized question-answer samples
        """
        try:
            logger.info(f"Loading HotpotQA dataset (split={split}, limit={limit})...")
            
            # Load dataset with limit
            if limit:
                dataset = hf_datasets.load_dataset(
                    self.dataset_name, 
                    self.config_name, 
                    split=f"{split}[:{limit}]",
                    trust_remote_code=True
                )
            else:
                dataset = hf_datasets.load_dataset(
                    self.dataset_name, 
                    self.config_name, 
                    split=split,
                    trust_remote_code=True
                )
            
            samples = []
            skipped = 0
            
            for item in dataset:
                # Handle both dict and string formats
                if isinstance(item, str) or not isinstance(item, dict):
                    skipped += 1
                    continue
                    
                try:
                    # Extract question and answer
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    context = item.get('context', [])
                    supporting_facts = item.get('supporting_facts', [])
                    
                    # Skip items without answer
                    if not answer or not question:
                        skipped += 1
                        continue
                    
                    # Process context
                    passages = []
                    if context:
                        # Handle dict format (context is dict with 'title' and 'sentences' keys)
                        if isinstance(context, dict):
                            titles = context.get('title', [])
                            sentences_list = context.get('sentences', [])
                            
                            # Each index represents one passage
                            for i in range(min(len(titles), len(sentences_list))):
                                title = titles[i]
                                sentences = sentences_list[i]
                                
                                # Join sentences into passage text
                                if isinstance(sentences, list):
                                    passage_text = " ".join(sentences)
                                else:
                                    passage_text = str(sentences)
                                
                                if passage_text.strip():  # Only add non-empty passages
                                    passages.append({
                                        'id': f"hotpot_passage_{len(samples)}_{i}",
                                        'title': title,
                                        'text': passage_text
                                    })
                        
                        # Handle list format (legacy structure - list of [title, sentences] pairs)
                        elif isinstance(context, list):
                            for i, context_item in enumerate(context):
                                if isinstance(context_item, (list, tuple)) and len(context_item) >= 2:
                                    title, sentences = context_item[0], context_item[1]
                                    passage_text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                                    if passage_text.strip():
                                        passages.append({
                                            'id': f"hotpot_passage_{len(samples)}_{i}",
                                            'title': title,
                                            'text': passage_text
                                        })
                    
                    # Create standardized format
                    sample = {
                        'id': f"hotpot_qa_{len(samples)}",
                        'query_id': item.get('id', f"hotpot_qa_query_{len(samples)}"),
                        'query': question,
                        'answer': answer,
                        'all_answers': [answer],  # HotpotQA has single answers
                        'passages': passages,
                        'source_type': 'hotpot_qa',
                        'metadata': {
                            'original_id': item.get('id', ''),
                            'level': item.get('level', 'unknown'),
                            'type': item.get('type', 'unknown'),
                            'supporting_facts': supporting_facts,
                            'num_passages': len(passages)
                        }
                    }
                    
                    samples.append(sample)
                    
                except Exception as e:
                    # Skip problematic items
                    logger.debug(f"Skipping malformed HotpotQA item: {e}")
                    skipped += 1
                    continue
            
            if skipped > 0:
                logger.warning(f"Skipped {skipped} malformed items")
            
            logger.info(f"✅ Loaded {len(samples)} HotpotQA samples from {split} split")
            return samples
            
        except Exception as e:
            raise DatasetError(
                f"Failed to load HotpotQA dataset: {e}",
                error_code="HOTPOT_QA_LOAD_ERROR",
                context={"split": split, "limit": limit, "skipped": skipped},
                suggestions=[
                    "Check internet connection",
                    "Verify HuggingFace datasets library is installed",
                    "Try a smaller limit value",
                    "Check dataset availability on HuggingFace"
                ]
            )


class BenchmarkDataManager:
    """Manager for loading and combining multiple benchmark datasets"""
    
    def __init__(self):
        self.loaders = {
            'ms_marco': MSMarcoLoader(),
            'hotpot_qa': HotpotQALoader()
        }
        logger.info("Benchmark data manager initialized")
    
    def load_mixed_dataset(self, datasets: List[str], samples_per_dataset: int = 50) -> List[Dict[str, Any]]:
        """
        Load a mixed dataset from multiple sources
        
        Args:
            datasets: List of dataset names to include
            samples_per_dataset: Number of samples from each dataset
            
        Returns:
            Combined and shuffled dataset
        """
        with logger.operation("load_mixed_dataset", datasets=datasets, samples=samples_per_dataset):
            all_samples = []
            
            for dataset_name in datasets:
                if dataset_name not in self.loaders:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue
                
                loader = self.loaders[dataset_name]
                
                # Load samples
                if dataset_name == 'ms_marco':
                    samples = loader.load(split="train", limit=samples_per_dataset)
                else:  # hotpot_qa
                    samples = loader.load(split="validation", limit=samples_per_dataset)
                
                all_samples.extend(samples)
            
            # Shuffle for random evaluation order
            random.shuffle(all_samples)
            
            logger.info(f"✅ Created mixed dataset with {len(all_samples)} samples")
            return all_samples
    
    def get_test_sample(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get a small test sample for validation"""
        return self.load_mixed_dataset(['ms_marco', 'hotpot_qa'], samples_per_dataset=n//2)
    
    def load_ms_marco(self, limit: int = 50, split: str = "train") -> List[Dict[str, Any]]:
        """Load MS MARCO samples directly"""
        return self.loaders['ms_marco'].load(split=split, limit=limit)
    
    def load_hotpot_qa(self, limit: int = 50, split: str = "validation") -> List[Dict[str, Any]]:
        """Load HotpotQA samples directly"""
        return self.loaders['hotpot_qa'].load(split=split, limit=limit)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing dataset loaders...")
    
    # Test MS MARCO
    try:
        ms_loader = MSMarcoLoader()
        ms_samples = ms_loader.get_sample(3)
        print(f"✅ MS MARCO loaded: {len(ms_samples)} samples")
        print(f"   Sample query: {ms_samples[0]['query'][:60]}...")
        print(f"   Passages: {ms_samples[0]['metadata']['num_passages']}")
    except Exception as e:
        print(f"❌ MS MARCO failed: {e}")
    
    # Test HotpotQA
    try:
        hq_loader = HotpotQALoader()
        hq_samples = hq_loader.get_sample(3)
        print(f"✅ HotpotQA loaded: {len(hq_samples)} samples")
        print(f"   Sample query: {hq_samples[0]['query'][:60]}...")
        print(f"   Passages: {hq_samples[0]['metadata']['num_passages']}")
    except Exception as e:
        print(f"❌ HotpotQA failed: {e}")
    
    # Test mixed dataset
    try:
        manager = BenchmarkDataManager()
        mixed = manager.get_test_sample(10)
        print(f"✅ Mixed dataset: {len(mixed)} samples")
    except Exception as e:
        print(f"❌ Mixed dataset failed: {e}")
    
    print("\n✅ Dataset loader tests complete!")
