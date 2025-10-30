"""
Document Adapter for LightRAG Benchmarking

Converts benchmark passages into LightRAG-compatible documents while
preserving metadata for evaluation tracking.

Key Features:
- Preserves passage IDs for answer evaluation
- Converts passages to documents for graph building
- Maintains query-passage relationships
- Tracks source information for debugging
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..utils.logging import get_logger
from ..utils.errors import DatasetError

logger = get_logger("dataset_adapter")


class LightRAGDocumentAdapter:
    """Adapter to convert benchmark passages into LightRAG documents"""
    
    def __init__(self, working_dir: Optional[Path] = None):
        """
        Initialize document adapter
        
        Args:
            working_dir: Optional working directory for document storage
        """
        self.working_dir = working_dir or Path("benchmarks/benchmark_storage")
        self.docs_dir = self.working_dir / "documents"
        self.mapping_file = self.working_dir / "passage_document_mapping.json"
        
        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping from passage IDs to document IDs
        self.passage_to_doc_mapping = {}
        self.doc_counter = 0
        
        logger.info(f"Document adapter initialized (working_dir={self.working_dir})")
    
    def convert_sample_to_documents(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a single benchmark sample's passages into documents
        
        Args:
            sample: Benchmark sample with passages
            
        Returns:
            List of document dictionaries compatible with LightRAG
        """
        documents = []
        passages = sample.get('passages', [])
        query = sample.get('query', '')
        query_id = sample.get('query_id', sample.get('id', ''))
        source_type = sample.get('source_type', 'unknown')
        
        for passage in passages:
            passage_id = passage.get('id', f"passage_{self.doc_counter}")
            passage_text = passage.get('text', '')
            
            if not passage_text.strip():
                logger.debug(f"Skipping empty passage: {passage_id}")
                continue
            
            # Create document ID
            doc_id = f"doc_{self.doc_counter}"
            self.doc_counter += 1
            
            # Create document with metadata
            document = {
                'id': doc_id,
                'text': passage_text,
                'metadata': {
                    'passage_id': passage_id,
                    'query_id': query_id,
                    'query': query,
                    'source_type': source_type,
                    'title': passage.get('title', ''),
                    'url': passage.get('url', ''),
                    'is_selected': passage.get('is_selected', 0),
                    'passage_index': passage.get('index', 0)
                }
            }
            
            # Store mapping
            self.passage_to_doc_mapping[passage_id] = {
                'doc_id': doc_id,
                'query_id': query_id,
                'source_type': source_type
            }
            
            documents.append(document)
        
        logger.debug(f"Converted {len(documents)} passages to documents for query {query_id}")
        return documents
    
    def convert_dataset_to_documents(self, dataset: List[Dict[str, Any]], 
                                    save_docs: bool = False) -> Dict[str, Any]:
        """
        Convert entire dataset to LightRAG documents
        
        Args:
            dataset: List of benchmark samples
            save_docs: Whether to save documents to disk
            
        Returns:
            Dictionary with documents and metadata
        """
        with logger.operation("convert_dataset_to_documents", samples=len(dataset)):
            all_documents = []
            query_doc_mapping = {}  # Map query_id to doc_ids
            
            for sample in dataset:
                query_id = sample.get('query_id', sample.get('id', ''))
                
                # Convert passages to documents
                documents = self.convert_sample_to_documents(sample)
                all_documents.extend(documents)
                
                # Track query-document relationship
                doc_ids = [doc['id'] for doc in documents]
                query_doc_mapping[query_id] = {
                    'doc_ids': doc_ids,
                    'query': sample.get('query', ''),
                    'answer': sample.get('answer', ''),
                    'all_answers': sample.get('all_answers', []),
                    'source_type': sample.get('source_type', 'unknown')
                }
            
            # Save documents if requested
            if save_docs:
                self.save_documents(all_documents)
                self.save_mappings(query_doc_mapping)
            
            logger.info(f"✅ Converted {len(dataset)} samples to {len(all_documents)} documents")
            
            return {
                'documents': all_documents,
                'query_doc_mapping': query_doc_mapping,
                'passage_doc_mapping': self.passage_to_doc_mapping,
                'total_documents': len(all_documents),
                'total_queries': len(dataset)
            }
    
    def save_documents(self, documents: List[Dict[str, Any]]):
        """Save documents to disk"""
        try:
            docs_file = self.docs_dir / "benchmark_documents.json"
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {len(documents)} documents to {docs_file}")
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
    
    def save_mappings(self, query_doc_mapping: Dict[str, Any]):
        """Save query-document mappings to disk"""
        try:
            # Save query mapping
            query_mapping_file = self.working_dir / "query_document_mapping.json"
            with open(query_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(query_doc_mapping, f, indent=2, ensure_ascii=False)
            
            # Save passage mapping
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.passage_to_doc_mapping, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved mappings to {self.working_dir}")
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")
    
    def load_mappings(self) -> Dict[str, Any]:
        """Load saved mappings from disk"""
        try:
            query_mapping_file = self.working_dir / "query_document_mapping.json"
            
            if query_mapping_file.exists():
                with open(query_mapping_file, 'r', encoding='utf-8') as f:
                    query_mapping = json.load(f)
                logger.info(f"Loaded query-document mappings for {len(query_mapping)} queries")
                return query_mapping
            else:
                logger.warning("No saved mappings found")
                return {}
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            return {}
    
    def get_documents_for_query(self, query_id: str, 
                               query_doc_mapping: Dict[str, Any]) -> List[str]:
        """Get document IDs associated with a query"""
        if query_id in query_doc_mapping:
            return query_doc_mapping[query_id].get('doc_ids', [])
        else:
            logger.warning(f"No documents found for query: {query_id}")
            return []
    
    def create_document_files(self, documents: List[Dict[str, Any]], 
                            format: str = "txt") -> List[Path]:
        """
        Create individual document files for LightRAG ingestion
        
        Args:
            documents: List of document dictionaries
            format: File format (txt or json)
            
        Returns:
            List of created file paths
        """
        with logger.operation("create_document_files", count=len(documents), format=format):
            file_paths = []
            
            for doc in documents:
                doc_id = doc['id']
                text = doc['text']
                
                if format == "txt":
                    # Create text file with metadata in header
                    file_path = self.docs_dir / f"{doc_id}.txt"
                    metadata = doc.get('metadata', {})
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        # Write metadata as comments
                        f.write(f"# Document ID: {doc_id}\n")
                        f.write(f"# Passage ID: {metadata.get('passage_id', 'N/A')}\n")
                        f.write(f"# Query ID: {metadata.get('query_id', 'N/A')}\n")
                        f.write(f"# Source: {metadata.get('source_type', 'N/A')}\n")
                        if metadata.get('title'):
                            f.write(f"# Title: {metadata['title']}\n")
                        f.write("\n")
                        f.write(text)
                    
                elif format == "json":
                    # Create JSON file with full document
                    file_path = self.docs_dir / f"{doc_id}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(doc, f, indent=2, ensure_ascii=False)
                
                file_paths.append(file_path)
            
            logger.info(f"✅ Created {len(file_paths)} document files")
            return file_paths
    
    def get_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about converted documents"""
        total_docs = len(documents)
        total_chars = sum(len(doc['text']) for doc in documents)
        avg_doc_length = total_chars / total_docs if total_docs > 0 else 0
        
        # Count by source type
        source_counts = {}
        for doc in documents:
            source = doc.get('metadata', {}).get('source_type', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'avg_document_length': avg_doc_length,
            'source_distribution': source_counts,
            'unique_queries': len(set(doc['metadata']['query_id'] for doc in documents if 'metadata' in doc))
        }


# Testing and example usage
if __name__ == "__main__":
    print("🧪 Testing LightRAG Document Adapter")
    
    # Create sample data
    sample_data = [
        {
            'id': 'test_query_1',
            'query_id': 'q1',
            'query': 'What is machine learning?',
            'answer': 'Machine learning is a subset of AI...',
            'passages': [
                {
                    'id': 'passage_1',
                    'text': 'Machine learning is a method of data analysis that automates analytical model building.',
                    'title': 'ML Introduction'
                },
                {
                    'id': 'passage_2',
                    'text': 'It is a branch of artificial intelligence based on the idea that systems can learn from data.',
                    'title': 'AI and ML'
                }
            ],
            'source_type': 'test'
        }
    ]
    
    # Test adapter
    adapter = LightRAGDocumentAdapter(working_dir=Path("test_adapter_output"))
    
    # Convert dataset
    result = adapter.convert_dataset_to_documents(sample_data, save_docs=True)
    
    print(f"✅ Converted {result['total_queries']} queries to {result['total_documents']} documents")
    print(f"   Documents: {[doc['id'] for doc in result['documents']]}")
    
    # Get statistics
    stats = adapter.get_statistics(result['documents'])
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Create document files
    file_paths = adapter.create_document_files(result['documents'][:1], format="txt")
    print(f"\n✅ Created {len(file_paths)} document files")
    print(f"   First file: {file_paths[0]}")
    
    print("\n✅ Document adapter tests complete!")
