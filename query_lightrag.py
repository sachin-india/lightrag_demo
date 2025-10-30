import os
import asyncio
import logging
import logging.config
import time
import json
from datetime import datetime
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

WORKING_DIR = "./rag_storage"
OUTPUT_DIR = "./output"  # Directory for all output files

# =============================================================================
# PERFORMANCE MONITORING CONFIGURATION
# =============================================================================
ENABLE_LATENCY_TRACKING = True  # Enable/disable latency monitoring
ENABLE_PERFORMANCE_REPORT = True  # Generate performance report file

# Output files will be automatically timestamped
# Example: answers_2025-10-30_054957.txt, performance_metrics_2025-10-30_054957.json

# =============================================================================
# ANSWER CONCISENESS CONFIGURATION
# =============================================================================
# Adjust these parameters to control answer length and detail:

# CONCISE ANSWERS (1-2 sentences):
# - Set CHUNK_TOP_K = 3-5
# - Set MAX_COMPLETION_TOKENS = 50-100
# - Set CONCISE_SYSTEM_PROMPT = True
# - Set QUERY_MODE = "naive"

# DETAILED ANSWERS (comprehensive responses):
# - Set CHUNK_TOP_K = 15-20  
# - Set MAX_COMPLETION_TOKENS = 2000-4000
# - Set CONCISE_SYSTEM_PROMPT = False
# - Set QUERY_MODE = "naive" or "global"

# Current settings (MODERATE - balanced approach):
CHUNK_TOP_K = 12           # Number of text chunks to retrieve (3-5=concise, 15-20=detailed)
TOP_K = 25                 # Number of entities/relations to retrieve (10=concise, 40=detailed)
MAX_COMPLETION_TOKENS = 250 # LLM output limit (50-100=concise, 2000-4000=detailed)
CONCISE_SYSTEM_PROMPT = True # True=force 1-2 sentences, False=natural length
QUERY_MODE = "global"       # "naive"=best for chunks, "global"=best for comprehensive

# Advanced settings (usually don't need to change):
MAX_ENTITY_TOKENS = 3000   # Context from entities (500=concise, 4000=detailed)
MAX_RELATION_TOKENS = 3000 # Context from relations (500=concise, 4000=detailed) 
MAX_TOTAL_TOKENS = 10000    # Total context limit (2000=concise, 12000=detailed)

# =============================================================================
# PERFORMANCE MONITORING CLASS
# =============================================================================
class PerformanceTracker:
    """Track and analyze latency metrics for LightRAG queries"""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
        self.end_time = None
    
    def start_query(self, query_id, query_text):
        """Start tracking a query"""
        self.start_time = time.time()
        return {
            'query_id': query_id,
            'query_text': query_text,
            'start_time': datetime.now().isoformat(),
            'start_timestamp': self.start_time
        }
    
    def end_query(self, query_id, response, answer_length=0):
        """End tracking a query and calculate metrics"""
        self.end_time = time.time()
        latency_ms = (self.end_time - self.start_time) * 1000
        
        metric = {
            'query_id': query_id,
            'end_time': datetime.now().isoformat(),
            'latency_ms': round(latency_ms, 2),
            'latency_seconds': round(latency_ms / 1000, 2),
            'response_length': len(str(response)),
            'answer_length': answer_length,
            'tokens_per_second': round(answer_length / (latency_ms / 1000), 2) if latency_ms > 0 else 0
        }
        
        self.metrics.append(metric)
        return metric
    
    def get_summary(self):
        """Calculate summary statistics"""
        if not self.metrics:
            return {}
        
        latencies = [m['latency_ms'] for m in self.metrics]
        
        return {
            'total_queries': len(self.metrics),
            'total_time_ms': sum(latencies),
            'total_time_seconds': round(sum(latencies) / 1000, 2),
            'avg_latency_ms': round(sum(latencies) / len(latencies), 2),
            'min_latency_ms': round(min(latencies), 2),
            'max_latency_ms': round(max(latencies), 2),
            'median_latency_ms': round(sorted(latencies)[len(latencies) // 2], 2),
        }
    
    def export_metrics(self, filename=None):
        """Export metrics to JSON file"""
        if not filename:
            filename = PERFORMANCE_LOG_FILE
        
        data = {
            'export_time': datetime.now().isoformat(),
            'individual_metrics': self.metrics,
            'summary': self.get_summary()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename

def configure_logging(timestamp):
    """Configure logging for the application with timestamped log file"""
    
    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicron.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Use output directory for log files
    log_dir = Path(OUTPUT_DIR)
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    log_file_path = log_dir / f"lightrag_query_{timestamp}.log"

    print(f"\nLightRAG query log file: {log_file_path}\n")

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_file_path),
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

async def create_custom_llm_func():
    """Create LLM function with configurable conciseness"""
    
    async def custom_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        # Apply concise system prompt if enabled
        if CONCISE_SYSTEM_PROMPT:
            concise_instruction = "IMPORTANT: Provide extremely concise answers in exactly 1-2 sentences. Be direct and specific. No references or additional explanations."
            if system_prompt:
                system_prompt = f"{concise_instruction}\n\n{system_prompt}"
            else:
                system_prompt = concise_instruction
        
        # Set token limit based on configuration
        kwargs['max_completion_tokens'] = MAX_COMPLETION_TOKENS
        
        return await gpt_4o_mini_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    return custom_llm_func

async def initialize_rag():
    """Initialize RAG instance with proper setup"""
    custom_llm = await create_custom_llm_func()
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=custom_llm,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def process_queries_from_file(queries_file_path, output_file_path=None):
    """Process queries from a text file with configurable conciseness and performance tracking"""
    
    # Check if OpenAI API key is configured (LightRAG uses LLM_BINDING_API_KEY)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        print("❌ Error: OpenAI API key is not set.")
        print("Please set it in your .env file:")
        print("  LLM_BINDING_API_KEY=your-api-key-here")
        print("  EMBEDDING_BINDING_API_KEY=your-api-key-here")
        return []

    # Initialize performance tracker
    perf_tracker = PerformanceTracker()
    
    rag = None
    try:
        # Print current configuration
        print("=" * 80)
        print("CURRENT CONFIGURATION:")
        print("=" * 80)
        print(f"Query Mode: {QUERY_MODE}")
        print(f"Chunk Retrieval: {CHUNK_TOP_K} chunks")
        print(f"Entity/Relation Retrieval: {TOP_K}")
        print(f"Max Output Tokens: {MAX_COMPLETION_TOKENS}")
        print(f"Concise System Prompt: {CONCISE_SYSTEM_PROMPT}")
        print(f"Answer Style: {'CONCISE (1-2 sentences)' if CONCISE_SYSTEM_PROMPT else 'NATURAL LENGTH'}")
        print(f"Performance Tracking: {'ENABLED ✓' if ENABLE_LATENCY_TRACKING else 'DISABLED'}")
        print("=" * 80)
        
        # Initialize RAG instance
        rag = await initialize_rag()
        
        # Test embedding function first
        test_text = ["Test embedding function"]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
        
        # Read queries from file
        with open(queries_file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = []
        
        print(f"\nProcessing {len(queries)} queries...\n")
        
        # Process all queries with configured parameters
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"Query {i}/{len(queries)}: {query}")
            print(f"{'='*80}")
            
            try:
                # Start performance tracking
                if ENABLE_LATENCY_TRACKING:
                    perf_tracker.start_query(i, query)
                
                # Create query parameters based on configuration
                query_params = QueryParam(
                    mode=QUERY_MODE,
                    chunk_top_k=CHUNK_TOP_K,
                    top_k=TOP_K,
                    max_entity_tokens=MAX_ENTITY_TOKENS,
                    max_relation_tokens=MAX_RELATION_TOKENS,
                    max_total_tokens=MAX_TOTAL_TOKENS,
                    include_references=not CONCISE_SYSTEM_PROMPT,  # Skip references if concise
                )
                
                response = await rag.aquery(query, param=query_params)
                
                # End performance tracking and get metrics
                perf_metrics = None
                if ENABLE_LATENCY_TRACKING:
                    answer_length = len(str(response).split())
                    perf_metrics = perf_tracker.end_query(i, response, answer_length)
                    
                    # Display latency info
                    print(f"\n⏱️  PERFORMANCE METRICS:")
                    print(f"   Latency: {perf_metrics['latency_ms']}ms ({perf_metrics['latency_seconds']}s)")
                    print(f"   Response Length: {perf_metrics['response_length']} chars")
                    print(f"   Answer Words: {perf_metrics['answer_length']}")
                    print(f"   Tokens/Second: {perf_metrics['tokens_per_second']}")
                
                print(f"\n📝 Answer:\n{response}")
                
                results.append({
                    'query_id': i,
                    'query': query,
                    'answer': response,
                    'performance': perf_metrics,
                    'config': {
                        'mode': QUERY_MODE,
                        'chunk_top_k': CHUNK_TOP_K,
                        'max_tokens': MAX_COMPLETION_TOKENS,
                        'concise': CONCISE_SYSTEM_PROMPT
                    }
                })
                
            except Exception as e:
                print(f"❌ Error processing query {i}: {str(e)}")
                results.append({
                    'query_id': i,
                    'query': query,
                    'answer': f"Error: {str(e)}",
                    'performance': None,
                    'config': {}
                })
        
        # Print performance summary
        if ENABLE_LATENCY_TRACKING:
            summary = perf_tracker.get_summary()
            print(f"\n{'='*80}")
            print("📊 PERFORMANCE SUMMARY")
            print(f"{'='*80}")
            print(f"Total Queries: {summary['total_queries']}")
            print(f"Total Time: {summary['total_time_seconds']}s ({summary['total_time_ms']}ms)")
            print(f"Average Latency: {summary['avg_latency_ms']}ms")
            print(f"Min Latency: {summary['min_latency_ms']}ms")
            print(f"Max Latency: {summary['max_latency_ms']}ms")
            print(f"Median Latency: {summary['median_latency_ms']}ms")
            print(f"{'='*80}")
        
        # Save results to file if specified
        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("LIGHTRAG QUERY RESULTS WITH PERFORMANCE METRICS\n") 
                f.write("=" * 80 + "\n")
                f.write(f"Configuration: Mode={QUERY_MODE}, Chunks={CHUNK_TOP_K}, ")
                f.write(f"Tokens={MAX_COMPLETION_TOKENS}, Concise={CONCISE_SYSTEM_PROMPT}\n")
                f.write(f"Performance Tracking: {ENABLE_LATENCY_TRACKING}\n")
                f.write("=" * 80 + "\n\n")
                
                for result in results:
                    f.write(f"Query {result['query_id']}: {result['query']}\n")
                    f.write(f"\nAnswer: {result['answer']}\n")
                    
                    if result['performance']:
                        f.write(f"\n⏱️  Performance Metrics:\n")
                        f.write(f"   Latency: {result['performance']['latency_ms']}ms ({result['performance']['latency_seconds']}s)\n")
                        f.write(f"   Response Length: {result['performance']['response_length']} chars\n")
                        f.write(f"   Answer Words: {result['performance']['answer_length']}\n")
                        f.write(f"   Tokens/Second: {result['performance']['tokens_per_second']}\n")
                    
                    f.write("-" * 80 + "\n\n")
                
                # Add summary section
                if ENABLE_LATENCY_TRACKING:
                    summary = perf_tracker.get_summary()
                    f.write("=" * 80 + "\n")
                    f.write("📊 PERFORMANCE SUMMARY\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Total Queries: {summary['total_queries']}\n")
                    f.write(f"Total Time: {summary['total_time_seconds']}s\n")
                    f.write(f"Average Latency: {summary['avg_latency_ms']}ms\n")
                    f.write(f"Min Latency: {summary['min_latency_ms']}ms\n")
                    f.write(f"Max Latency: {summary['max_latency_ms']}ms\n")
                    f.write(f"Median Latency: {summary['median_latency_ms']}ms\n")
                    f.write("=" * 80 + "\n")
            
            print(f"\n✅ Results saved to: {output_file_path}")
            
            # Export detailed metrics to JSON with timestamp
            if ENABLE_PERFORMANCE_REPORT:
                # Extract timestamp from output_file_path
                output_path = Path(output_file_path)
                timestamp = output_path.stem.replace("answers_", "")
                metrics_file = output_path.parent / f"performance_metrics_{timestamp}.json"
                perf_tracker.export_metrics(str(metrics_file))
                print(f"✅ Detailed metrics exported to: {metrics_file}")
        
        return results
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    finally:
        if rag:
            await rag.finalize_storages()

async def main(timestamp):
    """Main function to process queries"""
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Input and output files (using provided timestamp)
    queries_file = "queries.txt"
    output_file = output_dir / f"answers_{timestamp}.txt"
    
    await process_queries_from_file(queries_file, str(output_file))

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                        LIGHTRAG QUERY PROCESSOR                             ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    📝 CONFIGURATION GUIDE:
    
    🔹 FOR CONCISE ANSWERS (1-2 sentences):
       - Set CHUNK_TOP_K = 3-5
       - Set MAX_COMPLETION_TOKENS = 50-100  
       - Set CONCISE_SYSTEM_PROMPT = True
       - Set QUERY_MODE = "naive"
    
    🔹 FOR DETAILED ANSWERS (comprehensive):
       - Set CHUNK_TOP_K = 15-20
       - Set MAX_COMPLETION_TOKENS = 2000-4000
       - Set CONCISE_SYSTEM_PROMPT = False  
       - Set QUERY_MODE = "naive" or "global"
    
    📄 Input:  queries.txt (one query per line)
    📂 Output: output/answers_YYYY-MM-DD_HHMMSS.txt (formatted results)
    📂 Output: output/performance_metrics_YYYY-MM-DD_HHMMSS.json (detailed metrics)
    📂 Output: output/lightrag_query_YYYY-MM-DD_HHMMSS.log (debug logs)
    
    """)
    
    # Generate timestamp for this run (used by all output files)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Configure logging with timestamp before running the main function
    configure_logging(timestamp)
    asyncio.run(main(timestamp))
    print("\n✅ Query processing completed!")
    print(f"📁 All outputs saved to: {OUTPUT_DIR}/")
