import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

WORKING_DIR = "/home/sachi/AgenticAI/projects/lightrag/covid/rag_storage"

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
QUERY_MODE = "hybrid"       # "naive"=best for chunks, "global"=best for comprehensive

# Advanced settings (usually don't need to change):
MAX_ENTITY_TOKENS = 3000   # Context from entities (500=concise, 4000=detailed)
MAX_RELATION_TOKENS = 3000 # Context from relations (500=concise, 4000=detailed) 
MAX_TOTAL_TOKENS = 10000    # Total context limit (2000=concise, 12000=detailed)

def configure_logging():
    """Configure logging for the application"""
    
    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicron.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_query.log"))

    print(f"\nLightRAG query log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

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
                    "filename": log_file_path,
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
    """Process queries from a text file with configurable conciseness"""
    
    # Check if OpenAI API key is configured (LightRAG uses LLM_BINDING_API_KEY)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        print("❌ Error: OpenAI API key is not set.")
        print("Please set it in your .env file:")
        print("  LLM_BINDING_API_KEY=your-api-key-here")
        print("  EMBEDDING_BINDING_API_KEY=your-api-key-here")
        return []

    rag = None
    try:
        # Print current configuration
        print("=" * 60)
        print("CURRENT CONFIGURATION:")
        print("=" * 60)
        print(f"Query Mode: {QUERY_MODE}")
        print(f"Chunk Retrieval: {CHUNK_TOP_K} chunks")
        print(f"Entity/Relation Retrieval: {TOP_K}")
        print(f"Max Output Tokens: {MAX_COMPLETION_TOKENS}")
        print(f"Concise System Prompt: {CONCISE_SYSTEM_PROMPT}")
        print(f"Answer Style: {'CONCISE (1-2 sentences)' if CONCISE_SYSTEM_PROMPT else 'NATURAL LENGTH'}")
        print("=" * 60)
        
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
            print(f"Query {i}: {query}")
            
            try:
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
                
                print(f"Answer: {response}")
                print("-" * 60)
                
                results.append({
                    'query': query,
                    'answer': response,
                    'config': {
                        'mode': QUERY_MODE,
                        'chunk_top_k': CHUNK_TOP_K,
                        'max_tokens': MAX_COMPLETION_TOKENS,
                        'concise': CONCISE_SYSTEM_PROMPT
                    }
                })
                
            except Exception as e:
                print(f"Error processing query {i}: {str(e)}")
                results.append({
                    'query': query,
                    'answer': f"Error: {str(e)}",
                    'config': {}
                })
        
        # Save results to file if specified
        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("LIGHTRAG QUERY RESULTS\n") 
                f.write("=" * 80 + "\n")
                f.write(f"Configuration: Mode={QUERY_MODE}, Chunks={CHUNK_TOP_K}, ")
                f.write(f"Tokens={MAX_COMPLETION_TOKENS}, Concise={CONCISE_SYSTEM_PROMPT}\n")
                f.write("=" * 80 + "\n\n")
                
                for result in results:
                    f.write(f"Query: {result['query']}\n")
                    f.write(f"Answer: {result['answer']}\n")
                    f.write("-" * 80 + "\n")
            print(f"\nResults saved to: {output_file_path}")
        
        return results
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    finally:
        if rag:
            await rag.finalize_storages()

async def main():
    """Main function to process queries"""
    queries_file = "queries.txt"
    output_file = "answers.txt"
    
    await process_queries_from_file(queries_file, output_file)

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
    📄 Output: answers.txt (formatted results)
    
    """)
    
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\n✅ Query processing completed!")