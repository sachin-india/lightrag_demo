"""
Document Ingestion Script for LightRAG
Loads extracted JSON documents from knowledgebase/ into the LightRAG knowledge graph
"""

import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete

# Load environment variables
load_dotenv()

# Configuration
WORKING_DIR = "./rag_storage"
KNOWLEDGEBASE_DIR = "./knowledgebase"

async def create_custom_llm_func():
    """Create custom LLM function for document processing"""
    async def custom_llm_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs
    ) -> str:
        return await gpt_4o_mini_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    return custom_llm_func

async def ingest_documents():
    """Ingest all JSON documents from knowledgebase into LightRAG"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set in .env file")
        return
    
    print("=" * 80)
    print("LIGHTRAG DOCUMENT INGESTION")
    print("=" * 80)
    print(f"Working Directory: {WORKING_DIR}")
    print(f"Knowledgebase Directory: {KNOWLEDGEBASE_DIR}")
    print()
    
    # Initialize LightRAG
    print("🔧 Initializing LightRAG...")
    custom_llm = await create_custom_llm_func()
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=custom_llm,
    )
    
    await rag.initialize_storages()
    print("✅ LightRAG initialized\n")
    
    # Find all JSON files in knowledgebase
    kb_path = Path(KNOWLEDGEBASE_DIR)
    json_files = list(kb_path.glob("*_extracted.json"))
    
    if not json_files:
        print(f"⚠️ No *_extracted.json files found in {KNOWLEDGEBASE_DIR}")
        return
    
    print(f"📁 Found {len(json_files)} document(s) to ingest:\n")
    
    total_pages = 0
    total_words = 0
    
    # Process each JSON file
    for json_file in json_files:
        print(f"📄 Processing: {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Extract metadata
        filename = doc_data.get('filename', 'unknown')
        pages_count = doc_data.get('pages_count', 0)
        word_count = doc_data.get('word_count', 0)
        extraction_date = doc_data.get('extraction_date', 'unknown')
        
        print(f"   Source PDF: {filename}")
        print(f"   Pages: {pages_count}")
        print(f"   Words: {word_count:,}")
        print(f"   Extracted: {extraction_date}")
        
        total_pages += pages_count
        total_words += word_count
        
        # Combine all page texts into a single document
        pages = doc_data.get('pages', [])
        full_text_parts = []
        
        for page in pages:
            page_num = page.get('page_number', 0)
            text = page.get('text', '').strip()
            
            if text:
                # Add page marker for context
                full_text_parts.append(f"\n--- Page {page_num} ---\n{text}")
        
        full_text = "\n".join(full_text_parts)
        
        if not full_text.strip():
            print("   ⚠️ No text content found, skipping...")
            continue
        
        print(f"   📝 Ingesting {len(full_text)} characters...")
        
        # Insert document into LightRAG
        try:
            await rag.ainsert(full_text)
            print("   ✅ Successfully ingested into knowledge graph\n")
        except Exception as e:
            print(f"   ❌ Error during ingestion: {e}\n")
            continue
    
    print("=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)
    print(f"Documents Processed: {len(json_files)}")
    print(f"Total Pages: {total_pages}")
    print(f"Total Words: {total_words:,}")
    print()
    print("✅ Knowledge graph is now ready for queries!")
    print("   Run: python query_lightrag.py")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(ingest_documents())
