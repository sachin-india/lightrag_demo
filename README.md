# LightRAG Query Processor

A single, configurable script for querying your LightRAG knowledge graph with adjustable answer conciseness.

## 📁 Files

- **`query_lightrag.py`** - Main query processor script
- **`queries.txt`** - Input file with questions (one per line) 
- **`answers.txt`** - Output file with results (generated when script runs)
- **`README.md`** - This documentation

## 🚀 Quick Start

```bash
python query_lightrag.py
```

## 🎛️ Tuning Answer Conciseness

Edit the configuration section at the top of `query_lightrag.py`:

### For CONCISE Answers (1-2 sentences):
```python
CHUNK_TOP_K = 3             # Fewer chunks = less context
MAX_COMPLETION_TOKENS = 75  # Short output limit
CONCISE_SYSTEM_PROMPT = True # Force brief responses  
QUERY_MODE = "naive"        # Best for direct retrieval
```

### For DETAILED Answers (comprehensive):
```python
CHUNK_TOP_K = 20            # More chunks = rich context
MAX_COMPLETION_TOKENS = 3000 # Long output limit
CONCISE_SYSTEM_PROMPT = False # Natural length responses
QUERY_MODE = "global"       # Best for comprehensive analysis
```

### Current MODERATE Settings (balanced):
```python
CHUNK_TOP_K = 10           # Balanced context
MAX_COMPLETION_TOKENS = 500 # Medium length answers
CONCISE_SYSTEM_PROMPT = False # Natural responses
QUERY_MODE = "naive"       # Good chunk retrieval
```

## 📊 Parameter Guide

| Parameter | Concise | Moderate | Detailed | Description |
|-----------|---------|----------|----------|-------------|
| `CHUNK_TOP_K` | 3-5 | 8-12 | 15-20 | Text chunks retrieved |
| `MAX_COMPLETION_TOKENS` | 50-100 | 300-800 | 2000-4000 | Output length limit |
| `CONCISE_SYSTEM_PROMPT` | True | False | False | Force brief answers |
| `QUERY_MODE` | "naive" | "naive" | "global" | Search strategy |
| `TOP_K` | 10 | 15 | 40 | Entities/relations |

## 🔍 Query Modes

- **`naive`** - Pure vector search on text chunks (best for specific facts)
- **`local`** - Entity-focused search (good for relationships) 
- **`global`** - Comprehensive analysis (best for broad topics)
- **`hybrid`** - Combines local + global (balanced approach)

## 💡 Tips

1. **Start with current settings** - they work well for most cases
2. **For very short answers** - Set `CONCISE_SYSTEM_PROMPT = True`
3. **For rich detail** - Increase `CHUNK_TOP_K` and `MAX_COMPLETION_TOKENS`
4. **If answers seem incomplete** - Try "global" mode
5. **If answers are too verbose** - Reduce `CHUNK_TOP_K` and token limits

## 🔧 Advanced Settings

```python
MAX_ENTITY_TOKENS = 2000   # Context from entities
MAX_RELATION_TOKENS = 2000 # Context from relations  
MAX_TOTAL_TOKENS = 6000    # Total context limit
```

Lower these values for faster, more focused responses.

## 📝 Example Usage

1. Edit your questions in `queries.txt`:
```
What are the main COVID-19 symptoms?
How effective are mRNA vaccines?
Who wrote the paper on emerging variants?
```

2. Configure conciseness in `query_lightrag.py`

3. Run: `python query_lightrag.py`

4. Check results in `answers.txt`

## 🛠️ Requirements

- Python 3.7+
- lightrag-hku package
- openai package  
- Existing LightRAG storage in `rag_storage/`

## 📞 Configuration Examples

**Ultra Concise (Twitter-style)**:
```python
CHUNK_TOP_K = 3
MAX_COMPLETION_TOKENS = 50
CONCISE_SYSTEM_PROMPT = True
```

**Academic Detail**:
```python  
CHUNK_TOP_K = 20
MAX_COMPLETION_TOKENS = 4000
CONCISE_SYSTEM_PROMPT = False
QUERY_MODE = "global"
```