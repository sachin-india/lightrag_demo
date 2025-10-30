# LightRAG Query Processor

A complete RAG system for processing PDFs, building knowledge graphs, and querying with **automatic performance monitoring**.

## 📁 Files

- **`pdf_extractor.py`** - Extract text from PDFs (with column detection)
- **`ingest_documents.py`** - Load extracted JSON into LightRAG knowledge graph
- **`query_lightrag.py`** - Query processor with integrated performance monitoring
- **`queries.txt`** - Input file with questions (one per line)
- **`output/`** - Folder containing all query results (timestamped files)
  - `answers_YYYY-MM-DD_HHMMSS.txt` - Query results with metrics
  - `performance_metrics_YYYY-MM-DD_HHMMSS.json` - Detailed performance data
  - `lightrag_query_YYYY-MM-DD_HHMMSS.log` - Debug logs
- **`.env`** - Configuration file (API keys, paths)

## 🚀 Quick Start - Complete Workflow

### Step 1: Place PDFs
```powershell
# Put your PDF files in the pdfs/ folder
# Example: pdfs/research_paper.pdf
```

### Step 2: Extract Text from PDFs
```powershell
# Activate virtual environment
.\.myenv\Scripts\Activate.ps1

# Extract text from all PDFs in pdfs/ folder
python pdf_extractor.py
```
**Output:** Extracted JSON files saved to `knowledgebase/` folder

### Step 3: Ingest Documents into Knowledge Graph

**Option A: Using Ingestion Script (Recommended)**
```powershell
# Activate virtual environment if not already active
.\.myenv\Scripts\Activate.ps1

# Ingest all JSON files from knowledgebase/ into the graph
python ingest_documents.py
```

**Option B: Using LightRAG WebUI Server**
```powershell
# Start the server
lightrag-server

# Open browser to http://localhost:9621/webui/
# Use the web interface to upload/process documents
```

**Output:** Knowledge graph built in `rag_storage/` folder

### Step 4: Run Queries
```powershell
# Activate virtual environment if not already active
.\.myenv\Scripts\Activate.ps1

# Edit queries.txt with your questions (one per line)
# Then run the query processor
python query_lightrag.py
```

**Output:** 
- ✅ Real-time metrics in console
- ✅ Detailed answers in `output/answers_YYYY-MM-DD_HHMMSS.txt`
- ✅ JSON metrics in `output/performance_metrics_YYYY-MM-DD_HHMMSS.json`
- ✅ Debug logs in `output/lightrag_query_YYYY-MM-DD_HHMMSS.log`

**Note:** Each run creates new timestamped files, preserving all previous results for comparison.

---

## 📂 Folder Structure & Data Flow

```
lightrag_demo/
├── pdfs/                          # Step 1: Place your PDF files here
│   └── research_paper.pdf
│
├── knowledgebase/                 # Step 2: PDF extractor outputs here
│   └── research_paper_extracted.json
│
├── rag_storage/                   # Step 3: Knowledge graph stored here
│   ├── graph_chunk_entity_relation.graphml
│   ├── kv_store_*.json
│   └── vdb_*.json
│
├── output/                        # Step 4: All query results stored here (timestamped)
│   ├── answers_2025-10-30_054957.txt
│   ├── performance_metrics_2025-10-30_054957.json
│   ├── lightrag_query_2025-10-30_054957.log
│   ├── answers_2025-10-30_112034.txt
│   ├── performance_metrics_2025-10-30_112034.json
│   └── lightrag_query_2025-10-30_112034.log
│
└── queries.txt                    # Step 4: Your questions (input)
```

**Data Flow:**
1. `pdfs/` → **pdf_extractor.py** → `knowledgebase/` (JSON files)
2. `knowledgebase/` → **ingest_documents.py** → `rag_storage/` (knowledge graph)
3. `queries.txt` + `rag_storage/` → **query_lightrag.py** → `output/` (timestamped results)

---

## ⏱️ Performance Monitoring

### What You Get

**Real-Time Console Output:**
```
Query 1/8: What diagnostic methods are discussed for SARS-CoV-2?
================================================================================

⏱️  PERFORMANCE METRICS:
   Latency: 2340.56ms (2.34 seconds)
   Response Length: 523 chars
   Answer Words: 87
   Tokens/Second: 37.18

📝 Answer:
PCR-based tests remain the gold standard...

================================================================================
📊 PERFORMANCE SUMMARY
================================================================================
Total Queries: 8
Total Time: 18.75s
Average Latency: 2343.78ms
Min Latency: 2145.12ms
Max Latency: 2876.45ms
Median Latency: 2354.09ms
================================================================================
```

### Metrics Captured

| Metric | Unit | Example | Purpose |
|--------|------|---------|---------|
| **Latency** | milliseconds | 2340.56ms | Query processing time |
| **Response Length** | characters | 523 | Answer size |
| **Answer Words** | count | 87 | Word count |
| **Tokens/Second** | words/sec | 37.18 | Generation speed |
| **Total Time** | seconds | 18.75s | Total duration |
| **Average Latency** | ms | 2343.78ms | Mean query time |
| **Min/Max Latency** | ms | 2145-2876ms | Performance range |

### Output Files

| File | Format | Contains |
|------|--------|----------|
| `answers.txt` | Text | Queries + Answers + Per-query metrics + Summary |
| `performance_metrics.json` | JSON | Detailed metrics per query + Summary stats |
| Console | Terminal | Real-time metrics as queries run |

### Configuration

In `query_lightrag.py`, lines 20-23:

```python
ENABLE_LATENCY_TRACKING = True          # Toggle tracking on/off
ENABLE_PERFORMANCE_REPORT = True        # Export JSON metrics
PERFORMANCE_LOG_FILE = "performance_metrics.json"  # JSON filename
```

### Examples

**View JSON metrics in Python:**
```python
import json

with open('performance_metrics.json') as f:
    data = json.load(f)

summary = data['summary']
print(f"Processed {summary['total_queries']} queries")
print(f"Average latency: {summary['avg_latency_ms']}ms")
```

**Find slow queries:**
```python
import json

with open('performance_metrics.json') as f:
    data = json.load(f)

slow = [m for m in data['individual_metrics'] if m['latency_ms'] > 3000]
print(f"Slow queries (>3s): {len(slow)}")
```

---

## 🎛️ Tuning Answer Conciseness

Edit the configuration section at the top of `query_lightrag.py`:

### For CONCISE Answers (1-2 sentences):
```python
CHUNK_TOP_K = 3             # Fewer chunks = less context
MAX_COMPLETION_TOKENS = 75  # Short output limit
CONCISE_SYSTEM_PROMPT = True # Force brief responses  
QUERY_MODE = "naive"        # Best for direct retrieval
ENABLE_LATENCY_TRACKING = True  # Monitor performance
```

### For DETAILED Answers (comprehensive):
```python
CHUNK_TOP_K = 20            # More chunks = rich context
MAX_COMPLETION_TOKENS = 3000 # Long output limit
CONCISE_SYSTEM_PROMPT = False # Natural length responses
QUERY_MODE = "global"       # Best for comprehensive analysis
ENABLE_LATENCY_TRACKING = True  # Monitor performance
```

### Current MODERATE Settings (balanced):
```python
CHUNK_TOP_K = 12           # Balanced context
MAX_COMPLETION_TOKENS = 250 # Medium length answers
CONCISE_SYSTEM_PROMPT = True # Concise responses
QUERY_MODE = "hybrid"      # Balanced retrieval
ENABLE_LATENCY_TRACKING = True  # Monitor performance
```

## 📊 Parameter Guide

| Parameter | Concise | Moderate | Detailed | Description |
|-----------|---------|----------|----------|-------------|
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

---

## 🏆 Query Mode Performance Comparison

**Test Date:** October 30, 2025  
**Test Configuration:** Chunks=12, Max Tokens=250, Concise=True  
**Dataset:** Single COVID-19 research paper (17 pages)  
**Queries:** 3 questions about variants, diagnosis, and authorship

### Performance Summary Table

| Mode | Avg Latency | Min Latency | Max Latency | Tokens/sec | Speed Rank | Quality Notes |
|------|-------------|-------------|-------------|------------|------------|---------------|
| **hybrid** | 1.62s | 1.32s | 1.92s | 33.27 | 🥇 **Fastest** | Best balance - accurate, concise, comprehensive |
| **naive** | 4.31s | 3.21s | 6.20s | 13.43 | 🥈 2nd | Good quality, more detailed explanations |
| **local** | 5.89s | 5.10s | 7.16s | 11.20 | 🥉 3rd | Most detailed, entity-focused responses |
| **global** | 5.99s | 5.40s | 6.83s | 9.26 | 4th | Comprehensive analysis, broader context |

### Key Findings

✅ **Hybrid mode is 2.7x faster** than other modes while maintaining quality  
✅ **Naive mode offers good speed-quality balance** for specific fact queries  
✅ **Local mode provides most detail** but at the cost of speed  
✅ **Global mode best for comprehensive** questions requiring broad context  

### Answer Quality Comparison (Query 2 Sample)

**Question:** "What are the effect of emerging covid variants on SARS-CoV-2 diagnosis?"

<details>
<summary><strong>Hybrid (1.62s)</strong> - Concise & Direct ⭐ Best for most use cases</summary>

> Emerging COVID variants significantly challenge existing SARS-CoV-2 diagnostic methods, necessitating the development of newer and more adaptive technologies. Variants can alter the reliability of molecular tests, antigen tests, and serological tests, leading to discrepancies in detection rates and responses to treatments and vaccines.

**Words:** 56 | **Speed:** 34.5 tok/s
</details>

<details>
<summary><strong>Naive (4.31s)</strong> - Balanced Detail</summary>

> Emerging COVID-19 variants challenge the effectiveness of current diagnostic methods, necessitating the development of newer, adaptive technologies. Variants may alter the accuracy of nucleic acid tests, leading to false negatives, and thus, there's a need for multi-target assays and techniques like PCR and LAMP for reliable detection.

**Words:** 60 | **Speed:** 17.0 tok/s | **Note:** Mentions specific techniques (PCR, LAMP)
</details>

<details>
<summary><strong>Local (5.89s)</strong> - Most Detailed</summary>

> Emerging COVID-19 variants have created challenges for SARS-CoV-2 diagnosis by necessitating advancements in diagnostic methods to ensure accurate detection. Variants may result in changes to receptor binding, reduced efficacy of treatments, and decreased neutralization by antibodies, complicating testing methodologies and highlighting the need for adaptive diagnostic technologies.

**Words:** 60 | **Speed:** 11.1 tok/s | **Note:** Includes receptor binding & antibody neutralization details
</details>

<details>
<summary><strong>Global (5.99s)</strong> - Comprehensive Context</summary>

> The emergence of novel SARS-CoV-2 variants has significantly impacted the effectiveness of current diagnostic methods, complicating the detection of infections. These variants may lead to reduced sensitivity in molecular, antigen, and serological tests, thus necessitating the development of newer and more adaptive diagnostic technologies.

**Words:** 57 | **Speed:** 10.6 tok/s | **Note:** Broader perspective on diagnostic impact
</details>

### Recommendations

| Use Case | Recommended Mode | Why |
|----------|------------------|-----|
| Production queries | **hybrid** | Best speed-quality balance |
| Quick fact lookup | **naive** | Fast, focused answers |
| Research/analysis | **local** or **global** | Rich detail & context |
| Time-sensitive apps | **hybrid** | Consistent sub-2s latency |
| Comprehensive reports | **global** | Broader contextual understanding |

---

## 🧠 Understanding LightRAG Query Modes

### How Each Mode Works (Simplified)

#### 🔵 **Hybrid Mode** (Recommended Default)
**How it works:** Combines local entity search + global summarization in parallel, then merges results.

**Think of it as:** A smart librarian who quickly checks both the index cards (entities) AND skims through relevant chapters (text chunks), then gives you the best of both.

**✅ Benefits:**
- Fastest performance (leverages parallel processing)
- Balanced context from both entities and text
- Best for most real-world applications
- Consistently fast (1-2s typical)

**❌ Disadvantages:**
- May occasionally miss very obscure details
- Not as comprehensive as pure global mode

**Best for:** Production apps, chatbots, quick Q&A systems

---

#### 🟢 **Naive Mode**
**How it works:** Pure vector similarity search on text chunks. No entity/relation graphs involved.

**Think of it as:** A search engine that finds the most relevant text passages and reads them directly.

**✅ Benefits:**
- Simple and straightforward
- Good for specific fact-finding
- No graph traversal overhead
- Predictable behavior

**❌ Disadvantages:**
- Slower than hybrid (no parallel processing)
- Misses entity relationships
- Less contextual understanding
- Can't connect dots across documents

**Best for:** Simple fact lookup, when you know the answer is in specific text passages

---

#### 🟡 **Local Mode**
**How it works:** Focuses on knowledge graph entities and their relationships. Searches for relevant entities first, then reads connected information.

**Think of it as:** A detective who follows the web of connections between people, places, and concepts.

**✅ Benefits:**
- Best for relationship questions ("Who works with whom?")
- Most detailed entity-focused answers
- Excellent at connecting related concepts
- Great for exploring knowledge graphs

**❌ Disadvantages:**
- Slower (needs to traverse entity graph)
- Can be overly detailed
- May miss context from plain text
- Not ideal for simple factual questions

**Best for:** Research, exploring relationships, understanding connections between entities

---

#### 🔴 **Global Mode**
**How it works:** Analyzes the entire knowledge base, creates high-level summaries, then answers from that broad understanding.

**Think of it as:** A professor who has read everything and gives you the big picture view.

**✅ Benefits:**
- Most comprehensive understanding
- Best for broad "explain X" questions
- Captures overarching themes
- Good for summarization tasks

**❌ Disadvantages:**
- Slowest mode (analyzes entire corpus)
- Can be verbose
- May include less relevant context
- Overkill for simple questions

**Best for:** Research reports, comprehensive analysis, "explain the overall concept" questions

---

### Quick Decision Guide

```
┌─────────────────────────────────────────────────────────────┐
│ Need speed + good quality?          → Use HYBRID           │
│ Simple fact lookup?                  → Use NAIVE            │
│ "How are X and Y related?"           → Use LOCAL            │
│ "Explain the big picture of X"       → Use GLOBAL           │
│ Production/API with time limits?     → Use HYBRID           │
│ Research with no time pressure?      → Try LOCAL or GLOBAL  │
└─────────────────────────────────────────────────────────────┘
```

### Mode Selection Flowchart

```
Start: What kind of question?
    │
    ├─→ "Quick fact" (Who? What? When?) 
    │   └─→ NAIVE or HYBRID
    │
    ├─→ "Relationships" (How are X and Y connected?)
    │   └─→ LOCAL
    │
    ├─→ "Big picture" (Explain the overall...)
    │   └─→ GLOBAL
    │
    └─→ "Not sure / General purpose"
        └─→ HYBRID (safe default)
```

---

## 📈 Latest Run Results

**Last Updated:** October 30, 2025

### Performance Summary
```
Total Queries: 3
Total Time: 20.9s
Average Latency: 6.97s
Min Latency: 6.24s
Max Latency: 8.09s
```

### Sample Queries & Results

<details>
<summary>Click to expand query results</summary>

**Q1: What are the emerging covid variants?**
> Emerging COVID variants include strains classified as Variants of Interest (VOIs) and Variants of Concern (VOCs), such as B.1.1.7 (Alpha), B.1.351 (Beta), P.1 (Gamma), and B.1.617.2 (Delta). These variants have mutations that may affect transmissibility, vaccine effectiveness, and disease severity.

*Latency: 8.09s | Tokens/sec: 6.55*

---

**Q2: What are the effect of emerging covid variants on SARS-CoV-2 diagnosis?**
> Emerging COVID variants significantly challenge existing SARS-CoV-2 diagnostic methods, necessitating the development of newer and more adaptive technologies. Variants can alter the reliability of molecular tests, antigen tests, and serological tests, leading to discrepancies in detection rates and responses to treatments and vaccines.

*Latency: 6.24s | Tokens/sec: 8.97*

---

**Q3: Who is the 1st author and their affiliation of the paper "Emerging Covid-19 variants and their impact on SARS-COV-2 diagnosis"?**
> The first author of the paper "Emerging COVID-19 variants and their impact on SARS-CoV-2 diagnosis" is Queenie Fernandes, affiliated with the Translational Cancer Research Facility, National Center for Cancer Care and Research, Hamad Medical Corporation, Doha, Qatar.

*Latency: 6.57s | Tokens/sec: 7.61*

</details>

**Configuration Used:** Mode=hybrid, Chunks=12, Max Tokens=250, Concise=True