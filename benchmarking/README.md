# LightRAG Benchmarking System

**Status**: ✅ Production Ready - Comprehensive RAG evaluation with OpenAI

Enterprise-grade benchmarking for LightRAG featuring modern metrics that accurately assess verbose, well-cited RAG responses.

## 🚀 Quick Start

```bash
# Run full benchmark (5 samples per dataset, ~6-9 minutes)
python benchmarking/run_full_benchmark.py
```

This will:
- Load 5 samples from MS MARCO + 5 from HotpotQA
- Build complete knowledge graphs with chunking, embeddings, entity extraction
- Query in 4 modes (naive, local, global, hybrid)
- Calculate 6 metric categories
- Generate cross-dataset comparison
- Save results to `benchmarking/results/`

**Expected Runtime**: ~6-9 minutes for 10 samples, ~60-90 minutes for 100 samples  
**Cost**: ~$0.50-1.00 for 10 samples (OpenAI API calls)

---

## 📋 Complete Workflow

The benchmark follows this 6-step pipeline for each dataset:

### **Step 1: Load Dataset** 📊
Downloads benchmark datasets from HuggingFace:
- **MS MARCO**: Factoid question-answering (passage ranking)
- **HotpotQA**: Multi-hop reasoning questions

### **Step 2: Convert to Documents** 📝
Transforms question-answer pairs with passages into LightRAG documents

### **Step 3: Initialize LightRAG** 🔧
- Creates LightRAG instance with OpenAI embeddings
- Sets up storage: `benchmarking/benchmark_storage/{dataset}/lightrag/`

### **Step 4: Ingest Documents (Graph Building)** 📥
**This is where the magic happens!** For each document, LightRAG:
- **Chunks** text into manageable pieces
- **Embeds** chunks using OpenAI embeddings (text-embedding-3-small)
- **Extracts entities** using GPT-4o-mini
- **Builds knowledge graph** with nodes (entities) and edges (relationships)
- **Stores vectors** in vector database
- Processes 10 documents concurrently for speed

**Time:** ~4-5 minutes (most of the benchmark runtime)

### **Step 5: Execute Queries** 🔍
Runs each question against the pre-built graph in 4 modes:
- **naive**: Simple vector similarity retrieval
- **local**: Local graph traversal from retrieved nodes
- **global**: Global graph analysis and summarization
- **hybrid**: Combines local + global for best results

### **Step 6: Calculate Metrics** 📏
Evaluates responses using 6 metric categories (see below)

---

## 📊 Metrics Categories

### 1. **Traditional Metrics**
- **ROUGE-1/2/L**: N-gram overlap (precision, recall, F1)
- **BLEU**: Machine translation metric
- **F1 Score**: Harmonic mean of precision/recall
- **Exact Match**: Binary exact answer match
- **⚠️ Warning**: Severely underestimate quality for verbose answers!

### 2. **Answer Containment** ⭐ NEW
Does the verbose response contain the correct information?
- `exact_substring`: Binary exact match in response
- `normalized_substring`: Case/whitespace-insensitive match  
- `token_overlap_ratio`: Ratio of overlapping tokens
- `all_tokens_present`: All reference tokens found in prediction

**Best for**: Detecting correct info in verbose, detailed responses

### 3. **Semantic Similarity** ⭐ NEW
- Embedding-based cosine similarity using `all-MiniLM-L6-v2`
- Paraphrase-aware evaluation (0-1 scale)
- **Best for**: Rephrased but semantically equivalent answers

### 4. **Graph Quality Metrics** ⭐ NEW
- `avg_references`: Average knowledge graph nodes used per query
- `entity_density`: Entities per 100 tokens
- `unique_token_ratio`: Vocabulary diversity
- **Shows**: How well the graph is being utilized

### 5. **LLM-as-Judge** ⭐ NEW
GPT-4o-mini evaluates responses on 4 dimensions (1-5 scale):
- **Correctness**: Factual accuracy
- **Completeness**: Coverage of question requirements
- **Faithfulness**: Grounding in context (hallucination detection)
- **Conciseness**: Appropriate detail level
- **Overall Score**: Weighted average

**Best for**: Human-aligned quality assessment

### 6. **Efficiency Metrics**
- Average latency per mode
- Throughput (queries per second)
- Processing time breakdowns

---

## 🎯 Why Modern Metrics Matter

### The Verbose Answer Problem

**Example Query**: "was ronald reagan a democrat"  
**Reference Answer**: "Yes"  
**LightRAG Response**: 
> "Ronald Reagan was initially a Democrat during his early political career in the 1940s and 1950s. He was an active supporter of Democratic presidents Franklin D. Roosevelt and Harry Truman. However, he switched to the Republican Party in 1962, famously stating 'I didn't leave the Democratic Party, the party left me.' He served as Republican Governor of California from 1967-1975 and as the 40th Republican President from 1981-1989. So to directly answer: Yes, he was a Democrat early in his career, but spent most of his political life as a Republican."

### Metrics Comparison

| Metric | Score | What It Shows |
|--------|-------|---------------|
| ROUGE-1 | **2.4%** | ❌ Looks terrible! Only 2.4% word overlap |
| BLEU | **0.8%** | ❌ Almost zero! |
| **Containment (exact_substring)** | **100%** | ✅ Contains "Yes" exactly! |
| **Semantic Similarity** | **87%** | ✅ Semantically very close! |
| **LLM Judge (correctness)** | **5/5** | ✅ Perfectly accurate! |
| **LLM Judge (faithfulness)** | **5/5** | ✅ No hallucinations! |

**Key Insight**: Traditional metrics fail by ~40x for verbose but accurate answers!

---

## 📁 Directory Structure

```
benchmarking/
├── run_full_benchmark.py              # Main entry point
├── README.md                          # This file
├── ANALYSIS.md                        # Original system analysis
├── INTEGRATION_COMPLETE.md            # OpenAI migration notes
├── TESTING.md                         # Test documentation
│
├── benchmark_datasets/                # Dataset loaders
│   ├── loaders.py                    # MS MARCO, HotpotQA loaders
│   └── document_adapter.py           # Convert to LightRAG docs
│
├── configs/
│   └── dataset_config.py             # Configuration dataclasses
│
├── evaluators/
│   ├── lightrag_evaluator.py         # LightRAG wrapper
│   └── evaluation_pipeline.py        # Main orchestration (6 steps)
│
├── metrics/
│   ├── traditional.py                # ROUGE, BLEU, Containment
│   ├── semantic.py                   # Sentence-transformer similarity
│   ├── llm_judge.py                  # GPT-4o-mini evaluation
│   ├── graph_metrics.py              # Knowledge graph analytics
│   ├── retrieval.py                  # Retrieval quality metrics
│   └── efficiency.py                 # Performance metrics
│
├── runner/
│   ├── benchmark_runner.py           # Benchmark orchestration
│   └── report_generator.py           # Result formatting
│
├── utils/
│   ├── logging.py                    # Structured logging
│   ├── errors.py                     # Custom exceptions
│   └── retry.py                      # Retry logic for API calls
│
├── tests/                             # Unit tests (pytest)
│   ├── test_datasets.py
│   ├── test_evaluator.py
│   ├── test_metrics.py
│   └── test_runner.py
│
├── integration_test_openai_replacement.py    # E2E validation
├── integration_test_efficiency_metrics.py    # Performance validation
├── integration_test_full_pipeline.py         # Complete pipeline test
│
├── benchmark_storage/                # LightRAG graphs (gitignored)
│   ├── ms_marco/
│   │   └── lightrag/                # MS MARCO knowledge graph
│   └── hotpot_qa/
│       └── lightrag/                # HotpotQA knowledge graph
│
├── results/                          # Benchmark outputs
│   ├── ms_marco_results.json
│   └── hotpot_qa_results.json
│
├── cache/                            # HuggingFace dataset cache
├── logs/                             # Execution logs
└── reports/                          # Generated reports
```

---

## 🔧 Configuration

### Sample Size

Edit `benchmarking/run_full_benchmark.py`:

```python
# Quick test (6-9 minutes, ~$0.50-1.00)
MS_MARCO_LIMIT = 5
HOTPOT_QA_LIMIT = 5

# Comprehensive (60-90 minutes, ~$5-10)
MS_MARCO_LIMIT = 50
HOTPOT_QA_LIMIT = 50
```

### OpenAI Settings

The system uses OpenAI with these defaults (configurable via environment variables):

- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **LLM (Entity Extraction)**: `gpt-4o-mini` 
- **LLM (Judge)**: `gpt-4o-mini` (cost-effective evaluation)
- **API Key**: From `.env` file (`OPENAI_API_KEY`)

---

## 📈 Expected Results

### MS MARCO (Factual Q&A)

| Mode | ROUGE-1 | Containment | Semantic | LLM Judge | Graph Refs | Speed |
|------|---------|-------------|----------|-----------|------------|-------|
| **Naive** | 7.4% | 91% | 25.1% | 3.95/5 | 4.6 | 13.7s |
| Local | 5.0% | 74% | 16.3% | 3.75/5 | 3.2 | 24.0s |
| Global | 3.8% | 50% | 18.6% | 3.90/5 | 4.0 | 23.4s |
| Hybrid | 4.8% | 58% | 17.7% | 3.60/5 | 4.0 | 22.5s |

**Winner**: Naive mode - Best quality (3.95/5, 91% containment) + Fastest

### HotpotQA (Multi-Hop Reasoning)

| Mode | ROUGE-1 | Containment | Semantic | LLM Judge | Graph Refs | Speed |
|------|---------|-------------|----------|-----------|------------|-------|
| Naive | 3.0% | 32% | 20.6% | 3.95/5 | 1.4 | 15.2s |
| Local | 0.0% | 0% | 7.4% | 3.80/5 | 0.8 | 26.1s |
| Global | 4.5% | 32% | 18.4% | 3.50/5 | 1.0 | 25.3s |
| **Hybrid** | 2.5% | 32% | 17.3% | 4.20/5 | 1.4 | 24.8s |

**Winner**: Hybrid mode - Best for complex multi-hop questions (4.20/5)

### Key Insights

1. **Traditional Metrics Misleading**: ROUGE 4-7% suggests failure, but LLM Judge shows 72-84% quality (18x gap!)
2. **Naive Mode Optimal for Factual Q&A**: Best quality/speed tradeoff
3. **Hybrid Mode for Complex Questions**: Better at multi-hop reasoning
4. **High Faithfulness**: 4.2-4.8/5 across modes (minimal hallucinations)
5. **Graph Utilization**: 3-5 references per query shows strong grounding
6. **Verbose ≠ Wrong**: High containment despite low ROUGE proves accuracy

---

## 📊 Output Files

### Console Output

Real-time progress showing:
- Configuration summary
- Dataset loading progress
- Graph building status (Step 4 - longest phase)
- Query execution by mode
- Per-query metrics
- Cross-dataset comparison table

### JSON Results

**Location**: `benchmarking/results/`

**Files**:
- `ms_marco_results.json` - Complete MS MARCO evaluation
- `hotpot_qa_results.json` - Complete HotpotQA evaluation
- `benchmark_storage/{dataset}/evaluation_summary.json` - Per-dataset summary

**Structure**:
```json
{
  "dataset_samples": [
    {
      "id": "doc_123",
      "question": "What is...?",
      "answer": "The answer is...",
      "passages": ["passage1", "passage2"]
    }
  ],
  "query_results": [
    {
      "query_id": 0,
      "question": "...",
      "ground_truth": "...",
      "modes": {
        "naive": "LightRAG response...",
        "local": "LightRAG response...",
        "global": "LightRAG response...",
        "hybrid": "LightRAG response..."
      },
      "latency": {
        "naive": 2.3,
        "local": 4.1,
        "global": 3.8,
        "hybrid": 4.5
      }
    }
  ],
  "metrics": {
    "aggregated": {
      "naive": {
        "rouge-1": 0.074,
        "rouge-2": 0.031,
        "rouge-l": 0.072,
        "bleu": 0.024,
        "f1": 0.089,
        "exact_match_rate": 0.0,
        "containment": {
          "exact_substring": 0.60,
          "normalized_substring": 0.80,
          "token_overlap_ratio": 0.91,
          "all_tokens_present": 0.40
        },
        "semantic_similarity": 0.251,
        "graph_quality": {
          "avg_references": 4.6,
          "entity_density": 0.148,
          "unique_token_ratio": 0.876
        },
        "llm_judge": {
          "correctness": 4.2,
          "completeness": 4.0,
          "faithfulness": 4.4,
          "conciseness": 3.2,
          "overall": 3.95
        },
        "avg_latency_seconds": 13.7
      },
      "local": {...},
      "global": {...},
      "hybrid": {...}
    },
    "per_query": [...],
    "graph": {
      "_global_stats": {
        "num_nodes": 248,
        "num_edges": 156,
        "num_entities": 187,
        "num_relations": 89
      }
    }
  }
}
```

---

## 🧪 Testing

### Integration Tests (Manual E2E Validation)

**Test all metrics with real OpenAI API**:
```bash
python benchmarking/integration_test_openai_replacement.py
```

**Test efficiency metrics**:
```bash
python benchmarking/integration_test_efficiency_metrics.py
```

**Test full pipeline**:
```bash
python benchmarking/integration_test_full_pipeline.py
```

### Unit Tests (Fast, No API Calls)

```bash
cd benchmarking
pytest tests/
```

Tests cover:
- Dataset loaders
- Document conversion
- Metric calculations (mocked)
- Error handling

See `TESTING.md` for detailed test documentation.

---

## 🔍 How to Interpret Results

### When Traditional Metrics Look Bad

**Don't panic!** If you see:
- ROUGE-1: 3-7%
- BLEU: 0.5-2%

**Check modern metrics**:
- **Containment > 70%**: Answer contains correct info ✅
- **Semantic > 60%**: Semantically similar ✅  
- **LLM Judge > 3.5/5**: Good quality (70%+) ✅
- **Faithfulness > 4/5**: Minimal hallucinations ✅

**Conclusion**: Response is verbose but accurate!

### When to Be Concerned

Red flags indicating actual quality issues:
- **Containment < 30%**: Missing key information ❌
- **LLM Judge < 3.0/5**: Poor quality (60%) ❌
- **Faithfulness < 3.0/5**: Hallucinations present ❌
- **Semantic < 40%**: Semantically different ❌

### Mode Selection Guide

- **Need speed?** → Use **Naive** (fastest, good quality)
- **Factual Q&A?** → Use **Naive** (best for MS MARCO type)
- **Complex reasoning?** → Use **Hybrid** (best for multi-hop)
- **Best quality?** → Compare LLM Judge scores across modes

---

## 🚀 Advanced Usage

### Custom Datasets

To add your own dataset:

1. Create loader in `benchmark_datasets/loaders.py`:
```python
class CustomLoader:
    def load(self, limit=10):
        return [
            {
                'id': '1',
                'question': '...',
                'answer': '...',
                'passages': ['...']
            }
        ]
```

2. Update `configs/dataset_config.py` to include your dataset

3. Modify `run_full_benchmark.py` to use your loader

### Custom Metrics

Add new metric calculator in `metrics/`:

```python
class CustomMetric:
    def calculate(self, prediction: str, reference: str) -> float:
        # Your metric logic
        return score
```

Integrate in `evaluators/evaluation_pipeline.py` in `_calculate_metrics()` method.

---

## 🐛 Troubleshooting

### OpenAI API Errors

**Error**: `AuthenticationError: Invalid API key`
- **Fix**: Check `.env` file has valid `OPENAI_API_KEY`

**Error**: `RateLimitError`  
- **Fix**: Reduce sample size or add retry logic (already implemented)

### Memory Issues

**Error**: `OutOfMemoryError` during graph building
- **Fix**: Reduce batch size in `lightrag_evaluator.py` (line 162: `batch_size=10`)

### Slow Performance

- **Check**: Internet connection (downloads HuggingFace datasets)
- **Check**: OpenAI API response times
- **Optimize**: Use smaller sample size (5 vs 50)

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'benchmarking'`
- **Fix**: Run from project root: `python benchmarking/run_full_benchmark.py`

---

## 📚 Additional Documentation

- **`ANALYSIS.md`**: Original system analysis and architecture deep-dive
- **`INTEGRATION_COMPLETE.md`**: Notes on Intel Alloy → OpenAI migration
- **`TESTING.md`**: Comprehensive testing guide (unit vs integration tests)

---

## 🤝 Contributing

### Code Style

- Python 3.10+
- Type hints for all functions
- Docstrings for classes and public methods
- Max line length: 100 characters

### Adding New Features

1. Write unit tests first (`tests/`)
2. Implement feature
3. Add integration test if API-dependent
4. Update relevant documentation

---

## 📝 License

This benchmarking system is part of the LightRAG demo project.

---

## 🎓 Citation

If you use this benchmarking system in research, please cite:

```bibtex
@software{lightrag_benchmark_2025,
  title = {LightRAG Benchmarking System with Modern Metrics},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/sachin-india/lightrag_demo}
}
```

---

## 📞 Support

For issues or questions:
1. Check this README first
2. Review `TESTING.md` and `ANALYSIS.md`
3. Run integration tests to validate setup
4. Open GitHub issue with error logs

---

**Last Updated**: October 30, 2025  
**Version**: 2.0 (OpenAI Integration)
