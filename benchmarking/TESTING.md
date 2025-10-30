# Benchmarking Tests Overview

This folder contains two types of tests for the benchmarking system.

## Test Types

### 1. Unit Tests (`tests/` folder) ✅ Run These Regularly

**Framework:** pytest  
**Purpose:** Validate individual functions and methods in isolation  
**Speed:** Fast (<1 second)  
**Cost:** Free (no API calls)  
**CI/CD:** Yes ✅

**Run all unit tests:**
```bash
# Install pytest first
pip install pytest

# Run all tests
pytest benchmarking/tests/

# Run specific test file
pytest benchmarking/tests/test_metrics.py

# Run with verbose output
pytest benchmarking/tests/ -v

# Run with coverage
pytest benchmarking/tests/ --cov=benchmarking
```

**What they test:**
- ✅ Traditional metrics (ROUGE, BLEU, F1)
- ✅ Retrieval metrics (Precision@K, NDCG, MRR)
- ✅ Graph metrics (entity coverage, relation overlap)
- ✅ Utility functions (logging, error handling, retry logic)
- ✅ Runner and evaluator logic
- ✅ Dataset loading functions

**Characteristics:**
- No external API calls (mocked where needed)
- Test edge cases and error conditions
- Fast feedback during development
- Safe to run in CI/CD pipelines

---

### 2. Integration Tests (root folder) ⚠️ Run After Major Changes Only

**Framework:** Plain Python (asyncio)  
**Purpose:** Validate end-to-end workflows and external integrations  
**Speed:** Slow (5-30 seconds)  
**Cost:** Real API calls 💰 (OpenAI charges apply)  
**CI/CD:** No ❌ (costs money, requires API keys)

#### 2a. OpenAI Replacement Validation
**File:** `integration_test_openai_replacement.py`

**Purpose:** One-time validation that Alloy → OpenAI migration works

**Run:**
```bash
python benchmarking/integration_test_openai_replacement.py
```

**What it tests:**
- ✅ Traditional metrics calculation
- ✅ Semantic similarity (downloads sentence-transformers model ~90MB)
- ✅ Efficiency metrics with context manager
- ✅ LLM Judge with real OpenAI API calls 💰

**Requirements:**
- `OPENAI_API_KEY` in `.env` file
- Internet connection
- ~$0.01 cost per run (gpt-4o-mini)

#### 2b. Efficiency Metrics Test
**File:** `integration_test_efficiency_metrics.py`

**Purpose:** Validate efficiency metrics work with context manager pattern

**Run:**
```bash
python benchmarking/integration_test_efficiency_metrics.py
```

**What it tests:**
- ✅ Context manager usage
- ✅ Latency tracking
- ✅ Memory monitoring
- ✅ API call counting
- ✅ Metric aggregation

**Requirements:**
- None (pure Python, no API calls)
- Free and fast

#### 2c. Full Pipeline Test (⚠️ INCOMPLETE)
**File:** `integration_test_full_pipeline.py`

**Status:** Work in progress - document ingestion needs fixing

---

## When to Run Each

### Run Unit Tests When:
- ✅ During development (fast feedback)
- ✅ Before committing code
- ✅ In CI/CD pipelines
- ✅ After fixing bugs
- ✅ Every day

### Run Integration Tests When:
- ⚠️ After replacing external dependencies (like Alloy → OpenAI)
- ⚠️ Before major releases
- ⚠️ After significant architecture changes
- ⚠️ Manually, not in CI/CD
- ⚠️ Once per week maximum

---

## Test Coverage Summary

| Component | Unit Tests | Integration Tests |
|-----------|------------|-------------------|
| Traditional Metrics | ✅ Yes | ✅ Yes |
| Semantic Metrics | ✅ Yes | ✅ Yes |
| Retrieval Metrics | ✅ Yes | ❌ No |
| Graph Metrics | ✅ Yes | ❌ No |
| Efficiency Metrics | ✅ Yes | ✅ Yes |
| LLM Judge | ❌ Mocked only | ✅ Real API |
| LightRAG Evaluator | ✅ Yes | ⚠️ WIP |
| Dataset Loaders | ✅ Yes | ❌ No |
| Error Handling | ✅ Yes | ❌ No |
| Retry Logic | ✅ Yes | ❌ No |

---

## Migration Context

The integration tests were created during the **Intel Alloy → OpenAI migration** to validate that:
1. All metric modules still work after removing Alloy dependencies
2. OpenAI API integration functions correctly
3. No regressions were introduced during the refactoring

They are **not** meant to replace unit tests, but rather to complement them by testing the system as a whole.

---

## Best Practices

### For Development (Use Unit Tests)
```bash
# Fast feedback loop
pytest benchmarking/tests/test_metrics.py -v

# Watch mode (re-run on file changes)
pytest-watch benchmarking/tests/
```

### For Release Validation (Use Integration Tests)
```bash
# Before merging big changes
python benchmarking/integration_test_openai_replacement.py

# Verify efficiency tracking
python benchmarking/integration_test_efficiency_metrics.py
```

### For CI/CD (Unit Tests Only)
```yaml
# .github/workflows/test.yml
- name: Run unit tests
  run: pytest benchmarking/tests/
  
# DON'T run integration tests in CI (they cost money)
```

---

## Adding New Tests

### Adding Unit Tests
1. Create test file in `benchmarking/tests/`
2. Use pytest framework
3. Mock external dependencies
4. Test edge cases
5. Keep tests fast (<100ms each)

```python
# Example: benchmarking/tests/test_new_feature.py
import pytest
from metrics.new_feature import NewFeature

def test_new_feature_basic():
    feature = NewFeature()
    result = feature.process("test")
    assert result == expected_value
```

### Adding Integration Tests
1. Create in root `benchmarking/` folder
2. Prefix with `integration_test_`
3. Document API costs and requirements
4. Add to this README

```python
# Example: benchmarking/integration_test_new_api.py
"""
Integration Test - New API Integration
Costs: $0.XX per run
"""
async def main():
    # Test real API integration
    ...
```

---

## Summary

- **Unit tests** (`tests/`) = Fast, free, run always ✅
- **Integration tests** (root) = Slow, costs money, run occasionally ⚠️
- Both are valuable for different purposes
- Don't replace one with the other

For day-to-day development, focus on unit tests. Use integration tests to validate major changes or before releases.
