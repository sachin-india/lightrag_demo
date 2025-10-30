# LightRAG Benchmark Report: MS MARCO

**Generated**: 2025-10-30 09:11:39

---

## Dataset

- **Total Samples**: 1
- **Total Queries**: 1

## Knowledge Graph

- **Nodes**: 182
- **Edges**: 170
- **Entities**: 43
- **Relations**: 43

## Metrics by Query Mode

### NAIVE Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.1281 |
| ROUGE-2 | 0.0147 |
| ROUGE-L | 0.0687 |
| BLEU | 0.0566 |
| F1 Score | 0.1031 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 0.5417 |
| All Tokens Present | 0.0000 |

#### Semantic Similarity

- **Score**: 0.2523

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 6.00 |
| Entity Density | 0.161 |
| Token Diversity | 0.702 |
| Reference Usage Rate | 1.000 |

#### LLM Judge Evaluation (GPT-4o-mini)

| Dimension | Score |
|-----------|-------|
| Correctness | 5.00/5 |
| Completeness | 5.00/5 |
| Faithfulness | 5.00/5 |
| Conciseness | 4.00/5 |
| **Overall** | **4.75/5** |

#### Performance

- **Average Latency**: 9.20s

---

### LOCAL Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.0838 |
| ROUGE-2 | 0.0085 |
| ROUGE-L | 0.0623 |
| BLEU | 0.0390 |
| F1 Score | 0.0700 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 0.2917 |
| All Tokens Present | 0.0000 |

#### Semantic Similarity

- **Score**: 0.0335

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 5.00 |
| Entity Density | 0.259 |
| Token Diversity | 0.624 |
| Reference Usage Rate | 1.000 |

#### LLM Judge Evaluation (GPT-4o-mini)

| Dimension | Score |
|-----------|-------|
| Correctness | 1.00/5 |
| Completeness | 1.00/5 |
| Faithfulness | 1.00/5 |
| Conciseness | 2.00/5 |
| **Overall** | **1.25/5** |

#### Performance

- **Average Latency**: 10.21s

---

### GLOBAL Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.1000 |
| ROUGE-2 | 0.0000 |
| ROUGE-L | 0.0664 |
| BLEU | 0.0486 |
| F1 Score | 0.0853 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 0.2917 |
| All Tokens Present | 0.0000 |

#### Semantic Similarity

- **Score**: 0.1281

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 3.00 |
| Entity Density | 0.172 |
| Token Diversity | 0.628 |
| Reference Usage Rate | 1.000 |

#### LLM Judge Evaluation (GPT-4o-mini)

| Dimension | Score |
|-----------|-------|
| Correctness | 1.00/5 |
| Completeness | 1.00/5 |
| Faithfulness | 1.00/5 |
| Conciseness | 2.00/5 |
| **Overall** | **1.25/5** |

#### Performance

- **Average Latency**: 8.46s

---

### HYBRID Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.2652 |
| ROUGE-2 | 0.1875 |
| ROUGE-L | 0.1831 |
| BLEU | 0.1008 |
| F1 Score | 0.1831 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 1.0000 |
| All Tokens Present | 1.0000 |

#### Semantic Similarity

- **Score**: 0.1953

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 5.00 |
| Entity Density | 0.151 |
| Token Diversity | 0.629 |
| Reference Usage Rate | 1.000 |

#### LLM Judge Evaluation (GPT-4o-mini)

| Dimension | Score |
|-----------|-------|
| Correctness | 3.00/5 |
| Completeness | 3.00/5 |
| Faithfulness | 4.00/5 |
| Conciseness | 3.00/5 |
| **Overall** | **3.25/5** |

#### Performance

- **Average Latency**: 11.55s

---

## Sample Query Results

### Query 1

**Question**: what is rba

**Reference Answer**: Results-Based Accountability is a disciplined way of thinking and taking action that communities can use to improve the lives of children, youth, families, adults and the community as a whole....

**NAIVE Response**: RBA can refer to several different entities or concepts, depending on the context. Here are the most notable interpretations:

1. **Results-Based Accountability® (RBA)**:
   - This is a disciplined ap...

**LOCAL Response**: The **Reserve Bank of Australia (RBA)** is the central bank of Australia, which plays a critical role in managing the nation's currency and monetary policy. Established on **14 January 1960**, the RBA...

**GLOBAL Response**: The RBA, or Reserve Bank of Australia, is the central bank of Australia. Established on **14 January 1960**, the RBA is primarily responsible for issuing banknotes and managing the country's monetary ...

**HYBRID Response**: The **Reserve Bank of Australia (RBA)** is Australia’s central bank, established on **14 January 1960**. Its primary responsibilities include issuing banknotes and managing the country’s monetary poli...

---
