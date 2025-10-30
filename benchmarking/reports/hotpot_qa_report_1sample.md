# LightRAG Benchmark Report: HotpotQA

**Generated**: 2025-10-30 09:11:39

---

## Dataset

- **Total Samples**: 1
- **Total Queries**: 1

## Knowledge Graph

- **Nodes**: 94
- **Edges**: 70
- **Entities**: 53
- **Relations**: 53

## Metrics by Query Mode

### NAIVE Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.0000 |
| ROUGE-2 | 0.0000 |
| ROUGE-L | 0.0000 |
| BLEU | 0.0000 |
| F1 Score | 0.0000 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 0.0000 |
| All Tokens Present | 0.0000 |

#### Semantic Similarity

- **Score**: 0.0229

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 5.00 |
| Entity Density | 0.383 |
| Token Diversity | 0.767 |
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

- **Average Latency**: 2.91s

---

### LOCAL Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.0000 |
| ROUGE-2 | 0.0000 |
| ROUGE-L | 0.0000 |
| BLEU | 0.0000 |
| F1 Score | 0.0000 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 0.0000 |
| All Tokens Present | 0.0000 |

#### Semantic Similarity

- **Score**: 0.0328

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 2.00 |
| Entity Density | 0.355 |
| Token Diversity | 0.680 |
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

- **Average Latency**: 8.32s

---

### GLOBAL Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.0741 |
| ROUGE-2 | 0.0000 |
| ROUGE-L | 0.0541 |
| BLEU | 0.0278 |
| F1 Score | 0.0541 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 1.0000 |
| All Tokens Present | 1.0000 |

#### Semantic Similarity

- **Score**: 0.0939

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 2.00 |
| Entity Density | 0.452 |
| Token Diversity | 0.762 |
| Reference Usage Rate | 1.000 |

#### LLM Judge Evaluation (GPT-4o-mini)

| Dimension | Score |
|-----------|-------|
| Correctness | 5.00/5 |
| Completeness | 5.00/5 |
| Faithfulness | 5.00/5 |
| Conciseness | 5.00/5 |
| **Overall** | **5.00/5** |

#### Performance

- **Average Latency**: 4.53s

---

### HYBRID Mode

#### Traditional Metrics

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.0000 |
| ROUGE-2 | 0.0000 |
| ROUGE-L | 0.0000 |
| BLEU | 0.0000 |
| F1 Score | 0.0000 |
| Exact Match | 0.0000 |

#### Containment Metrics

| Metric | Value |
|--------|-------|
| Exact Match | 0.0000 |
| Normalized Match | 0.0000 |
| Token Overlap Ratio | 0.0000 |
| All Tokens Present | 0.0000 |

#### Semantic Similarity

- **Score**: 0.0323

#### Graph Quality

| Metric | Value |
|--------|-------|
| Avg References | 3.00 |
| Entity Density | 0.299 |
| Token Diversity | 0.655 |
| Reference Usage Rate | 1.000 |

#### LLM Judge Evaluation (GPT-4o-mini)

| Dimension | Score |
|-----------|-------|
| Correctness | 1.00/5 |
| Completeness | 3.00/5 |
| Faithfulness | 5.00/5 |
| Conciseness | 3.00/5 |
| **Overall** | **3.00/5** |

#### Performance

- **Average Latency**: 12.89s

---

## Sample Query Results

### Query 1

**Question**: Were Scott Derrickson and Ed Wood of the same nationality?

**Reference Answer**: yes...

**NAIVE Response**: Scott Derrickson and Ed Wood were both American, thus sharing the same nationality. Ed Wood was a cult filmmaker known for his work in Hollywood, whereas Scott Derrickson is a contemporary filmmaker a...

**LOCAL Response**: Scott Derrickson is an American filmmaker, while Ed Wood was also an American filmmaker. Therefore, both Scott Derrickson and Ed Wood shared the same nationality, which is American.

### References
- ...

**GLOBAL Response**: Scott Derrickson is an American filmmaker, while Ed Wood was also an American filmmaker. Therefore, yes, both Scott Derrickson and Ed Wood were of the same nationality, which is American.

### Referen...

**HYBRID Response**: No, Scott Derrickson and Ed Wood were not of the same nationality. 

- **Scott Derrickson** is an American filmmaker, known for directing films such as *Sinister* and *Doctor Strange*.
- **Ed Wood** w...

---
