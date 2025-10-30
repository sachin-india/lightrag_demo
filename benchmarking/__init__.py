"""
LightRAG Benchmarking System

Enterprise-grade benchmarking for graph-based RAG evaluation.
Completely separate from production code - zero impact on main system.

Features:
- Multi-dataset support (MS MARCO, HotpotQA)
- All 4 query modes benchmarked (naive, local, global, hybrid)
- Comprehensive metrics (ROUGE, BLEU, latency, graph stats)
- Separate storage from production
- Intel-compliant (Azure OpenAI only)
"""

__version__ = "1.0.0"
