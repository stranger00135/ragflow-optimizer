---
layout: default
title: "Stop Guessing Your RAG Chunking Parameters — Automate It"
---

# Stop Guessing Your RAG Chunking Parameters — Automate It

*By Tiger Guo — February 2026*

If you're building RAG applications with [RAGFlow](https://github.com/infiniflow/ragflow), you've probably hit this problem: **what chunk_size should I use?**

Too small? Lost context. Too large? Too much noise. And that's just one parameter — there's also overlap, auto-generated questions, auto-keywords, and the chunking method itself (naive, hierarchical, QA-based).

Most of us pick values by feel. Maybe we try a couple combinations manually. But we never really *know* if our config is optimal.

## The Problem is Bigger Than You Think

Different document types genuinely need different chunking configs. In my testing:

- **Manual/hierarchical chunking** with 1024 tokens beat general chunking at 512 tokens by **15%** on policy documents
- **auto_questions=5** improved retrieval for procedure documents but **hurt performance** on data tables
- Switching from naive to QA-based chunking improved MRR by **0.12** on FAQ-style content

If you're using one config for all your documents, you're leaving significant retrieval quality on the table.

## Introducing RAGFlow Optimizer

I built [**RAGFlow Optimizer**](https://github.com/stranger00135/ragflow-optimizer) — an open-source tool that **automatically finds the optimal chunking parameters** for your RAGFlow knowledge bases.

### How It Works

**Phase 1: Preset Screening**
- Tests different chunking methods (general, manual, QA, paper, book, etc.)
- Evaluates each with LLM-based relevance scoring
- Picks the winning preset

**Phase 2: Parameter Fine-Tuning**
- Takes the winning preset and systematically varies parameters
- Tests `chunk_token_num`, `auto_questions`, `auto_keywords`
- Uses a combined score: `(1-FailRate)×0.4 + Precision@3×0.3 + MRR×0.3`

### Key Design Decisions

**KB Reuse**: Files are uploaded once. When testing a new config, only the chunking step is repeated — no redundant uploads. This saves significant time on large document sets.

**Noise Injection**: The tool adds distractor documents to test robustness. A config that only works on clean data isn't useful in production.

**Per-Folder Optimization**: Different document types get independently optimized configs. Your SOPs, policy docs, and data tables each get their own optimal settings.

**LLM-Based Evaluation**: Instead of keyword matching, an LLM judges whether retrieved chunks actually answer the test queries. This catches semantic relevance that string matching misses.

## Quick Start

```bash
git clone https://github.com/stranger00135/ragflow-optimizer
cd ragflow-optimizer
pip install -r requirements.txt
cp .env.example .env  # Add your RAGFlow + LLM credentials
python main.py run
```

The tool generates a clean report with the optimal config and metrics for each document folder.

## What I Learned Building This

1. **RAGFlow's KB reuse is underrated.** Most RAG optimization tools re-upload everything for each experiment. RAGFlow's architecture lets you re-chunk without re-uploading, which makes automated optimization practical.

2. **Noise injection changes everything.** Configs that score well on clean benchmarks often fail when you add irrelevant documents. Testing with distractors is essential for production-ready configs.

3. **The "best" config doesn't exist.** There's no universal optimal chunk_size. The right answer depends on your documents, your queries, and your retrieval model. That's exactly why automated optimization matters.

## What's Next

- Support for more RAG platforms (LangChain, LlamaIndex)
- Automated query generation from documents
- Embedding model comparison alongside chunking optimization
- CI/CD integration for continuous config validation

The tool is MIT licensed and contributions are welcome.

**→ [GitHub: stranger00135/ragflow-optimizer](https://github.com/stranger00135/ragflow-optimizer)**

---

*If you found this useful, consider giving the repo a ⭐ — it helps others discover the tool.*
