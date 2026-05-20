## 0001 — Architecture Rationale

**Status:** Implemented  
**Date:** 2026-05-20

### Context

We need to produce high-quality Korean legal embeddings from a 306K-document
corpus (laws + precedents) for downstream similarity search. The system must:

- Run entirely on local hardware (Apple M1 Pro, 16 GB unified memory)
- Produce embeddings compatible with SQLite vec0 for the sibling `legal-ko` repo
- Enable iterative improvement without cloud dependencies
- Support flexible embedding dimensions for storage/quality tradeoffs

### Decisions

#### 1. Rust + Burn framework (not Python/PyTorch)

**Why:** The sibling repo ecosystem is Rust-native. Burn provides:
- Native Metal/wgpu backend for M1 acceleration without CUDA
- Type-safe tensor operations with compile-time dimension checking
- Single binary deployment, no Python environment management
- Composable training abstractions (Learner, SupervisedTraining)

**Tradeoff:** Smaller ecosystem than PyTorch. Fewer pretrained model loaders.
Mitigated by the distillation path (use llama.cpp for inference from pretrained
models, train only the student in Burn).

#### 2. Contrastive learning with InfoNCE (not generative/MLM)

**Why:** The task is embedding, not generation. InfoNCE directly optimizes the
metric we care about (cosine similarity between semantically related texts).
It's also batch-efficient — each batch provides B positive pairs and B*(B-1)
implicit negatives without additional sampling.

**Training pairs:**
- Precedents: 판결요지 (ruling summary) → 참조조문 (cited statute) or 사건명
- Laws: article heading → article body, law title → article heading

These are naturally co-occurring pairs where semantic relatedness is structural
rather than requiring human annotation.

#### 3. Matryoshka representation learning (not fixed-dim only)

**Why:** Computing InfoNCE at truncated prefix dimensions (384/256/128/64)
trains the model so that any prefix of the embedding vector is independently
useful. Benefits:

- Store 64-dim in vec0 for fast approximate search, 384-dim for reranking
- No separate models needed for different quality/speed tradeoffs
- Zero inference cost — just truncate at query time
- Compatible with vec0's fixed-dimension storage

**Implementation:** Average InfoNCE loss across 4 dimension levels. Each level
re-normalizes after truncation to maintain unit-norm invariant.

#### 4. Wgpu backend (not ndarray or CUDA)

**Why:** Apple M1 Pro has 16 Metal GPU cores. Wgpu maps to Metal natively,
providing 3-5x speedup over CPU ndarray for matrix operations. No NVIDIA
hardware available, so CUDA is not an option.

**Tradeoff:** Burn's wgpu backend is less mature than ndarray. Some trait
bounds require `recursion_limit = 256`. LossMetric adaptor doesn't work
with wgpu, so we drop per-epoch metric logging from SupervisedTraining.

#### 5. Distillation via llama.cpp (not fine-tuning a pretrained model in Burn)

**Why:** Burn lacks pretrained model loaders for Qwen/Gemma/etc. Rather than
implement weight loading from safetensors (complex, one-off work), we:

1. Use `llama-embedding` (llama.cpp CLI) to run a pretrained model and extract
   embeddings as teacher targets
2. Train the Burn student model with MSE loss against those targets

This gives us the knowledge of a 300M+ parameter model distilled into our
lightweight 6-layer transformer, without needing to load external weights
into Burn.

**Teacher choice:** EmbeddingGemma-300M Q4 — purpose-built embedding model,
768-dim output, 277MB GGUF, runs in <2s on M1. Teacher embeddings are
truncated from 768→384 for the student target.

**Operational constraints:**
- Texts are truncated to 512 characters before sending to `llama-embedding`
  (the model's context window cannot handle full 4.7KB average legal documents
  concatenated across a batch)
- Default batch size of 8 texts per `llama-embedding` invocation balances
  throughput against context overflow
- Each invocation reloads the model (~1.5s fixed cost), making small batches
  acceptable for the total volume (~930 unique texts for 500 pairs)

#### 6. vec0 index population during export (not separate indexing step)

**Why:** The `content_vectors_idx` virtual table is what downstream search
queries. If we only write JSON to `content_vectors`, the vec0 index stays
empty and search doesn't work. By writing both in the same export pass:

- Single source of truth for model version
- Atomic clear+rewrite prevents stale mixed-model results
- Raw f32 bytes for vec0 (no JSON parse overhead at query time)

#### 7. BPE tokenizer trained from corpus (not reusing a pretrained tokenizer)

**Why:** Korean legal text has domain-specific vocabulary (법률, 판결요지,
조문, etc.) that general-purpose tokenizers split suboptimally. Training
a 32K-vocab BPE from the actual corpus gives better token efficiency for
the 128-token sequence length budget.

**Tradeoff:** Not compatible with pretrained model vocabularies. Acceptable
because we train from scratch (contrastive) or distill (MSE on embeddings,
not tokens).

**Corpus sampling:** BPE merge computation is O(n^2) on corpus size. Training
on the full 306K documents takes hours (observed: 2 merges in 3 minutes for
32K vocab). The tokenizer trainer samples 10K evenly-spaced documents from
the corpus, which provides sufficient vocabulary coverage while completing
in ~1-2 minutes.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     legal-ko/.qmd/data.db                   │
│  documents (306K) │ content │ content_vectors │ vec0 index  │
└────────┬──────────────────────────────────────────┬─────────┘
         │ read                                     │ write
         ▼                                          ▲
┌─────────────────┐                      ┌──────────────────┐
│   data.rs       │                      │   training.rs    │
│ precedent pairs │                      │   export()       │
│ law pairs       │                      │ JSON + vec0 f32  │
└────────┬────────┘                      └────────┬─────────┘
         │                                        │
         ▼                                        │
┌─────────────────┐    ┌───────────────┐           │
│  tokenize.rs    │    │   model.rs    │───────────┘
│ BPE 32K vocab   │───▶│ 6L transformer│
│ MAX_SEQ_LEN=128 │    │ 384-dim embed │
└─────────────────┘    └──────┬────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼──────┐    ┌───────▼─────────┐
              │  loss.rs   │    │  distill.rs     │
              │ Matryoshka │    │ llama-embedding │
              │ InfoNCE    │    │ MSE student     │
              └────────────┘    └─────────────────┘
```

## Future Work

- **Qwen3-0.6B continued pretraining** — once Burn supports safetensors weight
  loading for Qwen architecture, or via a custom weight converter
- **Hard negative mining** — use `documents_fts` (FTS5) to find near-miss
  documents as hard negatives for contrastive training
- **Multi-collection weighting** — balance precedent vs law pairs by collection
  size or downstream task importance
- **Evaluation** — retrieval metrics (MRR, recall@k) on held-out pairs

## Training Time Estimates (M1 Pro, Wgpu/Metal)

Calibrated from 900 samples / batch 32 / small model = 26s/epoch.

| Scenario | Per Epoch | 10 Epochs |
|----------|-----------|-----------|
| Small (2L/128d), 10K pairs | ~5 min | ~48 min |
| Full (6L/384d), 10K pairs | ~43 min | ~7 hours |
| Small, full 230K pairs | ~110 min | ~18 hours |
| Full, full 230K pairs | ~16.5 hours | ~7 days |

**Best quality/compute tradeoff:** 10K teacher distillation samples → full model,
10 epochs ≈ 7 hours (`task distill:full`). Generates teachers via llama-embedding
(~30 min for 10K), then trains full 6L/384d student with MSE loss.

### Baseline (small distilled, 931 samples, 5 epochs)

- MRR: 0.055 | R@1: 1% | R@5: 5.5% | R@10: 9.5% (near-random on 200 pairs)
