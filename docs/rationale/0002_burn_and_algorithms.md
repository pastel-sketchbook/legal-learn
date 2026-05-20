## 0002 — Burn Library Usage & Algorithms

**Status:** Implemented  
**Date:** 2026-05-20

### Context

This document catalogs which parts of the Burn framework we use and the ML
algorithms implemented in the codebase. Useful for onboarding, upgrade planning,
and understanding which Burn features we depend on.

### Burn Crates & Modules

| Burn Module | Usage | Source |
|-------------|-------|--------|
| `burn::backend::Wgpu` | Primary compute backend (Metal on macOS) | `training.rs`, `distill.rs`, `eval.rs` |
| `burn::backend::Autodiff` | Automatic differentiation wrapper for training | `training.rs`, `distill.rs` |
| `burn::nn::transformer::TransformerEncoder{,Config}` | Core model architecture — multi-head self-attention layers | `model.rs` |
| `burn::nn::Embedding` | Token embedding lookup table (32K vocab → d_model) | `model.rs` |
| `burn::nn::Linear` | Positional embedding projection and final output head | `model.rs` |
| `burn::nn::LayerNorm` | Post-encoder normalization before pooling | `model.rs` |
| `burn::prelude::Tensor` | All tensor operations (matmul, narrow, normalize, etc.) | everywhere |
| `burn::tensor::TensorData` | CPU↔GPU data transfer for tokenized inputs | `dataset.rs`, `distill.rs`, `eval.rs` |
| `burn::tensor::loss::cross_entropy_with_logits` | Soft cross-entropy for InfoNCE loss | `loss.rs` |
| `burn::data::dataloader::DataLoaderBuilder` | Batched data loading with shuffle for training | `training.rs` |
| `burn::data::dataloader::batcher::Batcher` | Custom batch collation (padding, masking) | `dataset.rs` |
| `burn_dataset::Dataset` | Dataset trait implementation for text pairs | `dataset.rs` |
| `burn::optim::AdamConfig` | Adam optimizer for both contrastive and distillation training | `training.rs`, `distill.rs` |
| `burn::optim::GradientsParams` | Manual gradient application in distillation loop | `distill.rs` |
| `burn::optim::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig` | Cosine annealing learning rate decay | `training.rs` |
| `burn::optim::lr_scheduler::linear::LinearLrSchedulerConfig` | Linear warmup phase | `training.rs` |
| `burn::optim::lr_scheduler::composed::ComposedLrSchedulerConfig` | Warmup → cosine composed schedule | `training.rs` |
| `burn::train::Learner` | High-level training loop with validation and checkpointing | `training.rs` |
| `burn::train::SupervisedTraining` | Training strategy for `TrainStep`/`InferenceStep` impls | `training.rs` |
| `burn::train::{TrainStep, TrainOutput, InferenceStep}` | Step traits for model forward/backward | `model.rs` |
| `burn::train::RegressionOutput` | Loss output wrapper for the Learner | `model.rs` |
| `burn::record::CompactRecorder` | Checkpoint serialization (MessagePack format, `.mpk`) | `training.rs`, `distill.rs`, `eval.rs` |

### Algorithms

#### 1. InfoNCE (Noise-Contrastive Estimation) Loss

**File:** `loss.rs:13`  
**Purpose:** Contrastive objective that pushes paired (anchor, positive) embeddings
together while pushing all other in-batch negatives apart.

- Computes cosine similarity matrix `[B, B]` between anchor and positive embeddings
- Scales by temperature τ (default 0.07) to sharpen the distribution
- Uses soft cross-entropy against an identity target (diagonal = positive pair)
- In-batch negatives: every other sample in the batch serves as a negative

#### 2. Matryoshka Representation Learning

**File:** `loss.rs:40`  
**Purpose:** Trains nested embedding subspaces so that prefix truncations (384, 256,
128, 64 dims) are independently useful for retrieval.

- Truncates full embedding to each dimension level via `Tensor::narrow`
- Re-normalizes truncated vectors to unit length
- Computes InfoNCE at each level independently
- Averages all level losses equally

**Reference:** Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022)

#### 3. Knowledge Distillation (MSE)

**File:** `distill.rs`  
**Purpose:** Transfer knowledge from a large teacher model (EmbeddingGemma-300M via
llama.cpp) to the small Burn student model.

- Teacher: generates 768-dim embeddings offline via `llama-embedding` CLI
- Student target: truncated to student's d_model (128 or 384 dims)
- Loss: Mean Squared Error between student output and teacher target
- Manual training loop with `Autodiff` backend (not using Learner)
- Adam optimizer with per-step gradient updates

#### 4. Transformer Encoder with Mean Pooling

**File:** `model.rs`  
**Purpose:** The embedding model architecture.

- Token embeddings (32K vocab) + learned positional embeddings (512 positions)
- N-layer Transformer encoder (multi-head self-attention + FFN)
- Attention mask applied to exclude padding tokens
- Mean pooling over non-masked positions → fixed-size embedding
- L2 normalization of final output (unit sphere)

#### 5. BPE Tokenization

**File:** `tokenize.rs`  
**Purpose:** Byte-Pair Encoding tokenizer trained on the Korean legal corpus.

- Trained via HuggingFace `tokenizers` crate (BPE model + byte-level pre-tokenizer)
- 32K vocabulary, MAX_SEQ_LEN=128 tokens
- Corpus: 2K sampled documents × 1K chars (to bound training time)
- Fixed-length encoding with padding and attention mask generation

#### 6. Cosine Annealing with Linear Warmup

**File:** `training.rs`  
**Purpose:** Learning rate schedule for stable training.

- Linear warmup: LR ramps from 0 → target over N steps
- Cosine annealing: LR decays following cosine curve to near-zero
- Composed via Burn's `ComposedLrSchedulerConfig` with `SchedulerReduction`

#### 7. Retrieval Evaluation (MRR, Recall@k)

**File:** `eval.rs`  
**Purpose:** Measure embedding quality on held-out text pairs.

- Takes last N pairs from dataset as eval set (not seen early in training)
- Encodes all anchors and positives in batches of 64
- Computes full cosine similarity matrix (dot product of normalized vectors)
- For each anchor, ranks all positives by similarity
- Reports: Mean Reciprocal Rank, Recall@1, Recall@5, Recall@10

### Not Used (Burn features we deliberately avoid)

| Feature | Reason |
|---------|--------|
| `burn::backend::NdArray` (as primary) | Switched to Wgpu for Metal GPU acceleration |
| `burn::nn::loss::*` | Using custom InfoNCE; Burn's built-in losses don't cover contrastive |
| `burn::train::metric::*` | Custom loss logging; Burn metrics don't fit contrastive/distillation |
| `burn::data::dataset::SqliteDataset` | We use rusqlite directly for flexible pair extraction |
| Distributed training | Single-machine only (M1 Pro) |
