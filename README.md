# legal-learn

Burn-based Rust training pipeline for learning Korean legal embeddings from
laws and precedents.

Trains a local embedding model against the indexed corpus produced by the
sibling `legal-ko` repo at `../legal-ko/.qmd/data.db`.

## Goal

- Train a local Burn model in Rust over Korean legal text
- Start with a contrastive embedding model (384-dim, Matryoshka)
- Distill knowledge from larger models via llama.cpp (EmbeddingGemma-300M, Qwen3-0.6B)
- Export embeddings into SQLite vec0 index for similarity search
- Future: continued pretraining with `Qwen3-0.6B-Base`

## Current Pipeline

1. **Data extraction** — reads precedent and law documents from `data.db`; extracts
   contrastive pairs (판결요지→참조조문, article heading→body, law title→article)
2. **Tokenization** — trains/loads a BPE tokenizer from the corpus
3. **Contrastive training** — Matryoshka InfoNCE loss at 384/256/128/64 dims with
   cosine annealing LR scheduler, wgpu (Metal) backend
4. **Distillation** — generates teacher embeddings via `llama-embedding`, trains
   student model with MSE loss against teacher targets
5. **Export** — writes 384-dim vectors to `content_vectors` and `content_vectors_idx`
   (vec0) for downstream similarity search

## Dataset

- Database: `../legal-ko/.qmd/data.db`
- Tables used: `documents`, `content`, `content_vectors`, `content_vectors_idx`
- Collections: `precedents` (판결요지 pairs), `laws` (article-level pairs)
- Scale: ~306K documents, 4 collections

## Usage

```bash
# Contrastive training (Metal-accelerated)
cargo run -- train --db ../legal-ko/.qmd/data.db

# Fast iteration with small model
cargo run -- train --db ../legal-ko/.qmd/data.db --small --limit 64 --epochs 5

# Generate teacher embeddings from EmbeddingGemma-300M
cargo run -- distill-generate --db ../legal-ko/.qmd/data.db --limit 500

# Train student via distillation
cargo run -- distill-train --db ../legal-ko/.qmd/data.db

# Export embeddings (includes vec0 index)
cargo run -- export --db ../legal-ko/.qmd/data.db --checkpoint checkpoints

# Inspect corpus stats
cargo run -- inspect --db ../legal-ko/.qmd/data.db --json
```

Or use Task:

```bash
task train              # Full contrastive training
task train:fast         # Small model, 64 pairs, 5 epochs
task distill:generate   # Teacher embedding generation (full corpus)
task distill:generate:fast  # Teacher generation, --limit 500
task distill:train      # Distillation training (full model, 10 epochs)
task distill:train:fast # Distillation, small model, 5 epochs
task distill            # Full pipeline (generate + train)
task export             # Write embeddings to database
task inspect            # Corpus statistics
```

## Architecture

```
src/
  main.rs      — CLI entry point (inspect, train, export, distill-generate, distill-train)
  data.rs      — SQLite loading; precedent + law pair extraction
  dataset.rs   — Burn dataset + batch collation for contrastive training
  loss.rs      — Matryoshka InfoNCE loss (384/256/128/64 dim levels)
  model.rs     — Transformer encoder embedding model (384 dims, L2-normalized)
  tokenize.rs  — BPE tokenizer training/loading and fixed-length encoding
  training.rs  — Burn training loop, checkpointing, vec0 export
  distill.rs   — llama.cpp teacher generation + MSE distillation training
```

## Hardware

Tested on Apple M1 Pro (16 GB). Backend: `burn::backend::Wgpu` (Metal 4).

## Notes

- Backend: wgpu for Metal GPU acceleration (previously ndarray)
- Optimizer: Adam with cosine annealing (optional linear warmup)
- Distillation teacher: EmbeddingGemma-300M Q4 via llama-embedding (768→384 truncation)
- vec0 index: populated during export with raw f32 little-endian bytes
- Treat `../legal-ko/.qmd/data.db` as an external dependency owned by the sibling repo
