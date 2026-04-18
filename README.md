# legal-learn

Burn-based Rust training pipeline for learning from Korean laws and precedents.

This repo trains a local model against the indexed corpus produced by the
sibling `legal-ko` repo at `../legal-ko/.qmd/data.db`.

## Goal

- train a local Burn model in Rust over Korean legal text
- start with a practical local-sized target such as `Qwen3-0.6B-Base` for future
  continued pretraining work
- keep the current pipeline focused on corpus preparation, contrastive training,
  and embedding export back into SQLite

## Current Pipeline

- reads precedent documents from `../legal-ko/.qmd/data.db`
- extracts training pairs from `판결요지 -> 참조조문`, with fallback to `사건명`
- trains a BPE tokenizer from the local corpus when needed
- trains a 384-dim transformer embedding model with Burn
- exports learned vectors into `content_vectors`

## Dataset

- database: `../legal-ko/.qmd/data.db`
- tables used: `documents`, `content`, `content_vectors`
- source corpus: Korean laws and precedents indexed by `legal-ko`

## Usage

```bash
# Train the embedding model
cargo run -- train --db ../legal-ko/.qmd/data.db --output checkpoints

# Export learned embeddings back into the database
cargo run -- export --db ../legal-ko/.qmd/data.db --checkpoint checkpoints
```

Or use Task:

```bash
task train
task export
```

## Architecture

- `src/main.rs` - CLI entry point
- `src/data.rs` - SQLite loading and precedent pair extraction
- `src/dataset.rs` - Burn dataset and batch collation
- `src/loss.rs` - InfoNCE loss
- `src/model.rs` - 384-dim transformer encoder embedding model
- `src/tokenize.rs` - tokenizer training/loading and fixed-length encoding
- `src/training.rs` - training loop and embedding export

## Notes

- training is written in Rust with Burn
- the current backend is `ndarray`, with room to change backends later
- treat `../legal-ko/.qmd/data.db` as an external dependency owned by the sibling repo
