---
description: Rust project conventions for legal-learn.
globs: "*.rs, Cargo.toml, Cargo.lock, Taskfile.yml"
alwaysApply: true
---

# Rust — legal-learn

Burn-based Rust training project for learning Korean laws and precedents from
the indexed corpus in `../legal-ko/.qmd/data.db`.
Use `cargo` for build/test/lint tasks and `task` for the common training flows.

## Repo Goal

- Train a local model in Rust with Burn against the Korean legal corpus staged by
  the sibling `legal-ko` repo.
- The current codebase trains a 384-dim embedding model and can export learned
  embeddings back into the SQLite corpus.
- Preferred first local model target for future continued-pretraining work:
  `Qwen3-0.6B-Base`, sized realistically for local iteration before moving up.

## Dataset

- Primary dataset: `../legal-ko/.qmd/data.db`
- SQLite tables currently used by this repo: `documents`, `content`,
  `content_vectors`
- Current training pairs come from precedent documents:
  `anchor = 판결요지`, `positive = 참조조문`, with fallback to `사건명`

## Build & Run

- `cargo build` to compile the project.
- `cargo build --release` to build the release binary.
- `cargo test` to run tests.
- `cargo clippy -- -D warnings` for lints.
- `cargo fmt --all` to format code.
- `cargo run -- train --db ../legal-ko/.qmd/data.db` to train on the local corpus.
- `cargo run -- export --db ../legal-ko/.qmd/data.db --checkpoint checkpoints` to
  write embeddings back into the corpus.
- `task train` to run the default training flow.
- `task export` to export embeddings from the default checkpoint directory.

## Architecture

```
src/
  main.rs      — CLI entry point with `train` and `export` subcommands
  data.rs      — SQLite loading and precedent pair extraction
  dataset.rs   — Burn dataset + batch collation for contrastive training
  loss.rs      — InfoNCE loss over normalized embeddings
  model.rs     — Transformer encoder embedding model (384 dims)
  tokenize.rs  — BPE tokenizer training/loading and fixed-length encoding
  training.rs  — Burn training loop, checkpointing, and embedding export
```

## Training Design

- Backend: Burn `ndarray` today; keep code portable for later backend upgrades.
- Tokenizer: train or reuse a local BPE tokenizer saved under the checkpoint
  output directory.
- Model: small transformer encoder with mean pooling and L2-normalized output.
- Objective: contrastive InfoNCE over legal precedent-derived text pairs.
- Export path: writes JSON-encoded 384-dim vectors into `content_vectors` under a
  repo-owned model name.

## Conventions

- Use `anyhow::Result` for fallible functions.
- Use `tracing` instead of `println!` / `eprintln!`.
- Keep changes small and local unless a broader refactor is clearly needed.
- Prefer Burn-native implementations for model, training, checkpointing, and
  tensor operations.
- Treat `../legal-ko/.qmd/data.db` as an external dataset dependency; do not
  mutate its schema unless explicitly asked.
