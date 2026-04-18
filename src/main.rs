mod data;
mod dataset;
mod loss;
mod model;
mod tokenize;
mod training;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "legal-learn", about = "Train Korean legal embeddings with Burn")]
struct Cli {
  #[command(subcommand)]
  command: Command,
}

#[derive(Subcommand)]
enum Command {
  /// Inspect the dataset and report training-relevant stats
  Inspect {
    /// Path to .qmd/data.db
    #[arg(long, default_value = "../legal-ko/.qmd/data.db")]
    db: String,

    /// Output as JSON
    #[arg(long)]
    json: bool,
  },

  /// Train the embedding model from legal-ko data
  Train {
    /// Path to .qmd/data.db
    #[arg(long, default_value = "../legal-ko/.qmd/data.db")]
    db: String,

    /// Output directory for checkpoints
    #[arg(long, default_value = "checkpoints")]
    output: String,

    /// Number of training epochs
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,

    /// Limit the number of extracted training pairs for fast debugging
    #[arg(long)]
    limit: Option<usize>,

    /// Use a small model (2 layers, 128 dim) for fast iteration
    #[arg(long)]
    small: bool,
  },

  /// Export trained embeddings back into data.db
  Export {
    /// Path to .qmd/data.db
    #[arg(long, default_value = "../legal-ko/.qmd/data.db")]
    db: String,

    /// Checkpoint directory to load model from
    #[arg(long)]
    checkpoint: String,
  },
}

fn main() -> Result<()> {
  tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  let cli = Cli::parse();

  match cli.command {
    Command::Inspect { db, json } => {
      let stats = data::inspect_corpus(&db)?;
      if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
      } else {
        println!("DB: {db}");
        println!("Active documents: {}", stats.documents);
        println!("Precedent documents: {}", stats.precedent_documents);
        println!("Law documents: {}", stats.law_documents);
        println!("Training pairs: {}", stats.training_pairs);
      }
    }
    Command::Train {
      db,
      output,
      epochs,
      batch_size,
      lr,
      limit,
      small,
    } => {
      tracing::info!(db = %db, epochs, batch_size, lr, limit, small, "starting training");
      training::run(&db, &output, epochs, batch_size, lr, limit, small)?;
    }
    Command::Export { db, checkpoint } => {
      tracing::info!(db = %db, checkpoint = %checkpoint, "exporting embeddings");
      training::export(&db, &checkpoint)?;
    }
  }

  Ok(())
}
