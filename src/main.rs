#![recursion_limit = "256"]

mod data;
mod dataset;
mod distill;
mod eval;
mod loss;
mod model;
mod tokenize;
mod training;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "legal-learn",
    about = "Train Korean legal embeddings with Burn"
)]
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

        /// Number of linear warmup steps (0 = no warmup, cosine only)
        #[arg(long, default_value_t = 0)]
        warmup_steps: usize,
    },

    /// Export trained embeddings back into data.db
    Export {
        /// Path to .qmd/data.db
        #[arg(long, default_value = "../legal-ko/.qmd/data.db")]
        db: String,

        /// Checkpoint directory to load model from
        #[arg(long)]
        checkpoint: String,

        /// Use small model config (must match training)
        #[arg(long)]
        small: bool,
    },

    /// Generate teacher embeddings using llama-embedding
    DistillGenerate {
        /// Path to .qmd/data.db
        #[arg(long, default_value = "../legal-ko/.qmd/data.db")]
        db: String,

        /// Output JSONL file for teacher embeddings
        #[arg(long, default_value = "checkpoints/teacher_embeddings.jsonl")]
        output: String,

        /// Path to GGUF model for llama-embedding
        #[arg(long)]
        model: Option<String>,

        /// Batch size for llama-embedding calls
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        /// Limit number of training pairs
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Train student model via distillation from teacher embeddings
    DistillTrain {
        /// Path to .qmd/data.db (for tokenizer training)
        #[arg(long, default_value = "../legal-ko/.qmd/data.db")]
        db: String,

        /// Path to teacher embeddings JSONL
        #[arg(long, default_value = "checkpoints/teacher_embeddings.jsonl")]
        teacher: String,

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

        /// Use a small model
        #[arg(long)]
        small: bool,
    },

    /// Evaluate retrieval quality (MRR, recall@k) on held-out pairs
    Eval {
        /// Path to .qmd/data.db
        #[arg(long, default_value = "../legal-ko/.qmd/data.db")]
        db: String,

        /// Checkpoint directory to load model from
        #[arg(long, default_value = "checkpoints")]
        checkpoint: String,

        /// Use small model config
        #[arg(long)]
        small: bool,

        /// Number of eval pairs (sampled from tail of dataset)
        #[arg(long, default_value_t = 100)]
        n: usize,
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
            warmup_steps,
        } => {
            tracing::info!(db = %db, epochs, batch_size, lr, limit, small, warmup_steps, "starting training");
            training::run(&training::TrainConfig {
                db_path: db,
                output_dir: output,
                epochs,
                batch_size,
                lr,
                pair_limit: limit,
                small,
                warmup_steps,
            })?;
        }
        Command::Export {
            db,
            checkpoint,
            small,
        } => {
            tracing::info!(db = %db, checkpoint = %checkpoint, small, "exporting embeddings");
            training::export(&db, &checkpoint, small)?;
        }
        Command::DistillGenerate {
            db,
            output,
            model,
            batch_size,
            limit,
        } => {
            let model_path = model.unwrap_or_else(distill::default_teacher_model);
            tracing::info!(db = %db, model = %model_path, "generating teacher embeddings");
            distill::generate_teachers(&distill::TeacherConfig {
                db_path: db,
                output_path: output,
                model_path,
                batch_size,
                limit,
            })?;
        }
        Command::DistillTrain {
            db,
            teacher,
            output,
            epochs,
            batch_size,
            lr,
            small,
        } => {
            tracing::info!(teacher = %teacher, epochs, "starting distillation training");
            distill::train_distill(&distill::DistillTrainConfig {
                db_path: db,
                teacher_path: teacher,
                output_dir: output,
                epochs,
                batch_size,
                lr,
                small,
            })?;
        }
        Command::Eval {
            db,
            checkpoint,
            small,
            n,
        } => {
            tracing::info!(checkpoint = %checkpoint, n, small, "evaluating retrieval quality");
            eval::evaluate(&db, &checkpoint, small, n)?;
        }
    }

    Ok(())
}
