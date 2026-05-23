use std::sync::Arc;

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::optim::lr_scheduler::composed::{ComposedLrSchedulerConfig, SchedulerReduction};
use burn::optim::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::train::{Learner, SupervisedTraining};

use crate::data;
use crate::dataset::{PairBatcher, PairDataset};
use crate::model::EmbeddingModelConfig;
use crate::tokenize::{self, MAX_SEQ_LEN, VOCAB_SIZE};

/// Default backend for training (wgpu = Metal on macOS).
type TrainBackend = Autodiff<burn::backend::Wgpu>;
type InnerBackend = burn::backend::Wgpu;

/// Training run configuration.
pub struct TrainConfig {
    pub db_path: String,
    pub output_dir: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub pair_limit: Option<usize>,
    pub small: bool,
    pub warmup_steps: usize,
}

/// Run the training loop.
pub fn run(cfg: &TrainConfig) -> Result<()> {
    let TrainConfig {
        db_path,
        output_dir,
        epochs,
        batch_size,
        lr,
        pair_limit,
        small,
        warmup_steps,
    } = cfg;
    let (epochs, batch_size, lr, warmup_steps) = (*epochs, *batch_size, *lr, *warmup_steps);
    let device = Device::<InnerBackend>::default();

    // 1. Tokenizer
    let tokenizer_path = format!("{output_dir}/tokenizer.json");
    std::fs::create_dir_all(output_dir.as_str())?;
    let tokenizer = Arc::new(tokenize::get_tokenizer(db_path, &tokenizer_path)?);

    // 2. Load pairs and split train/valid (90/10)
    let mut pairs = data::load_pairs_limited(db_path, *pair_limit)?;
    let split = (pairs.len() as f64 * 0.9) as usize;
    let valid_pairs = pairs.split_off(split);
    let train_pairs = pairs;
    let train_pairs_len = train_pairs.len();

    tracing::info!(
        train = train_pairs.len(),
        valid = valid_pairs.len(),
        "dataset split"
    );

    // 3. Datasets
    let train_dataset = PairDataset::new(train_pairs, tokenizer.clone());
    let valid_dataset = PairDataset::new(valid_pairs, tokenizer);

    // 4. Dataloaders
    let batcher = PairBatcher::new();

    let dataloader_train = DataLoaderBuilder::<TrainBackend, _, _>::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::<InnerBackend, _, _>::new(batcher)
        .batch_size(batch_size)
        .build(valid_dataset);

    // 5. Model
    let config = if *small {
        tracing::info!("using small model (2 layers, 128 dim)");
        EmbeddingModelConfig {
            vocab_size: VOCAB_SIZE,
            max_seq_len: MAX_SEQ_LEN,
            d_model: 128,
            n_layers: 2,
            n_heads: 4,
            d_ff: 512,
            dropout: 0.1,
        }
    } else {
        EmbeddingModelConfig {
            vocab_size: VOCAB_SIZE,
            max_seq_len: MAX_SEQ_LEN,
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            d_ff: 1536,
            dropout: 0.1,
        }
    };
    let model = config.init::<TrainBackend>(&device);

    // 6. Optimizer + LR scheduler
    let optim = AdamConfig::new().init();

    let iters_per_epoch = train_pairs_len.div_ceil(batch_size);
    let total_iters = epochs * iters_per_epoch;

    let scheduler = if warmup_steps > 0 {
        tracing::info!(
            warmup_steps,
            total_iters,
            "using linear warmup + cosine annealing"
        );
        let cosine_iters = total_iters.saturating_sub(warmup_steps).max(1);
        ComposedLrSchedulerConfig::new()
            .with_reduction(SchedulerReduction::Prod)
            .linear(LinearLrSchedulerConfig::new(1e-7, 1.0, warmup_steps))
            .cosine(CosineAnnealingLrSchedulerConfig::new(lr, cosine_iters))
            .init()
            .map_err(|e| anyhow::anyhow!("scheduler config: {e}"))?
    } else {
        tracing::info!(total_iters, "using cosine annealing (no warmup)");
        ComposedLrSchedulerConfig::new()
            .cosine(CosineAnnealingLrSchedulerConfig::new(
                lr,
                total_iters.max(1),
            ))
            .init()
            .map_err(|e| anyhow::anyhow!("scheduler config: {e}"))?
    };

    // 7. Learner
    let learner = Learner::new(model, optim, scheduler);

    // 8. Supervised training
    let result = SupervisedTraining::new(output_dir, dataloader_train, dataloader_valid)
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(epochs)
        .summary()
        .launch(learner);

    // 9. Save final model
    let trained_model = result.model;
    let final_path = format!("{output_dir}/model_final");
    trained_model
        .save_file(&final_path, &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("saving model: {e}"))?;

    tracing::info!(path = %final_path, "training complete, model saved");
    Ok(())
}

/// Export embeddings from a trained checkpoint back into data.db.
pub fn export(db_path: &str, checkpoint_dir: &str, small: bool) -> Result<()> {
    let device = Device::<InnerBackend>::default();

    // Load tokenizer
    let tokenizer_path = format!("{checkpoint_dir}/tokenizer.json");
    let tokenizer = tokenize::load_tokenizer(&tokenizer_path)?;

    // Load model
    let config = if small {
        EmbeddingModelConfig {
            vocab_size: VOCAB_SIZE,
            max_seq_len: MAX_SEQ_LEN,
            d_model: 128,
            n_layers: 2,
            n_heads: 4,
            d_ff: 512,
            dropout: 0.1,
        }
    } else {
        EmbeddingModelConfig {
            vocab_size: VOCAB_SIZE,
            max_seq_len: MAX_SEQ_LEN,
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            d_ff: 1536,
            dropout: 0.1,
        }
    };
    let model: crate::model::EmbeddingModel<InnerBackend> = config.init(&device);

    // Try model_final first, then model_distilled
    let model_path = {
        let final_path = format!("{checkpoint_dir}/model_final");
        let distilled_path = format!("{checkpoint_dir}/model_distilled");
        if std::path::Path::new(&format!("{final_path}.mpk")).exists()
            || std::path::Path::new(&format!("{final_path}.mpk.gz")).exists()
        {
            final_path
        } else {
            distilled_path
        }
    };
    tracing::info!(path = %model_path, "loading model checkpoint");
    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .map_err(|e| anyhow::anyhow!("loading model: {e}"))?;
    let model = model.load_record(record);

    // Load sqlite-vec extension for vec0 virtual table support
    // SAFETY: sqlite3_vec_init has the correct signature for sqlite3_auto_extension.
    // The transmute converts the function pointer to the expected Option<extern "C" fn()> type.
    unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }

    // Open database
    let conn = rusqlite::Connection::open(db_path)
        .with_context(|| format!("opening database for export: {db_path}"))?;

    // Iterate all documents and compute embeddings
    let mut stmt = conn.prepare(
        "SELECT d.hash, c.doc FROM content c
     JOIN documents d ON d.hash = c.hash
     WHERE d.active = 1",
    )?;

    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(Result::ok)
        .collect();

    tracing::info!(documents = rows.len(), "computing embeddings");

    let model_name = "legal-learn-v1";
    let now = chrono::Utc::now().to_rfc3339();

    // Clear old embeddings from this model
    conn.execute("DELETE FROM content_vectors WHERE model = ?1", [model_name])?;
    conn.execute(
        "DELETE FROM content_vectors_idx WHERE model = ?1",
        [model_name],
    )?;

    conn.execute_batch("BEGIN")?;

    let mut insert_stmt = conn.prepare(
        "INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedding, embedded_at)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    let mut insert_idx_stmt = conn.prepare(
        "INSERT INTO content_vectors_idx (embedding, hash, model, seq, pos)
     VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;

    let batch_size = 64;
    for chunk_start in (0..rows.len()).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(rows.len());
        let chunk = &rows[chunk_start..chunk_end];
        let actual_batch = chunk.len();

        // Tokenize batch
        let mut all_ids: Vec<i64> = Vec::with_capacity(actual_batch * MAX_SEQ_LEN);
        let mut all_mask: Vec<bool> = Vec::with_capacity(actual_batch * MAX_SEQ_LEN);
        for (_hash, doc) in chunk {
            let (ids, mask) = tokenize::encode(&tokenizer, doc);
            all_ids.extend(ids.iter().map(|&v| v as i64));
            all_mask.extend_from_slice(&mask);
        }

        let ids_tensor = Tensor::<InnerBackend, 2, Int>::from_data(
            TensorData::new(all_ids, [actual_batch, MAX_SEQ_LEN]),
            &device,
        );
        let mask_tensor = Tensor::<InnerBackend, 2, Bool>::from_data(
            TensorData::new(all_mask, [actual_batch, MAX_SEQ_LEN]),
            &device,
        );

        // Forward pass for entire batch → [batch, d_model]
        let embeddings = model.forward(ids_tensor, mask_tensor);
        let emb_data: Vec<f32> = embeddings
            .into_data()
            .to_vec()
            .context("extracting embedding tensor data")?;

        let d_model = if small { 128 } else { 384 };
        for (j, (hash, _doc)) in chunk.iter().enumerate() {
            let emb = &emb_data[j * d_model..(j + 1) * d_model];
            let emb_json = serde_json::to_string(emb)?;

            insert_stmt.execute(rusqlite::params![hash, 0, 0, model_name, emb_json, now])?;

            let emb_bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
            insert_idx_stmt.execute(rusqlite::params![emb_bytes, hash, model_name, 0, 0])?;
        }

        if chunk_end % 10000 < batch_size {
            tracing::info!(
                progress = chunk_end,
                total = rows.len(),
                "embedding documents"
            );
        }
    }

    conn.execute_batch("COMMIT")?;
    tracing::info!("export complete: {} embeddings written", rows.len());
    Ok(())
}
