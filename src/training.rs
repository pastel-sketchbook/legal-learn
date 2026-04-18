use std::sync::Arc;

use anyhow::Result;
use burn::backend::Autodiff;
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};

use crate::data;
use crate::dataset::{PairBatcher, PairDataset};
use crate::model::EmbeddingModelConfig;
use crate::tokenize::{self, MAX_SEQ_LEN, VOCAB_SIZE};

/// Default backend for training.
type TrainBackend = Autodiff<burn::backend::NdArray>;
type InnerBackend = burn::backend::NdArray;

/// Run the training loop.
pub fn run(
  db_path: &str,
  output_dir: &str,
  epochs: usize,
  batch_size: usize,
  lr: f64,
) -> Result<()> {
  let device = <InnerBackend as Backend>::Device::default();

  // 1. Tokenizer
  let tokenizer_path = format!("{output_dir}/tokenizer.json");
  std::fs::create_dir_all(output_dir)?;
  let tokenizer = Arc::new(tokenize::get_tokenizer(db_path, &tokenizer_path)?);

  // 2. Load pairs and split train/valid (90/10)
  let mut pairs = data::load_pairs(db_path)?;
  let split = (pairs.len() as f64 * 0.9) as usize;
  let valid_pairs = pairs.split_off(split);
  let train_pairs = pairs;

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
  let config = EmbeddingModelConfig {
    vocab_size: VOCAB_SIZE,
    max_seq_len: MAX_SEQ_LEN,
    d_model: 384,
    n_layers: 6,
    n_heads: 6,
    d_ff: 1536,
    dropout: 0.1,
  };
  let model = config.init::<TrainBackend>(&device);

  // 6. Optimizer + scheduler
  let optim = AdamConfig::new().init();

  // 7. Learner
  let learner = Learner::new(model, optim, lr);

  // 8. Supervised training
  let result = SupervisedTraining::new(output_dir, dataloader_train, dataloader_valid)
    .metric_train_numeric(LossMetric::<InnerBackend>::new())
    .metric_valid_numeric(LossMetric::<InnerBackend>::new())
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
pub fn export(db_path: &str, checkpoint_dir: &str) -> Result<()> {
  let device = <InnerBackend as Backend>::Device::default();

  // Load tokenizer
  let tokenizer_path = format!("{checkpoint_dir}/tokenizer.json");
  let tokenizer = tokenize::load_tokenizer(&tokenizer_path)?;

  // Load model
  let config = EmbeddingModelConfig {
    vocab_size: VOCAB_SIZE,
    max_seq_len: MAX_SEQ_LEN,
    d_model: 384,
    n_layers: 6,
    n_heads: 6,
    d_ff: 1536,
    dropout: 0.1,
  };
  let model: crate::model::EmbeddingModel<InnerBackend> = config.init(&device);

  let model_path = format!("{checkpoint_dir}/model_final");
  let record = CompactRecorder::new()
    .load(model_path.into(), &device)
    .map_err(|e| anyhow::anyhow!("loading model: {e}"))?;
  let model = model.load_record(record);

  // Open database
  let conn = rusqlite::Connection::open(db_path)?;

  // Iterate all documents and compute embeddings
  let mut stmt = conn.prepare(
    "SELECT d.hash, c.doc FROM content c
     JOIN documents d ON d.hash = c.hash
     WHERE d.active = 1",
  )?;

  let rows: Vec<(String, String)> = stmt
    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
    .filter_map(|r| r.ok())
    .collect();

  tracing::info!(documents = rows.len(), "computing embeddings");

  let model_name = "legal-learn-v1";
  let now = chrono::Utc::now().to_rfc3339();

  // Clear old embeddings from this model
  conn.execute(
    "DELETE FROM content_vectors WHERE model = ?1",
    [model_name],
  )?;

  let mut insert_stmt = conn.prepare(
    "INSERT INTO content_vectors (hash, seq, pos, model, embedding, embedded_at)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
  )?;

  for (i, (hash, doc)) in rows.iter().enumerate() {
    let (ids, mask) = tokenize::encode(&tokenizer, doc);

    let ids_tensor = Tensor::<InnerBackend, 1, Int>::from_data(
      TensorData::new(ids.iter().map(|&v| v as i64).collect::<Vec<_>>(), [MAX_SEQ_LEN]),
      &device,
    )
    .unsqueeze::<2>();

    let mask_tensor = Tensor::<InnerBackend, 1, Bool>::from_data(
      TensorData::from(mask.as_slice()),
      &device,
    )
    .unsqueeze::<2>();

    let embedding = model.forward(ids_tensor, mask_tensor);
    let emb_data: Vec<f32> = embedding
      .squeeze::<1>()
      .into_data()
      .to_vec()
      .unwrap();

    let emb_json = serde_json::to_string(&emb_data)?;

    insert_stmt.execute(rusqlite::params![hash, 0, 0, model_name, emb_json, now])?;

    if (i + 1) % 1000 == 0 {
      tracing::info!(progress = i + 1, total = rows.len(), "embedding documents");
    }
  }

  tracing::info!("export complete: {} embeddings written", rows.len());
  Ok(())
}
