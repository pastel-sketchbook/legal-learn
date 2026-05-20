//! Distillation from a llama.cpp teacher model.
//!
//! Uses `llama-embedding` to generate teacher embeddings from a large model
//! (e.g. EmbeddingGemma-300M or Qwen3-0.6B), then trains the Burn student model
//! to reproduce them via MSE loss.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::TensorData;

use crate::model::EmbeddingModelConfig;
use crate::tokenize::{self, MAX_SEQ_LEN, VOCAB_SIZE};
use crate::data;

type TrainBackend = Autodiff<burn::backend::Wgpu>;
type InnerBackend = burn::backend::Wgpu;

/// Default teacher model path (EmbeddingGemma 300M Q4)
pub const DEFAULT_TEACHER_MODEL: &str = "/Users/AD9C65/.cache/huggingface/hub/models--ggml-org--embeddinggemma-300M-qat-q4_0-GGUF/snapshots/8dd0ca2a66a8f14470acb0e2a71f801afbc5fb73/embeddinggemma-300M-qat-Q4_0.gguf";

/// Configuration for teacher embedding generation.
pub struct TeacherConfig {
    pub db_path: String,
    pub output_path: String,
    pub model_path: String,
    pub batch_size: usize,
    pub limit: Option<usize>,
}

/// Configuration for distillation training.
pub struct DistillTrainConfig {
    pub db_path: String,
    pub teacher_path: String,
    pub output_dir: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub small: bool,
}

/// A single distillation sample: text + teacher embedding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TeacherSample {
    pub text: String,
    pub embedding: Vec<f32>,
}

/// Generate teacher embeddings for all training texts using llama-embedding.
pub fn generate_teachers(cfg: &TeacherConfig) -> Result<()> {
    // Collect all unique texts from training pairs
    let pairs = data::load_pairs_limited(&cfg.db_path, cfg.limit)?;
    let mut texts: Vec<String> = Vec::new();
    for pair in &pairs {
        texts.push(pair.anchor.clone());
        texts.push(pair.positive.clone());
    }
    // Deduplicate
    texts.sort();
    texts.dedup();

    tracing::info!(texts = texts.len(), "generating teacher embeddings");

    let mut samples: Vec<TeacherSample> = Vec::new();

    // Process in batches using llama-embedding
    for chunk in texts.chunks(cfg.batch_size) {
        let embeddings = run_llama_embedding(&cfg.model_path, chunk)?;
        for (text, emb) in chunk.iter().zip(embeddings) {
            samples.push(TeacherSample {
                text: text.clone(),
                embedding: emb,
            });
        }

        if samples.len() % 1000 < cfg.batch_size {
            tracing::info!(progress = samples.len(), total = texts.len(), "generating");
        }
    }

    // Save to JSON lines file
    std::fs::create_dir_all(
        std::path::Path::new(&cfg.output_path)
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    )?;
    let file = std::fs::File::create(&cfg.output_path)
        .with_context(|| format!("creating output: {}", cfg.output_path))?;
    let mut writer = std::io::BufWriter::new(file);
    for sample in &samples {
        serde_json::to_writer(&mut writer, sample)?;
        writeln!(writer)?;
    }

    tracing::info!(
        count = samples.len(),
        path = %cfg.output_path,
        "teacher embeddings saved"
    );
    Ok(())
}

/// Run llama-embedding on a batch of texts and return their embeddings.
fn run_llama_embedding(model_path: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let separator = "<#sep#>";
    let combined = texts.join(separator);

    let output = Command::new("llama-embedding")
        .args([
            "--model",
            model_path,
            "--pooling",
            "mean",
            "--embd-normalize",
            "2",
            "--embd-output-format",
            "json",
            "--embd-separator",
            separator,
            "--prompt",
            &combined,
            "--log-disable",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("failed to run llama-embedding")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("llama-embedding failed: {stderr}");
    }

    let stdout = String::from_utf8(output.stdout).context("invalid utf8 from llama-embedding")?;

    // Parse JSON array of embedding objects
    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&stdout).context("parsing llama-embedding output")?;

    let mut embeddings = Vec::new();
    for obj in parsed {
        let emb = obj
            .get("embedding")
            .and_then(|v| v.as_array())
            .context("missing embedding field")?;
        let values: Vec<f32> = emb
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        embeddings.push(values);
    }

    Ok(embeddings)
}

/// Train the student model via distillation against teacher embeddings.
pub fn train_distill(cfg: &DistillTrainConfig) -> Result<()> {
    let device = <InnerBackend as Backend>::Device::default();

    // Load teacher embeddings
    let file = std::fs::File::open(&cfg.teacher_path)
        .with_context(|| format!("opening teacher file: {}", cfg.teacher_path))?;
    let reader = BufReader::new(file);

    let mut samples: Vec<TeacherSample> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.is_empty() {
            let sample: TeacherSample =
                serde_json::from_str(&line).context("parsing teacher sample")?;
            samples.push(sample);
        }
    }

    tracing::info!(samples = samples.len(), "loaded teacher embeddings");

    let teacher_dim = samples.first().map(|s| s.embedding.len()).unwrap_or(384);
    tracing::info!(teacher_dim, "teacher embedding dimension");

    // Tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", cfg.output_dir);
    std::fs::create_dir_all(&cfg.output_dir)?;
    let tokenizer = Arc::new(tokenize::get_tokenizer(&cfg.db_path, &tokenizer_path)?);

    // Model
    let model_config = if cfg.small {
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
    let mut model = model_config.init::<TrainBackend>(&device);

    // Optimizer
    let mut optim = AdamConfig::new().init();

    for epoch in 0..cfg.epochs {
        let mut epoch_loss = 0.0f64;
        let mut batch_count = 0usize;

        for batch_samples in samples.chunks(cfg.batch_size) {
            let batch_size = batch_samples.len();

            // Tokenize student inputs
            let mut ids_flat = Vec::with_capacity(batch_size * MAX_SEQ_LEN);
            let mut mask_flat = Vec::with_capacity(batch_size * MAX_SEQ_LEN);
            let mut teacher_flat = Vec::with_capacity(batch_size * 384);

            for sample in batch_samples {
                let (ids, mask) = tokenize::encode(&tokenizer, &sample.text);
                ids_flat.extend(ids.iter().map(|&v| v as i64));
                mask_flat.extend(mask.iter().copied());
                // Truncate or pad teacher embedding to 384 dims
                let mut emb = sample.embedding.clone();
                emb.resize(384, 0.0);
                teacher_flat.extend_from_slice(&emb[..384]);
            }

            let ids_tensor = Tensor::<TrainBackend, 1, Int>::from_data(
                TensorData::new(ids_flat, [batch_size * MAX_SEQ_LEN]),
                &device,
            )
            .reshape([batch_size, MAX_SEQ_LEN]);

            let mask_tensor = Tensor::<TrainBackend, 1, Bool>::from_data(
                TensorData::from(mask_flat.as_slice()),
                &device,
            )
            .reshape([batch_size, MAX_SEQ_LEN]);

            let teacher_tensor = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(teacher_flat, [batch_size, 384]),
                &device,
            );

            // Forward pass
            let student_emb = model.forward(ids_tensor, mask_tensor);

            // MSE loss between student and teacher
            let diff = student_emb - teacher_tensor;
            let mse = diff.powf_scalar(2.0).mean();

            // Extract loss value before backward
            let loss_value: f64 = mse.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;

            // Backward and optimize
            let grads = mse.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.lr, model, grads_params);

            epoch_loss += loss_value;
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count.max(1) as f64;
        tracing::info!(epoch = epoch + 1, avg_loss, "distillation epoch complete");
    }

    // Save model
    let final_path = format!("{}/model_distilled", cfg.output_dir);
    model
        .save_file(&final_path, &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("saving distilled model: {e}"))?;

    tracing::info!(path = %final_path, "distillation complete");
    Ok(())
}
