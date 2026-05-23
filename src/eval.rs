use anyhow::{Context, Result};
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};

use crate::data;
use crate::model::EmbeddingModelConfig;
use crate::tokenize::{self, MAX_SEQ_LEN, VOCAB_SIZE};

type B = burn::backend::Wgpu;

/// Evaluate retrieval quality using held-out text pairs.
/// Reports MRR and recall@1/5/10.
pub fn evaluate(db_path: &str, checkpoint_dir: &str, small: bool, n: usize) -> Result<()> {
    let device = Device::<B>::default();

    // Load tokenizer and model
    let tokenizer_path = format!("{checkpoint_dir}/tokenizer.json");
    let tokenizer = tokenize::load_tokenizer(&tokenizer_path)?;

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
    let model: crate::model::EmbeddingModel<B> = config.init(&device);

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
    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .map_err(|e| anyhow::anyhow!("loading model: {e}"))?;
    let model = model.load_record(record);

    // Load pairs and take the last N as eval set (not seen early in training)
    let pairs = data::load_pairs(db_path)?;
    let eval_pairs = if pairs.len() > n {
        &pairs[pairs.len() - n..]
    } else {
        &pairs[..]
    };
    let eval_n = eval_pairs.len();
    tracing::info!(
        eval_n,
        total_pairs = pairs.len(),
        "evaluating on held-out pairs"
    );

    let d_model = if small { 128 } else { 384 };

    // Encode all anchors and positives in batches
    let anchor_embs = encode_batch(
        &model,
        &tokenizer,
        eval_pairs.iter().map(|p| &p.anchor),
        &device,
        d_model,
    )?;
    let positive_embs = encode_batch(
        &model,
        &tokenizer,
        eval_pairs.iter().map(|p| &p.positive),
        &device,
        d_model,
    )?;

    // Compute cosine similarity matrix [eval_n x eval_n]
    // anchor_embs and positive_embs are already L2-normalized by the model
    let mut mrr_sum = 0.0f64;
    let mut recall_at_1 = 0usize;
    let mut recall_at_5 = 0usize;
    let mut recall_at_10 = 0usize;

    for i in 0..eval_n {
        // Compute similarities of anchor[i] to all positives
        let mut sims: Vec<(usize, f32)> = (0..eval_n)
            .map(|j| {
                let dot: f32 = anchor_embs[i * d_model..(i + 1) * d_model]
                    .iter()
                    .zip(&positive_embs[j * d_model..(j + 1) * d_model])
                    .map(|(a, b)| a * b)
                    .sum();
                (j, dot)
            })
            .collect();
        // L2-normalized embeddings have no NaN, so partial_cmp always succeeds
        sims.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("no NaN in L2-normalized cosine sim")
        });

        // i is always in 0..eval_n, so it must appear in sims
        let rank = sims
            .iter()
            .position(|(j, _)| *j == i)
            .expect("query index must exist in sims")
            + 1;
        mrr_sum += 1.0 / rank as f64;
        if rank <= 1 {
            recall_at_1 += 1;
        }
        if rank <= 5 {
            recall_at_5 += 1;
        }
        if rank <= 10 {
            recall_at_10 += 1;
        }
    }

    let mrr = mrr_sum / eval_n as f64;
    let r1 = recall_at_1 as f64 / eval_n as f64;
    let r5 = recall_at_5 as f64 / eval_n as f64;
    let r10 = recall_at_10 as f64 / eval_n as f64;

    println!("Retrieval Evaluation (n={eval_n})");
    println!("  MRR:        {mrr:.4}");
    println!("  Recall@1:   {r1:.4} ({recall_at_1}/{eval_n})");
    println!("  Recall@5:   {r5:.4} ({recall_at_5}/{eval_n})");
    println!("  Recall@10:  {r10:.4} ({recall_at_10}/{eval_n})");

    Ok(())
}

fn encode_batch<'a, B2: Backend>(
    model: &crate::model::EmbeddingModel<B2>,
    tokenizer: &tokenizers::Tokenizer,
    texts: impl Iterator<Item = &'a String>,
    device: &B2::Device,
    d_model: usize,
) -> Result<Vec<f32>> {
    let texts: Vec<&String> = texts.collect();
    let n = texts.len();
    let batch_size = 64;
    let mut all_embs = Vec::with_capacity(n * d_model);

    for chunk_start in (0..n).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(n);
        let actual = chunk_end - chunk_start;

        let mut all_ids: Vec<i64> = Vec::with_capacity(actual * MAX_SEQ_LEN);
        let mut all_mask: Vec<bool> = Vec::with_capacity(actual * MAX_SEQ_LEN);
        for text in &texts[chunk_start..chunk_end] {
            let (ids, mask) = tokenize::encode(tokenizer, text);
            all_ids.extend(ids.iter().map(|&v| v as i64));
            all_mask.extend_from_slice(&mask);
        }

        let ids_tensor = Tensor::<B2, 2, Int>::from_data(
            TensorData::new(all_ids, [actual, MAX_SEQ_LEN]),
            device,
        );
        let mask_tensor = Tensor::<B2, 2, Bool>::from_data(
            TensorData::new(all_mask, [actual, MAX_SEQ_LEN]),
            device,
        );

        let embs = model.forward(ids_tensor, mask_tensor);
        let emb_data: Vec<f32> = embs.into_data().to_vec().context("extracting embeddings")?;
        all_embs.extend_from_slice(&emb_data);
    }

    Ok(all_embs)
}
