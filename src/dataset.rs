use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn_dataset::Dataset;

use crate::data::TextPair;
use crate::tokenize::{self, MAX_SEQ_LEN};

/// A single tokenized pair — ready for batching.
#[derive(Debug, Clone)]
pub struct TokenizedPair {
  pub anchor_ids: Vec<u32>,
  pub anchor_mask: Vec<bool>,
  pub positive_ids: Vec<u32>,
  pub positive_mask: Vec<bool>,
}

/// Dataset wrapping a Vec<TextPair>, tokenized on-the-fly.
pub struct PairDataset {
  pairs: Vec<TextPair>,
  tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
}

impl PairDataset {
  pub fn new(pairs: Vec<TextPair>, tokenizer: std::sync::Arc<tokenizers::Tokenizer>) -> Self {
    Self { pairs, tokenizer }
  }
}

impl Dataset<TokenizedPair> for PairDataset {
  fn get(&self, index: usize) -> Option<TokenizedPair> {
    let pair = self.pairs.get(index)?;
    let (anchor_ids, anchor_mask) = tokenize::encode(&self.tokenizer, &pair.anchor);
    let (positive_ids, positive_mask) = tokenize::encode(&self.tokenizer, &pair.positive);
    Some(TokenizedPair {
      anchor_ids,
      anchor_mask,
      positive_ids,
      positive_mask,
    })
  }

  fn len(&self) -> usize {
    self.pairs.len()
  }
}

/// The batched tensor input to the model.
#[derive(Debug, Clone)]
pub struct TrainingBatch<B: Backend> {
  /// Anchor token IDs: [batch, seq_len]
  pub anchor_ids: Tensor<B, 2, Int>,
  /// Anchor attention mask: [batch, seq_len]
  pub anchor_mask: Tensor<B, 2, Bool>,
  /// Positive token IDs: [batch, seq_len]
  pub positive_ids: Tensor<B, 2, Int>,
  /// Positive attention mask: [batch, seq_len]
  pub positive_mask: Tensor<B, 2, Bool>,
}

/// Batcher that collates TokenizedPairs into tensor batches.
#[derive(Clone)]
pub struct PairBatcher;

impl PairBatcher {
  pub fn new() -> Self {
    Self
  }
}

impl<B: Backend> Batcher<B, TokenizedPair, TrainingBatch<B>> for PairBatcher {
  fn batch(&self, items: Vec<TokenizedPair>, device: &B::Device) -> TrainingBatch<B> {
    let batch_size = items.len();
    let seq_len = MAX_SEQ_LEN;

    let mut anchor_ids_flat = Vec::with_capacity(batch_size * seq_len);
    let mut anchor_mask_flat = Vec::with_capacity(batch_size * seq_len);
    let mut positive_ids_flat = Vec::with_capacity(batch_size * seq_len);
    let mut positive_mask_flat = Vec::with_capacity(batch_size * seq_len);

    for item in &items {
      anchor_ids_flat.extend(item.anchor_ids.iter().map(|&v| v as i64));
      anchor_mask_flat.extend(item.anchor_mask.iter().copied());
      positive_ids_flat.extend(item.positive_ids.iter().map(|&v| v as i64));
      positive_mask_flat.extend(item.positive_mask.iter().copied());
    }

    let anchor_ids = Tensor::<B, 1, Int>::from_data(
      TensorData::new(anchor_ids_flat, [batch_size * seq_len]),
      device,
    )
    .reshape([batch_size, seq_len]);

    let anchor_mask = Tensor::<B, 1, Bool>::from_data(
      TensorData::from(anchor_mask_flat.as_slice()),
      device,
    )
    .reshape([batch_size, seq_len]);

    let positive_ids = Tensor::<B, 1, Int>::from_data(
      TensorData::new(positive_ids_flat, [batch_size * seq_len]),
      device,
    )
    .reshape([batch_size, seq_len]);

    let positive_mask = Tensor::<B, 1, Bool>::from_data(
      TensorData::from(positive_mask_flat.as_slice()),
      device,
    )
    .reshape([batch_size, seq_len]);

    TrainingBatch {
      anchor_ids,
      anchor_mask,
      positive_ids,
      positive_mask,
    }
  }
}
