use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::RegressionOutput;
use burn::train::{TrainOutput, TrainStep, InferenceStep};

use crate::dataset::TrainingBatch;
use crate::loss;

/// Embedding dimension — must match legal-ko's vec0 index (384).
pub const EMBED_DIM: usize = 384;

/// Temperature for InfoNCE loss.
const TAU: f64 = 0.07;

/// Small transformer encoder for sentence embeddings.
#[derive(Module, Debug)]
pub struct EmbeddingModel<B: Backend> {
  token_embedding: Embedding<B>,
  position_embedding: Embedding<B>,
  encoder: TransformerEncoder<B>,
  norm: LayerNorm<B>,
  projection: Linear<B>,
}

#[derive(Config, Debug)]
pub struct EmbeddingModelConfig {
  #[config(default = 32000)]
  pub vocab_size: usize,
  #[config(default = 512)]
  pub max_seq_len: usize,
  #[config(default = 384)]
  pub d_model: usize,
  #[config(default = 6)]
  pub n_layers: usize,
  #[config(default = 6)]
  pub n_heads: usize,
  #[config(default = 1536)]
  pub d_ff: usize,
  #[config(default = 0.1)]
  pub dropout: f64,
}

impl EmbeddingModelConfig {
  pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingModel<B> {
    let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
    let position_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);

    let encoder = TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_heads, self.n_layers)
      .with_dropout(self.dropout)
      .init(device);

    let norm = LayerNormConfig::new(self.d_model).init(device);
    let projection = LinearConfig::new(self.d_model, EMBED_DIM).init(device);

    EmbeddingModel {
      token_embedding,
      position_embedding,
      encoder,
      norm,
      projection,
    }
  }
}

impl<B: Backend> EmbeddingModel<B> {
  /// Forward pass: token IDs → 384-dim normalized embedding.
  pub fn forward(
    &self,
    input: Tensor<B, 2, Int>,
    mask: Tensor<B, 2, Bool>,
  ) -> Tensor<B, 2> {
    let [batch, seq_len] = input.dims();
    let device = input.device();

    let tok_emb = self.token_embedding.forward(input);
    let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
      .unsqueeze::<2>()
      .expand([batch, seq_len]);
    let pos_emb = self.position_embedding.forward(positions);
    let hidden = tok_emb + pos_emb;

    let encoder_input = TransformerEncoderInput::new(hidden)
      .mask_pad(mask.clone());
    let encoded = self.encoder.forward(encoder_input);

    let d_model = self.norm.gamma.dims()[0];
    let mask_f: Tensor<B, 3> = mask.float().unsqueeze_dim::<3>(2).expand([batch, seq_len, d_model]);
    let summed: Tensor<B, 2> = (encoded * mask_f.clone()).sum_dim(1).squeeze_dim::<2>(1);
    let counts: Tensor<B, 2> = mask_f.sum_dim(1).squeeze_dim::<2>(1).clamp_min(1e-9);
    let pooled = summed / counts;

    let normed = self.norm.forward(pooled);
    let projected = self.projection.forward(normed);

    let norms: Tensor<B, 2> = projected.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-9);
    projected / norms
  }

  /// Forward pass for a contrastive pair, returning the loss.
  pub fn forward_pair(&self, batch: TrainingBatch<B>) -> Tensor<B, 1> {
    let anchor_emb = self.forward(batch.anchor_ids, batch.anchor_mask);
    let positive_emb = self.forward(batch.positive_ids, batch.positive_mask);
    loss::info_nce_loss(anchor_emb, positive_emb, TAU)
  }
}

// --- TrainStep ---

impl<B: AutodiffBackend> TrainStep for EmbeddingModel<B> {
  type Input = TrainingBatch<B>;
  type Output = RegressionOutput<B>;

  fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
    let loss = self.forward_pair(batch);
    let grads = loss.backward();
    let output = RegressionOutput::new(
      loss.clone().detach(),
      loss.clone().detach().unsqueeze(),
      loss.detach().unsqueeze(),
    );
    TrainOutput::new(self, grads, output)
  }
}

// --- InferenceStep ---

impl<B: Backend> InferenceStep for EmbeddingModel<B> {
  type Input = TrainingBatch<B>;
  type Output = RegressionOutput<B>;

  fn step(&self, batch: Self::Input) -> Self::Output {
    let loss = self.forward_pair(batch);
    let detached = loss.detach();
    RegressionOutput::new(detached.clone(), detached.clone().unsqueeze(), detached.unsqueeze())
  }
}
