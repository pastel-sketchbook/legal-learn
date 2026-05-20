use burn::prelude::*;

/// Matryoshka dimension levels for nested representation learning.
pub const MATRYOSHKA_DIMS: &[usize] = &[384, 256, 128, 64];

/// InfoNCE contrastive loss.
///
/// Given anchor embeddings `a` [B, D] and positive embeddings `p` [B, D],
/// computes the cosine similarity matrix [B, B] and uses cross-entropy
/// where the diagonal entries are the positive pairs.
///
/// Temperature `tau` controls the sharpness (default: 0.07).
pub fn info_nce_loss<B: Backend>(
    anchors: Tensor<B, 2>,
    positives: Tensor<B, 2>,
    tau: f64,
) -> Tensor<B, 1> {
    let [batch_size, _dim] = anchors.dims();

    // Cosine similarity matrix: [B, B]
    // anchors and positives are already L2-normalized by the model
    let sim_matrix = anchors.matmul(positives.transpose());

    // Scale by temperature
    let logits = sim_matrix / tau;

    // Build one-hot target probabilities: identity matrix [B, B]
    // Diagonal = 1.0, off-diagonal = 0.0
    let targets = Tensor::<B, 2>::eye(batch_size, &logits.device());

    // Cross-entropy with soft targets
    burn::tensor::loss::cross_entropy_with_logits(logits, targets)
}

/// Matryoshka InfoNCE loss: average InfoNCE across multiple dimension truncations.
///
/// The embedding model produces 384-dim outputs. We truncate to each level in
/// `MATRYOSHKA_DIMS`, re-normalize, compute InfoNCE, and average. This trains
/// prefix subsets of dimensions to be independently useful.
pub fn matryoshka_loss<B: Backend>(
    anchors: &Tensor<B, 2>,
    positives: &Tensor<B, 2>,
    tau: f64,
) -> Tensor<B, 1> {
    let device = anchors.device();
    let [_batch, full_dim] = anchors.dims();

    let mut total_loss: Option<Tensor<B, 1>> = None;
    let mut count = 0usize;

    for &dim in MATRYOSHKA_DIMS {
        if dim > full_dim {
            continue;
        }

        // Truncate to first `dim` dimensions: [B, dim]
        let a = anchors.clone().narrow(1, 0, dim);
        let p = positives.clone().narrow(1, 0, dim);

        // Re-normalize after truncation
        let a_norm = a.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-9);
        let a = a / a_norm;
        let p_norm = p.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-9);
        let p = p / p_norm;

        let loss = info_nce_loss(a, p, tau);
        total_loss = Some(match total_loss {
            Some(acc) => acc + loss,
            None => loss,
        });
        count += 1;
    }

    let total = total_loss.unwrap_or_else(|| Tensor::zeros([1], &device));
    total / (count as f64)
}
