use burn::prelude::*;

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
