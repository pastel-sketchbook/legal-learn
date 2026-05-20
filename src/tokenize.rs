use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::models::bpe::trainer::BpeTrainer;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::Trainer;
use tokenizers::{AddedToken, Tokenizer};

pub const PAD_TOKEN: &str = "[PAD]";
pub const UNK_TOKEN: &str = "[UNK]";
pub const PAD_ID: u32 = 0;
pub const VOCAB_SIZE: usize = 32000;
pub const MAX_SEQ_LEN: usize = 512;

/// Train a BPE tokenizer from the legal corpus and save it.
pub fn train_tokenizer(db_path: &str, output_path: &str) -> Result<Tokenizer> {
    tracing::info!("collecting corpus texts from {db_path}");

    let conn = rusqlite::Connection::open(db_path)
        .with_context(|| format!("opening database: {db_path}"))?;

    let mut stmt = conn.prepare(
        "SELECT c.doc FROM content c
     JOIN documents d ON d.hash = c.hash
     WHERE d.active = 1",
    )?;

    let mut texts: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    // Cap corpus size for tokenizer training (BPE merge is expensive on large text volume)
    const MAX_TOKENIZER_DOCS: usize = 2_000;
    const MAX_DOC_CHARS: usize = 1_000;
    if texts.len() > MAX_TOKENIZER_DOCS {
        tracing::info!(
            original = texts.len(),
            sampled = MAX_TOKENIZER_DOCS,
            "sampling corpus for tokenizer training"
        );
        let step = texts.len() / MAX_TOKENIZER_DOCS;
        texts = texts.into_iter().step_by(step).take(MAX_TOKENIZER_DOCS).collect();
    }
    // Truncate each doc to limit total token volume
    texts.iter_mut().for_each(|t| {
        if t.len() > MAX_DOC_CHARS {
            *t = t.chars().take(MAX_DOC_CHARS).collect();
        }
    });

    tracing::info!(docs = texts.len(), "training BPE tokenizer");

    let mut model = BPE::default();
    let mut trainer = BpeTrainer::builder()
        .vocab_size(VOCAB_SIZE)
        .special_tokens(vec![
            AddedToken::from(PAD_TOKEN, true),
            AddedToken::from(UNK_TOKEN, true),
        ])
        .build();

    trainer
        .feed(texts.iter(), |text| Ok(vec![text.to_string()]))
        .map_err(|e| anyhow::anyhow!("BPE trainer feed failed: {e}"))?;
    let special_tokens = trainer
        .train(&mut model)
        .map_err(|e| anyhow::anyhow!("BPE training failed: {e}"))?;

    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(Some(NFC));
    tokenizer.with_pre_tokenizer(Some(Whitespace));

    tokenizer.add_special_tokens(&special_tokens);

    // Add post-processing (optional: CLS/SEP not needed for embeddings)
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_SEQ_LEN,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(MAX_SEQ_LEN),
        pad_id: PAD_ID,
        pad_token: PAD_TOKEN.to_string(),
        ..Default::default()
    }));

    tokenizer
        .save(output_path, true)
        .map_err(|e| anyhow::anyhow!("saving tokenizer: {e}"))?;

    tracing::info!(path = output_path, "tokenizer saved");
    Ok(tokenizer)
}

/// Load a previously trained tokenizer.
pub fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    let tok = Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {path}: {e}"))?;
    Ok(tok)
}

/// Load or train the tokenizer.
pub fn get_tokenizer(db_path: &str, tokenizer_path: &str) -> Result<Tokenizer> {
    if Path::new(tokenizer_path).exists() {
        tracing::info!(path = tokenizer_path, "loading existing tokenizer");
        load_tokenizer(tokenizer_path)
    } else {
        tracing::info!("no tokenizer found, training from corpus");
        train_tokenizer(db_path, tokenizer_path)
    }
}

/// Tokenize text and return (token_ids, attention_mask).
/// Both are vectors of length MAX_SEQ_LEN.
pub fn encode(tokenizer: &Tokenizer, text: &str) -> (Vec<u32>, Vec<bool>) {
    // The tokenizer is pre-validated at load/train time; encode only fails on
    // internal bugs (e.g. missing normalizer), so panic is acceptable here.
    let encoding = tokenizer
        .encode(text, true)
        .expect("tokenization should not fail for a validated tokenizer");

    let ids = encoding.get_ids().to_vec();
    let mask: Vec<bool> = encoding
        .get_attention_mask()
        .iter()
        .map(|&v| v == 1)
        .collect();

    (ids, mask)
}
