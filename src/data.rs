use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Serialize;

/// A training pair extracted from precedent data.
/// anchor = 판결요지 (ruling summary), positive = 참조조문 section or 사건명.
#[derive(Debug, Clone)]
pub struct TextPair {
    pub anchor: String,
    pub positive: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorpusStats {
    pub documents: usize,
    pub precedent_documents: usize,
    pub law_documents: usize,
    pub training_pairs: usize,
}

/// Load (판결요지, 참조조문/사건명) pairs from precedents in data.db.
pub fn load_pairs(db_path: &str) -> Result<Vec<TextPair>> {
    load_pairs_limited(db_path, None)
}

/// Load precedent-derived training pairs with an optional cap for fast debugging.
pub fn load_pairs_limited(db_path: &str, limit: Option<usize>) -> Result<Vec<TextPair>> {
    let conn = Connection::open(db_path).with_context(|| format!("opening database: {db_path}"))?;

    let mut stmt = conn.prepare(
        "SELECT c.doc FROM content c
     JOIN documents d ON d.hash = c.hash
     WHERE d.collection = 'precedents' AND d.active = 1",
    )?;

    let mut pairs = Vec::new();

    let rows = stmt.query_map([], |row| {
        let doc: String = row.get(0)?;
        Ok(doc)
    })?;

    for row in rows {
        let doc = row?;
        if let Some(pair) = extract_pair(&doc) {
            pairs.push(pair);
            if limit.is_some_and(|limit| pairs.len() >= limit) {
                break;
            }
        }
    }

    tracing::info!(count = pairs.len(), "loaded training pairs");
    Ok(pairs)
}

pub fn inspect_corpus(db_path: &str) -> Result<CorpusStats> {
    let conn = Connection::open(db_path).with_context(|| format!("opening database: {db_path}"))?;

    let documents = count_active_documents(&conn, None)?;
    let precedent_documents = count_active_documents(&conn, Some("precedents"))?;
    let law_documents = count_active_documents(&conn, Some("laws"))?;
    let training_pairs = load_pairs(db_path)?.len();

    Ok(CorpusStats {
        documents,
        precedent_documents,
        law_documents,
        training_pairs,
    })
}

fn count_active_documents(conn: &Connection, collection: Option<&str>) -> Result<usize> {
    let sql = if collection.is_some() {
        "SELECT COUNT(*) FROM documents WHERE active = 1 AND collection = ?1"
    } else {
        "SELECT COUNT(*) FROM documents WHERE active = 1"
    };

    let count = match collection {
        Some(collection) => conn.query_row(sql, [collection], |row| row.get(0))?,
        None => conn.query_row(sql, [], |row| row.get(0))?,
    };

    Ok(count)
}

/// Extract structured fields from a precedent markdown document.
fn extract_pair(doc: &str) -> Option<TextPair> {
    let summary = extract_section(doc, "판결요지")?;
    // Prefer 참조조문 as the positive; fall back to 사건명 from frontmatter
    let positive =
        extract_section(doc, "참조조문").or_else(|| extract_frontmatter_field(doc, "사건명"))?;

    if summary.len() < 20 || positive.len() < 5 {
        return None;
    }

    Some(TextPair {
        anchor: summary,
        positive,
    })
}

/// Extract text under a `## heading` section.
fn extract_section(doc: &str, heading: &str) -> Option<String> {
    let marker = format!("## {heading}");
    let start = doc.find(&marker)?;
    let after = start + marker.len();
    let rest = &doc[after..];

    // Take text until the next heading or end
    let end = rest.find("\n## ").unwrap_or(rest.len());
    let text = rest[..end].trim();

    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Extract a YAML frontmatter field value.
fn extract_frontmatter_field(doc: &str, field: &str) -> Option<String> {
    if !doc.starts_with("---") {
        return None;
    }
    let end = doc[3..].find("---")?;
    let frontmatter = &doc[3..3 + end];

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(field) {
            let rest = rest.trim_start_matches(':').trim();
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    None
}
