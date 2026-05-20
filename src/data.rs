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

/// Load training pairs from all collections with an optional cap for fast debugging.
pub fn load_pairs_limited(db_path: &str, limit: Option<usize>) -> Result<Vec<TextPair>> {
    let conn = Connection::open(db_path).with_context(|| format!("opening database: {db_path}"))?;

    let mut pairs = Vec::new();

    // 1. Precedent pairs: 판결요지 → 참조조문/사건명
    load_precedent_pairs(&conn, &mut pairs, limit)?;
    let precedent_count = pairs.len();

    // 2. Law pairs: article heading → article body
    let law_limit = limit.map(|l| l.saturating_sub(pairs.len()));
    if law_limit != Some(0) {
        load_law_pairs(&conn, &mut pairs, law_limit)?;
    }

    tracing::info!(
        precedent = precedent_count,
        law = pairs.len() - precedent_count,
        total = pairs.len(),
        "loaded training pairs"
    );
    Ok(pairs)
}

fn load_precedent_pairs(
    conn: &Connection,
    pairs: &mut Vec<TextPair>,
    limit: Option<usize>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT c.doc FROM content c
     JOIN documents d ON d.hash = c.hash
     WHERE d.collection = 'precedents' AND d.active = 1",
    )?;

    let rows = stmt.query_map([], |row| {
        let doc: String = row.get(0)?;
        Ok(doc)
    })?;

    for row in rows {
        let doc = row?;
        if let Some(pair) = extract_precedent_pair(&doc) {
            pairs.push(pair);
            if limit.is_some_and(|l| pairs.len() >= l) {
                break;
            }
        }
    }
    Ok(())
}

fn load_law_pairs(
    conn: &Connection,
    pairs: &mut Vec<TextPair>,
    limit: Option<usize>,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT c.doc FROM content c
     JOIN documents d ON d.hash = c.hash
     WHERE d.collection = 'laws' AND d.active = 1",
    )?;

    let rows = stmt.query_map([], |row| {
        let doc: String = row.get(0)?;
        Ok(doc)
    })?;

    for row in rows {
        let doc = row?;
        let law_pairs = extract_law_pairs(&doc);
        for pair in law_pairs {
            pairs.push(pair);
            if limit.is_some_and(|l| pairs.len() >= l) {
                return Ok(());
            }
        }
    }
    Ok(())
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
fn extract_precedent_pair(doc: &str) -> Option<TextPair> {
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

/// Extract (article heading, article body) pairs from a law document.
/// Each `##### 제N조 (title)` becomes a pair: title → body.
fn extract_law_pairs(doc: &str) -> Vec<TextPair> {
    let mut pairs = Vec::new();
    let title = extract_frontmatter_field(doc, "제목").unwrap_or_default();

    // Find article headings: ##### 제N조 (...)
    let lines: Vec<&str> = doc.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("##### 제") && line.contains('조') {
            let heading = line.trim_start_matches('#').trim().to_string();
            // Collect body until next heading
            let mut body = String::new();
            i += 1;
            while i < lines.len() {
                let next = lines[i];
                if next.trim().starts_with('#') {
                    break;
                }
                if !next.trim().is_empty() {
                    if !body.is_empty() {
                        body.push(' ');
                    }
                    body.push_str(next.trim());
                }
                i += 1;
            }

            if heading.len() >= 5 && body.len() >= 20 {
                // Pair 1: article heading → article body
                pairs.push(TextPair {
                    anchor: heading.clone(),
                    positive: body.clone(),
                });
                // Pair 2: law title → article heading (cross-reference)
                if !title.is_empty() {
                    pairs.push(TextPair {
                        anchor: title.clone(),
                        positive: heading,
                    });
                }
            }
        } else {
            i += 1;
        }
    }
    pairs
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
