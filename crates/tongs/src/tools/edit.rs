//! The edit tool: targeted exact-text replacements with a fuzzy fallback.
//!
//! The matching/application core is pure, ported from TS Pi's `edit-diff.ts`:
//! exact match first; if that fails, both content and needle are normalized
//! (per-line trailing whitespace stripped, smart quotes / Unicode dashes /
//! special spaces mapped to ASCII) and matched in that space. Every edit must
//! match exactly once; edits must not overlap; replacements apply in reverse
//! offset order; the result must differ. Line endings (and a UTF-8 BOM) are
//! detected up front and restored on write.
//!
//! Divergence from TS: no NFKC normalization pass (Rust std has no Unicode
//! normalization; the quote/dash/space maps cover the practical cases).

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use super::support::resolve_path;
use super::{Tool, ToolEffects, ToolOutput, ToolUpdate};
use crate::{Error, Result};

pub(crate) struct EditTool {
    cwd: PathBuf,
}

impl EditTool {
    pub(crate) fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Edit {
    #[serde(rename = "oldText")]
    pub old_text: String,
    #[serde(rename = "newText")]
    pub new_text: String,
}

// ---------------------------------------------------------------------------
// Pure core.
// ---------------------------------------------------------------------------

pub(crate) fn detect_line_ending(content: &str) -> &'static str {
    let crlf = content.find("\r\n");
    let lf = content.find('\n');
    match (crlf, lf) {
        (Some(crlf), Some(lf)) if crlf < lf => "\r\n",
        _ => "\n",
    }
}

pub(crate) fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub(crate) fn restore_line_endings(text: &str, ending: &str) -> String {
    if ending == "\r\n" {
        text.replace('\n', "\r\n")
    } else {
        text.to_string()
    }
}

/// Splits a UTF-8 BOM off the front, if present.
pub(crate) fn strip_bom(content: &str) -> (&str, &str) {
    match content.strip_prefix('\u{FEFF}') {
        Some(text) => ("\u{FEFF}", text),
        None => ("", content),
    }
}

/// Normalizes text for fuzzy matching: per-line trailing whitespace stripped,
/// smart quotes / dashes / special spaces mapped to ASCII.
pub(crate) fn normalize_for_fuzzy(text: &str) -> String {
    let stripped: String = text
        .split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    stripped
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

struct Match {
    edit_index: usize,
    index: usize,
    length: usize,
    new_text: String,
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count()
}

/// Applies the edits to LF-normalized content. Returns the new content.
pub(crate) fn apply_edits(normalized: &str, edits: &[Edit], path: &str) -> Result<String> {
    let total = edits.len();
    let edits: Vec<Edit> = edits
        .iter()
        .map(|edit| Edit {
            old_text: normalize_to_lf(&edit.old_text),
            new_text: normalize_to_lf(&edit.new_text),
        })
        .collect();

    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(Error::Tool(if total == 1 {
                format!("oldText must not be empty in {path}.")
            } else {
                format!("edits[{index}].oldText must not be empty in {path}.")
            }));
        }
    }

    // If any edit needs the fuzzy space, the whole operation runs there.
    let any_fuzzy = edits.iter().any(|edit| {
        !normalized.contains(&edit.old_text)
            && normalize_for_fuzzy(normalized).contains(&normalize_for_fuzzy(&edit.old_text))
    });
    let base = if any_fuzzy {
        normalize_for_fuzzy(normalized)
    } else {
        normalized.to_string()
    };

    let mut matches: Vec<Match> = Vec::new();
    for (index, edit) in edits.iter().enumerate() {
        let (found_at, length) = match base.find(&edit.old_text) {
            Some(at) => (Some(at), edit.old_text.len()),
            None => {
                let fuzzy_needle = normalize_for_fuzzy(&edit.old_text);
                let fuzzy_base = normalize_for_fuzzy(&base);
                // `base` is already fuzzy when any_fuzzy, so this re-normalize
                // is a no-op there and a fresh fallback probe otherwise.
                match fuzzy_base.find(&fuzzy_needle) {
                    Some(at) if any_fuzzy => (Some(at), fuzzy_needle.len()),
                    _ => (None, 0),
                }
            }
        };
        let Some(at) = found_at else {
            return Err(Error::Tool(if total == 1 {
                format!(
                    "Could not find the exact text in {path}. The old text must match \
                     exactly including all whitespace and newlines."
                )
            } else {
                format!(
                    "Could not find edits[{index}] in {path}. The oldText must match \
                     exactly including all whitespace and newlines."
                )
            }));
        };

        let occurrences = count_occurrences(
            &normalize_for_fuzzy(&base),
            &normalize_for_fuzzy(&edit.old_text),
        );
        if occurrences > 1 {
            return Err(Error::Tool(if total == 1 {
                format!(
                    "Found {occurrences} occurrences of the text in {path}. The text must \
                     be unique. Please provide more context to make it unique."
                )
            } else {
                format!(
                    "Found {occurrences} occurrences of edits[{index}] in {path}. Each \
                     oldText must be unique. Please provide more context to make it unique."
                )
            }));
        }

        matches.push(Match {
            edit_index: index,
            index: at,
            length,
            new_text: edit.new_text.clone(),
        });
    }

    matches.sort_by_key(|m| m.index);
    for pair in matches.windows(2) {
        if pair[0].index + pair[0].length > pair[1].index {
            return Err(Error::Tool(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or \
                 target disjoint regions.",
                pair[0].edit_index, pair[1].edit_index
            )));
        }
    }

    let mut new_content = base.clone();
    for m in matches.iter().rev() {
        new_content.replace_range(m.index..m.index + m.length, &m.new_text);
    }

    if new_content == base {
        return Err(Error::Tool(if total == 1 {
            format!(
                "No changes made to {path}. The replacement produced identical content. \
                 This might indicate an issue with special characters or the text not \
                 existing as expected."
            )
        } else {
            format!("No changes made to {path}. The replacements produced identical content.")
        }));
    }

    Ok(new_content)
}

/// Tolerates the legacy/odd input shapes models produce: a top-level
/// `oldText`/`newText` pair, and `edits` sent as a JSON string.
fn prepare_input(mut input: serde_json::Value) -> serde_json::Value {
    let Some(object) = input.as_object_mut() else {
        return input;
    };
    if let Some(serde_json::Value::String(raw)) = object.get("edits")
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw)
        && parsed.is_array()
    {
        object.insert("edits".to_string(), parsed);
    }
    let old_text = object
        .get("oldText")
        .and_then(|v| v.as_str().map(str::to_string));
    let new_text = object
        .get("newText")
        .and_then(|v| v.as_str().map(str::to_string));
    if let (Some(old_text), Some(new_text)) = (old_text, new_text) {
        let edits = object
            .entry("edits")
            .or_insert_with(|| serde_json::Value::Array(Vec::new()));
        if let Some(array) = edits.as_array_mut() {
            array.push(json!({ "oldText": old_text, "newText": new_text }));
        }
        object.remove("oldText");
        object.remove("newText");
    }
    input
}

#[derive(Deserialize)]
struct EditInput {
    path: String,
    edits: Vec<Edit>,
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Apply one or more targeted text replacements to a file. Each oldText must \
         match exactly once in the original file (whitespace included); edits must \
         not overlap."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute)"
                },
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements. Each edit is \
                        matched against the original file, not incrementally. Do not \
                        include overlapping or nested edits. If two changes touch the \
                        same block or nearby lines, merge them into one edit instead.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text for one targeted replacement. \
                                    It must be unique in the original file and must not \
                                    overlap with any other edits[].oldText in the same call."
                            },
                            "newText": {
                                "type": "string",
                                "description": "Replacement text for this targeted edit."
                            }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["path", "edits"]
        })
    }

    fn effects(&self) -> ToolEffects {
        ToolEffects::write()
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        input: serde_json::Value,
        _on_update: Option<Box<dyn Fn(ToolUpdate) + Send + Sync>>,
    ) -> Result<ToolOutput> {
        let input: EditInput = serde_json::from_value(prepare_input(input))
            .map_err(|error| Error::Tool(format!("edit: invalid input: {error}")))?;
        if input.edits.is_empty() {
            return Err(Error::Tool(
                "Edit tool input is invalid. edits must contain at least one replacement."
                    .to_string(),
            ));
        }
        let absolute = resolve_path(&self.cwd, &input.path);
        let display_path = input.path.clone();
        let replacements = input.edits.len();

        skein::runtime::spawn_blocking({
            let absolute = absolute.clone();
            move || -> Result<()> {
                let raw = std::fs::read_to_string(&absolute).map_err(|error| {
                    Error::Tool(format!("reading {}: {error}", absolute.display()))
                })?;
                let (bom, text) = strip_bom(&raw);
                let ending = detect_line_ending(text);
                let normalized = normalize_to_lf(text);
                let new_content = apply_edits(&normalized, &input.edits, &display_path)?;
                let restored = format!("{bom}{}", restore_line_endings(&new_content, ending));
                std::fs::write(&absolute, restored).map_err(|error| {
                    Error::Tool(format!("writing {}: {error}", absolute.display()))
                })
            }
        })
        .await?;

        Ok(ToolOutput::text(format!(
            "Edited {} ({replacements} replacement{})",
            input.path,
            if replacements == 1 { "" } else { "s" }
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(old_text: &str, new_text: &str) -> Vec<Edit> {
        vec![Edit {
            old_text: old_text.to_string(),
            new_text: new_text.to_string(),
        }]
    }

    #[test]
    fn applies_exact_replacement() {
        let result = apply_edits("fn main() {}\n", &one("main", "start"), "f").unwrap();
        assert_eq!(result, "fn start() {}\n");
    }

    #[test]
    fn missing_text_errors() {
        let error = apply_edits("abc", &one("xyz", "q"), "f").unwrap_err();
        assert!(format!("{error}").contains("Could not find the exact text in f"));
    }

    #[test]
    fn duplicate_text_errors() {
        let error = apply_edits("x x", &one("x", "y"), "f").unwrap_err();
        assert!(format!("{error}").contains("Found 2 occurrences"));
    }

    #[test]
    fn empty_old_text_errors() {
        let error = apply_edits("abc", &one("", "y"), "f").unwrap_err();
        assert!(format!("{error}").contains("must not be empty"));
    }

    #[test]
    fn identical_replacement_errors() {
        let error = apply_edits("abc", &one("b", "b"), "f").unwrap_err();
        assert!(format!("{error}").contains("No changes made"));
    }

    #[test]
    fn fuzzy_matches_trailing_whitespace_and_smart_quotes() {
        // File has trailing spaces and a smart quote; needle is the clean form.
        let content = "let s = \u{201C}hi\u{201D};   \nnext\n";
        let result = apply_edits(content, &one("let s = \"hi\";", "let s = \"yo\";"), "f").unwrap();
        assert!(result.contains("let s = \"yo\";"));
    }

    #[test]
    fn multiple_edits_apply_in_offset_order() {
        let content = "alpha\nbeta\ngamma\n";
        let edits = vec![
            Edit {
                old_text: "gamma".to_string(),
                new_text: "GAMMA".to_string(),
            },
            Edit {
                old_text: "alpha".to_string(),
                new_text: "ALPHA".to_string(),
            },
        ];
        assert_eq!(
            apply_edits(content, &edits, "f").unwrap(),
            "ALPHA\nbeta\nGAMMA\n"
        );
    }

    #[test]
    fn overlapping_edits_error() {
        let content = "abcdef";
        let edits = vec![
            Edit {
                old_text: "abcd".to_string(),
                new_text: "x".to_string(),
            },
            Edit {
                old_text: "cdef".to_string(),
                new_text: "y".to_string(),
            },
        ];
        let error = apply_edits(content, &edits, "f").unwrap_err();
        assert!(format!("{error}").contains("overlap"));
    }

    #[test]
    fn line_ending_round_trip() {
        assert_eq!(detect_line_ending("a\r\nb"), "\r\n");
        assert_eq!(detect_line_ending("a\nb"), "\n");
        let normalized = normalize_to_lf("a\r\nb\rc");
        assert_eq!(normalized, "a\nb\nc");
        assert_eq!(restore_line_endings("a\nb", "\r\n"), "a\r\nb");
    }

    #[test]
    fn bom_is_preserved_by_strip_restore() {
        let (bom, text) = strip_bom("\u{FEFF}hello");
        assert_eq!(bom, "\u{FEFF}");
        assert_eq!(text, "hello");
    }

    #[test]
    fn legacy_top_level_old_new_text_is_accepted() {
        let input = json!({"path": "f", "oldText": "a", "newText": "b"});
        let prepared = prepare_input(input);
        assert_eq!(prepared["edits"][0]["oldText"], "a");
        assert!(prepared.get("oldText").is_none());
    }

    #[test]
    fn edits_as_json_string_is_accepted() {
        let input = json!({"path": "f", "edits": "[{\"oldText\":\"a\",\"newText\":\"b\"}]"});
        let prepared = prepare_input(input);
        assert_eq!(prepared["edits"][0]["newText"], "b");
    }
}
