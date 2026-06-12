//! The ls tool: directory listing, dirs suffixed `/`, dotfiles included.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use super::support::{DEFAULT_MAX_BYTES, resolve_path, truncate_head};
use super::{Tool, ToolEffects, ToolOutput, ToolUpdate};
use crate::{Error, Result};

const DEFAULT_LIMIT: usize = 500;

pub(crate) struct LsTool {
    cwd: PathBuf,
}

impl LsTool {
    pub(crate) fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[derive(Deserialize)]
struct LsInput {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

/// Pure: order entries (case-insensitive) and render with `/` dir suffixes
/// and the limit notice.
fn render_listing(mut entries: Vec<(String, bool)>, limit: usize) -> String {
    entries.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
    let total = entries.len();
    let shown: Vec<String> = entries
        .into_iter()
        .take(limit)
        .map(|(name, is_dir)| if is_dir { format!("{name}/") } else { name })
        .collect();
    let mut output = shown.join("\n");
    if total > limit {
        output.push_str(&format!(
            "\n\n[Showing {limit} of {total} entries. Use limit={total} to see all.]"
        ));
    }
    if output.is_empty() {
        output = "(empty directory)".to_string();
    }
    output
}

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List the entries of a directory (directories get a trailing /, dotfiles included)."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (default: .)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of entries (default: 500)"
                }
            }
        })
    }

    fn effects(&self) -> ToolEffects {
        ToolEffects::read()
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        input: serde_json::Value,
        _on_update: Option<Box<dyn Fn(ToolUpdate) + Send + Sync>>,
    ) -> Result<ToolOutput> {
        let input: LsInput = serde_json::from_value(input)
            .map_err(|error| Error::Tool(format!("ls: invalid input: {error}")))?;
        let directory = resolve_path(&self.cwd, input.path.as_deref().unwrap_or("."));
        let limit = input.limit.unwrap_or(DEFAULT_LIMIT).max(1);

        let entries = skein::runtime::spawn_blocking({
            let directory = directory.clone();
            move || -> std::io::Result<Vec<(String, bool)>> {
                let mut entries = Vec::new();
                for entry in std::fs::read_dir(&directory)? {
                    let entry = entry?;
                    let is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
                    entries.push((entry.file_name().to_string_lossy().into_owned(), is_dir));
                }
                Ok(entries)
            }
        })
        .await
        .map_err(|error| Error::Tool(format!("listing {}: {error}", directory.display())))?;

        let listing = render_listing(entries, limit);
        let truncation = truncate_head(&listing, usize::MAX, DEFAULT_MAX_BYTES);
        Ok(ToolOutput::text(truncation.content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_case_insensitively_and_marks_dirs() {
        let entries = vec![
            ("Zeta".to_string(), false),
            (".hidden".to_string(), false),
            ("alpha".to_string(), true),
        ];
        assert_eq!(render_listing(entries, 10), ".hidden\nalpha/\nZeta");
    }

    #[test]
    fn enforces_limit_with_notice() {
        let entries = (0..5).map(|i| (format!("f{i}"), false)).collect();
        let output = render_listing(entries, 2);
        assert!(output.starts_with("f0\nf1"));
        assert!(output.contains("[Showing 2 of 5 entries. Use limit=5 to see all.]"));
    }

    #[test]
    fn empty_directory_is_stated() {
        assert_eq!(render_listing(Vec::new(), 10), "(empty directory)");
    }
}
