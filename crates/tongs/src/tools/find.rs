//! The find tool: glob file search, .gitignore-aware, relative POSIX paths.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use super::support::{DEFAULT_MAX_BYTES, resolve_path, truncate_head};
use super::{Tool, ToolEffects, ToolOutput, ToolUpdate};
use crate::{Error, Result};

const DEFAULT_LIMIT: usize = 1000;

pub(crate) struct FindTool {
    cwd: PathBuf,
}

impl FindTool {
    pub(crate) fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[derive(Deserialize)]
struct FindInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

/// Pure: sort, cap, and render found paths (dirs get a trailing `/`).
fn render_found(mut found: Vec<(String, bool)>, limit: usize) -> String {
    found.sort_by(|a, b| a.0.cmp(&b.0));
    let total = found.len();
    let shown: Vec<String> = found
        .into_iter()
        .take(limit)
        .map(|(path, is_dir)| if is_dir { format!("{path}/") } else { path })
        .collect();
    if shown.is_empty() {
        return "No files found.".to_string();
    }
    let mut output = shown.join("\n");
    if total > limit {
        output.push_str(&format!(
            "\n\n[Showing {limit} of {total} results. Use limit={total} to see all.]"
        ));
    }
    output
}

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files by glob pattern. Respects .gitignore. Returns paths relative to \
         the search directory."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json'"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: .)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of results (default: 1000)"
                }
            },
            "required": ["pattern"]
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
        let input: FindInput = serde_json::from_value(input)
            .map_err(|error| Error::Tool(format!("find: invalid input: {error}")))?;
        let root = resolve_path(&self.cwd, input.path.as_deref().unwrap_or("."));
        let limit = input.limit.unwrap_or(DEFAULT_LIMIT).max(1);

        // Default globset semantics let `*` cross separators, so `*.rs`
        // matches nested files — the fd-like behavior the description implies.
        let matcher = globset::Glob::new(&input.pattern)
            .map_err(|error| Error::Tool(format!("find: invalid pattern: {error}")))?
            .compile_matcher();

        let found = skein::runtime::spawn_blocking({
            let root = root.clone();
            move || -> Vec<(String, bool)> {
                let mut found = Vec::new();
                for entry in ignore::WalkBuilder::new(&root).build() {
                    let Ok(entry) = entry else { continue };
                    let path = entry.path();
                    let Ok(relative) = path.strip_prefix(&root) else {
                        continue;
                    };
                    if relative.as_os_str().is_empty() {
                        continue;
                    }
                    if matcher.is_match(relative) {
                        let is_dir = entry.file_type().is_some_and(|kind| kind.is_dir());
                        found.push((relative.to_string_lossy().into_owned(), is_dir));
                    }
                }
                found
            }
        })
        .await;

        let rendered = render_found(found, limit);
        Ok(ToolOutput::text(
            truncate_head(&rendered, usize::MAX, DEFAULT_MAX_BYTES).content,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_sorted_with_dir_suffix() {
        let found = vec![
            ("src/z.rs".to_string(), false),
            ("src".to_string(), true),
            ("src/a.rs".to_string(), false),
        ];
        assert_eq!(render_found(found, 10), "src/\nsrc/a.rs\nsrc/z.rs");
    }

    #[test]
    fn caps_results_with_notice() {
        let found = (0..4).map(|i| (format!("f{i}"), false)).collect();
        let output = render_found(found, 2);
        assert!(output.contains("[Showing 2 of 4 results. Use limit=4 to see all.]"));
    }

    #[test]
    fn empty_results_say_so() {
        assert_eq!(render_found(Vec::new(), 10), "No files found.");
    }
}
