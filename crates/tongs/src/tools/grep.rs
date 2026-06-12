//! The grep tool: regex search over the tree, .gitignore-aware.
//!
//! The walk runs on the blocking pool (`ignore` crate: standard filters, so
//! hidden files and gitignored paths are skipped); match filtering, context
//! grouping, and truncation are pure.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use super::support::{
    DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH, resolve_path, truncate_head,
};
use super::{Tool, ToolEffects, ToolOutput, ToolUpdate};
use crate::{Error, Result};

const DEFAULT_LIMIT: usize = 100;

pub(crate) struct GrepTool {
    cwd: PathBuf,
}

impl GrepTool {
    pub(crate) fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[derive(Deserialize)]
struct GrepInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default, rename = "ignoreCase")]
    ignore_case: bool,
    #[serde(default)]
    literal: bool,
    #[serde(default)]
    context: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

/// One file's matched line numbers (0-indexed), with its lines.
struct FileMatches {
    relative_path: String,
    lines: Vec<String>,
    matched: Vec<usize>,
}

/// Pure: renders grouped matches with optional context, rg-style.
fn render_matches(files: &[FileMatches], context: usize, limit: usize) -> (String, usize) {
    let mut output: Vec<String> = Vec::new();
    let mut shown = 0usize;
    let mut first_group = true;

    'outer: for file in files {
        for &line_index in &file.matched {
            if shown == limit {
                break 'outer;
            }
            if context > 0 && !first_group {
                output.push("--".to_string());
            }
            first_group = false;
            let start = line_index.saturating_sub(context);
            let end = (line_index + context + 1).min(file.lines.len());
            for index in start..end {
                let text = clip_line(&file.lines[index]);
                let display = index + 1;
                if index == line_index {
                    output.push(format!("{}:{display}: {text}", file.relative_path));
                } else {
                    output.push(format!("{}-{display}- {text}", file.relative_path));
                }
            }
            shown += 1;
        }
    }
    (output.join("\n"), shown)
}

fn clip_line(line: &str) -> String {
    if line.chars().count() <= GREP_MAX_LINE_LENGTH {
        return line.to_string();
    }
    let clipped: String = line.chars().take(GREP_MAX_LINE_LENGTH).collect();
    format!("{clipped} [line truncated]")
}

/// True when the glob matches the candidate (full relative path when the
/// glob contains a separator, the file name otherwise).
fn glob_matches(glob: &globset::GlobMatcher, has_separator: bool, relative: &Path) -> bool {
    if has_separator {
        glob.is_match(relative)
    } else {
        relative
            .file_name()
            .is_some_and(|name| glob.is_match(Path::new(name)))
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents for a regex (or literal) pattern. Respects .gitignore. \
         Output format: path:line: text."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal string)"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: .)"
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.ts'"
                },
                "ignoreCase": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)"
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string (default: false)"
                },
                "context": {
                    "type": "number",
                    "description": "Lines before/after each match (default: 0)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of matches to return (default: 100)"
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
        let input: GrepInput = serde_json::from_value(input)
            .map_err(|error| Error::Tool(format!("grep: invalid input: {error}")))?;
        let root = resolve_path(&self.cwd, input.path.as_deref().unwrap_or("."));
        let limit = input.limit.unwrap_or(DEFAULT_LIMIT).max(1);
        let context = input.context.unwrap_or(0);

        let pattern = if input.literal {
            regex::escape(&input.pattern)
        } else {
            input.pattern.clone()
        };
        let regex = regex::RegexBuilder::new(&pattern)
            .case_insensitive(input.ignore_case)
            .build()
            .map_err(|error| Error::Tool(format!("grep: invalid pattern: {error}")))?;

        let glob = input
            .glob
            .as_deref()
            .map(|glob| {
                globset::Glob::new(glob)
                    .map(|g| (g.compile_matcher(), glob.contains('/')))
                    .map_err(|error| Error::Tool(format!("grep: invalid glob: {error}")))
            })
            .transpose()?;

        let (rendered, shown, hit_limit) = skein::runtime::spawn_blocking({
            let root = root.clone();
            move || -> Result<(String, usize, bool)> {
                let mut files: Vec<FileMatches> = Vec::new();
                let mut total_matches = 0usize;
                let walker = ignore::WalkBuilder::new(&root).build();
                for entry in walker {
                    let Ok(entry) = entry else { continue };
                    if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                        continue;
                    }
                    let path = entry.path();
                    let relative = path.strip_prefix(&root).unwrap_or(path);
                    if let Some((matcher, has_separator)) = &glob
                        && !glob_matches(matcher, *has_separator, relative)
                    {
                        continue;
                    }
                    let Ok(bytes) = std::fs::read(path) else { continue };
                    if bytes.contains(&0) {
                        // Binary file.
                        continue;
                    }
                    let text = String::from_utf8_lossy(&bytes);
                    let lines: Vec<String> = text.split('\n').map(str::to_string).collect();
                    let matched: Vec<usize> = lines
                        .iter()
                        .enumerate()
                        .filter(|(_, line)| regex.is_match(line))
                        .map(|(index, _)| index)
                        .collect();
                    if matched.is_empty() {
                        continue;
                    }
                    total_matches += matched.len();
                    files.push(FileMatches {
                        relative_path: relative.to_string_lossy().into_owned(),
                        lines,
                        matched,
                    });
                    // Collect a little past the limit so the notice can say so,
                    // without walking the whole tree for huge result sets.
                    if total_matches > limit {
                        break;
                    }
                }
                files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
                let (rendered, shown) = render_matches(&files, context, limit);
                Ok((rendered, shown, total_matches > limit))
            }
        })
        .await?;

        if shown == 0 {
            return Ok(ToolOutput::text("No matches found."));
        }
        let mut output = truncate_head(&rendered, usize::MAX, DEFAULT_MAX_BYTES).content;
        if hit_limit {
            output.push_str(&format!(
                "\n\n[{shown} matches shown ({limit} limit). Narrow the pattern or raise limit.]"
            ));
        }
        Ok(ToolOutput::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, lines: &[&str], matched: &[usize]) -> FileMatches {
        FileMatches {
            relative_path: path.to_string(),
            lines: lines.iter().map(|s| s.to_string()).collect(),
            matched: matched.to_vec(),
        }
    }

    #[test]
    fn renders_simple_matches() {
        let files = vec![file("src/a.rs", &["one", "two", "three"], &[1])];
        let (output, shown) = render_matches(&files, 0, 10);
        assert_eq!(output, "src/a.rs:2: two");
        assert_eq!(shown, 1);
    }

    #[test]
    fn renders_context_groups() {
        let files = vec![file("a", &["l1", "l2", "l3", "l4"], &[1, 3])];
        let (output, _) = render_matches(&files, 1, 10);
        let expected = "a-1- l1\na:2: l2\na-3- l3\n--\na-3- l3\na:4: l4";
        assert_eq!(output, expected);
    }

    #[test]
    fn enforces_match_limit() {
        let files = vec![file("a", &["x", "x", "x"], &[0, 1, 2])];
        let (output, shown) = render_matches(&files, 0, 2);
        assert_eq!(shown, 2);
        assert_eq!(output.lines().count(), 2);
    }

    #[test]
    fn clips_very_long_lines() {
        let long = "y".repeat(600);
        let clipped = clip_line(&long);
        assert!(clipped.ends_with("[line truncated]"));
        assert!(clipped.len() < 600);
    }

    #[test]
    fn glob_name_vs_path_matching() {
        let name_glob = globset::Glob::new("*.rs").unwrap().compile_matcher();
        assert!(glob_matches(&name_glob, false, Path::new("deep/dir/x.rs")));
        let path_glob = globset::Glob::new("src/*.rs").unwrap().compile_matcher();
        assert!(glob_matches(&path_glob, true, Path::new("src/x.rs")));
        assert!(!glob_matches(&path_glob, true, Path::new("other/x.rs")));
    }
}
