//! Pure helpers shared by the tools: path resolution and output truncation.
//!
//! Truncation semantics ported from TS Pi's `truncate.ts`: two independent
//! limits (lines and bytes), whichever is hit first wins, and output never
//! contains partial lines.

use std::path::{Path, PathBuf};

/// Default line limit for tool output.
pub(crate) const DEFAULT_MAX_LINES: usize = 2000;
/// Default byte limit for tool output (50KB).
pub(crate) const DEFAULT_MAX_BYTES: usize = 50 * 1024;
/// Max characters kept per grep match line.
pub(crate) const GREP_MAX_LINE_LENGTH: usize = 500;

/// Resolves a tool `path` argument against the tool's `cwd` (absolute paths
/// pass through).
pub(crate) fn resolve_path(cwd: &Path, path: &str) -> PathBuf {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        cwd.join(candidate)
    }
}

/// How a truncation ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TruncatedBy {
    Lines,
    Bytes,
}

#[derive(Clone, Debug)]
pub(crate) struct Truncation {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    /// Lines in the original content (asserted by tests; read.rs derives its
    /// own whole-file count because it truncates an offset slice).
    #[allow(dead_code)]
    pub total_lines: usize,
    pub output_lines: usize,
    /// The first line alone exceeds the byte limit (head truncation only).
    pub first_line_exceeds_limit: bool,
}

fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// Keeps the head of `content`, up to the line and byte limits.
pub(crate) fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    if total_lines <= max_lines && content.len() <= max_bytes {
        return Truncation {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            output_lines: total_lines,
            first_line_exceeds_limit: false,
        };
    }

    if lines.first().is_some_and(|line| line.len() > max_bytes) {
        return Truncation {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            output_lines: 0,
            first_line_exceeds_limit: true,
        };
    }

    let mut output: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    for (index, line) in lines.iter().enumerate() {
        if index >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(index > 0);
        if bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output.push(line);
        bytes += line_bytes;
    }
    let output_lines = output.len();
    Truncation {
        content: output.join("\n"),
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        output_lines,
        first_line_exceeds_limit: false,
    }
}

/// Keeps the tail of `content` (bash output: the most recent lines matter),
/// up to the line and byte limits.
pub(crate) fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    if total_lines <= max_lines && content.len() <= max_bytes {
        return Truncation {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            output_lines: total_lines,
            first_line_exceeds_limit: false,
        };
    }

    let mut output: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    for (taken, line) in lines.iter().rev().enumerate() {
        if taken >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(taken > 0);
        if bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output.push(line);
        bytes += line_bytes;
    }
    output.reverse();
    let output_lines = output.len();
    Truncation {
        content: output.join("\n"),
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        output_lines,
        first_line_exceeds_limit: false,
    }
}

/// Human-readable byte size.
pub(crate) fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.0}KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_no_truncation() {
        let result = truncate_head("a\nb\nc", 10, 1000);
        assert!(!result.truncated);
        assert_eq!(result.content, "a\nb\nc");
        assert_eq!(result.total_lines, 3);
    }

    #[test]
    fn head_line_limit() {
        let result = truncate_head("a\nb\nc\nd", 2, 1000);
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.content, "a\nb");
        assert_eq!(result.output_lines, 2);
        assert_eq!(result.total_lines, 4);
    }

    #[test]
    fn head_byte_limit_keeps_whole_lines() {
        // 9-byte budget: "aaaa" (4) + "\nbbbb" (5) fits exactly; "\ncccc" breaks it.
        let result = truncate_head("aaaa\nbbbb\ncccc", 10, 9);
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(result.content, "aaaa\nbbbb");
    }

    #[test]
    fn head_first_line_too_big() {
        let result = truncate_head(&"x".repeat(100), 10, 50);
        assert!(result.first_line_exceeds_limit);
        assert_eq!(result.content, "");
    }

    #[test]
    fn tail_keeps_last_lines() {
        let result = truncate_tail("a\nb\nc\nd", 2, 1000);
        assert!(result.truncated);
        assert_eq!(result.content, "c\nd");
    }

    #[test]
    fn trailing_newline_not_counted_as_line() {
        let result = truncate_head("a\nb\n", 10, 1000);
        assert_eq!(result.total_lines, 2);
    }

    #[test]
    fn resolves_relative_and_absolute() {
        let cwd = Path::new("/work");
        assert_eq!(resolve_path(cwd, "src/x.rs"), PathBuf::from("/work/src/x.rs"));
        assert_eq!(resolve_path(cwd, "/etc/hosts"), PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn formats_sizes() {
        assert_eq!(format_size(10), "10B");
        assert_eq!(format_size(51200), "50KB");
        assert_eq!(format_size(2 * 1024 * 1024), "2.0MB");
    }
}
