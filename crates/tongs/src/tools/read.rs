//! The read tool: file contents with offset/limit paging and head truncation.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use super::support::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, resolve_path, truncate_head,
};
use super::{Tool, ToolEffects, ToolOutput, ToolUpdate};
use crate::model::{ContentBlock, ImageContent, TextContent};
use crate::{Error, Result};

pub(crate) struct ReadTool {
    cwd: PathBuf,
    description: String,
}

impl ReadTool {
    pub(crate) fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            description: format!(
                "Read the contents of a file. Supports text files and images (jpg, png, \
                 gif, webp). Images are sent as attachments. For text files, output is \
                 truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). \
                 Use offset/limit for large files. When you need the full file, continue \
                 with offset until complete.",
                DEFAULT_MAX_BYTES / 1024
            ),
        }
    }
}

#[derive(Deserialize)]
struct ReadInput {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

/// MIME type for paths with a supported image extension.
fn image_mime_type(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

/// Pure: page + truncate file text per the offset/limit/truncation contract.
fn render_read(
    text: &str,
    path: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Result<String> {
    let all_lines: Vec<&str> = text.split('\n').collect();
    let total_lines = all_lines.len();
    let start = offset.map(|offset| offset.saturating_sub(1)).unwrap_or(0);
    let start_display = start + 1;
    if start >= total_lines {
        return Err(Error::Tool(format!(
            "Offset {} is beyond end of file ({total_lines} lines total)",
            offset.unwrap_or(0)
        )));
    }

    let (selected, user_limited) = match limit {
        Some(limit) => {
            let end = (start + limit).min(total_lines);
            (all_lines[start..end].join("\n"), Some(end - start))
        }
        None => (all_lines[start..].join("\n"), None),
    };

    let truncation = truncate_head(&selected, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    if truncation.first_line_exceeds_limit {
        let first_line_size = format_size(all_lines[start].len());
        return Ok(format!(
            "[Line {start_display} is {first_line_size}, exceeds {} limit. Use bash: \
             sed -n '{start_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]",
            format_size(DEFAULT_MAX_BYTES)
        ));
    }
    if truncation.truncated {
        let end_display = start_display + truncation.output_lines.saturating_sub(1);
        let next_offset = end_display + 1;
        let notice = match truncation.truncated_by {
            Some(super::support::TruncatedBy::Lines) => format!(
                "[Showing lines {start_display}-{end_display} of {total_lines}. \
                 Use offset={next_offset} to continue.]"
            ),
            _ => format!(
                "[Showing lines {start_display}-{end_display} of {total_lines} ({} limit). \
                 Use offset={next_offset} to continue.]",
                format_size(DEFAULT_MAX_BYTES)
            ),
        };
        return Ok(format!("{}\n\n{notice}", truncation.content));
    }
    if let Some(taken) = user_limited
        && start + taken < total_lines
    {
        let remaining = total_lines - (start + taken);
        let next_offset = start + taken + 1;
        return Ok(format!(
            "{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]",
            truncation.content
        ));
    }
    Ok(truncation.content)
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)"
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
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
        let input: ReadInput = serde_json::from_value(input)
            .map_err(|error| Error::Tool(format!("read: invalid input: {error}")))?;
        let absolute = resolve_path(&self.cwd, &input.path);

        if let Some(mime_type) = image_mime_type(&absolute) {
            let bytes = skein::runtime::spawn_blocking({
                let absolute = absolute.clone();
                move || std::fs::read(&absolute)
            })
            .await
            .map_err(|error| Error::Tool(format!("reading {}: {error}", absolute.display())))?;
            return Ok(ToolOutput {
                content: vec![
                    ContentBlock::Text(TextContent {
                        text: format!("Read image file [{mime_type}]"),
                        text_signature: None,
                    }),
                    ContentBlock::Image(ImageContent {
                        data: crate::util::base64_encode(&bytes),
                        mime_type: mime_type.to_string(),
                    }),
                ],
                details: None,
                is_error: false,
            });
        }

        let text = skein::runtime::spawn_blocking({
            let absolute = absolute.clone();
            move || {
                std::fs::read(&absolute).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            }
        })
        .await
        .map_err(|error| Error::Tool(format!("reading {}: {error}", absolute.display())))?;

        let output = render_read(&text, &input.path, input.offset, input.limit)?;
        Ok(ToolOutput::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_whole_file() {
        assert_eq!(render_read("a\nb", "f", None, None).unwrap(), "a\nb");
    }

    #[test]
    fn offset_and_limit_page_through() {
        let text = "1\n2\n3\n4\n5";
        let page = render_read(text, "f", Some(2), Some(2)).unwrap();
        assert!(page.starts_with("2\n3"));
        // Lines 2-3 shown; line 4 is next.
        assert!(page.contains("2 more lines in file. Use offset=4 to continue."));
    }

    #[test]
    fn offset_beyond_eof_errors() {
        let error = render_read("a\nb", "f", Some(10), None).unwrap_err();
        assert!(format!("{error}").contains("beyond end of file"));
    }

    #[test]
    fn long_files_get_truncation_notice() {
        let text = (1..=3000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let output = render_read(&text, "f", None, None).unwrap();
        assert!(output.contains("[Showing lines 1-2000 of 3000. Use offset=2001 to continue.]"));
    }

    #[test]
    fn image_extensions_detected() {
        assert_eq!(image_mime_type(Path::new("x.PNG")), Some("image/png"));
        assert_eq!(image_mime_type(Path::new("x.jpeg")), Some("image/jpeg"));
        assert_eq!(image_mime_type(Path::new("x.rs")), None);
    }
}
