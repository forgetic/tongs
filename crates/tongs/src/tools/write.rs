//! The write tool: whole-file writes, creating parent directories.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use super::support::resolve_path;
use super::{Tool, ToolEffects, ToolOutput, ToolUpdate};
use crate::{Error, Result};

pub(crate) struct WriteTool {
    cwd: PathBuf,
}

impl WriteTool {
    pub(crate) fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[derive(Deserialize)]
struct WriteInput {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file, creating parent directories as needed and \
         overwriting any existing file."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
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
        let input: WriteInput = serde_json::from_value(input)
            .map_err(|error| Error::Tool(format!("write: invalid input: {error}")))?;
        let absolute = resolve_path(&self.cwd, &input.path);
        let bytes = input.content.len();

        skein::runtime::spawn_blocking({
            let absolute = absolute.clone();
            move || -> std::io::Result<()> {
                if let Some(parent) = absolute.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&absolute, input.content)
            }
        })
        .await
        .map_err(|error| Error::Tool(format!("writing {}: {error}", absolute.display())))?;

        Ok(ToolOutput::text(format!(
            "Wrote {bytes} bytes to {}",
            input.path
        )))
    }
}
