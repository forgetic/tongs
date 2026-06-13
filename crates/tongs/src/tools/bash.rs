//! The bash tool: run a shell command, stream-merge stdout+stderr, enforce a
//! timeout, and tail-truncate the output (the most recent lines matter for
//! build/test runs).
//!
//! stderr is merged inside the child (`exec 2>&1`) so interleaving is true to
//! what a terminal would show. The child is killed on timeout and on drop;
//! grandchildren that re-daemonize are not chased (no process-group kill —
//! divergence from TS Pi's killProcessTree).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use skein::io::{AsyncRead, ReadBuf};
use skein::process::{Command, Stdio};

use super::support::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail};
use super::{Tool, ToolEffects, ToolOutput, ToolUpdate};
use crate::{Error, Result};

/// Cap on bytes kept in memory while reading (tail ring; final output is
/// further truncated to the standard limits).
const TAIL_RING_BYTES: usize = 256 * 1024;

pub(crate) struct BashTool {
    cwd: PathBuf,
}

impl BashTool {
    pub(crate) fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[derive(Deserialize)]
struct BashInput {
    command: String,
    #[serde(default)]
    timeout: Option<u64>,
}

/// A bounded byte ring keeping the most recent output.
struct TailRing {
    bytes: VecDeque<u8>,
    capacity: usize,
    dropped: bool,
}

impl TailRing {
    fn new(capacity: usize) -> Self {
        Self {
            bytes: VecDeque::new(),
            capacity,
            dropped: false,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        self.bytes.extend(chunk);
        while self.bytes.len() > self.capacity {
            self.bytes.pop_front();
            self.dropped = true;
        }
    }

    fn into_text(self) -> (String, bool) {
        let bytes: Vec<u8> = self.bytes.into();
        (String::from_utf8_lossy(&bytes).into_owned(), self.dropped)
    }
}

/// Pure: assemble the model-facing output from the captured tail, exit
/// status, and timeout flag.
fn render_outcome(
    text: &str,
    ring_dropped: bool,
    exit_code: Option<i32>,
    timed_out: Option<u64>,
) -> (String, bool) {
    let truncation = truncate_tail(text, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let mut output = String::new();
    if truncation.truncated || ring_dropped {
        output.push_str(&format!(
            "[Output truncated: showing the last {} lines]\n",
            truncation.output_lines
        ));
    }
    output.push_str(&truncation.content);
    if output.is_empty() {
        output.push_str("(no output)");
    }

    if let Some(seconds) = timed_out {
        output.push_str(&format!("\n\nCommand timed out after {seconds} seconds"));
        return (output, true);
    }
    match exit_code {
        Some(0) => (output, false),
        Some(code) => {
            output.push_str(&format!("\n\nExit code: {code}"));
            (output, true)
        }
        None => {
            output.push_str("\n\nCommand terminated by signal");
            (output, true)
        }
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the workspace. stdout and stderr are merged; \
         long output is truncated to the most recent lines."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)"
                }
            },
            "required": ["command"]
        })
    }

    fn effects(&self) -> ToolEffects {
        ToolEffects::process()
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        input: serde_json::Value,
        _on_update: Option<Box<dyn Fn(ToolUpdate) + Send + Sync>>,
    ) -> Result<ToolOutput> {
        let input: BashInput = serde_json::from_value(input)
            .map_err(|error| Error::Tool(format!("bash: invalid input: {error}")))?;

        let mut command = Command::new("bash");
        command
            .arg("-c")
            // Merge stderr into stdout inside the shell so interleaving is real.
            .arg(format!("exec 2>&1\n{}", input.command))
            .current_dir(&self.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = command
            .spawn()
            .map_err(|error| Error::Tool(format!("bash: spawn failed: {error}")))?;
        let mut stdout = child
            .stdout()
            .ok_or_else(|| Error::Tool("bash: child stdout unavailable".to_string()))?;

        let deadline = input
            .timeout
            .filter(|seconds| *seconds > 0)
            .map(|seconds| crate::runtime::engine_now() + Duration::from_secs(seconds));

        let mut ring = TailRing::new(TAIL_RING_BYTES);
        let mut buffer = [0u8; 8192];
        let mut timed_out = false;
        loop {
            let read = match deadline {
                Some(deadline) => {
                    let read = Box::pin(read_some(&mut stdout, &mut buffer));
                    match skein::time::timeout_at(deadline, read).await {
                        Ok(read) => read,
                        Err(_) => {
                            timed_out = true;
                            break;
                        }
                    }
                }
                None => read_some(&mut stdout, &mut buffer).await,
            };
            match read {
                Ok(0) => break,
                Ok(n) => ring.push(&buffer[..n]),
                Err(error) => {
                    return Err(Error::Tool(format!("bash: reading output: {error}")));
                }
            }
        }

        if timed_out {
            let _ = child.kill();
            let (text, ring_dropped) = ring.into_text();
            let (output, _) = render_outcome(&text, ring_dropped, None, input.timeout);
            return Ok(ToolOutput {
                is_error: true,
                ..ToolOutput::text(output)
            });
        }

        // stdout hit EOF: the process is exiting; poll for its status without
        // blocking the loop thread.
        let exit_code = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status.code(),
                Ok(None) => {
                    if let Some(deadline) = deadline
                        && crate::runtime::engine_now() >= deadline
                    {
                        let _ = child.kill();
                        let (text, ring_dropped) = ring.into_text();
                        let (output, _) = render_outcome(&text, ring_dropped, None, input.timeout);
                        return Ok(ToolOutput {
                            is_error: true,
                            ..ToolOutput::text(output)
                        });
                    }
                    skein::time::sleep(crate::runtime::engine_now(), Duration::from_millis(10))
                        .await;
                }
                Err(error) => {
                    return Err(Error::Tool(format!("bash: waiting for exit: {error}")));
                }
            }
        };

        let (text, ring_dropped) = ring.into_text();
        let (output, is_error) = render_outcome(&text, ring_dropped, exit_code, None);
        Ok(ToolOutput {
            is_error,
            ..ToolOutput::text(output)
        })
    }
}

/// One read into `buffer`, returning the byte count (0 = EOF).
async fn read_some<R: AsyncRead + Unpin>(
    reader: &mut R,
    buffer: &mut [u8],
) -> std::io::Result<usize> {
    std::future::poll_fn(|cx| {
        let mut read_buf = ReadBuf::new(buffer);
        match std::pin::Pin::new(&mut *reader).poll_read(cx, &mut read_buf) {
            std::task::Poll::Ready(Ok(())) => std::task::Poll::Ready(Ok(read_buf.filled().len())),
            std::task::Poll::Ready(Err(error)) => std::task::Poll::Ready(Err(error)),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_renders_plain_output() {
        let (output, is_error) = render_outcome("hello\n", false, Some(0), None);
        assert_eq!(output, "hello\n");
        assert!(!is_error);
    }

    #[test]
    fn nonzero_exit_appends_code_and_errors() {
        let (output, is_error) = render_outcome("boom\n", false, Some(2), None);
        assert!(output.contains("Exit code: 2"));
        assert!(is_error);
    }

    #[test]
    fn timeout_message_and_error() {
        let (output, is_error) = render_outcome("partial", false, None, Some(5));
        assert!(output.contains("timed out after 5 seconds"));
        assert!(is_error);
    }

    #[test]
    fn empty_output_is_stated() {
        let (output, _) = render_outcome("", false, Some(0), None);
        assert_eq!(output, "(no output)");
    }

    #[test]
    fn long_output_tail_truncates() {
        let text: String = (0..3000).map(|i| format!("line{i}\n")).collect();
        let (output, _) = render_outcome(&text, false, Some(0), None);
        assert!(output.starts_with("[Output truncated: showing the last 2000 lines]"));
        assert!(output.contains("line2999"));
        assert!(!output.contains("line0\n"));
    }

    #[test]
    fn tail_ring_keeps_most_recent_bytes() {
        let mut ring = TailRing::new(8);
        ring.push(b"0123456789");
        let (text, dropped) = ring.into_text();
        assert_eq!(text, "23456789");
        assert!(dropped);
    }
}
