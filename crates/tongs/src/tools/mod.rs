//! The tool abstraction and the seven classic coding tools.
//!
//! Each tool follows the sans-IO split: matching/filtering/truncation logic
//! is pure (tested without a filesystem), the shell does fs/process work on
//! skein (`spawn_blocking` for filesystem walks, the async child API for
//! bash). Semantics ported from TS Pi's `packages/coding-agent` tools.

mod bash;
mod edit;
mod find;
mod grep;
mod ls;
mod read;
mod support;
mod write;

use std::path::Path;

use async_trait::async_trait;

use crate::Result;
use crate::model::{ContentBlock, TextContent};
use crate::provider::ToolDef;

/// A progress update emitted by a running tool.
#[derive(Clone, Debug)]
pub struct ToolUpdate {
    pub text: String,
}

/// What a tool returns to the model.
#[derive(Clone, Debug, Default)]
pub struct ToolOutput {
    pub content: Vec<ContentBlock>,
    /// Structured details for observers; not sent to the model.
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
}

impl ToolOutput {
    /// A plain-text output.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
        }
    }

    /// A plain-text error output (the model sees it as a failed tool call).
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::text(text)
        }
    }
}

/// A tool's declared effects, used by agent loops to plan which adjacent
/// tool calls may run concurrently: only mutually read-only calls batch;
/// any write/network/process effect is a serialization barrier.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ToolEffects {
    pub reads: bool,
    pub writes: bool,
    pub network: bool,
    pub process: bool,
}

impl ToolEffects {
    /// Read-only: safe to run in parallel with other read-only tools.
    pub fn read() -> Self {
        Self {
            reads: true,
            ..Self::default()
        }
    }

    /// Mutates the workspace: serializes.
    pub fn write() -> Self {
        Self {
            reads: true,
            writes: true,
            ..Self::default()
        }
    }

    /// Talks to the network: serializes.
    pub fn network() -> Self {
        Self {
            network: true,
            ..Self::default()
        }
    }

    /// Runs a subprocess (arbitrary effects): serializes.
    pub fn process() -> Self {
        Self {
            reads: true,
            writes: true,
            process: true,
            ..Self::default()
        }
    }

    /// Whether this effect set is safe to run alongside others.
    pub fn parallel_safe(self) -> bool {
        !self.writes && !self.network && !self.process
    }

    /// Whether two effect sets may share a parallel batch.
    pub fn compatible_with(self, other: Self) -> bool {
        self.parallel_safe() && other.parallel_safe()
    }

    /// The combined effects of a batch.
    pub fn union(self, other: Self) -> Self {
        Self {
            reads: self.reads || other.reads,
            writes: self.writes || other.writes,
            network: self.network || other.network,
            process: self.process || other.process,
        }
    }
}

/// A callable tool.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The wire name the model calls.
    fn name(&self) -> &str;

    /// A short human label (defaults to the name).
    fn label(&self) -> &str {
        self.name()
    }

    fn description(&self) -> &str;

    /// JSON Schema for the arguments object.
    fn parameters(&self) -> serde_json::Value;

    fn effects(&self) -> ToolEffects;

    async fn execute(
        &self,
        tool_call_id: &str,
        input: serde_json::Value,
        on_update: Option<Box<dyn Fn(ToolUpdate) + Send + Sync>>,
    ) -> Result<ToolOutput>;
}

/// The tool set offered to one agent run.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_tools(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }

    pub fn push(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn tools(&self) -> &[Box<dyn Tool>] {
        &self.tools
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|tool| tool.name() == name)
            .map(AsRef::as_ref)
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.tools.iter().map(|tool| tool.name()))
            .finish()
    }
}

/// The provider-facing definition of a tool.
pub fn tool_to_definition(tool: &dyn Tool) -> ToolDef {
    ToolDef {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        parameters: tool.parameters(),
    }
}

pub fn create_read_tool(cwd: impl AsRef<Path>) -> Box<dyn Tool> {
    Box::new(read::ReadTool::new(cwd.as_ref()))
}

pub fn create_ls_tool(cwd: impl AsRef<Path>) -> Box<dyn Tool> {
    Box::new(ls::LsTool::new(cwd.as_ref()))
}

pub fn create_grep_tool(cwd: impl AsRef<Path>) -> Box<dyn Tool> {
    Box::new(grep::GrepTool::new(cwd.as_ref()))
}

pub fn create_find_tool(cwd: impl AsRef<Path>) -> Box<dyn Tool> {
    Box::new(find::FindTool::new(cwd.as_ref()))
}

pub fn create_bash_tool(cwd: impl AsRef<Path>) -> Box<dyn Tool> {
    Box::new(bash::BashTool::new(cwd.as_ref()))
}

pub fn create_edit_tool(cwd: impl AsRef<Path>) -> Box<dyn Tool> {
    Box::new(edit::EditTool::new(cwd.as_ref()))
}

pub fn create_write_tool(cwd: impl AsRef<Path>) -> Box<dyn Tool> {
    Box::new(write::WriteTool::new(cwd.as_ref()))
}
