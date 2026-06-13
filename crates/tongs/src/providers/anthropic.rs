//! The Anthropic Messages wire format.
//!
//! Ported from TS Pi's `anthropic.ts`. Both auth modes are supported: API key
//! (`x-api-key`) and OAuth (`sk-ant-oat…` bearer), where the OAuth
//! subscription path requires the Claude Code identity as the first system
//! block, the Claude Code identity headers ([`claude_code_headers`]), and
//! Claude Code canonical tool-name casing.
//!
//! Divergences from TS, by intent: no `eager_input_streaming` on tools (on
//! the API-key path the legacy `fine-grained-tool-streaming-2025-05-14` beta
//! is sent instead when tools are present); thinking is configured only when
//! the caller sets a thinking level (None omits the field entirely). The
//! OAuth path sends the **full** Claude Code identity header set — beta
//! flags, `user-agent`, the `X-Stainless-*` family, and fresh per-request
//! UUIDs — ported (with attribution) from anvil's
//! `anvil-temper-agent/src/provider/anthropic_oauth.rs`, the canonical
//! smith-derived subscription workaround.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::http::Client;
use crate::model::{
    AssistantMessage, ContentBlock, Message, Model, StopReason, StreamEvent, TextContent,
    ThinkingContent, ThinkingLevel, ToolCall, Usage, UserContent,
};
use crate::provider::{Context, EventStream, ModelEntry, Provider, StreamOptions, ToolDef};
use crate::providers::wire::{WireAdapter, sse_event_stream};
use crate::sse::SseEvent;
use crate::util::now_ms;
use crate::{Error, Result};

/// The identity the Claude subscription (OAuth) path requires as the first
/// system block.
pub const CLAUDE_CODE_SYSTEM_IDENTITY: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";

const ANTHROPIC_VERSION: &str = "2023-06-01";
const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

/// The `anthropic-beta` flag set Claude Code sends on the subscription OAuth
/// path. Copied from anvil's `anthropic_oauth.rs` so requests match the real
/// client's wire shape.
const OAUTH_BETAS: &str = concat!(
    "claude-code-20250219,",
    "oauth-2025-04-20,",
    "interleaved-thinking-2025-05-14,",
    "context-management-2025-06-27,",
    "prompt-caching-scope-2026-01-05,",
    "advisor-tool-2026-03-01,",
    "advanced-tool-use-2025-11-20,",
    "context-1m-2025-08-07,",
    "effort-2025-11-24,",
    "extended-cache-ttl-2025-04-11"
);

/// The user-agent Claude Code sends; kept verbatim with the rest of the
/// identity header set.
const CLAUDE_CODE_USER_AGENT: &str = "claude-cli/2.1.139 (external, sdk-cli)";

/// Claude Code-compatible identity headers for the subscription OAuth path.
///
/// Carries no token material — only client identity. The per-request UUIDs
/// (`x-client-request-id`, `X-Claude-Code-Session-Id`) are fresh on every
/// call. The OAuth branch of [`Provider::stream`] sends these by default;
/// model/entry/per-request headers still override any of them.
///
/// Ported from anvil's `request_headers()` (see module docs).
pub fn claude_code_headers() -> Vec<(String, String)> {
    vec![
        (
            "x-client-request-id".to_string(),
            uuid::Uuid::new_v4().to_string(),
        ),
        ("anthropic-beta".to_string(), OAUTH_BETAS.to_string()),
        (
            "anthropic-version".to_string(),
            ANTHROPIC_VERSION.to_string(),
        ),
        ("user-agent".to_string(), CLAUDE_CODE_USER_AGENT.to_string()),
        ("x-app".to_string(), "cli".to_string()),
        (
            "X-Claude-Code-Session-Id".to_string(),
            uuid::Uuid::new_v4().to_string(),
        ),
        ("X-Stainless-Arch".to_string(), "x64".to_string()),
        ("X-Stainless-Lang".to_string(), "js".to_string()),
        ("X-Stainless-OS".to_string(), "Linux".to_string()),
        (
            "X-Stainless-Package-Version".to_string(),
            "0.93.0".to_string(),
        ),
        ("X-Stainless-Retry-Count".to_string(), "0".to_string()),
        ("X-Stainless-Runtime".to_string(), "node".to_string()),
        (
            "X-Stainless-Runtime-Version".to_string(),
            "v24.3.0".to_string(),
        ),
        ("X-Stainless-Timeout".to_string(), "600".to_string()),
    ]
}

/// Claude Code 2.x canonical tool names; the OAuth path expects this casing.
const CLAUDE_CODE_TOOLS: [&str; 17] = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// True for Anthropic subscription OAuth access tokens.
pub(crate) fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// Canonical Claude Code casing for a tool name, when it matches one.
fn to_claude_code_name(name: &str) -> String {
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|tool| tool.eq_ignore_ascii_case(name))
        .map(|tool| tool.to_string())
        .unwrap_or_else(|| name.to_string())
}

/// Maps a wire tool name back to the registered tool's spelling.
fn from_claude_code_name(name: &str, tools: &[ToolDef]) -> String {
    tools
        .iter()
        .find(|tool| tool.name.eq_ignore_ascii_case(name))
        .map(|tool| tool.name.clone())
        .unwrap_or_else(|| name.to_string())
}

/// Anthropic requires tool ids matching `^[a-zA-Z0-9_-]+$`, max 64 chars.
fn normalize_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

// ---------------------------------------------------------------------------
// Pure: request building.
// ---------------------------------------------------------------------------

/// Builds the Messages request body.
pub(crate) fn build_anthropic_request(
    model: &Model,
    context: &Context<'_>,
    options: &StreamOptions,
    oauth: bool,
) -> Value {
    let mut body = json!({
        "model": model.id,
        "messages": convert_anthropic_messages(model, context, oauth),
        "max_tokens": options.max_tokens.unwrap_or(model.max_tokens).max(1),
        "stream": true,
    });
    let object = body.as_object_mut().expect("body is an object");

    let mut system: Vec<Value> = Vec::new();
    if oauth {
        system.push(json!({ "type": "text", "text": CLAUDE_CODE_SYSTEM_IDENTITY }));
    }
    if let Some(prompt) = context.system_prompt.as_deref()
        && !prompt.is_empty()
        // Callers that already use the identity as their system prompt (the
        // anvil OAuth adapter) must not send it twice.
        && !(oauth && prompt == CLAUDE_CODE_SYSTEM_IDENTITY)
    {
        system.push(json!({ "type": "text", "text": prompt }));
    }
    if !system.is_empty() {
        object.insert("system".to_string(), Value::Array(system));
    }

    // Temperature is incompatible with extended thinking.
    let thinking_enabled =
        matches!(options.thinking_level, Some(level) if level != ThinkingLevel::Off);
    if let Some(temperature) = options.temperature
        && !thinking_enabled
    {
        object.insert("temperature".to_string(), json!(temperature));
    }

    if !context.tools.is_empty() {
        let tools: Vec<Value> = context
            .tools
            .iter()
            .map(|tool| {
                let properties = tool
                    .parameters
                    .get("properties")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let required = tool
                    .parameters
                    .get("required")
                    .cloned()
                    .unwrap_or_else(|| json!([]));
                let name = if oauth {
                    to_claude_code_name(&tool.name)
                } else {
                    tool.name.clone()
                };
                json!({
                    "name": name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                })
            })
            .collect();
        object.insert("tools".to_string(), Value::Array(tools));
    }

    if model.reasoning
        && let Some(level) = options.thinking_level
    {
        let thinking = match level {
            ThinkingLevel::Off => json!({ "type": "disabled" }),
            // Modern Claude models use adaptive thinking with an effort knob.
            other => {
                let effort = match other {
                    ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
                    ThinkingLevel::Medium => "medium",
                    ThinkingLevel::XHigh => "xhigh",
                    _ => "high",
                };
                object.insert("output_config".to_string(), json!({ "effort": effort }));
                json!({ "type": "adaptive", "display": "summarized" })
            }
        };
        object.insert("thinking".to_string(), thinking);
    }

    body
}

/// Converts the unified conversation into Anthropic `messages`.
pub(crate) fn convert_anthropic_messages(
    model: &Model,
    context: &Context<'_>,
    oauth: bool,
) -> Vec<Value> {
    let supports_images = model.input.contains(&crate::model::InputType::Image);
    let mut messages: Vec<Value> = Vec::new();
    // Consecutive tool results merge into one user turn.
    let mut pending_tool_results: Vec<Value> = Vec::new();

    let flush_tool_results = |messages: &mut Vec<Value>, pending: &mut Vec<Value>| {
        if !pending.is_empty() {
            messages.push(json!({ "role": "user", "content": std::mem::take(pending) }));
        }
    };

    for message in context.messages.iter() {
        match message {
            Message::ToolResult(result) => {
                let content: Vec<Value> = result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) => {
                            Some(json!({ "type": "text", "text": text.text }))
                        }
                        ContentBlock::Image(image) if supports_images => Some(json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.mime_type,
                                "data": image.data,
                            },
                        })),
                        _ => None,
                    })
                    .collect();
                pending_tool_results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": normalize_tool_call_id(&result.tool_call_id),
                    "content": content,
                    "is_error": result.is_error,
                }));
            }
            Message::User(user) => {
                flush_tool_results(&mut messages, &mut pending_tool_results);
                match &user.content {
                    UserContent::Text(text) => {
                        if !text.trim().is_empty() {
                            messages.push(json!({ "role": "user", "content": text }));
                        }
                    }
                    UserContent::Blocks(blocks) => {
                        let content: Vec<Value> = blocks
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                                    Some(json!({ "type": "text", "text": text.text }))
                                }
                                ContentBlock::Image(image) if supports_images => Some(json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image.mime_type,
                                        "data": image.data,
                                    },
                                })),
                                _ => None,
                            })
                            .collect();
                        if !content.is_empty() {
                            messages.push(json!({ "role": "user", "content": content }));
                        }
                    }
                }
            }
            Message::Assistant(assistant) => {
                flush_tool_results(&mut messages, &mut pending_tool_results);
                let mut blocks: Vec<Value> = Vec::new();
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text(text) => {
                            if !text.text.trim().is_empty() {
                                blocks.push(json!({ "type": "text", "text": text.text }));
                            }
                        }
                        ContentBlock::Thinking(thinking) => {
                            if thinking.redacted {
                                if let Some(signature) = thinking.thinking_signature.as_deref() {
                                    blocks.push(json!({
                                        "type": "redacted_thinking",
                                        "data": signature,
                                    }));
                                }
                                continue;
                            }
                            if thinking.thinking.trim().is_empty() {
                                continue;
                            }
                            match thinking.thinking_signature.as_deref() {
                                // A missing signature (e.g. an aborted stream)
                                // replays as plain text.
                                None | Some("") => blocks.push(json!({
                                    "type": "text",
                                    "text": thinking.thinking,
                                })),
                                Some(signature) => blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": thinking.thinking,
                                    "signature": signature,
                                })),
                            }
                        }
                        ContentBlock::ToolCall(call) => {
                            let name = if oauth {
                                to_claude_code_name(&call.name)
                            } else {
                                call.name.clone()
                            };
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": normalize_tool_call_id(&call.id),
                                "name": name,
                                "input": call.arguments,
                            }));
                        }
                        ContentBlock::Image(_) => {}
                    }
                }
                if !blocks.is_empty() {
                    messages.push(json!({ "role": "assistant", "content": blocks }));
                }
            }
        }
    }
    flush_tool_results(&mut messages, &mut pending_tool_results);
    messages
}

/// Normalizes a base URL to the `…/v1/messages` endpoint.
pub(crate) fn resolve_anthropic_url(base_url: &str) -> String {
    let raw = if base_url.trim().is_empty() {
        "https://api.anthropic.com"
    } else {
        base_url
    };
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/v1/messages") {
        normalized.to_string()
    } else if normalized.ends_with("/v1") {
        format!("{normalized}/messages")
    } else {
        format!("{normalized}/v1/messages")
    }
}

// ---------------------------------------------------------------------------
// Pure: SSE folding.
// ---------------------------------------------------------------------------

pub(crate) struct AnthropicAdapter {
    output: AssistantMessage,
    model_cost: crate::model::ModelCost,
    /// Registered tool names, for mapping Claude Code casing back.
    tool_names: Vec<ToolDef>,
    oauth: bool,
    /// Wire `index` of each open block → position in `output.content`, plus
    /// the tool-call scratch JSON for tool blocks.
    open_blocks: HashMap<u64, OpenBlock>,
    started: bool,
    saw_message_start: bool,
    finished: bool,
}

struct OpenBlock {
    content_index: usize,
    partial_json: String,
}

impl AnthropicAdapter {
    pub(crate) fn new(model: &Model, tools: &[ToolDef], oauth: bool) -> Self {
        Self {
            output: AssistantMessage {
                content: Vec::new(),
                api: "anthropic-messages".to_string(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: now_ms(),
            },
            model_cost: model.cost,
            tool_names: tools.to_vec(),
            oauth,
            open_blocks: HashMap::new(),
            started: false,
            saw_message_start: false,
            finished: false,
        }
    }

    fn fail(&mut self, message: String) -> Vec<StreamEvent> {
        self.finished = true;
        self.output.stop_reason = StopReason::Error;
        self.output.error_message = Some(message);
        vec![StreamEvent::Error {
            reason: StopReason::Error,
            error: self.output.clone(),
        }]
    }

    fn apply_usage(&mut self, usage: &Value) {
        let mut read = |key: &str, slot: fn(&mut Usage) -> &mut u64| {
            if let Some(value) = usage.get(key).and_then(Value::as_u64) {
                *slot(&mut self.output.usage) = value;
            }
        };
        read("input_tokens", |usage| &mut usage.input);
        read("output_tokens", |usage| &mut usage.output);
        read("cache_read_input_tokens", |usage| &mut usage.cache_read);
        read("cache_creation_input_tokens", |usage| {
            &mut usage.cache_write
        });
        let totals = &mut self.output.usage;
        totals.total_tokens = totals.input + totals.output + totals.cache_read + totals.cache_write;
        self.output.usage.price_with(self.model_cost);
    }

    fn on_json(&mut self, event: &Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(StreamEvent::Start);
        }
        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "message_start" => {
                self.saw_message_start = true;
                if let Some(usage) = event.pointer("/message/usage") {
                    self.apply_usage(usage);
                }
            }
            "content_block_start" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let Some(block) = event.get("content_block") else {
                    return events;
                };
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        self.output
                            .content
                            .push(ContentBlock::Text(TextContent::default()));
                        self.open_blocks.insert(
                            index,
                            OpenBlock {
                                content_index: self.output.content.len() - 1,
                                partial_json: String::new(),
                            },
                        );
                        events.push(StreamEvent::TextStart {
                            content_index: self.output.content.len() - 1,
                        });
                    }
                    Some("thinking") => {
                        self.output
                            .content
                            .push(ContentBlock::Thinking(ThinkingContent {
                                thinking: String::new(),
                                thinking_signature: Some(String::new()),
                                redacted: false,
                            }));
                        self.open_blocks.insert(
                            index,
                            OpenBlock {
                                content_index: self.output.content.len() - 1,
                                partial_json: String::new(),
                            },
                        );
                        events.push(StreamEvent::ThinkingStart {
                            content_index: self.output.content.len() - 1,
                        });
                    }
                    Some("redacted_thinking") => {
                        let data = block
                            .get("data")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        self.output
                            .content
                            .push(ContentBlock::Thinking(ThinkingContent {
                                thinking: "[Reasoning redacted]".to_string(),
                                thinking_signature: Some(data),
                                redacted: true,
                            }));
                        self.open_blocks.insert(
                            index,
                            OpenBlock {
                                content_index: self.output.content.len() - 1,
                                partial_json: String::new(),
                            },
                        );
                        events.push(StreamEvent::ThinkingStart {
                            content_index: self.output.content.len() - 1,
                        });
                    }
                    Some("tool_use") => {
                        let id = block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let wire_name = block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let name = if self.oauth {
                            from_claude_code_name(wire_name, &self.tool_names)
                        } else {
                            wire_name.to_string()
                        };
                        let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                        self.output.content.push(ContentBlock::ToolCall(ToolCall {
                            id,
                            name,
                            arguments: if input.is_object() { input } else { json!({}) },
                        }));
                        self.open_blocks.insert(
                            index,
                            OpenBlock {
                                content_index: self.output.content.len() - 1,
                                partial_json: String::new(),
                            },
                        );
                        events.push(StreamEvent::ToolCallStart {
                            content_index: self.output.content.len() - 1,
                        });
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let Some(open) = self.open_blocks.get_mut(&index) else {
                    return events;
                };
                let content_index = open.content_index;
                let Some(delta) = event.get("delta") else {
                    return events;
                };
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str)
                            && let Some(ContentBlock::Text(block)) =
                                self.output.content.get_mut(content_index)
                        {
                            block.text.push_str(text);
                            events.push(StreamEvent::TextDelta {
                                content_index,
                                delta: text.to_string(),
                            });
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(text) = delta.get("thinking").and_then(Value::as_str)
                            && let Some(ContentBlock::Thinking(block)) =
                                self.output.content.get_mut(content_index)
                        {
                            block.thinking.push_str(text);
                            events.push(StreamEvent::ThinkingDelta {
                                content_index,
                                delta: text.to_string(),
                            });
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                            open.partial_json.push_str(partial);
                            events.push(StreamEvent::ToolCallDelta {
                                content_index,
                                delta: partial.to_string(),
                            });
                        }
                    }
                    Some("signature_delta") => {
                        if let Some(signature) = delta.get("signature").and_then(Value::as_str)
                            && let Some(ContentBlock::Thinking(block)) =
                                self.output.content.get_mut(content_index)
                        {
                            block
                                .thinking_signature
                                .get_or_insert_with(String::new)
                                .push_str(signature);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let Some(open) = self.open_blocks.remove(&index) else {
                    return events;
                };
                match self.output.content.get_mut(open.content_index) {
                    Some(ContentBlock::Text(block)) => {
                        events.push(StreamEvent::TextEnd {
                            content_index: open.content_index,
                            content: block.text.clone(),
                        });
                    }
                    Some(ContentBlock::Thinking(block)) => {
                        events.push(StreamEvent::ThinkingEnd {
                            content_index: open.content_index,
                            content: block.thinking.clone(),
                        });
                    }
                    Some(ContentBlock::ToolCall(block)) => {
                        if !open.partial_json.is_empty() {
                            block.arguments = serde_json::from_str(&open.partial_json)
                                .unwrap_or_else(|_| json!({}));
                        }
                        events.push(StreamEvent::ToolCallEnd {
                            content_index: open.content_index,
                            tool_call: block.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    let (stop, error_message): (StopReason, Option<String>) = match reason {
                        "end_turn" | "pause_turn" | "stop_sequence" => (StopReason::Stop, None),
                        "max_tokens" => (StopReason::Length, None),
                        "tool_use" => (StopReason::ToolUse, None),
                        "refusal" => (
                            StopReason::Error,
                            Some(
                                event
                                    .pointer("/delta/stop_details/explanation")
                                    .and_then(Value::as_str)
                                    .unwrap_or("The model refused to complete the request")
                                    .to_string(),
                            ),
                        ),
                        other => (
                            StopReason::Error,
                            Some(format!("unhandled stop reason: {other}")),
                        ),
                    };
                    self.output.stop_reason = stop;
                    if let Some(message) = error_message {
                        self.output.error_message = Some(message);
                    }
                }
                if let Some(usage) = event.get("usage") {
                    self.apply_usage(usage);
                }
            }
            "message_stop" => {
                self.finished = true;
                if matches!(
                    self.output.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) {
                    let reason = self.output.stop_reason;
                    if self.output.error_message.is_none() {
                        self.output.error_message = Some("an unknown error occurred".to_string());
                    }
                    events.push(StreamEvent::Error {
                        reason,
                        error: self.output.clone(),
                    });
                } else {
                    events.push(StreamEvent::Done {
                        reason: self.output.stop_reason,
                        message: self.output.clone(),
                    });
                }
            }
            _ => {}
        }
        events
    }
}

impl WireAdapter for AnthropicAdapter {
    fn on_sse(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        if event.event.as_deref() == Some("error") {
            return Ok(self.fail(event.data));
        }
        // Only the six message events carry protocol state; ping et al. skip.
        const MESSAGE_EVENTS: [&str; 6] = [
            "message_start",
            "message_delta",
            "message_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ];
        let Some(name) = event.event.as_deref() else {
            return Ok(Vec::new());
        };
        if !MESSAGE_EVENTS.contains(&name) {
            return Ok(Vec::new());
        }
        let value: Value = serde_json::from_str(&event.data).map_err(|error| {
            Error::Decode(format!("invalid Anthropic SSE event {name}: {error}"))
        })?;
        Ok(self.on_json(&value))
    }

    fn on_eof(&mut self) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        if self.saw_message_start {
            return Ok(self.fail("Anthropic stream ended before message_stop".to_string()));
        }
        Ok(self.fail("Anthropic stream ended without a message".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Shell: the provider.
// ---------------------------------------------------------------------------

/// The Anthropic Messages provider.
pub struct AnthropicProvider {
    entry: ModelEntry,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(entry: ModelEntry, client: Client) -> Self {
        Self { entry, client }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn api(&self) -> &str {
        "anthropic-messages"
    }

    async fn stream(&self, context: &Context<'_>, options: &StreamOptions) -> Result<EventStream> {
        let model = &self.entry.model;
        let token = options
            .api_key
            .as_deref()
            .or(self.entry.api_key.as_deref())
            .ok_or_else(|| Error::Auth(format!("no API key for provider: {}", model.provider)))?;
        let oauth = is_oauth_token(token);
        let body = build_anthropic_request(model, context, options, oauth);

        // Default headers first; entry and per-request headers override them
        // (case-insensitively), so callers can replace e.g. anthropic-beta.
        let mut headers: Vec<(String, String)> =
            vec![("accept".to_string(), "text/event-stream".to_string())];
        if oauth {
            // The full Claude Code identity set (betas, user-agent, the
            // X-Stainless family, fresh per-request UUIDs) — the subscription
            // path is gated on looking like the real client.
            headers.extend(claude_code_headers());
            headers.push(("Authorization".to_string(), format!("Bearer {token}")));
        } else {
            headers.push((
                "anthropic-version".to_string(),
                ANTHROPIC_VERSION.to_string(),
            ));
            if !context.tools.is_empty() {
                headers.push((
                    "anthropic-beta".to_string(),
                    FINE_GRAINED_TOOL_STREAMING_BETA.to_string(),
                ));
            }
            headers.push(("x-api-key".to_string(), token.to_string()));
        }
        let mut set_header = |name: &str, value: &str| {
            if let Some(slot) = headers
                .iter_mut()
                .find(|(existing, _)| existing.eq_ignore_ascii_case(name))
            {
                slot.1 = value.to_string();
            } else {
                headers.push((name.to_string(), value.to_string()));
            }
        };
        for (name, value) in &model.headers {
            set_header(name, value);
        }
        for (name, value) in &self.entry.headers {
            set_header(name, value);
        }
        for (name, value) in &options.headers {
            set_header(name, value);
        }

        let mut request = self
            .client
            .post(&resolve_anthropic_url(&model.base_url))
            .json(&body)?;
        for (name, value) in headers {
            request = request.header(name, value);
        }

        let response = request.send_streaming().await?;
        if !(200..300).contains(&response.status()) {
            let status = response.status();
            let body = response.read_to_end().await.unwrap_or_default();
            let text = String::from_utf8_lossy(&body);
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|value| {
                    value
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_else(|| {
                    if text.is_empty() {
                        format!("request failed with HTTP {status}")
                    } else {
                        text.into_owned()
                    }
                });
            return Err(Error::Api { status, message });
        }

        let adapter = AnthropicAdapter::new(model, &context.tools, oauth);
        Ok(sse_event_stream(response.into_body(), adapter))
    }
}

#[cfg(test)]
#[path = "anthropic_tests.rs"]
mod tests;
