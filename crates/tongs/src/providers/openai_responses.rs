//! The OpenAI Responses wire format, in its ChatGPT Codex variant.
//!
//! Ported from TS Pi's `openai-codex-responses.ts` + `openai-responses-shared.ts`
//! (SSE transport only; the WebSocket transport is deliberately not ported).
//! Split sans-IO: request building and SSE-event folding are pure and tested
//! with synthetic events; [`CodexProvider`] is the thin HTTP shell.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::http::Client;
use crate::model::{
    AssistantMessage, ContentBlock, Message, Model, StopReason, StreamEvent, ToolCall, Usage,
    UserContent,
};
use crate::provider::{Context, EventStream, ModelEntry, Provider, StreamOptions, ToolDef};
use crate::providers::wire::{WireAdapter, sse_event_stream};
use crate::sse::SseEvent;
use crate::util::{base64url_decode, now_ms, short_hash};
use crate::{Error, Result};

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
/// Providers whose `call_id|item_id` tool-call ids replay natively.
const TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];

// ---------------------------------------------------------------------------
// Pure: request building.
// ---------------------------------------------------------------------------

/// Builds the Codex Responses request body.
pub(crate) fn build_codex_request(
    model: &Model,
    context: &Context<'_>,
    options: &StreamOptions,
) -> Value {
    let mut body = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": context
            .system_prompt
            .as_deref()
            .filter(|prompt| !prompt.is_empty())
            .unwrap_or("You are a helpful assistant."),
        "input": convert_responses_messages(model, context, false),
        "text": { "verbosity": "low" },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });
    let object = body.as_object_mut().expect("body is an object");

    if let Some(temperature) = options.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(session_id) = options.session_id.as_deref() {
        object.insert("prompt_cache_key".to_string(), json!(session_id));
    }
    if !context.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(convert_responses_tools(&context.tools, None)),
        );
    }
    if let Some(level) = options.thinking_level {
        let effort = match level {
            crate::model::ThinkingLevel::Off => "none",
            other => other.as_str(),
        };
        object.insert(
            "reasoning".to_string(),
            json!({ "effort": effort, "summary": "auto" }),
        );
    }
    body
}

/// Converts tool definitions to Responses `function` tools. `strict: None`
/// serializes as an explicit `false`, matching the codex CLI's request shape
/// (jig's authoritative codex recordings send `"strict": false`, never null —
/// a divergence the tongs subject-conformance T3 gate caught).
pub(crate) fn convert_responses_tools(tools: &[ToolDef], strict: Option<bool>) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": strict.unwrap_or(false),
            })
        })
        .collect()
}

/// Converts the unified conversation into Responses `input` items.
pub(crate) fn convert_responses_messages(
    model: &Model,
    context: &Context<'_>,
    include_system_prompt: bool,
) -> Vec<Value> {
    let mut items = Vec::new();

    if include_system_prompt && let Some(prompt) = context.system_prompt.as_deref() {
        let role = if model.reasoning {
            "developer"
        } else {
            "system"
        };
        items.push(json!({ "role": role, "content": prompt }));
    }

    // First pass: map original tool-call ids to their normalized replay form,
    // so tool results reference the same ids their calls were sent under.
    let mut id_map: HashMap<String, String> = HashMap::new();
    for message in context.messages.iter() {
        if let Message::Assistant(assistant) = message {
            for block in &assistant.content {
                if let ContentBlock::ToolCall(call) = block {
                    id_map.insert(
                        call.id.clone(),
                        normalize_tool_call_id(&call.id, model, assistant),
                    );
                }
            }
        }
    }

    let supports_images = model.input.contains(&crate::model::InputType::Image);

    for (msg_index, message) in context.messages.iter().enumerate() {
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    items.push(json!({
                        "role": "user",
                        "content": [{ "type": "input_text", "text": text }],
                    }));
                }
                UserContent::Blocks(blocks) => {
                    let content: Vec<Value> = blocks
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(text) => {
                                Some(json!({ "type": "input_text", "text": text.text }))
                            }
                            ContentBlock::Image(image) if supports_images => Some(json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!(
                                    "data:{};base64,{}",
                                    image.mime_type, image.data
                                ),
                            })),
                            ContentBlock::Image(_) => Some(json!({
                                "type": "input_text",
                                "text": "(image omitted: model does not support images)",
                            })),
                            _ => None,
                        })
                        .collect();
                    if !content.is_empty() {
                        items.push(json!({ "role": "user", "content": content }));
                    }
                }
            },
            Message::Assistant(assistant) => {
                // Items from a different model on the same provider/api must
                // not replay `fc_…` item ids (OpenAI validates their pairing
                // with reasoning items).
                let different_model = assistant.model != model.id
                    && assistant.provider == model.provider
                    && assistant.api == model.api;
                let mut text_block_index = 0usize;
                for block in &assistant.content {
                    match block {
                        ContentBlock::Thinking(thinking) => {
                            if let Some(signature) = thinking.thinking_signature.as_deref()
                                && let Ok(item) = serde_json::from_str::<Value>(signature)
                            {
                                items.push(item);
                            }
                        }
                        ContentBlock::Text(text) => {
                            let fallback = if text_block_index == 0 {
                                format!("msg_pi_{msg_index}")
                            } else {
                                format!("msg_pi_{msg_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            let parsed = parse_text_signature(text.text_signature.as_deref());
                            let mut id = parsed
                                .as_ref()
                                .map(|signature| signature.id.clone())
                                .unwrap_or(fallback);
                            if id.len() > 64 {
                                id = format!("msg_{}", short_hash(&id));
                            }
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": text.text,
                                    "annotations": [],
                                }],
                                "status": "completed",
                                "id": id,
                            });
                            if let Some(phase) = parsed
                                .as_ref()
                                .and_then(|signature| signature.phase.clone())
                            {
                                item.as_object_mut()
                                    .expect("item is an object")
                                    .insert("phase".to_string(), json!(phase));
                            }
                            items.push(item);
                        }
                        ContentBlock::ToolCall(call) => {
                            let normalized = id_map.get(&call.id).cloned().unwrap_or_else(|| {
                                normalize_tool_call_id(&call.id, model, assistant)
                            });
                            let (call_id, item_id) = split_tool_call_id(&normalized);
                            let item_id = match item_id {
                                Some(id) if different_model && id.starts_with("fc_") => None,
                                other => other,
                            };
                            items.push(json!({
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": call.name,
                                "arguments": call.arguments.to_string(),
                            }));
                        }
                        ContentBlock::Image(_) => {}
                    }
                }
            }
            Message::ToolResult(result) => {
                let normalized = id_map
                    .get(&result.tool_call_id)
                    .cloned()
                    .unwrap_or_else(|| normalize_id_part(&result.tool_call_id));
                let (call_id, _) = split_tool_call_id(&normalized);
                let text: String = result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = result
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Image(_)));

                let output: Value = if has_images && supports_images {
                    let mut parts = Vec::new();
                    if !text.is_empty() {
                        parts.push(json!({ "type": "input_text", "text": text }));
                    }
                    for block in &result.content {
                        if let ContentBlock::Image(image) = block {
                            parts.push(json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!(
                                    "data:{};base64,{}",
                                    image.mime_type, image.data
                                ),
                            }));
                        }
                    }
                    Value::Array(parts)
                } else if text.is_empty() && has_images {
                    json!("(see attached image)")
                } else {
                    json!(text)
                };

                items.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));
            }
        }
    }

    items
}

/// A parsed `textSignature` (v1 JSON or a legacy bare id).
struct TextSignature {
    id: String,
    phase: Option<String>,
}

fn parse_text_signature(signature: Option<&str>) -> Option<TextSignature> {
    let signature = signature?;
    if signature.starts_with('{')
        && let Ok(value) = serde_json::from_str::<Value>(signature)
        && value.get("v").and_then(Value::as_i64) == Some(1)
        && let Some(id) = value.get("id").and_then(Value::as_str)
    {
        let phase = value
            .get("phase")
            .and_then(Value::as_str)
            .filter(|phase| *phase == "commentary" || *phase == "final_answer")
            .map(str::to_string);
        return Some(TextSignature {
            id: id.to_string(),
            phase,
        });
    }
    Some(TextSignature {
        id: signature.to_string(),
        phase: None,
    })
}

fn normalize_id_part(part: &str) -> String {
    let sanitized: String = part
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let truncated = if sanitized.len() > 64 {
        sanitized[..64].to_string()
    } else {
        sanitized
    };
    truncated.trim_end_matches('_').to_string()
}

/// Normalizes a `call_id|item_id` pair for replay on this model.
fn normalize_tool_call_id(id: &str, model: &Model, source: &AssistantMessage) -> String {
    if !TOOL_CALL_PROVIDERS.contains(&model.provider.as_str()) {
        return normalize_id_part(id);
    }
    let Some((call_id, item_id)) = id.split_once('|') else {
        return normalize_id_part(id);
    };
    let normalized_call_id = normalize_id_part(call_id);
    let foreign = source.provider != model.provider || source.api != model.api;
    let mut normalized_item_id = if foreign {
        let hashed = format!("fc_{}", short_hash(item_id));
        if hashed.len() > 64 {
            hashed[..64].to_string()
        } else {
            hashed
        }
    } else {
        normalize_id_part(item_id)
    };
    // The Responses API requires item ids to start with "fc".
    if !normalized_item_id.starts_with("fc_") {
        normalized_item_id = normalize_id_part(&format!("fc_{normalized_item_id}"));
    }
    format!("{normalized_call_id}|{normalized_item_id}")
}

fn split_tool_call_id(id: &str) -> (&str, Option<&str>) {
    match id.split_once('|') {
        Some((call_id, item_id)) => (call_id, Some(item_id)),
        None => (id, None),
    }
}

// ---------------------------------------------------------------------------
// Pure: SSE event folding.
// ---------------------------------------------------------------------------

/// What kind of output item is currently streaming.
enum CurrentItem {
    Reasoning {
        /// A summary part was opened (deltas only count inside one).
        has_summary_part: bool,
    },
    Message {
        /// The type of the last `content_part.added` ("output_text"/"refusal").
        last_part: Option<String>,
    },
    FunctionCall {
        call_id: String,
        item_id: String,
        name: String,
        partial_json: String,
    },
}

/// Folds Codex Responses SSE events into unified stream events.
pub(crate) struct ResponsesAdapter {
    output: AssistantMessage,
    model_cost: crate::model::ModelCost,
    current: Option<CurrentItem>,
    started: bool,
    finished: bool,
}

impl ResponsesAdapter {
    pub(crate) fn new(model: &Model, api: &str) -> Self {
        Self {
            output: AssistantMessage {
                content: Vec::new(),
                api: api.to_string(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: now_ms(),
            },
            model_cost: model.cost,
            current: None,
            started: false,
            finished: false,
        }
    }

    fn block_index(&self) -> usize {
        self.output.content.len().saturating_sub(1)
    }

    /// The terminal error event (provider-reported failure).
    fn fail(&mut self, message: String) -> Vec<StreamEvent> {
        self.finished = true;
        self.output.stop_reason = StopReason::Error;
        self.output.error_message = Some(message);
        vec![StreamEvent::Error {
            reason: StopReason::Error,
            error: self.output.clone(),
        }]
    }

    fn on_json(&mut self, event: &Value) -> Result<Vec<StreamEvent>> {
        let Some(kind) = event.get("type").and_then(Value::as_str) else {
            return Ok(Vec::new());
        };
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(StreamEvent::Start);
        }

        match kind {
            "error" => {
                let code = event.get("code").and_then(Value::as_str).unwrap_or("");
                let message = event.get("message").and_then(Value::as_str).unwrap_or("");
                let detail = if message.is_empty() && code.is_empty() {
                    event.to_string()
                } else if message.is_empty() {
                    code.to_string()
                } else {
                    message.to_string()
                };
                events.extend(self.fail(format!("Codex error: {detail}")));
            }
            "response.failed" => {
                let error = event.pointer("/response/error");
                let message = match error {
                    Some(error) => format!(
                        "{}: {}",
                        error
                            .get("code")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown"),
                        error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("no message"),
                    ),
                    None => "Codex response failed".to_string(),
                };
                events.extend(self.fail(message));
            }
            "response.output_item.added" => {
                let Some(item) = event.get("item") else {
                    return Ok(events);
                };
                match item.get("type").and_then(Value::as_str) {
                    Some("reasoning") => {
                        self.current = Some(CurrentItem::Reasoning {
                            has_summary_part: false,
                        });
                        self.output.content.push(ContentBlock::Thinking(
                            crate::model::ThinkingContent::default(),
                        ));
                        events.push(StreamEvent::ThinkingStart {
                            content_index: self.block_index(),
                        });
                    }
                    Some("message") => {
                        self.current = Some(CurrentItem::Message { last_part: None });
                        self.output
                            .content
                            .push(ContentBlock::Text(crate::model::TextContent::default()));
                        events.push(StreamEvent::TextStart {
                            content_index: self.block_index(),
                        });
                    }
                    Some("function_call") => {
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let item_id = item
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let partial_json = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        self.output.content.push(ContentBlock::ToolCall(ToolCall {
                            id: format!("{call_id}|{item_id}"),
                            name: name.clone(),
                            arguments: json!({}),
                        }));
                        self.current = Some(CurrentItem::FunctionCall {
                            call_id,
                            item_id,
                            name,
                            partial_json,
                        });
                        events.push(StreamEvent::ToolCallStart {
                            content_index: self.block_index(),
                        });
                    }
                    _ => {}
                }
            }
            "response.reasoning_summary_part.added" => {
                if let Some(CurrentItem::Reasoning { has_summary_part }) = &mut self.current {
                    *has_summary_part = true;
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(CurrentItem::Reasoning {
                    has_summary_part: true,
                }) = &self.current
                    && let Some(delta) = event.get("delta").and_then(Value::as_str)
                {
                    events.extend(self.push_thinking_delta(delta));
                }
            }
            "response.reasoning_summary_part.done" => {
                if let Some(CurrentItem::Reasoning {
                    has_summary_part: true,
                }) = &self.current
                {
                    events.extend(self.push_thinking_delta("\n\n"));
                }
            }
            "response.reasoning_text.delta" => {
                if matches!(self.current, Some(CurrentItem::Reasoning { .. }))
                    && let Some(delta) = event.get("delta").and_then(Value::as_str)
                {
                    events.extend(self.push_thinking_delta(delta));
                }
            }
            "response.content_part.added" => {
                if let Some(CurrentItem::Message { last_part }) = &mut self.current {
                    let part_type = event
                        .pointer("/part/type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if part_type == "output_text" || part_type == "refusal" {
                        *last_part = Some(part_type.to_string());
                    }
                }
            }
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    match &mut self.current {
                        Some(CurrentItem::Message {
                            last_part: Some(part),
                        }) if part == "output_text" => {
                            events.extend(self.push_text_delta(delta));
                        }
                        // Tolerate streams that skip the item/part bookkeeping
                        // (e.g. test fakes): implicitly open a text block.
                        Some(CurrentItem::Message { last_part }) if last_part.is_none() => {
                            *last_part = Some("output_text".to_string());
                            events.extend(self.push_text_delta(delta));
                        }
                        None => {
                            self.current = Some(CurrentItem::Message {
                                last_part: Some("output_text".to_string()),
                            });
                            self.output
                                .content
                                .push(ContentBlock::Text(crate::model::TextContent::default()));
                            events.push(StreamEvent::TextStart {
                                content_index: self.block_index(),
                            });
                            events.extend(self.push_text_delta(delta));
                        }
                        _ => {}
                    }
                }
            }
            "response.refusal.delta" => {
                if let Some(CurrentItem::Message {
                    last_part: Some(part),
                }) = &self.current
                    && part == "refusal"
                    && let Some(delta) = event.get("delta").and_then(Value::as_str)
                {
                    events.extend(self.push_text_delta(delta));
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(CurrentItem::FunctionCall { partial_json, .. }) = &mut self.current
                    && let Some(delta) = event.get("delta").and_then(Value::as_str)
                {
                    partial_json.push_str(delta);
                    events.push(StreamEvent::ToolCallDelta {
                        content_index: self.output.content.len().saturating_sub(1),
                        delta: delta.to_string(),
                    });
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(CurrentItem::FunctionCall { partial_json, .. }) = &mut self.current
                    && let Some(arguments) = event.get("arguments").and_then(Value::as_str)
                {
                    *partial_json = arguments.to_string();
                }
            }
            "response.output_item.done" => {
                events.extend(self.on_item_done(event.get("item")));
            }
            "response.completed" => {
                events.extend(self.on_completed(event.get("response")));
            }
            _ => {}
        }
        Ok(events)
    }

    fn push_thinking_delta(&mut self, delta: &str) -> Vec<StreamEvent> {
        let index = self.block_index();
        if let Some(ContentBlock::Thinking(block)) = self.output.content.last_mut() {
            block.thinking.push_str(delta);
            vec![StreamEvent::ThinkingDelta {
                content_index: index,
                delta: delta.to_string(),
            }]
        } else {
            Vec::new()
        }
    }

    fn push_text_delta(&mut self, delta: &str) -> Vec<StreamEvent> {
        let index = self.block_index();
        if let Some(ContentBlock::Text(block)) = self.output.content.last_mut() {
            block.text.push_str(delta);
            vec![StreamEvent::TextDelta {
                content_index: index,
                delta: delta.to_string(),
            }]
        } else {
            Vec::new()
        }
    }

    fn on_item_done(&mut self, item: Option<&Value>) -> Vec<StreamEvent> {
        let Some(item) = item else {
            return Vec::new();
        };
        let index = self.block_index();
        let current = self.current.take();
        match (item.get("type").and_then(Value::as_str), current) {
            (Some("reasoning"), Some(CurrentItem::Reasoning { .. })) => {
                let joined = |key: &str| -> String {
                    item.get(key)
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .filter_map(|part| part.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        })
                        .unwrap_or_default()
                };
                let summary = joined("summary");
                let content_text = joined("content");
                if let Some(ContentBlock::Thinking(block)) = self.output.content.last_mut() {
                    if !summary.is_empty() {
                        block.thinking = summary;
                    } else if !content_text.is_empty() {
                        block.thinking = content_text;
                    }
                    block.thinking_signature = Some(item.to_string());
                    return vec![StreamEvent::ThinkingEnd {
                        content_index: index,
                        content: block.thinking.clone(),
                    }];
                }
                Vec::new()
            }
            (Some("message"), Some(CurrentItem::Message { .. })) => {
                let text: String = item
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                                Some("output_text") => part.get("text").and_then(Value::as_str),
                                Some("refusal") => part.get("refusal").and_then(Value::as_str),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();
                if let Some(ContentBlock::Text(block)) = self.output.content.last_mut() {
                    block.text = text;
                    let mut signature = json!({
                        "v": 1,
                        "id": item.get("id").cloned().unwrap_or(Value::Null),
                    });
                    if let Some(phase) = item.get("phase").and_then(Value::as_str) {
                        signature
                            .as_object_mut()
                            .expect("signature is an object")
                            .insert("phase".to_string(), json!(phase));
                    }
                    block.text_signature = Some(signature.to_string());
                    return vec![StreamEvent::TextEnd {
                        content_index: index,
                        content: block.text.clone(),
                    }];
                }
                Vec::new()
            }
            (Some("function_call"), current) => {
                let (call_id, item_id, name, partial_json) = match current {
                    Some(CurrentItem::FunctionCall {
                        call_id,
                        item_id,
                        name,
                        partial_json,
                    }) => (call_id, item_id, name, partial_json),
                    _ => (
                        item.get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        item.get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        item.get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        item.get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}")
                            .to_string(),
                    ),
                };
                let arguments =
                    serde_json::from_str::<Value>(&partial_json).unwrap_or_else(|_| json!({}));
                let tool_call = ToolCall {
                    id: format!("{call_id}|{item_id}"),
                    name,
                    arguments,
                };
                if let Some(ContentBlock::ToolCall(block)) = self.output.content.last_mut() {
                    *block = tool_call.clone();
                }
                vec![StreamEvent::ToolCallEnd {
                    content_index: index,
                    tool_call,
                }]
            }
            _ => Vec::new(),
        }
    }

    fn on_completed(&mut self, response: Option<&Value>) -> Vec<StreamEvent> {
        self.finished = true;
        if let Some(usage) = response.and_then(|response| response.get("usage")) {
            let read = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
            let cached = usage
                .pointer("/input_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            self.output.usage = Usage {
                // Cached tokens are included in input_tokens; report them apart.
                input: read("input_tokens").saturating_sub(cached),
                output: read("output_tokens"),
                cache_read: cached,
                cache_write: 0,
                total_tokens: read("total_tokens"),
                cost: Default::default(),
            };
            self.output.usage.price_with(self.model_cost);
        }
        let status = response
            .and_then(|response| response.get("status"))
            .and_then(Value::as_str);
        self.output.stop_reason = match status {
            Some("incomplete") => StopReason::Length,
            Some("failed" | "cancelled") => StopReason::Error,
            // completed / in_progress / queued / absent.
            _ => StopReason::Stop,
        };
        if self.output.stop_reason == StopReason::Stop
            && self
                .output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
        {
            self.output.stop_reason = StopReason::ToolUse;
        }
        if self.output.stop_reason == StopReason::Error {
            self.output.error_message = Some("response failed".to_string());
            return vec![StreamEvent::Error {
                reason: StopReason::Error,
                error: self.output.clone(),
            }];
        }
        vec![StreamEvent::Done {
            reason: self.output.stop_reason,
            message: self.output.clone(),
        }]
    }
}

impl WireAdapter for ResponsesAdapter {
    fn on_sse(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(Vec::new());
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|error| Error::Decode(format!("invalid Codex SSE JSON: {error}")))?;
        // The codex stream's `response.done` / `response.incomplete` are
        // aliases of `response.completed` with a status inside.
        if let Some(kind) = value.get("type").and_then(Value::as_str)
            && (kind == "response.done" || kind == "response.incomplete")
        {
            let mut normalized = value.clone();
            normalized
                .as_object_mut()
                .expect("event is an object")
                .insert("type".to_string(), json!("response.completed"));
            return self.on_json(&normalized);
        }
        self.on_json(&value)
    }

    fn on_eof(&mut self) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        Ok(self.fail("model stream ended before response.completed".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Shell: the provider.
// ---------------------------------------------------------------------------

/// The ChatGPT Codex Responses provider.
pub struct CodexProvider {
    entry: ModelEntry,
    client: Client,
}

impl CodexProvider {
    pub fn new(entry: ModelEntry, client: Client) -> Self {
        Self { entry, client }
    }
}

#[async_trait]
impl Provider for CodexProvider {
    fn api(&self) -> &str {
        "openai-codex-responses"
    }

    async fn stream(&self, context: &Context<'_>, options: &StreamOptions) -> Result<EventStream> {
        let model = &self.entry.model;
        let token = options
            .api_key
            .as_deref()
            .or(self.entry.api_key.as_deref())
            .ok_or_else(|| Error::Auth(format!("no API key for provider: {}", model.provider)))?;
        let account_id = extract_account_id(token)?;
        let body = build_codex_request(model, context, options);

        let mut request = self
            .client
            .post(&resolve_codex_url(&model.base_url))
            .json(&body)?;
        for (name, value) in &model.headers {
            request = request.header(name.clone(), value.clone());
        }
        for (name, value) in &self.entry.headers {
            request = request.header(name.clone(), value.clone());
        }
        for (name, value) in &options.headers {
            request = request.header(name.clone(), value.clone());
        }
        request = request
            .header("Authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", account_id)
            .header("originator", "pi")
            .header(
                "User-Agent",
                format!("pi ({}; {})", std::env::consts::OS, std::env::consts::ARCH),
            )
            .header("OpenAI-Beta", "responses=experimental")
            .header("accept", "text/event-stream");
        if let Some(session_id) = options.session_id.as_deref() {
            request = request
                .header("session-id", session_id)
                .header("x-client-request-id", session_id);
        }

        let response = request.send_streaming().await?;
        if !(200..300).contains(&response.status()) {
            let status = response.status();
            let body = response.read_to_end().await.unwrap_or_default();
            return Err(Error::Api {
                status,
                message: friendly_api_error(status, &String::from_utf8_lossy(&body)),
            });
        }

        let adapter = ResponsesAdapter::new(model, self.api());
        Ok(sse_event_stream(response.into_body(), adapter))
    }
}

/// Normalizes a base URL to the canonical `…/backend-api/codex/responses`
/// endpoint. A bare host (e.g. a test fake) gets the full backend path, so
/// overriding the base URL keeps the production request path.
pub(crate) fn resolve_codex_url(base_url: &str) -> String {
    let raw = if base_url.trim().is_empty() {
        DEFAULT_CODEX_BASE_URL
    } else {
        base_url
    };
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_string()
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else if normalized.ends_with("/backend-api") {
        format!("{normalized}/codex/responses")
    } else {
        format!("{normalized}/backend-api/codex/responses")
    }
}

/// Extracts the `chatgpt_account_id` claim from the OAuth access-token JWT.
pub(crate) fn extract_account_id(token: &str) -> Result<String> {
    let failure = || Error::Auth("failed to extract accountId from token".to_string());
    let mut parts = token.split('.');
    let (Some(_), Some(payload), Some(_), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(failure());
    };
    let decoded = base64url_decode(payload).ok_or_else(failure)?;
    let claims: Value = serde_json::from_slice(&decoded).map_err(|_| failure())?;
    claims
        .get(JWT_CLAIM_PATH)
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .ok_or_else(failure)
}

/// Renders a non-2xx response body into a human-useful message (usage-limit
/// errors get the friendly form).
pub(crate) fn friendly_api_error(status: u16, body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().and_then(|value| value.get("error"));
    let Some(error) = error else {
        return if body.is_empty() {
            format!("request failed with HTTP {status}")
        } else {
            body.to_string()
        };
    };
    let code = error
        .get("code")
        .or_else(|| error.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let message = error.get("message").and_then(Value::as_str).unwrap_or("");

    let usage_limited = status == 429
        || code.contains("usage_limit_reached")
        || code.contains("usage_not_included")
        || code.contains("rate_limit_exceeded");
    if usage_limited {
        let plan = error
            .get("plan_type")
            .and_then(Value::as_str)
            .map(|plan| format!(" ({} plan)", plan.to_lowercase()))
            .unwrap_or_default();
        let when = error
            .get("resets_at")
            .and_then(Value::as_u64)
            .map(|resets_at| {
                let now = now_ms() / 1000;
                let minutes = resets_at.saturating_sub(now) / 60;
                format!(" Try again in ~{minutes} min.")
            })
            .unwrap_or_default();
        if message.is_empty() {
            return format!("You have hit your ChatGPT usage limit{plan}.{when}");
        }
    }
    if !message.is_empty() {
        message.to_string()
    } else if !code.is_empty() {
        code.to_string()
    } else {
        format!("request failed with HTTP {status}")
    }
}

#[cfg(test)]
#[path = "openai_responses_tests.rs"]
mod tests;
