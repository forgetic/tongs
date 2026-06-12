//! The OpenAI-compatible chat-completions wire format (DeepSeek and friends).
//!
//! Ported from TS Pi's `openai-completions.ts`, trimmed to the dialect our
//! consumers use: `{base_url}/chat/completions`, streamed `choices[].delta`
//! chunks with indexed tool-call assembly, `reasoning_content` thinking
//! deltas, and a `[DONE]` sentinel.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::http::Client;
use crate::model::{
    AssistantMessage, ContentBlock, Message, Model, StopReason, StreamEvent, TextContent,
    ThinkingContent, ToolCall, Usage, UserContent,
};
use crate::provider::{Context, EventStream, ModelEntry, Provider, StreamOptions, ToolDef};
use crate::providers::wire::{WireAdapter, sse_event_stream};
use crate::sse::SseEvent;
use crate::util::now_ms;
use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Pure: request building.
// ---------------------------------------------------------------------------

/// Builds the chat-completions request body.
pub(crate) fn build_completions_request(
    model: &Model,
    context: &Context<'_>,
    options: &StreamOptions,
) -> Value {
    let mut body = json!({
        "model": model.id,
        "messages": convert_completions_messages(model, context),
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    let object = body.as_object_mut().expect("body is an object");

    if let Some(temperature) = options.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    let max_tokens = options.max_tokens.unwrap_or(model.max_tokens);
    if max_tokens > 0 {
        object.insert("max_tokens".to_string(), json!(max_tokens));
    }
    if !context.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(convert_completions_tools(&context.tools)),
        );
    }
    body
}

pub(crate) fn convert_completions_tools(tools: &[ToolDef]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        })
        .collect()
}

/// Converts the unified conversation into chat-completions `messages`.
pub(crate) fn convert_completions_messages(model: &Model, context: &Context<'_>) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(prompt) = context.system_prompt.as_deref() {
        messages.push(json!({ "role": "system", "content": prompt }));
    }

    for message in context.messages.iter() {
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    messages.push(json!({ "role": "user", "content": text }));
                }
                UserContent::Blocks(blocks) => {
                    let supports_images = model
                        .input
                        .contains(&crate::model::InputType::Image);
                    let parts: Vec<Value> = blocks
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(text) => {
                                Some(json!({ "type": "text", "text": text.text }))
                            }
                            ContentBlock::Image(image) if supports_images => Some(json!({
                                "type": "image_url",
                                "image_url": { "url": format!(
                                    "data:{};base64,{}",
                                    image.mime_type, image.data
                                )},
                            })),
                            ContentBlock::Image(_) => Some(json!({
                                "type": "text",
                                "text": "(image omitted: model does not support images)",
                            })),
                            _ => None,
                        })
                        .collect();
                    if !parts.is_empty() {
                        messages.push(json!({ "role": "user", "content": parts }));
                    }
                }
            },
            Message::Assistant(assistant) => {
                let text: String = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let tool_calls: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolCall(call) => Some(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments.to_string(),
                            },
                        })),
                        _ => None,
                    })
                    .collect();
                let has_tool_calls = !tool_calls.is_empty();
                let mut item = json!({ "role": "assistant" });
                let object = item.as_object_mut().expect("item is an object");
                if !text.is_empty() {
                    object.insert("content".to_string(), json!(text));
                }
                if has_tool_calls {
                    object.insert("tool_calls".to_string(), Value::Array(tool_calls));
                }
                if !text.is_empty() || has_tool_calls {
                    messages.push(item);
                }
            }
            Message::ToolResult(result) => {
                let text: String = result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": text,
                }));
            }
        }
    }
    messages
}

// ---------------------------------------------------------------------------
// Pure: SSE folding.
// ---------------------------------------------------------------------------

/// A tool-call slot being assembled from indexed deltas.
#[derive(Default)]
struct ToolCallSlot {
    id: String,
    name: String,
    arguments: String,
    /// Index of its block in the output content.
    block_index: Option<usize>,
}

pub(crate) struct CompletionsAdapter {
    output: AssistantMessage,
    model_cost: crate::model::ModelCost,
    slots: Vec<ToolCallSlot>,
    started: bool,
    finished: bool,
    finish_reason: Option<StopReason>,
}

impl CompletionsAdapter {
    pub(crate) fn new(model: &Model) -> Self {
        Self {
            output: AssistantMessage {
                content: Vec::new(),
                api: "openai-completions".to_string(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: now_ms(),
            },
            model_cost: model.cost,
            slots: Vec::new(),
            started: false,
            finished: false,
            finish_reason: None,
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

    /// Appends to the trailing text block (opening one as needed).
    fn push_text(&mut self, delta: &str, events: &mut Vec<StreamEvent>) {
        if !matches!(self.output.content.last(), Some(ContentBlock::Text(_))) {
            self.output
                .content
                .push(ContentBlock::Text(TextContent::default()));
            events.push(StreamEvent::TextStart {
                content_index: self.output.content.len() - 1,
            });
        }
        let index = self.output.content.len() - 1;
        if let Some(ContentBlock::Text(block)) = self.output.content.last_mut() {
            block.text.push_str(delta);
        }
        events.push(StreamEvent::TextDelta {
            content_index: index,
            delta: delta.to_string(),
        });
    }

    fn push_thinking(&mut self, delta: &str, events: &mut Vec<StreamEvent>) {
        if !matches!(self.output.content.last(), Some(ContentBlock::Thinking(_))) {
            self.output
                .content
                .push(ContentBlock::Thinking(ThinkingContent::default()));
            events.push(StreamEvent::ThinkingStart {
                content_index: self.output.content.len() - 1,
            });
        }
        let index = self.output.content.len() - 1;
        if let Some(ContentBlock::Thinking(block)) = self.output.content.last_mut() {
            block.thinking.push_str(delta);
        }
        events.push(StreamEvent::ThinkingDelta {
            content_index: index,
            delta: delta.to_string(),
        });
    }

    fn on_tool_call_delta(&mut self, delta: &Value, events: &mut Vec<StreamEvent>) {
        let index = delta.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        while self.slots.len() <= index {
            self.slots.push(ToolCallSlot::default());
        }
        let needs_block = self.slots[index].block_index.is_none();
        if needs_block {
            self.output.content.push(ContentBlock::ToolCall(ToolCall {
                id: String::new(),
                name: String::new(),
                arguments: json!({}),
            }));
            self.slots[index].block_index = Some(self.output.content.len() - 1);
            events.push(StreamEvent::ToolCallStart {
                content_index: self.output.content.len() - 1,
            });
        }
        let slot = &mut self.slots[index];
        if let Some(id) = delta.get("id").and_then(Value::as_str) {
            slot.id.push_str(id);
        }
        if let Some(name) = delta.pointer("/function/name").and_then(Value::as_str) {
            slot.name.push_str(name);
        }
        if let Some(arguments) = delta.pointer("/function/arguments").and_then(Value::as_str)
            && !arguments.is_empty()
        {
            slot.arguments.push_str(arguments);
            events.push(StreamEvent::ToolCallDelta {
                content_index: slot.block_index.expect("slot has a block"),
                delta: arguments.to_string(),
            });
        }
    }

    /// Finalizes tool-call slots into blocks and emits the terminal event.
    fn finish(&mut self) -> Vec<StreamEvent> {
        self.finished = true;
        let mut events = Vec::new();
        for slot in std::mem::take(&mut self.slots) {
            let Some(block_index) = slot.block_index else {
                continue;
            };
            let arguments =
                serde_json::from_str::<Value>(&slot.arguments).unwrap_or_else(|_| json!({}));
            let call = ToolCall {
                id: slot.id,
                name: slot.name,
                arguments,
            };
            if let Some(ContentBlock::ToolCall(block)) = self.output.content.get_mut(block_index) {
                *block = call.clone();
            }
            events.push(StreamEvent::ToolCallEnd {
                content_index: block_index,
                tool_call: call,
            });
        }

        let mut reason = self.finish_reason.unwrap_or(StopReason::Stop);
        if reason == StopReason::Stop
            && self
                .output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
        {
            reason = StopReason::ToolUse;
        }
        self.output.stop_reason = reason;
        self.output.usage.price_with(self.model_cost);
        events.push(StreamEvent::Done {
            reason,
            message: self.output.clone(),
        });
        events
    }

    fn on_json(&mut self, chunk: &Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(StreamEvent::Start);
        }

        if let Some(error) = chunk.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("provider returned an error chunk")
                .to_string();
            events.extend(self.fail(message));
            return events;
        }

        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            let read = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
            let cached = usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let cache_write = usage
                .pointer("/prompt_tokens_details/cache_write_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            self.output.usage = Usage {
                input: read("prompt_tokens").saturating_sub(cached),
                output: read("completion_tokens"),
                cache_read: cached,
                cache_write,
                total_tokens: read("total_tokens"),
                cost: Default::default(),
            };
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return events;
        };

        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(Value::as_str)
                && !content.is_empty()
            {
                self.push_text(content, &mut events);
            }
            // llama.cpp/DeepSeek-style reasoning fields, first non-empty wins.
            for key in ["reasoning_content", "reasoning", "reasoning_text"] {
                if let Some(thinking) = delta.get(key).and_then(Value::as_str)
                    && !thinking.is_empty()
                {
                    self.push_thinking(thinking, &mut events);
                    break;
                }
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    self.on_tool_call_delta(tool_call, &mut events);
                }
            }
        }

        if let Some(finish) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(match finish {
                "length" => StopReason::Length,
                "tool_calls" | "function_call" => StopReason::ToolUse,
                // stop / content_filter / anything else.
                _ => StopReason::Stop,
            });
        }
        events
    }
}

impl WireAdapter for CompletionsAdapter {
    fn on_sse(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let data = event.data.trim();
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data == "[DONE]" {
            if self.finish_reason.is_none() {
                return Ok(self.fail("stream ended without finish_reason".to_string()));
            }
            return Ok(self.finish());
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|error| Error::Decode(format!("invalid completions SSE JSON: {error}")))?;
        Ok(self.on_json(&value))
    }

    fn on_eof(&mut self) -> Result<Vec<StreamEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        if self.finish_reason.is_some() {
            // Terminal chunk arrived but the [DONE] sentinel was cut off.
            return Ok(self.finish());
        }
        Ok(self.fail("stream ended without finish_reason".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Shell: the provider.
// ---------------------------------------------------------------------------

/// An OpenAI-compatible chat-completions provider.
pub struct CompletionsProvider {
    entry: ModelEntry,
    client: Client,
}

impl CompletionsProvider {
    pub fn new(entry: ModelEntry, client: Client) -> Self {
        Self { entry, client }
    }
}

#[async_trait]
impl Provider for CompletionsProvider {
    fn api(&self) -> &str {
        "openai-completions"
    }

    async fn stream(
        &self,
        context: &Context<'_>,
        options: &StreamOptions,
    ) -> Result<EventStream> {
        let model = &self.entry.model;
        let body = build_completions_request(model, context, options);
        let url = format!(
            "{}/chat/completions",
            model.base_url.trim_end_matches('/')
        );

        let mut request = self.client.post(&url).json(&body)?;
        for (name, value) in &model.headers {
            request = request.header(name.clone(), value.clone());
        }
        for (name, value) in &self.entry.headers {
            request = request.header(name.clone(), value.clone());
        }
        for (name, value) in &options.headers {
            request = request.header(name.clone(), value.clone());
        }
        if self.entry.auth_header
            && let Some(token) = options.api_key.as_deref().or(self.entry.api_key.as_deref())
        {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        request = request.header("accept", "text/event-stream");

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

        let adapter = CompletionsAdapter::new(model);
        Ok(sse_event_stream(response.into_body(), adapter))
    }
}

#[cfg(test)]
#[path = "openai_completions_tests.rs"]
mod tests;
