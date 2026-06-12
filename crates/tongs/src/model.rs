//! The unified model/message types.
//!
//! These mirror the TypeScript Pi unified LLM API (`packages/ai/src/types.ts`):
//! one message shape shared by every provider, with provider-specific wire
//! formats handled entirely inside the provider adapters. Serde names follow
//! the TypeScript spelling (camelCase, `role`/`type` tags) so serialized
//! conversations interoperate.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// A reasoning-effort level, mapped by each provider to its own vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

impl ThinkingLevel {
    /// The provider-facing effort string (the TS pi level names).
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingLevel::Off => "off",
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::XHigh => "xhigh",
        }
    }
}

/// Why a model turn ended.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    #[default]
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

/// Token accounting for one model turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: UsageCost,
}

/// Dollar cost breakdown for one model turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// A text block in a message.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    /// Opaque provider metadata replayed on later turns (e.g. the OpenAI
    /// Responses output-item id). Not content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

/// A model "thinking" (extended reasoning) block.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    /// Opaque payload replayed for multi-turn reasoning continuity (e.g. the
    /// Anthropic signature, or the whole OpenAI reasoning item as JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// True when safety filters redacted the content; the encrypted payload
    /// lives in `thinking_signature`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
}

/// An image block (base64 payload).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub data: String,
    pub mime_type: String,
}

/// A tool invocation requested by the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// The parsed JSON arguments object.
    pub arguments: serde_json::Value,
}

/// One block of assistant (or tool-result) content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    Text(TextContent),
    Thinking(ThinkingContent),
    Image(ImageContent),
    ToolCall(ToolCall),
}

/// What a user turn carries: plain text or mixed text/image blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A user turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: UserContent,
    /// Unix milliseconds.
    pub timestamp: u64,
}

/// A completed assistant turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    /// The wire API that produced it (e.g. `anthropic-messages`).
    pub api: String,
    /// The provider id (e.g. `openai-codex`).
    pub provider: String,
    /// The requested model id.
    pub model: String,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix milliseconds.
    pub timestamp: u64,
}

/// The result of one tool call, replayed to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    /// Tool-specific structured details, not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
    /// Unix milliseconds.
    pub timestamp: u64,
}

/// One conversation message. Assistant and tool-result messages are shared
/// (`Arc`) because the agent loop clones histories per turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User(UserMessage),
    Assistant(Arc<AssistantMessage>),
    ToolResult(Arc<ToolResultMessage>),
}

/// A live event from a streaming model response.
///
/// Streams emit `Start` first, deltas while blocks arrive, and terminate with
/// exactly one of `Done` (success) or `Error` (the carried message has
/// `stop_reason` `Error`/`Aborted` and an `error_message`).
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    Start,
    TextStart {
        content_index: usize,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    TextEnd {
        content_index: usize,
        content: String,
    },
    ThinkingStart {
        content_index: usize,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
    },
    ToolCallStart {
        content_index: usize,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
    },
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

/// A model's identity and capabilities, consumed by the provider factory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    /// The wire API: `openai-completions`, `openai-codex-responses`,
    /// `openai-responses`, `anthropic-messages`.
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<InputType>,
    pub cost: ModelCost,
    pub context_window: usize,
    pub max_tokens: usize,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// Input modalities a model accepts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    Text,
    Image,
}

/// Per-million-token pricing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl Usage {
    /// Fills the dollar cost from per-million-token pricing.
    pub fn price_with(&mut self, cost: ModelCost) {
        let per = 1_000_000.0;
        self.cost.input = (self.input as f64 / per) * cost.input;
        self.cost.output = (self.output as f64 / per) * cost.output;
        self.cost.cache_read = (self.cache_read as f64 / per) * cost.cache_read;
        self.cost.cache_write = (self.cache_write as f64 / per) * cost.cache_write;
        self.cost.total =
            self.cost.input + self.cost.output + self.cost.cache_read + self.cost.cache_write;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_serde_round_trip() {
        let message = Message::Assistant(Arc::new(AssistantMessage {
            content: vec![
                ContentBlock::Text(TextContent {
                    text: "hello".to_string(),
                    text_signature: None,
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::json!({"path": "x.rs"}),
                }),
            ],
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            model: "claude".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 1,
        }));
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(json.contains("\"stopReason\":\"toolUse\""));
        assert!(json.contains("\"type\":\"toolCall\""));
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, message);
    }

    #[test]
    fn user_content_untagged() {
        let text = UserMessage {
            content: UserContent::Text("hi".to_string()),
            timestamp: 0,
        };
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"content\":\"hi\""));
        let back: UserMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, text);
    }

    #[test]
    fn usage_pricing() {
        let mut usage = Usage {
            input: 2_000_000,
            output: 1_000_000,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 3_000_000,
            cost: UsageCost::default(),
        };
        usage.price_with(ModelCost {
            input: 1.0,
            output: 5.0,
            cache_read: 0.0,
            cache_write: 0.0,
        });
        assert_eq!(usage.cost.input, 2.0);
        assert_eq!(usage.cost.output, 5.0);
        assert_eq!(usage.cost.total, 7.0);
    }
}
