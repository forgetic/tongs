use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::model::{
    AssistantMessage, ContentBlock, InputType, Message, Model, ModelCost, StopReason, StreamEvent,
    TextContent, ThinkingContent, ToolCall, ToolResultMessage, Usage, UserContent, UserMessage,
};
use crate::provider::{Context, StreamOptions, ToolDef};
use crate::providers::wire::WireAdapter;
use crate::sse::SseEvent;

fn claude_model() -> Model {
    Model {
        id: "claude-opus-4-8".to_string(),
        name: "claude-opus-4-8".to_string(),
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        base_url: "https://api.anthropic.com".to_string(),
        reasoning: true,
        input: vec![InputType::Text, InputType::Image],
        cost: ModelCost::default(),
        context_window: 1_000_000,
        max_tokens: 128_000,
        headers: HashMap::new(),
    }
}

fn named(event: &str, value: Value) -> SseEvent {
    SseEvent {
        event: Some(event.to_string()),
        data: value.to_string(),
    }
}

#[test]
fn oauth_request_has_identity_first_and_canonical_tool_names() {
    let model = claude_model();
    let context = Context {
        system_prompt: Some(Cow::Borrowed("be terse")),
        messages: Cow::Owned(vec![Message::User(UserMessage {
            content: UserContent::Text("hi".to_string()),
            timestamp: 0,
        })]),
        tools: Cow::Owned(vec![ToolDef {
            name: "read".to_string(),
            description: "read a file".to_string(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        }]),
    };
    let options = StreamOptions::default();
    let body = build_anthropic_request(&model, &context, &options, true);

    assert_eq!(body["system"][0]["text"], CLAUDE_CODE_SYSTEM_IDENTITY);
    assert_eq!(body["system"][1]["text"], "be terse");
    // OAuth canonical casing for known Claude Code tools.
    assert_eq!(body["tools"][0]["name"], "Read");
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(body["max_tokens"], 128_000);
    assert_eq!(body["stream"], true);
    assert!(body.get("thinking").is_none());
}

#[test]
fn identity_system_prompt_is_not_duplicated() {
    let model = claude_model();
    let context = Context {
        system_prompt: Some(Cow::Borrowed(CLAUDE_CODE_SYSTEM_IDENTITY)),
        messages: Cow::Owned(Vec::new()),
        tools: Cow::Owned(Vec::new()),
    };
    let body = build_anthropic_request(&model, &context, &StreamOptions::default(), true);
    let system = body["system"].as_array().unwrap();
    assert_eq!(system.len(), 1);
    assert_eq!(system[0]["text"], CLAUDE_CODE_SYSTEM_IDENTITY);
}

#[test]
fn api_key_request_keeps_tool_names_and_temperature() {
    let model = claude_model();
    let context = Context {
        system_prompt: None,
        messages: Cow::Owned(Vec::new()),
        tools: Cow::Owned(vec![ToolDef {
            name: "read".to_string(),
            description: "read".to_string(),
            parameters: json!({"type": "object"}),
        }]),
    };
    let options = StreamOptions {
        temperature: Some(0.0),
        ..StreamOptions::default()
    };
    let body = build_anthropic_request(&model, &context, &options, false);
    assert_eq!(body["tools"][0]["name"], "read");
    assert_eq!(body["temperature"], 0.0);
    assert!(body.get("system").is_none());
}

#[test]
fn replays_history_with_merged_tool_results() {
    let model = claude_model();
    let messages = vec![
        Message::User(UserMessage {
            content: UserContent::Text("go".to_string()),
            timestamp: 0,
        }),
        Message::Assistant(Arc::new(AssistantMessage {
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "mull".to_string(),
                    thinking_signature: Some("sig-1".to_string()),
                    redacted: false,
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "toolu_1".to_string(),
                    name: "read".to_string(),
                    arguments: json!({"path": "a"}),
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "toolu_2".to_string(),
                    name: "read".to_string(),
                    arguments: json!({"path": "b"}),
                }),
            ],
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-opus-4-8".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 0,
        })),
        Message::ToolResult(Arc::new(ToolResultMessage {
            tool_call_id: "toolu_1".to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "alpha".to_string(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
        })),
        Message::ToolResult(Arc::new(ToolResultMessage {
            tool_call_id: "toolu_2".to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "beta".to_string(),
                text_signature: None,
            })],
            details: None,
            is_error: true,
            timestamp: 0,
        })),
    ];
    let context = Context {
        system_prompt: None,
        messages: Cow::Owned(messages),
        tools: Cow::Owned(Vec::new()),
    };
    let converted = convert_anthropic_messages(&model, &context, false);

    assert_eq!(converted.len(), 3);
    assert_eq!(converted[1]["role"], "assistant");
    assert_eq!(converted[1]["content"][0]["type"], "thinking");
    assert_eq!(converted[1]["content"][0]["signature"], "sig-1");
    assert_eq!(converted[1]["content"][1]["type"], "tool_use");
    assert_eq!(converted[1]["content"][1]["input"], json!({"path": "a"}));
    // Both tool results merge into one user turn.
    assert_eq!(converted[2]["role"], "user");
    let results = converted[2]["content"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["tool_use_id"], "toolu_1");
    assert_eq!(results[1]["is_error"], true);
}

#[test]
fn unsigned_thinking_replays_as_text() {
    let model = claude_model();
    let context = Context {
        system_prompt: None,
        messages: Cow::Owned(vec![Message::Assistant(Arc::new(AssistantMessage {
            content: vec![ContentBlock::Thinking(ThinkingContent {
                thinking: "loose thought".to_string(),
                thinking_signature: None,
                redacted: false,
            })],
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-opus-4-8".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }))]),
        tools: Cow::Owned(Vec::new()),
    };
    let converted = convert_anthropic_messages(&model, &context, false);
    assert_eq!(converted[0]["content"][0]["type"], "text");
    assert_eq!(converted[0]["content"][0]["text"], "loose thought");
}

#[test]
fn adapter_folds_jig_shaped_stream() {
    let model = claude_model();
    let mut adapter = AnthropicAdapter::new(&model, &[], false);
    let mut events = Vec::new();
    events.extend(
        adapter
            .on_sse(named(
                "message_start",
                json!({"type": "message_start", "message": {
                    "id": "msg_1", "usage": {"input_tokens": 9, "output_tokens": 0},
                }}),
            ))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(named(
                "content_block_start",
                json!({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}}),
            ))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(named(
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0,
                       "delta": {"type": "text_delta", "text": "{\"action\":\"advance\"}"}}),
            ))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(named(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ))
            .unwrap(),
    );
    // A tool block at index 1, jig-style single-fragment input.
    events.extend(
        adapter
            .on_sse(named(
                "content_block_start",
                json!({"type": "content_block_start", "index": 1, "content_block": {
                    "type": "tool_use", "id": "toolu_9", "name": "write", "input": {}}}),
            ))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(named(
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 1,
                       "delta": {"type": "input_json_delta", "partial_json": "{\"path\":\"out.txt\"}"}}),
            ))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(named(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 1}),
            ))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(named(
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                       "usage": {"output_tokens": 5}}),
            ))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(named("message_stop", json!({"type": "message_stop"})))
            .unwrap(),
    );
    assert!(adapter.on_eof().unwrap().is_empty());

    assert!(matches!(events[0], StreamEvent::Start));
    let StreamEvent::Done { reason, message } = events.last().unwrap() else {
        panic!("expected Done, got {:?}", events.last());
    };
    assert_eq!(*reason, StopReason::ToolUse);
    assert_eq!(message.usage.input, 9);
    assert_eq!(message.usage.output, 5);
    assert_eq!(message.usage.total_tokens, 14);
    let ContentBlock::ToolCall(call) = &message.content[1] else {
        panic!("expected tool call block");
    };
    assert_eq!(call.id, "toolu_9");
    assert_eq!(call.arguments, json!({"path": "out.txt"}));
}

#[test]
fn refusal_stop_becomes_error_event() {
    let model = claude_model();
    let mut adapter = AnthropicAdapter::new(&model, &[], false);
    let _ = adapter
        .on_sse(named(
            "message_start",
            json!({"type": "message_start", "message": {"usage": {}}}),
        ))
        .unwrap();
    let _ = adapter
        .on_sse(named(
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {}}),
        ))
        .unwrap();
    let events = adapter
        .on_sse(named("message_stop", json!({"type": "message_stop"})))
        .unwrap();
    let StreamEvent::Error { reason, error } = events.last().unwrap() else {
        panic!("expected error event");
    };
    assert_eq!(*reason, StopReason::Error);
    assert!(error.error_message.as_deref().unwrap().contains("refused"));
}

#[test]
fn truncated_stream_is_an_error() {
    let model = claude_model();
    let mut adapter = AnthropicAdapter::new(&model, &[], false);
    let _ = adapter
        .on_sse(named(
            "message_start",
            json!({"type": "message_start", "message": {"usage": {}}}),
        ))
        .unwrap();
    let events = adapter.on_eof().unwrap();
    let StreamEvent::Error { error, .. } = events.last().unwrap() else {
        panic!("expected error");
    };
    assert!(
        error
            .error_message
            .as_deref()
            .unwrap()
            .contains("before message_stop")
    );
}

#[test]
fn oauth_tool_names_map_back_to_registered_spelling() {
    let model = claude_model();
    let tools = vec![ToolDef {
        name: "read".to_string(),
        description: "read".to_string(),
        parameters: json!({}),
    }];
    let mut adapter = AnthropicAdapter::new(&model, &tools, true);
    let events = adapter
        .on_sse(named(
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {
                "type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}}),
        ))
        .unwrap();
    assert!(!events.is_empty());
    let _ = adapter
        .on_sse(named(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ))
        .unwrap();
    let ContentBlock::ToolCall(call) = &adapter.output.content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.name, "read");
}

#[test]
fn resolves_anthropic_urls() {
    assert_eq!(
        resolve_anthropic_url(""),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        resolve_anthropic_url("https://api.anthropic.com"),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        resolve_anthropic_url("http://127.0.0.1:9000/"),
        "http://127.0.0.1:9000/v1/messages"
    );
    assert_eq!(resolve_anthropic_url("http://h/v1"), "http://h/v1/messages");
}

#[test]
fn detects_oauth_tokens() {
    assert!(is_oauth_token("sk-ant-oat01-xyz"));
    assert!(!is_oauth_token("sk-ant-api03-xyz"));
}

#[test]
fn claude_code_headers_match_identity_without_tokens() {
    let headers: HashMap<String, String> = claude_code_headers().into_iter().collect();
    assert_eq!(
        headers.get("anthropic-version").map(String::as_str),
        Some("2023-06-01")
    );
    assert_eq!(headers.get("x-app").map(String::as_str), Some("cli"));
    assert_eq!(
        headers.get("user-agent").map(String::as_str),
        Some("claude-cli/2.1.139 (external, sdk-cli)")
    );
    assert_eq!(
        headers.get("X-Stainless-Runtime").map(String::as_str),
        Some("node")
    );
    let beta = headers.get("anthropic-beta").expect("beta header");
    for flag in [
        "claude-code-20250219",
        "oauth-2025-04-20",
        "context-1m-2025-08-07",
        "effort-2025-11-24",
    ] {
        assert!(beta.contains(flag), "missing beta flag {flag}");
    }
    assert!(uuid::Uuid::parse_str(headers.get("x-client-request-id").unwrap()).is_ok());
    assert!(uuid::Uuid::parse_str(headers.get("X-Claude-Code-Session-Id").unwrap()).is_ok());
    // No token material leaks into the identity headers.
    let rendered = format!("{headers:?}");
    assert!(!rendered.contains("sk-ant"));
    assert!(!rendered.contains("refresh"));
}

#[test]
fn claude_code_headers_use_fresh_ids() {
    let first: HashMap<String, String> = claude_code_headers().into_iter().collect();
    let second: HashMap<String, String> = claude_code_headers().into_iter().collect();
    assert_ne!(
        first.get("x-client-request-id"),
        second.get("x-client-request-id")
    );
    assert_ne!(
        first.get("X-Claude-Code-Session-Id"),
        second.get("X-Claude-Code-Session-Id")
    );
}

#[test]
fn system_identity_is_the_exact_required_line() {
    // The literal the subscription path requires as the first system block; a
    // drift here is a generic 429 at request time, so pin it.
    assert_eq!(
        CLAUDE_CODE_SYSTEM_IDENTITY,
        "You are Claude Code, Anthropic's official CLI for Claude."
    );
}

#[test]
fn cache_breakpoints_mark_system_and_last_two_user_turns() {
    let model = claude_model();
    let ephemeral = json!({ "type": "ephemeral" });
    let context = Context {
        system_prompt: Some(Cow::Borrowed("be terse")),
        messages: Cow::Owned(vec![
            Message::User(UserMessage {
                content: UserContent::Text("first".to_string()),
                timestamp: 0,
            }),
            Message::Assistant(Arc::new(AssistantMessage {
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "toolu_1".to_string(),
                    name: "read".to_string(),
                    arguments: json!({"path": "a"}),
                })],
                api: "anthropic-messages".to_string(),
                provider: "anthropic".to_string(),
                model: "claude-opus-4-8".to_string(),
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                error_message: None,
                timestamp: 0,
            })),
            Message::ToolResult(Arc::new(ToolResultMessage {
                tool_call_id: "toolu_1".to_string(),
                tool_name: "read".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "alpha".to_string(),
                    text_signature: None,
                })],
                details: None,
                is_error: false,
                timestamp: 0,
            })),
        ]),
        tools: Cow::Owned(vec![]),
    };
    let body = build_anthropic_request(&model, &context, &StreamOptions::default(), false);

    // The static prefix: one marker on the last system block.
    assert_eq!(body["system"][0]["cache_control"], ephemeral);

    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 3);
    // Both user-role turns (the string-shorthand user message converts to
    // block form, and the merged tool-result turn) carry the sliding markers.
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"][0]["cache_control"], ephemeral);
    // The assistant turn stays unmarked.
    assert!(messages[1]["content"][0].get("cache_control").is_none());
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"][0]["cache_control"], ephemeral);
}
