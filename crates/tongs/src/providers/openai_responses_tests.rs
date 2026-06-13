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

fn codex_model() -> Model {
    Model {
        id: "gpt-5.5".to_string(),
        name: "gpt-5.5".to_string(),
        api: "openai-codex-responses".to_string(),
        provider: "openai-codex".to_string(),
        base_url: String::new(),
        reasoning: true,
        input: vec![InputType::Text],
        cost: ModelCost::default(),
        context_window: 400_000,
        max_tokens: 0,
        headers: HashMap::new(),
    }
}

fn user_context(text: &str) -> Context<'static> {
    Context {
        system_prompt: Some(Cow::Owned("be brief".to_string())),
        messages: Cow::Owned(vec![Message::User(UserMessage {
            content: UserContent::Text(text.to_string()),
            timestamp: 0,
        })]),
        tools: Cow::Owned(vec![ToolDef {
            name: "read".to_string(),
            description: "read a file".to_string(),
            parameters: json!({"type": "object"}),
        }]),
    }
}

fn data_event(value: Value) -> SseEvent {
    SseEvent {
        event: None,
        data: value.to_string(),
    }
}

#[test]
fn builds_codex_request_body() {
    let model = codex_model();
    let options = StreamOptions {
        thinking_level: Some(crate::model::ThinkingLevel::XHigh),
        ..StreamOptions::default()
    };
    let body = build_codex_request(&model, &user_context("hi"), &options);

    assert_eq!(body["model"], "gpt-5.5");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["instructions"], "be brief");
    assert_eq!(body["text"]["verbosity"], "low");
    assert_eq!(body["include"][0], "reasoning.encrypted_content");
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["reasoning"]["effort"], "xhigh");
    assert_eq!(body["reasoning"]["summary"], "auto");
    // The system prompt rides in `instructions`, not in input.
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][0]["text"], "hi");
    // Codex tools carry an explicit null strict.
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "read");
    assert!(body["tools"][0]["strict"].is_null());
    // No temperature unless set.
    assert!(body.get("temperature").is_none());
}

#[test]
fn off_thinking_maps_to_none_effort() {
    let model = codex_model();
    let options = StreamOptions {
        thinking_level: Some(crate::model::ThinkingLevel::Off),
        ..StreamOptions::default()
    };
    let body = build_codex_request(&model, &user_context("hi"), &options);
    assert_eq!(body["reasoning"]["effort"], "none");
}

fn assistant_with(content: Vec<ContentBlock>) -> Arc<AssistantMessage> {
    Arc::new(AssistantMessage {
        content,
        api: "openai-codex-responses".to_string(),
        provider: "openai-codex".to_string(),
        model: "gpt-5.5".to_string(),
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 0,
    })
}

#[test]
fn replays_assistant_turn_and_tool_result() {
    let model = codex_model();
    let assistant = assistant_with(vec![
        ContentBlock::Thinking(ThinkingContent {
            thinking: "hmm".to_string(),
            thinking_signature: Some(json!({"type": "reasoning", "id": "rs_1"}).to_string()),
            redacted: false,
        }),
        ContentBlock::Text(TextContent {
            text: "running".to_string(),
            text_signature: Some(json!({"v": 1, "id": "msg_1"}).to_string()),
        }),
        ContentBlock::ToolCall(ToolCall {
            id: "call_9|fc_9".to_string(),
            name: "read".to_string(),
            arguments: json!({"path": "a"}),
        }),
    ]);
    let messages = vec![
        Message::User(UserMessage {
            content: UserContent::Text("go".to_string()),
            timestamp: 0,
        }),
        Message::Assistant(assistant),
        Message::ToolResult(Arc::new(ToolResultMessage {
            tool_call_id: "call_9|fc_9".to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "contents".to_string(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
        })),
    ];
    let context = Context {
        system_prompt: None,
        messages: Cow::Owned(messages),
        tools: Cow::Owned(Vec::new()),
    };
    let items = convert_responses_messages(&model, &context, false);

    assert_eq!(items.len(), 5);
    assert_eq!(items[0]["role"], "user");
    // Thinking replays the stored reasoning item verbatim.
    assert_eq!(items[1]["type"], "reasoning");
    assert_eq!(items[1]["id"], "rs_1");
    // Text replays with its signature id.
    assert_eq!(items[2]["type"], "message");
    assert_eq!(items[2]["id"], "msg_1");
    assert_eq!(items[2]["content"][0]["text"], "running");
    // The tool call splits into call_id + item id.
    assert_eq!(items[3]["type"], "function_call");
    assert_eq!(items[3]["call_id"], "call_9");
    assert_eq!(items[3]["id"], "fc_9");
    assert_eq!(items[3]["arguments"], "{\"path\":\"a\"}");
    // The result references the call id only.
    assert_eq!(items[4]["type"], "function_call_output");
    assert_eq!(items[4]["call_id"], "call_9");
    assert_eq!(items[4]["output"], "contents");
}

#[test]
fn foreign_tool_call_ids_are_rebuilt() {
    let model = codex_model();
    let mut assistant = assistant_with(vec![ContentBlock::ToolCall(ToolCall {
        id: "toolu_123|item!!".to_string(),
        name: "read".to_string(),
        arguments: json!({}),
    })]);
    Arc::get_mut(&mut assistant).unwrap().provider = "anthropic".to_string();
    Arc::get_mut(&mut assistant).unwrap().api = "anthropic-messages".to_string();
    let context = Context {
        system_prompt: None,
        messages: Cow::Owned(vec![Message::Assistant(assistant)]),
        tools: Cow::Owned(Vec::new()),
    };
    let items = convert_responses_messages(&model, &context, false);
    assert_eq!(items[0]["call_id"], "toolu_123");
    let item_id = items[0]["id"].as_str().unwrap();
    assert!(item_id.starts_with("fc_"), "foreign item id: {item_id}");
}

#[test]
fn adapter_folds_full_turn() {
    let model = codex_model();
    let mut adapter = ResponsesAdapter::new(&model, "openai-codex-responses");
    let mut events = Vec::new();
    let feed = |adapter: &mut ResponsesAdapter, value: Value| -> Vec<StreamEvent> {
        adapter
            .on_sse(data_event(value))
            .expect("adapter accepts event")
    };

    events.extend(feed(
        &mut adapter,
        json!({"type": "response.created", "response": {"id": "resp_1"}}),
    ));
    // Reasoning item.
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.reasoning_summary_part.added", "part": {"type": "summary_text", "text": ""}}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.reasoning_summary_text.delta", "delta": "thinking…"}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_item.done", "item": {
            "type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "thinking…"}],
        }}),
    ));
    // Text item.
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_text.delta", "delta": "hel"}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_text.delta", "delta": "lo"}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_item.done", "item": {
            "type": "message", "id": "msg_1", "phase": "final_answer",
            "content": [{"type": "output_text", "text": "hello"}],
        }}),
    ));
    // Tool call item.
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_item.added", "item": {
            "type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "read", "arguments": "",
        }}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.function_call_arguments.delta", "delta": "{\"path\":"}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.function_call_arguments.done", "arguments": "{\"path\":\"a.rs\"}"}),
    ));
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.output_item.done", "item": {
            "type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "read", "arguments": "{\"path\":\"a.rs\"}",
        }}),
    ));
    // Terminal (codex alias `response.done`).
    events.extend(feed(
        &mut adapter,
        json!({"type": "response.done", "response": {
            "status": "completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 30,
                "total_tokens": 130,
                "input_tokens_details": {"cached_tokens": 40},
            },
        }}),
    ));
    // Anything after the terminal is ignored.
    assert!(
        adapter
            .on_sse(data_event(json!({"type": "x"})))
            .unwrap()
            .is_empty()
    );
    assert!(adapter.on_eof().unwrap().is_empty());

    assert!(matches!(events[0], StreamEvent::Start));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::ThinkingDelta { delta, .. } if delta == "thinking…"))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { delta, .. } if delta == "lo"))
    );
    let tool_end = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::ToolCallEnd { tool_call, .. } => Some(tool_call.clone()),
            _ => None,
        })
        .expect("tool call end");
    assert_eq!(tool_end.id, "call_1|fc_1");
    assert_eq!(tool_end.name, "read");
    assert_eq!(tool_end.arguments, json!({"path": "a.rs"}));

    let StreamEvent::Done { reason, message } = events.last().unwrap() else {
        panic!("expected terminal Done, got {:?}", events.last());
    };
    assert_eq!(*reason, StopReason::ToolUse);
    assert_eq!(message.content.len(), 3);
    assert_eq!(message.usage.input, 60);
    assert_eq!(message.usage.cache_read, 40);
    assert_eq!(message.usage.output, 30);
    assert_eq!(message.usage.total_tokens, 130);
    let ContentBlock::Thinking(thinking) = &message.content[0] else {
        panic!("first block should be thinking");
    };
    assert_eq!(thinking.thinking, "thinking…");
    assert!(thinking.thinking_signature.is_some());
    let ContentBlock::Text(text) = &message.content[1] else {
        panic!("second block should be text");
    };
    assert_eq!(text.text, "hello");
    assert!(
        text.text_signature
            .as_deref()
            .unwrap()
            .contains("final_answer")
    );
}

#[test]
fn adapter_surfaces_api_error_event() {
    let model = codex_model();
    let mut adapter = ResponsesAdapter::new(&model, "openai-codex-responses");
    let events = adapter
        .on_sse(data_event(
            json!({"type": "error", "code": "rate_limit", "message": "slow down"}),
        ))
        .unwrap();
    let StreamEvent::Error { reason, error } = events.last().unwrap() else {
        panic!("expected error event");
    };
    assert_eq!(*reason, StopReason::Error);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert!(
        error
            .error_message
            .as_deref()
            .unwrap()
            .contains("slow down")
    );
}

#[test]
fn adapter_errors_on_truncated_stream() {
    let model = codex_model();
    let mut adapter = ResponsesAdapter::new(&model, "openai-codex-responses");
    let _ = adapter
        .on_sse(data_event(json!({"type": "response.created"})))
        .unwrap();
    let events = adapter.on_eof().unwrap();
    let StreamEvent::Error { error, .. } = events.last().unwrap() else {
        panic!("expected error event at EOF");
    };
    assert!(
        error
            .error_message
            .as_deref()
            .unwrap()
            .contains("before response.completed")
    );
}

#[test]
fn done_markers_and_blank_data_are_skipped() {
    let model = codex_model();
    let mut adapter = ResponsesAdapter::new(&model, "openai-codex-responses");
    assert!(
        adapter
            .on_sse(SseEvent {
                event: None,
                data: "[DONE]".to_string()
            })
            .unwrap()
            .is_empty()
    );
    assert!(
        adapter
            .on_sse(SseEvent {
                event: None,
                data: "  ".to_string()
            })
            .unwrap()
            .is_empty()
    );
}

#[test]
fn resolves_codex_urls() {
    assert_eq!(
        resolve_codex_url(""),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        resolve_codex_url("https://example.com/base/"),
        "https://example.com/base/backend-api/codex/responses"
    );
    assert_eq!(
        resolve_codex_url("http://127.0.0.1:9000"),
        "http://127.0.0.1:9000/backend-api/codex/responses"
    );
    assert_eq!(
        resolve_codex_url("https://example.com/codex"),
        "https://example.com/codex/responses"
    );
    assert_eq!(
        resolve_codex_url("https://example.com/codex/responses"),
        "https://example.com/codex/responses"
    );
}

#[test]
fn extracts_account_id_from_jwt() {
    let header = "e30"; // {}
    let claims = json!({
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-77"}
    })
    .to_string();
    let payload = base64url_encode_for_test(claims.as_bytes());
    let token = format!("{header}.{payload}.sig");
    assert_eq!(extract_account_id(&token).unwrap(), "acct-77");
    assert!(extract_account_id("not-a-jwt").is_err());
}

#[test]
fn friendly_usage_limit_error() {
    let body = json!({
        "error": {"code": "usage_limit_reached", "plan_type": "Plus"}
    })
    .to_string();
    let message = friendly_api_error(429, &body);
    assert!(message.contains("usage limit"));
    assert!(message.contains("plus plan"));
}

/// Test-only base64url encoder (the crate only needs decoding).
fn base64url_encode_for_test(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[n as usize & 63] as char);
        }
    }
    out
}
