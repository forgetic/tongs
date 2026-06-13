use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::model::{
    AssistantMessage, ContentBlock, InputType, Message, Model, ModelCost, StopReason, StreamEvent,
    TextContent, ToolCall, ToolResultMessage, Usage, UserContent, UserMessage,
};
use crate::provider::{Context, StreamOptions, ToolDef};
use crate::providers::wire::WireAdapter;
use crate::sse::SseEvent;

fn deepseek_model() -> Model {
    Model {
        id: "deepseek-chat".to_string(),
        name: "deepseek-chat".to_string(),
        api: "openai-completions".to_string(),
        provider: "deepseek".to_string(),
        base_url: "https://api.deepseek.com/v1".to_string(),
        reasoning: false,
        input: vec![InputType::Text],
        cost: ModelCost::default(),
        context_window: 64_000,
        max_tokens: 8_192,
        headers: HashMap::new(),
    }
}

fn data(value: Value) -> SseEvent {
    SseEvent {
        event: None,
        data: value.to_string(),
    }
}

fn done() -> SseEvent {
    SseEvent {
        event: None,
        data: "[DONE]".to_string(),
    }
}

#[test]
fn builds_completions_request() {
    let model = deepseek_model();
    let messages = vec![
        Message::User(UserMessage {
            content: UserContent::Text("hello".to_string()),
            timestamp: 0,
        }),
        Message::Assistant(Arc::new(AssistantMessage {
            content: vec![
                ContentBlock::Text(TextContent {
                    text: "checking".to_string(),
                    text_signature: None,
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "read".to_string(),
                    arguments: json!({"path": "x"}),
                }),
            ],
            api: "openai-completions".to_string(),
            provider: "deepseek".to_string(),
            model: "deepseek-chat".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 0,
        })),
        Message::ToolResult(Arc::new(ToolResultMessage {
            tool_call_id: "call_1".to_string(),
            tool_name: "read".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                text: "data".to_string(),
                text_signature: None,
            })],
            details: None,
            is_error: false,
            timestamp: 0,
        })),
    ];
    let context = Context {
        system_prompt: Some(Cow::Borrowed("be terse")),
        messages: Cow::Owned(messages),
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
    let body = build_completions_request(&model, &context, &options);

    assert_eq!(body["model"], "deepseek-chat");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][2]["role"], "assistant");
    assert_eq!(body["messages"][2]["content"], "checking");
    assert_eq!(body["messages"][2]["tool_calls"][0]["id"], "call_1");
    assert_eq!(
        body["messages"][2]["tool_calls"][0]["function"]["arguments"],
        "{\"path\":\"x\"}"
    );
    assert_eq!(body["messages"][3]["role"], "tool");
    assert_eq!(body["messages"][3]["tool_call_id"], "call_1");
    assert_eq!(body["tools"][0]["function"]["name"], "read");
}

#[test]
fn folds_text_and_tool_call_stream() {
    let model = deepseek_model();
    let mut adapter = CompletionsAdapter::new(&model);
    let mut events = Vec::new();

    events.extend(
        adapter
            .on_sse(data(json!({
                "choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]
            })))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(data(json!({
                "choices": [{"delta": {"content": "let me look"}, "finish_reason": null}]
            })))
            .unwrap(),
    );
    // Header delta then arguments delta — the jig split.
    events.extend(
        adapter
            .on_sse(data(json!({
                "choices": [{"delta": {"tool_calls": [{
                    "index": 0, "id": "call_9", "type": "function",
                    "function": {"name": "read", "arguments": ""},
                }]}, "finish_reason": null}]
            })))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(data(json!({
                "choices": [{"delta": {"tool_calls": [{
                    "index": 0, "function": {"arguments": "{\"path\":\"a.rs\"}"},
                }]}, "finish_reason": null}]
            })))
            .unwrap(),
    );
    events.extend(
        adapter
            .on_sse(data(json!({
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            })))
            .unwrap(),
    );
    events.extend(adapter.on_sse(done()).unwrap());

    assert!(matches!(events[0], StreamEvent::Start));
    assert!(events.iter().any(
        |event| matches!(event, StreamEvent::TextDelta { delta, .. } if delta == "let me look")
    ));
    let StreamEvent::Done { reason, message } = events.last().unwrap() else {
        panic!("expected Done, got {:?}", events.last());
    };
    assert_eq!(*reason, StopReason::ToolUse);
    assert_eq!(message.usage.input, 11);
    assert_eq!(message.usage.output, 7);
    let calls: Vec<&ToolCall> = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_9");
    assert_eq!(calls[0].name, "read");
    assert_eq!(calls[0].arguments, json!({"path": "a.rs"}));
}

#[test]
fn reasoning_content_becomes_thinking() {
    let model = deepseek_model();
    let mut adapter = CompletionsAdapter::new(&model);
    let events = adapter
        .on_sse(data(json!({
            "choices": [{"delta": {"reasoning_content": "hmm"}, "finish_reason": null}]
        })))
        .unwrap();
    assert!(
        events.iter().any(
            |event| matches!(event, StreamEvent::ThinkingDelta { delta, .. } if delta == "hmm")
        )
    );
}

#[test]
fn missing_finish_reason_is_an_error() {
    let model = deepseek_model();
    let mut adapter = CompletionsAdapter::new(&model);
    let _ = adapter
        .on_sse(data(json!({
            "choices": [{"delta": {"content": "partial"}, "finish_reason": null}]
        })))
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
            .contains("without finish_reason")
    );
}

#[test]
fn finish_without_done_sentinel_still_completes() {
    let model = deepseek_model();
    let mut adapter = CompletionsAdapter::new(&model);
    let _ = adapter
        .on_sse(data(json!({
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
        })))
        .unwrap();
    let events = adapter.on_eof().unwrap();
    assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
}
