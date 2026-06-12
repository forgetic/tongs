//! End-to-end provider tests against jig's fake LLM.
//!
//! Each test starts an in-process [`jig_server::FakeLlm`] serving a scripted
//! reply, points a provider's `Model.base_url` at it, and runs a real
//! streaming turn: tongs' skein HTTP client on the wire, jig's recorded-
//! traffic-shaped SSE coming back, folded through the real adapters into
//! [`StreamEvent`]s. Request bodies are asserted via jig's normalized
//! [`jig_core::RequestView`], so these tests cover both directions of the
//! wire format — the layer the unit tests (which inject pre-parsed
//! `SseEvent`s) cannot reach.

use std::collections::HashMap;

use futures_core::Stream;
use jig_core::{Dialect, Reply, Script, Turn};
use jig_server::FakeLlm;
use serde_json::json;
use tongs::model::{
    ContentBlock, InputType, Message, Model, ModelCost, StopReason, StreamEvent, TextContent,
    UserContent, UserMessage,
};
use tongs::provider::{Context, ModelEntry, StreamOptions, ToolDef};
use tongs::providers::create_provider;

/// A bearer that satisfies the Codex provider's JWT account-id extraction:
/// `e30` is `{}`, the payload decodes to
/// `{"https://api.openai.com/auth":{"chatgpt_account_id":"acct-e2e"}}`.
const FAKE_CODEX_JWT: &str =
    "e30.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC1lMmUifX0.sig";

/// The three dialects under test: tongs `api` string, provider id, credential,
/// and the path + dialect jig should see the request on.
struct DialectCase {
    api: &'static str,
    provider: &'static str,
    api_key: &'static str,
    expected_path: &'static str,
    expected_dialect: Dialect,
}

const ANTHROPIC: DialectCase = DialectCase {
    api: "anthropic-messages",
    provider: "anthropic",
    api_key: "test-key",
    expected_path: "/v1/messages",
    expected_dialect: Dialect::Anthropic,
};

const COMPLETIONS: DialectCase = DialectCase {
    api: "openai-completions",
    provider: "openai",
    api_key: "test-key",
    expected_path: "/chat/completions",
    expected_dialect: Dialect::OpenAi,
};

const CODEX: DialectCase = DialectCase {
    api: "openai-codex-responses",
    provider: "openai-codex",
    api_key: FAKE_CODEX_JWT,
    expected_path: "/backend-api/codex/responses",
    expected_dialect: Dialect::Codex,
};

fn entry_for(case: &DialectCase, base_url: String) -> ModelEntry {
    ModelEntry {
        model: Model {
            id: "fake-model".to_string(),
            name: "Fake Model".to_string(),
            api: case.api.to_string(),
            provider: case.provider.to_string(),
            base_url,
            reasoning: false,
            input: vec![InputType::Text],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 4_096,
            headers: HashMap::new(),
        },
        api_key: Some(case.api_key.to_string()),
        headers: HashMap::new(),
        auth_header: true,
        compat: None,
        oauth_config: None,
    }
}

fn user_text(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: 0,
    })
}

fn write_tool() -> ToolDef {
    ToolDef {
        name: "write".to_string(),
        description: "Write a file".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
        }),
    }
}

fn text_reply(text: &str) -> Reply {
    Reply {
        turns: vec![Turn::Text(text.to_string())],
        usage: jig_core::Usage {
            prompt_tokens: 7,
            completion_tokens: 3,
        },
        stop: jig_core::StopReason::Stop,
    }
}

fn tool_call_reply() -> Reply {
    Reply {
        turns: vec![Turn::ToolCall {
            id: "call_1".to_string(),
            name: "write".to_string(),
            args: json!({ "path": "out.txt" }),
        }],
        usage: jig_core::Usage {
            prompt_tokens: 7,
            completion_tokens: 3,
        },
        stop: jig_core::StopReason::ToolCalls,
    }
}

/// Runs one streaming turn and collects every event up to the terminal one.
async fn drive(entry: &ModelEntry, context: &Context<'_>) -> Vec<StreamEvent> {
    let provider = create_provider(entry, None).expect("provider builds");
    let mut stream = provider
        .stream(context, &StreamOptions::default())
        .await
        .expect("stream starts");
    let mut events = Vec::new();
    while let Some(event) = std::future::poll_fn(|cx| std::pin::Pin::new(&mut stream).poll_next(cx))
        .await
    {
        let event = event.expect("stream event");
        let terminal = matches!(event, StreamEvent::Done { .. } | StreamEvent::Error { .. });
        events.push(event);
        if terminal {
            break;
        }
    }
    events
}

fn collected_text(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect()
}

fn done_message(events: &[StreamEvent]) -> &tongs::model::AssistantMessage {
    match events.last() {
        Some(StreamEvent::Done { message, .. }) => message,
        other => panic!("expected a terminal Done event, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Text round trip: request out, SSE in, unified events + usage out the other
// side; jig's view of the request asserted on the way.
// ---------------------------------------------------------------------------

fn text_round_trip(case: &DialectCase) {
    let fake = FakeLlm::start(Script::Fixed(text_reply("hello from jig"))).expect("fake starts");
    let entry = entry_for(case, fake.base_url());

    let events = tongs::runtime::block_on(async move {
        let messages = vec![user_text("hi")];
        let context = Context {
            system_prompt: Some("You are a terse fake.".into()),
            messages: messages.as_slice().into(),
            tools: Vec::new().into(),
        };
        drive(&entry, &context).await
    });

    assert_eq!(events.first(), Some(&StreamEvent::Start), "events: {events:?}");
    assert_eq!(collected_text(&events), "hello from jig");

    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.usage.input, 7, "usage: {:?}", message.usage);
    assert_eq!(message.usage.output, 3, "usage: {:?}", message.usage);
    let text: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "hello from jig");

    // What tongs actually sent, as jig projected it.
    let requests = fake.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].path, case.expected_path);
    assert_eq!(requests[0].method, "POST");
    let view = requests[0].view.as_ref().expect("request projects a view");
    assert_eq!(view.dialect, case.expected_dialect);
    assert_eq!(view.last_message().map(|m| m.content.as_str()), Some("hi"));
}

#[test]
fn anthropic_text_round_trip() {
    text_round_trip(&ANTHROPIC);
}

#[test]
fn completions_text_round_trip() {
    text_round_trip(&COMPLETIONS);
}

#[test]
fn codex_text_round_trip() {
    text_round_trip(&CODEX);
}

// ---------------------------------------------------------------------------
// Tool-call round trip: a scripted tool call arrives as one assembled
// ToolCallEnd with parseable arguments and a tool-use stop reason.
// ---------------------------------------------------------------------------

fn tool_call_round_trip(case: &DialectCase) {
    let fake = FakeLlm::start(Script::Fixed(tool_call_reply())).expect("fake starts");
    let entry = entry_for(case, fake.base_url());

    let events = tongs::runtime::block_on(async move {
        let messages = vec![user_text("write out.txt")];
        let tools = vec![write_tool()];
        let context = Context {
            system_prompt: None,
            messages: messages.as_slice().into(),
            tools: tools.as_slice().into(),
        };
        drive(&entry, &context).await
    });

    let tool_call = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ToolCallEnd { tool_call, .. } => Some(tool_call),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a ToolCallEnd event, got: {events:?}"));
    assert_eq!(tool_call.name, "write");
    assert_eq!(tool_call.arguments, json!({ "path": "out.txt" }));
    assert!(
        tool_call.id.contains("call_1"),
        "tool call id should carry the wire id: {}",
        tool_call.id
    );

    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::ToolUse);
}

#[test]
fn anthropic_tool_call_round_trip() {
    tool_call_round_trip(&ANTHROPIC);
}

#[test]
fn completions_tool_call_round_trip() {
    tool_call_round_trip(&COMPLETIONS);
}

#[test]
fn codex_tool_call_round_trip() {
    tool_call_round_trip(&CODEX);
}

// ---------------------------------------------------------------------------
// Full tool loop: turn 1 hands off a tool call; the assistant message tongs
// assembled is replayed verbatim with a tool result; jig's rule sees the
// result in the transcript and answers. This exercises the history→wire
// conversion that only multi-turn requests reach.
// ---------------------------------------------------------------------------

fn tool_loop_round_trip(case: &DialectCase) {
    let script = Script::rule(|view| {
        if view.prior_tool_results == 0 {
            tool_call_reply()
        } else {
            text_reply("all done")
        }
    });
    let fake = FakeLlm::start(script).expect("fake starts");
    let entry = entry_for(case, fake.base_url());

    let final_text = tongs::runtime::block_on(async move {
        let tools = vec![write_tool()];
        let mut messages = vec![user_text("write out.txt")];

        // Turn 1: the model hands off a tool call.
        let context = Context {
            system_prompt: None,
            messages: messages.as_slice().into(),
            tools: tools.as_slice().into(),
        };
        let events = drive(&entry, &context).await;
        let message = done_message(&events).clone();
        assert_eq!(message.stop_reason, StopReason::ToolUse);
        let tool_call = message
            .content
            .iter()
            .find_map(|block| match block {
                ContentBlock::ToolCall(tool_call) => Some(tool_call.clone()),
                _ => None,
            })
            .expect("assistant message carries the tool call");

        // Replay tongs' own assembled assistant message plus the tool result.
        messages.push(Message::Assistant(std::sync::Arc::new(message)));
        messages.push(Message::ToolResult(std::sync::Arc::new(
            tongs::model::ToolResultMessage {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                content: vec![ContentBlock::Text(TextContent {
                    text: "ok".to_string(),
                    text_signature: None,
                })],
                details: None,
                is_error: false,
                timestamp: 0,
            },
        )));

        // Turn 2: the transcript now carries the tool result.
        let context = Context {
            system_prompt: None,
            messages: messages.as_slice().into(),
            tools: tools.as_slice().into(),
        };
        let events = drive(&entry, &context).await;
        assert_eq!(done_message(&events).stop_reason, StopReason::Stop);
        collected_text(&events)
    });
    assert_eq!(final_text, "all done");

    let requests = fake.requests();
    assert_eq!(requests.len(), 2);
    let views: Vec<usize> = requests
        .iter()
        .map(|request| request.view.as_ref().expect("view").prior_tool_results)
        .collect();
    assert_eq!(views, vec![0, 1], "jig should see the tool result on turn 2");
}

#[test]
fn anthropic_tool_loop_round_trip() {
    tool_loop_round_trip(&ANTHROPIC);
}

#[test]
fn completions_tool_loop_round_trip() {
    tool_loop_round_trip(&COMPLETIONS);
}

#[test]
fn codex_tool_loop_round_trip() {
    tool_loop_round_trip(&CODEX);
}
