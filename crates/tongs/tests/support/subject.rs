//! The tongs **subject** driver: build a tongs provider pointed at jig's
//! passthrough recorder and run the conformance scenarios through it against
//! the **real** backends, for all three wire dialects.
//!
//! Each `(dialect, scenario)` cell records what tongs actually sends on the
//! wire as a redacted `role: subject` recording under
//! `tests/fixtures/<dialect>/<scenario>/recordings/tongs/`. The offline
//! conformance suite (`tests/subject_conformance.rs`) then validates those
//! committed recordings against jig's **authoritative** templates (derived
//! from official-client recordings): T3 asserts tongs' request *grammar* is
//! conformant, T4 best-effort compares the reply shape.
//!
//! Unlike the historical pi-sdk harness this driver needs **no** Anthropic
//! special-casing: tongs' anthropic provider applies the subscription-OAuth
//! workaround natively (Claude Code identity block, identity headers, tool
//! name casing) when the bearer is an `sk-ant-oat…` token — which is exactly
//! the wire behaviour the subject recording is meant to capture.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures_core::Stream;
use jig_record::{CapturePump, Provenance, Role, build_recording};
use tongs::auth::AuthFile;
use tongs::http::Client;
use tongs::model::{
    AssistantMessage, ContentBlock, InputType, Message, Model, ModelCost, StopReason, StreamEvent,
    TextContent, ToolCall, ToolResultMessage, Usage, UserContent, UserMessage,
};
use tongs::provider::{Context, ModelEntry, StreamOptions, ToolDef};
use tongs::providers::create_provider;

use super::auth::resolve_bearer;

/// One wire dialect tongs can be pointed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// OpenAI chat-completions, recorded against DeepSeek (an OpenAI-compatible
    /// backend the shared auth file has a key for).
    OpenAi,
    /// Anthropic messages (subscription OAuth; tongs applies the Claude Code
    /// workaround natively).
    Anthropic,
    /// OpenAI Codex responses.
    Codex,
}

impl Dialect {
    /// The fixture-tree slug (`openai` / `anthropic` / `codex`).
    pub fn slug(self) -> &'static str {
        match self {
            Dialect::OpenAi => "openai",
            Dialect::Anthropic => "anthropic",
            Dialect::Codex => "codex",
        }
    }

    /// The tongs `api` string selecting the request encoder.
    pub fn api(self) -> &'static str {
        match self {
            Dialect::OpenAi => "openai-completions",
            Dialect::Anthropic => "anthropic-messages",
            Dialect::Codex => "openai-codex-responses",
        }
    }

    /// The tongs provider id. All canonical: tongs' native anthropic provider
    /// *is* the subscription workaround, and the codex provider does the
    /// `chatgpt_account_id` claim extraction the responses path needs.
    pub fn provider(self) -> &'static str {
        match self {
            Dialect::OpenAi => "deepseek",
            Dialect::Anthropic => "anthropic",
            Dialect::Codex => "openai-codex",
        }
    }

    /// The jig/recorder route the request resolves to.
    pub fn route(self) -> &'static str {
        match self {
            Dialect::OpenAi => "/chat/completions",
            Dialect::Anthropic => "/v1/messages",
            Dialect::Codex => "/backend-api/codex/responses",
        }
    }

    /// The recorder upstream-host override, where the dialect's default is not
    /// the backend the auth file has a credential for.
    pub fn upstream_override(self) -> Option<&'static str> {
        match self {
            Dialect::OpenAi => Some("api.deepseek.com"),
            Dialect::Anthropic | Dialect::Codex => None,
        }
    }

    /// A **currently-valid** model id for the real backend this dialect
    /// records against. The request must name a model the backend accepts or
    /// it 400s; the *requested* model is a harness choice, not SDK wire
    /// behaviour (jig's authoritative templates mask it), so this only has to
    /// satisfy the live backend.
    pub fn model_id(self) -> &'static str {
        match self {
            Dialect::OpenAi => "deepseek-v4-flash",
            Dialect::Anthropic => "claude-sonnet-4-5",
            Dialect::Codex => "gpt-5.5",
        }
    }

    /// All three dialects, in fixture-tree order.
    pub fn all() -> [Dialect; 3] {
        [Dialect::OpenAi, Dialect::Anthropic, Dialect::Codex]
    }

    /// Parse a fixture-tree slug (`openai`/`anthropic`/`codex`) into a dialect.
    pub fn parse(slug: &str) -> Option<Dialect> {
        match slug {
            "openai" => Some(Dialect::OpenAi),
            "anthropic" => Some(Dialect::Anthropic),
            "codex" => Some(Dialect::Codex),
            _ => None,
        }
    }
}

/// The scenarios the subject driver records, mirroring the cells of jig's
/// authoritative fixture matrix that can be driven deterministically.
/// `thinking-text` is intentionally omitted: the scenario is steered only via
/// the prompt and forcing a reasoning turn out of the SDK is not reliable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// A single user turn → one text reply.
    SingleText,
    /// A single user turn → one tool call (tool-call request grammar).
    ToolCall,
    /// user + assistant(tool_call) + tool_result → final text (the multi-turn
    /// request grammar: how tongs echoes a prior tool call and feeds a result).
    ToolResultFinal,
    /// A single user turn → **two** tool calls in one assistant turn.
    /// Best-effort: the parallel shape is elicited by the prompt naming two
    /// cities; a dialect that does not produce it is a reviewed gap, not a
    /// hard failure (see the subject matrix guard).
    ParallelToolCalls,
}

impl Scenario {
    /// The fixture-tree scenario slug.
    pub fn slug(self) -> &'static str {
        match self {
            Scenario::SingleText => "single-text",
            Scenario::ToolCall => "tool-call",
            Scenario::ToolResultFinal => "tool-result-final",
            Scenario::ParallelToolCalls => "parallel-tool-calls",
        }
    }

    /// All subject scenarios, in fixture-tree order.
    pub fn all() -> [Scenario; 4] {
        [
            Scenario::SingleText,
            Scenario::ToolCall,
            Scenario::ToolResultFinal,
            Scenario::ParallelToolCalls,
        ]
    }

    /// Parse a fixture-tree scenario slug into a [`Scenario`].
    pub fn parse(slug: &str) -> Option<Scenario> {
        match slug {
            "single-text" => Some(Scenario::SingleText),
            "tool-call" => Some(Scenario::ToolCall),
            "tool-result-final" => Some(Scenario::ToolResultFinal),
            "parallel-tool-calls" => Some(Scenario::ParallelToolCalls),
            _ => None,
        }
    }
}

/// The single tool the tool scenarios expose. A fixed, minimal function schema
/// so the recorded request's tool grammar is deterministic and reviewable.
pub fn weather_tool() -> ToolDef {
    ToolDef {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    }
}

/// Build a tongs [`ModelEntry`] for `dialect` pointed at `base_url`, carrying
/// `api_key` as the bearer.
pub fn model_entry(dialect: Dialect, base_url: &str, api_key: String) -> ModelEntry {
    ModelEntry {
        model: Model {
            id: dialect.model_id().to_string(),
            name: dialect.model_id().to_string(),
            api: dialect.api().to_string(),
            provider: dialect.provider().to_string(),
            base_url: base_url.to_string(),
            reasoning: false,
            input: vec![InputType::Text],
            cost: ModelCost::default(),
            context_window: 200_000,
            max_tokens: 8_192,
            headers: HashMap::new(),
        },
        api_key: Some(api_key),
        headers: HashMap::new(),
        auth_header: true,
        compat: None,
        oauth_config: None,
    }
}

/// Build the [`Context`] for a scenario: a plain system prompt for every
/// dialect (the anthropic provider injects the Claude Code identity block
/// itself on the OAuth path) plus the scenario's message/tool shape.
pub fn context_for(dialect: Dialect, scenario: Scenario) -> Context<'static> {
    let role = "You are a terse test assistant. Follow the instruction exactly.";

    let tools = match scenario {
        Scenario::SingleText => vec![],
        Scenario::ToolCall | Scenario::ToolResultFinal | Scenario::ParallelToolCalls => {
            vec![weather_tool()]
        }
    };

    let messages = match scenario {
        Scenario::SingleText => vec![user("Reply with exactly: hello")],
        Scenario::ToolCall => vec![user(
            "Call the get_weather tool for the city Paris. Do not reply with text.",
        )],
        Scenario::ParallelToolCalls => vec![user(
            "Call the get_weather tool once for Paris and once for London, \
             both in the same turn (two parallel tool calls). Do not reply with text.",
        )],
        Scenario::ToolResultFinal => {
            // The prior assistant tool call + its result, fed back so tongs
            // encodes the multi-turn request grammar (how it echoes a tool
            // call and a tool result). The follow-up asks for the final text.
            let call = ToolCall {
                id: "call_jig_subject_1".to_string(),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({ "city": "Paris" }),
            };
            vec![
                user("What is the weather in Paris?"),
                Message::Assistant(Arc::new(AssistantMessage {
                    content: vec![ContentBlock::ToolCall(call.clone())],
                    api: dialect.api().to_string(),
                    provider: dialect.provider().to_string(),
                    model: dialect.model_id().to_string(),
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                    error_message: None,
                    timestamp: 0,
                })),
                Message::ToolResult(Arc::new(ToolResultMessage {
                    tool_call_id: call.id,
                    tool_name: call.name,
                    content: vec![ContentBlock::Text(TextContent {
                        text: "sunny, 24C".to_string(),
                        text_signature: None,
                    })],
                    details: None,
                    is_error: false,
                    timestamp: 0,
                })),
                user("Now tell me the weather in one short sentence."),
            ]
        }
    };

    Context {
        system_prompt: Some(role.into()),
        messages: messages.into(),
        tools: tools.into(),
    }
}

/// A user message with text content.
fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: 0,
    })
}

/// Record one `(dialect, scenario)` **subject** cell against the real backend:
/// resolve the dialect bearer, stand up the capture pump, drive one tongs
/// completion through it, and write the redacted `subject` recording under
/// `fixtures_root`. Returns the captured HTTP status so the caller can flag a
/// non-2xx (a finding, not a fixture).
///
/// Online and credential-driven — never part of the offline `cargo test`
/// suite.
pub fn record_subject_cell(
    dialect: Dialect,
    scenario: Scenario,
    fixtures_root: &Path,
    captured: &str,
    recorder_sha: &str,
    auth_file: &AuthFile,
) -> io::Result<u16> {
    // Resolve the real bearer FIRST, before standing up the recorder, so a
    // credential failure (e.g. an OAuth refresh rejection) is an immediate,
    // clean error rather than a half-built capture. The OAuth paths may
    // refresh a near-expiry token over the network and write it back.
    let auth = auth_file.clone();
    let api_key = tongs::runtime::block_on(async move {
        let client = Client::new();
        resolve_bearer(dialect, &client, &auth).await
    })
    .map_err(|e| io::Error::other(format!("resolve bearer: {e}")))?;

    // The recorder accepts concurrently on its own runtime thread while this
    // thread drives tongs on its own runtime.
    let pump = CapturePump::start(dialect.upstream_override().map(str::to_string))?;

    let entry = model_entry(dialect, &pump.base_url(), api_key);
    let context = context_for(dialect, scenario);
    tongs::runtime::block_on(async move {
        let provider = match create_provider(&entry, None) {
            Ok(provider) => provider,
            Err(e) => {
                eprintln!("  [provider build error, nothing driven] {e}");
                return;
            }
        };
        // Drain the stream so the full request/response round-trips through
        // the recorder. Stream errors (e.g. a 4xx from the backend) are
        // tolerated: the recorder still captured the exchange, which is the
        // finding.
        match provider.stream(&context, &StreamOptions::default()).await {
            Ok(mut stream) => {
                while let Some(event) =
                    std::future::poll_fn(|cx| Pin::new(&mut stream).poll_next(cx)).await
                {
                    match event {
                        Ok(event) => {
                            if matches!(event, StreamEvent::Done { .. } | StreamEvent::Error { .. })
                            {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("  [stream event error, capture still written] {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => eprintln!("  [stream start error, capture still written] {e}"),
        }
    });

    // Give the pump a beat to finish recording the exchange, then stop it.
    std::thread::sleep(Duration::from_millis(300));
    let exchanges = pump.stop();
    if exchanges.len() != 1 {
        eprintln!(
            "  [warning] expected exactly one routable exchange, captured {}; using the last",
            exchanges.len()
        );
    }
    let (request, response, route) = exchanges
        .into_iter()
        .next_back()
        .ok_or_else(|| io::Error::other("no routable exchange captured"))?;

    let status = response.status;
    let provenance = Provenance {
        client: "tongs".to_string(),
        role: Role::Subject,
        scenario: scenario.slug().to_string(),
        client_version: Some(format!("tongs {}", env!("CARGO_PKG_VERSION"))),
        captured: captured.to_string(),
        recorder_sha: recorder_sha.to_string(),
    };
    let recording = build_recording(&request, &response, &route, &provenance);
    let dir = recording.write(fixtures_root)?;
    eprintln!(
        "  wrote {}/{} subject recording -> {} (HTTP {status})",
        dialect.slug(),
        scenario.slug(),
        dir.display()
    );
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_slug_api_route_are_consistent() {
        assert_eq!(Dialect::OpenAi.route(), "/chat/completions");
        assert_eq!(Dialect::Anthropic.route(), "/v1/messages");
        assert_eq!(Dialect::Codex.route(), "/backend-api/codex/responses");
        assert_eq!(
            Dialect::all().map(|d| d.slug()),
            ["openai", "anthropic", "codex"]
        );
        for dialect in Dialect::all() {
            assert_eq!(Dialect::parse(dialect.slug()), Some(dialect));
        }
        for scenario in Scenario::all() {
            assert_eq!(Scenario::parse(scenario.slug()), Some(scenario));
        }
    }

    #[test]
    fn every_dialect_gets_a_plain_system_prompt() {
        // tongs applies the Claude Code identity itself on the anthropic OAuth
        // path; the harness must not duplicate it.
        for dialect in Dialect::all() {
            let ctx = context_for(dialect, Scenario::SingleText);
            assert_eq!(
                ctx.system_prompt.as_deref(),
                Some("You are a terse test assistant. Follow the instruction exactly.")
            );
        }
    }

    #[test]
    fn tool_scenarios_expose_the_weather_tool() {
        let single = context_for(Dialect::OpenAi, Scenario::SingleText);
        assert!(single.tools.is_empty());
        for scenario in [
            Scenario::ToolCall,
            Scenario::ToolResultFinal,
            Scenario::ParallelToolCalls,
        ] {
            let ctx = context_for(Dialect::OpenAi, scenario);
            assert_eq!(ctx.tools.len(), 1);
            assert_eq!(ctx.tools[0].name, "get_weather");
        }
    }

    #[test]
    fn tool_result_final_carries_the_full_multi_turn_grammar() {
        let ctx = context_for(Dialect::OpenAi, Scenario::ToolResultFinal);
        // user, assistant(tool_call), tool_result, user.
        assert_eq!(ctx.messages.len(), 4);
        assert!(matches!(ctx.messages[1], Message::Assistant(_)));
        assert!(matches!(ctx.messages[2], Message::ToolResult(_)));
    }

    #[test]
    fn model_entry_points_at_base_url_with_dialect_api() {
        let entry = model_entry(Dialect::Codex, "http://127.0.0.1:9999", "jwt".to_string());
        assert_eq!(entry.model.base_url, "http://127.0.0.1:9999");
        assert_eq!(entry.model.api, "openai-codex-responses");
        assert_eq!(entry.model.provider, "openai-codex");
        assert_eq!(entry.api_key.as_deref(), Some("jwt"));
    }
}
