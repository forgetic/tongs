//! Repro harness: hammer the real Anthropic Messages API through tongs/skein
//! with a large request body, looking for "connection reset by peer".
//!
//! ```text
//! cargo run -p tongs --example anthropic_reset_repro
//! ```
//! Env: TONGS_AUTH_FILE (default ~/.pi/agent/auth.json), REPRO_N (default 8),
//! REPRO_KB (approx user-content size in KB, default 200), REPRO_PARALLEL=1 to
//! fire all requests concurrently (mimics a sub-agent fan-out).

use std::collections::HashMap;

use futures_core::Stream;
use tongs::auth::{AuthFile, resolve_anthropic_bearer};
use tongs::http::Client;
use tongs::model::{InputType, Message, Model, ModelCost, StreamEvent, UserContent, UserMessage};
use tongs::provider::{Context, ModelEntry, StreamOptions};
use tongs::providers::create_provider;

fn main() {
    let outcome = tongs::runtime::block_on(run());
    if let Err(error) = outcome {
        eprintln!("repro failed: {error}");
        std::process::exit(1);
    }
}

fn anthropic_headers() -> HashMap<String, String> {
    HashMap::from([
        ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ("anthropic-beta".to_string(), "oauth-2025-04-20".to_string()),
        (
            "user-agent".to_string(),
            "claude-cli/2.1.139 (external, sdk-cli)".to_string(),
        ),
        ("x-app".to_string(), "cli".to_string()),
    ])
}

async fn one(
    provider: &dyn tongs::provider::Provider,
    bearer: &str,
    kb: usize,
    idx: usize,
) -> Result<usize, String> {
    let big = "Review note about the tongs crate. ".repeat(kb * 30);
    let messages = vec![Message::User(UserMessage {
        content: UserContent::Text(format!("Summarize in one word:\n{big}")),
        timestamp: 0,
    })];
    let context = Context {
        system_prompt: Some(std::borrow::Cow::Borrowed(
            "You are Claude Code, Anthropic's official CLI for Claude.",
        )),
        messages: messages.as_slice().into(),
        tools: Vec::new().into(),
    };
    let options = StreamOptions {
        api_key: Some(bearer.to_string()),
        max_tokens: Some(64),
        headers: anthropic_headers(),
        ..StreamOptions::default()
    };
    let mut stream = match provider.stream(&context, &options).await {
        Ok(s) => s,
        Err(e) => return Err(format!("req{idx} stream-start error: {e}")),
    };
    let mut n = 0usize;
    while let Some(ev) =
        std::future::poll_fn(|cx| std::pin::Pin::new(&mut stream).poll_next(cx)).await
    {
        match ev {
            Ok(StreamEvent::Done { .. }) => return Ok(n),
            Ok(StreamEvent::Error { error, .. }) => {
                return Err(format!("req{idx} terminal error: {error:?}"));
            }
            Ok(_) => n += 1,
            Err(e) => return Err(format!("req{idx} stream error after {n} events: {e}")),
        }
    }
    Err(format!("req{idx} stream ended without terminal event"))
}

async fn run() -> Result<(), String> {
    let auth_file = match std::env::var("TONGS_AUTH_FILE") {
        Ok(p) if !p.trim().is_empty() => AuthFile::new(p),
        _ => AuthFile::default_location(),
    };
    let client = Client::new();
    let bearer = resolve_anthropic_bearer(&client, &auth_file)
        .await
        .map_err(|e| format!("bearer: {e}"))?;
    let model_id = std::env::var("REPRO_MODEL").unwrap_or_else(|_| "claude-haiku-4-5".to_string());
    let entry = ModelEntry {
        model: Model {
            id: model_id.clone(),
            name: model_id.clone(),
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            base_url: String::new(),
            reasoning: true,
            input: vec![InputType::Text],
            cost: ModelCost::default(),
            context_window: 200_000,
            max_tokens: 64_000,
            headers: HashMap::new(),
        },
        api_key: None,
        headers: HashMap::new(),
        auth_header: true,
        compat: None,
        oauth_config: None,
    };
    let provider = create_provider(&entry, Some(client)).map_err(|e| e.to_string())?;

    let n: usize = std::env::var("REPRO_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let kb: usize = std::env::var("REPRO_KB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let parallel = std::env::var("REPRO_PARALLEL").is_ok();
    eprintln!("model={model_id} n={n} ~{kb}KB/req parallel={parallel}");

    let _ = parallel; // sequential is enough to reproduce a per-request reset
    let mut ok = 0;
    let mut fail = 0;
    for i in 0..n {
        match one(provider.as_ref(), &bearer, kb, i).await {
            Ok(ev) => {
                ok += 1;
                eprintln!("  req{i} OK ({ev} events)");
            }
            Err(e) => {
                fail += 1;
                eprintln!("  req{i} FAIL: {e}");
            }
        }
    }
    eprintln!("DONE ok={ok} fail={fail}");
    Ok(())
}
