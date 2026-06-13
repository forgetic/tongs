//! Live one-turn smoke test against the ChatGPT Codex backend.
//!
//! Needs a real `openai-codex` login in `~/.pi/agent/auth.json` (or the file
//! named by `TONGS_AUTH_FILE`). Run with:
//!
//! ```text
//! cargo run -p tongs --example codex_smoke
//! ```
//!
//! Override the model with `TONGS_SMOKE_MODEL` (default `gpt-5.5`).

use std::collections::HashMap;

use futures_core::Stream;
use tongs::auth::{AuthFile, resolve_chatgpt_bearer};
use tongs::http::Client;
use tongs::model::{InputType, Message, Model, ModelCost, StreamEvent, UserContent, UserMessage};
use tongs::provider::{Context, ModelEntry, StreamOptions};
use tongs::providers::create_provider;

fn main() {
    let outcome = tongs::runtime::block_on(run());
    if let Err(error) = outcome {
        eprintln!("codex smoke failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> tongs::Result<()> {
    let auth_file = match std::env::var("TONGS_AUTH_FILE") {
        Ok(path) if !path.trim().is_empty() => AuthFile::new(path),
        _ => AuthFile::default_location(),
    };
    let client = Client::new();
    let bearer = resolve_chatgpt_bearer(&client, &auth_file).await?;
    eprintln!("bearer resolved from {}", auth_file.path().display());

    let model_id = std::env::var("TONGS_SMOKE_MODEL").unwrap_or_else(|_| "gpt-5.5".to_string());
    let entry = ModelEntry {
        model: Model {
            id: model_id.clone(),
            name: model_id,
            api: "openai-codex-responses".to_string(),
            provider: "openai-codex".to_string(),
            base_url: String::new(),
            reasoning: true,
            input: vec![InputType::Text],
            cost: ModelCost::default(),
            context_window: 400_000,
            max_tokens: 0,
            headers: HashMap::new(),
        },
        api_key: None,
        headers: HashMap::new(),
        auth_header: true,
        compat: None,
        oauth_config: None,
    };
    let provider = create_provider(&entry, Some(client))?;

    let messages = vec![Message::User(UserMessage {
        content: UserContent::Text("Reply with exactly the word: pong".to_string()),
        timestamp: 0,
    })];
    let context = Context {
        system_prompt: Some("You are a terse smoke test.".into()),
        messages: messages.as_slice().into(),
        tools: Vec::new().into(),
    };
    let options = StreamOptions {
        api_key: Some(bearer),
        thinking_level: Some(tongs::model::ThinkingLevel::Low),
        ..StreamOptions::default()
    };

    let mut stream = provider.stream(&context, &options).await?;
    let mut terminal = None;
    while let Some(event) =
        std::future::poll_fn(|cx| std::pin::Pin::new(&mut stream).poll_next(cx)).await
    {
        match event? {
            StreamEvent::TextDelta { delta, .. } => eprint!("{delta}"),
            StreamEvent::ThinkingDelta { delta, .. } => eprint!("\x1b[2m{delta}\x1b[0m"),
            StreamEvent::Done { message, .. } => {
                terminal = Some(message);
                break;
            }
            StreamEvent::Error { error, .. } => {
                return Err(tongs::Error::Other(format!(
                    "model error: {}",
                    error.error_message.unwrap_or_default()
                )));
            }
            _ => {}
        }
    }
    eprintln!();

    let message = terminal
        .ok_or_else(|| tongs::Error::Other("stream ended without a terminal event".to_string()))?;
    println!(
        "stop={:?} usage: input={} cached={} output={}",
        message.stop_reason, message.usage.input, message.usage.cache_read, message.usage.output
    );
    let text: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            tongs::model::ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    println!("text: {text}");
    Ok(())
}
