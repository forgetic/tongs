//! The provider abstraction: one trait, unified request context and options,
//! and the event stream a provider returns.
//!
//! Provider implementations live in [`crate::providers`]; each one is a pure
//! request-builder / response-parser pair around its wire format, with the
//! HTTP transport shared through [`crate::http`].

use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::model::{Message, Model, StreamEvent, ThinkingLevel};

/// A tool definition advertised to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: serde_json::Value,
}

/// One model request: the system prompt, conversation, and tool definitions.
/// Borrowed (`Cow`) so per-turn calls need no cloning.
#[derive(Clone, Debug, Default)]
pub struct Context<'a> {
    pub system_prompt: Option<Cow<'a, str>>,
    pub messages: Cow<'a, [Message]>,
    pub tools: Cow<'a, [ToolDef]>,
}

/// Per-request options.
#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    /// The bearer credential (API key or OAuth access token).
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    /// Reasoning effort for models that support it.
    pub thinking_level: Option<ThinkingLevel>,
    /// Extra request headers; caller values override provider defaults.
    pub headers: HashMap<String, String>,
    /// Session identifier for providers with session-affine caching.
    pub session_id: Option<String>,
}

/// A streaming model provider for one wire API.
#[async_trait]
pub trait Provider: Send + Sync {
    /// The wire API id (e.g. `anthropic-messages`).
    fn api(&self) -> &str;

    /// Starts one streaming completion.
    ///
    /// Transport failures *before* the stream exists surface as `Err`; once a
    /// stream is returned, terminal failures arrive as
    /// [`StreamEvent::Error`] items carrying an assistant message with an
    /// error stop reason (mirroring the TS Pi stream contract).
    async fn stream(
        &self,
        context: &Context<'_>,
        options: &StreamOptions,
    ) -> Result<EventStream>;
}

/// The stream of [`StreamEvent`]s for one model response.
///
/// Pull-based: polling it drives the underlying HTTP read, so dropping it
/// closes the connection. Implements [`futures_core::Stream`], consumable
/// with any `StreamExt::next`.
pub struct EventStream {
    inner: Pin<Box<dyn futures_core::Stream<Item = Result<StreamEvent>> + Send>>,
}

impl EventStream {
    pub fn new(
        inner: impl futures_core::Stream<Item = Result<StreamEvent>> + Send + 'static,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }

    /// A stream that yields the given items verbatim (tests, synthesized
    /// terminal errors).
    pub fn from_events(events: Vec<Result<StreamEvent>>) -> Self {
        Self::new(Iter {
            items: events.into_iter().collect(),
        })
    }
}

impl futures_core::Stream for EventStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl std::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EventStream { .. }")
    }
}

/// A trivial queue-backed stream.
struct Iter {
    items: std::collections::VecDeque<Result<StreamEvent>>,
}

impl futures_core::Stream for Iter {
    type Item = Result<StreamEvent>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.items.pop_front())
    }
}

/// A provider entry: the model plus transport/auth knobs the factory consumes.
#[derive(Clone, Debug)]
pub struct ModelEntry {
    pub model: Model,
    /// A static credential; per-request `StreamOptions::api_key` overrides it.
    pub api_key: Option<String>,
    /// Extra headers always sent for this entry.
    pub headers: HashMap<String, String>,
    /// Whether the credential is sent as `Authorization: Bearer …`.
    pub auth_header: bool,
    /// Compatibility overrides (provider-specific JSON; reserved).
    pub compat: Option<serde_json::Value>,
    /// OAuth configuration (reserved).
    pub oauth_config: Option<serde_json::Value>,
}
