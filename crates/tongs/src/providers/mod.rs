//! Provider implementations, one per wire API, plus the factory.
//!
//! Each provider is split sans-IO style: a pure adapter builds the request
//! body and folds SSE events into unified [`crate::model::StreamEvent`]s; the
//! shared wire driver ([`wire`]) does the HTTP/SSE transport on skein.

pub mod openai_responses;
pub(crate) mod wire;

use std::sync::Arc;

use crate::http::Client;
use crate::provider::{ModelEntry, Provider};
use crate::{Error, Result};

pub use openai_responses::CodexProvider;

/// Builds the provider for a model entry.
///
/// Routing mirrors the entry semantics our consumers already rely on: the
/// `openai-codex` provider id selects the Codex Responses route regardless of
/// the `api` string; otherwise the `api` string decides.
pub fn create_provider(entry: &ModelEntry, client: Option<Client>) -> Result<Arc<dyn Provider>> {
    let client = client.unwrap_or_default();
    if entry.model.provider == "openai-codex" {
        return Ok(Arc::new(CodexProvider::new(entry.clone(), client)));
    }
    match entry.model.api.as_str() {
        "openai-codex-responses" => Ok(Arc::new(CodexProvider::new(entry.clone(), client))),
        other => Err(Error::Other(format!(
            "unsupported provider api `{other}` (tongs supports: openai-codex-responses)"
        ))),
    }
}
