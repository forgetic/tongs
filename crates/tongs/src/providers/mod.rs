//! Provider implementations, one per wire API, plus the factory.
//!
//! Each provider is split sans-IO style: a pure adapter builds the request
//! body and folds SSE events into unified [`crate::model::StreamEvent`]s; the
//! shared wire driver ([`wire`]) does the HTTP/SSE transport on skein.

pub(crate) mod wire;
