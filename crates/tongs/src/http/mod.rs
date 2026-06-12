//! The HTTP shell: a thin client over skein's h1 + TLS stack.
//!
//! Everything here is I/O glue; no protocol logic lives in this module. The
//! buffered [`client::Client`] serves OAuth/token flows; the streaming entry
//! point feeds provider SSE adapters.

pub mod client;

pub use client::{Client, RequestBuilder, Response};
