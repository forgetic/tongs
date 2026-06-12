//! tongs: a minimal, sans-IO agent SDK on skein.
//!
//! Clean-room port of the semantics of the MIT TypeScript Pi
//! (`earendil-works/pi`) — see FORK_NOTES.md at the repository root for the
//! provenance rule. Core logic is pure and synchronously testable; async I/O
//! lives in thin shells on skein.

pub mod auth;
pub mod config;
pub mod error;
pub mod http;
pub mod model;
pub mod provider;
pub mod providers;
pub mod runtime;
pub mod sse;
mod util;

pub use error::{Error, Result};
