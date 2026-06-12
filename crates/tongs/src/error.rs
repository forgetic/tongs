//! The crate-wide error type.
//!
//! One enum, string-carrying by design: tongs sits between wire protocols and
//! an agent loop that mostly wants to relay a failure line to a human or feed
//! it back to the model, so structured sub-errors would be ceremony without a
//! consumer. Variants exist where callers genuinely branch.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Transport-level failure: DNS, TCP, TLS, malformed HTTP.
    Http(String),
    /// The provider answered with a non-success status.
    Api { status: u16, message: String },
    /// Credentials missing, expired beyond refresh, or unreadable.
    Auth(String),
    /// A response or auth file did not parse.
    Decode(String),
    /// Tool execution failure surfaced to the model as a tool error.
    Tool(String),
    /// The operation was aborted by the caller.
    Aborted,
    /// Anything else.
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// A tool-execution failure attributed to `name`.
    pub fn tool(name: impl Into<String>, message: impl std::fmt::Display) -> Self {
        Error::Tool(format!("{}: {message}", name.into()))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http(message) => write!(f, "http error: {message}"),
            Error::Api { status, message } => write!(f, "api error ({status}): {message}"),
            Error::Auth(message) => write!(f, "auth error: {message}"),
            Error::Decode(message) => write!(f, "decode error: {message}"),
            Error::Tool(message) => write!(f, "tool error: {message}"),
            Error::Aborted => write!(f, "aborted"),
            Error::Other(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Http(error.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Decode(error.to_string())
    }
}
