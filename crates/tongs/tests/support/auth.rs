//! Real-credential resolution for the subject **recording** harness.
//!
//! The online harness (`tests/subject_record.rs`, `#[ignore]`) drives tongs
//! against the real backends, so it needs the real bearer per dialect from the
//! shared `~/.pi/agent/auth.json`:
//!
//! - **OpenAI/DeepSeek** — the `deepseek` plain API key. tongs'
//!   [`AuthFile::read_oauth`] deliberately rejects non-OAuth entries, so the
//!   key is read here with a small bespoke reader.
//! - **Codex** — the `openai-codex` OAuth access JWT (which carries the
//!   `chatgpt_account_id` claim the codex provider extracts itself), resolved
//!   through [`resolve_chatgpt_bearer`] so a near-expiry token is refreshed
//!   and written back.
//! - **Anthropic** — the subscription OAuth bearer, resolved through
//!   [`resolve_anthropic_bearer`] (refresh + write-back as well); the
//!   provider's native Claude Code workaround keys off the `sk-ant-oat`
//!   bearer.
//!
//! These functions touch the real credential file (and, on refresh, the
//! network), so they are used **only** by the manual recording leg — never by
//! the offline `cargo test` suite. Resolved values are secrets: never logged,
//! never echoed in errors.

use serde_json::Value;
use tongs::auth::{AuthFile, resolve_anthropic_bearer, resolve_chatgpt_bearer};
use tongs::http::Client;
use tongs::{Error, Result};

use super::subject::Dialect;

/// Read the `deepseek` plain API key from the auth file. Offline + pure (no
/// refresh; DeepSeek keys do not expire).
pub fn deepseek_api_key(auth_file: &AuthFile) -> Result<String> {
    let raw = std::fs::read_to_string(auth_file.path())
        .map_err(|e| Error::Auth(format!("reading {}: {e}", auth_file.path().display())))?;
    let root: Value = serde_json::from_str(&raw)
        .map_err(|e| Error::Auth(format!("parsing {}: {e}", auth_file.path().display())))?;
    root.get("deepseek")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::Auth("no `deepseek` entry in auth file".to_string()))?
        .get("key")
        .and_then(Value::as_str)
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
        .ok_or_else(|| Error::Auth("`deepseek` entry is missing its `key`".to_string()))
}

/// Resolve the bearer for `dialect` against the real auth file, refreshing the
/// OAuth dialects' tokens in place when near expiry. Async because the OAuth
/// paths may hit the network; the deepseek path is immediate.
pub async fn resolve_bearer(
    dialect: Dialect,
    client: &Client,
    auth_file: &AuthFile,
) -> Result<String> {
    match dialect {
        Dialect::OpenAi => deepseek_api_key(auth_file),
        Dialect::Codex => resolve_chatgpt_bearer(client, auth_file).await,
        Dialect::Anthropic => resolve_anthropic_bearer(client, auth_file).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(contents: Value) -> (tempdir::Dir, AuthFile) {
        let dir = tempdir::Dir::new("tongs-subject-auth");
        let path = dir.path().join("auth.json");
        std::fs::write(&path, contents.to_string()).unwrap();
        (dir, AuthFile::new(path))
    }

    /// A minimal self-cleaning temp dir so these tests need no dev-dependency.
    mod tempdir {
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(0);

        pub struct Dir(PathBuf);

        impl Dir {
            pub fn new(prefix: &str) -> Dir {
                let id = NEXT.fetch_add(1, Ordering::SeqCst);
                let path =
                    std::env::temp_dir().join(format!("{prefix}-{}-{id}", std::process::id()));
                std::fs::create_dir_all(&path).unwrap();
                Dir(path)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn reads_deepseek_api_key() {
        let (_dir, auth) = fixture(serde_json::json!({
            "deepseek": { "type": "api_key", "key": "sk-deepseek-123" }
        }));
        assert_eq!(deepseek_api_key(&auth).unwrap(), "sk-deepseek-123");
    }

    #[test]
    fn missing_deepseek_is_an_error() {
        let (_dir, auth) = fixture(serde_json::json!({ "anthropic": {} }));
        assert!(deepseek_api_key(&auth).is_err());
    }

    #[test]
    fn errors_do_not_leak_token_material() {
        // A malformed deepseek entry must not echo any neighbouring secret.
        let (_dir, auth) = fixture(serde_json::json!({
            "deepseek": { "type": "api_key" },
            "anthropic": { "access": "sk-ant-oat-super-secret" }
        }));
        let err = deepseek_api_key(&auth).unwrap_err();
        assert!(!format!("{err}").contains("sk-ant-oat-super-secret"));
    }
}
