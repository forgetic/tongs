//! OAuth credential handling against the shared auth file.
//!
//! The file (`~/.pi/agent/auth.json`, see [`crate::config::Config`]) has a
//! **dual on-disk schema**: the nodejs pi writes `{ type: "oauth", access,
//! refresh, expires }` while the older Rust SDK wrote `{ type: "o_auth",
//! access_token, refresh_token, expires }`. The reader here tolerates both,
//! and a refresh writes the entry back **in the schema it was read in**,
//! preserving unknown fields (e.g. `accountId`) and every other provider's
//! entry.
//!
//! Secrets discipline: token bytes never appear in errors — failures carry
//! only the provider key, path, and HTTP status.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::http::Client;
use crate::util::now_ms;
use crate::{Error, Result};

/// OpenAI Codex OAuth constants (the public CLI client).
pub const CHATGPT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const CHATGPT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const CHATGPT_OAUTH_SCOPES: &str = "openid profile email";
/// The auth-file key the codex credential lives under.
pub const CHATGPT_PROVIDER_KEY: &str = "openai-codex";

/// Anthropic OAuth constants (the public Claude client).
pub const ANTHROPIC_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
pub const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// The auth-file key the Anthropic credential lives under.
pub const ANTHROPIC_PROVIDER_KEY: &str = "anthropic";

/// Refresh once a token is within this many ms of expiry.
const REFRESH_WINDOW_MS: u64 = 5 * 60 * 1000;
/// Safety margin subtracted from a freshly issued token's lifetime.
const EXPIRY_SAFETY_MS: u64 = 5 * 60 * 1000;

/// The shared credentials file.
#[derive(Clone, Debug)]
pub struct AuthFile {
    path: PathBuf,
}

impl AuthFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The default location (`~/.pi/agent/auth.json`).
    pub fn default_location() -> Self {
        Self::new(crate::config::Config::auth_path())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads and tolerantly parses one provider's OAuth entry.
    pub fn read_oauth(&self, provider_key: &str) -> Result<OAuthEntry> {
        let raw = std::fs::read_to_string(&self.path)
            .map_err(|error| Error::Auth(format!("reading {}: {error}", self.path.display())))?;
        let root: Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Auth(format!("parsing {}: {error}", self.path.display())))?;
        let entry = root
            .get(provider_key)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                Error::Auth(format!(
                    "no `{provider_key}` entry in {}",
                    self.path.display()
                ))
            })?;

        let kind = entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if kind != "oauth" && kind != "o_auth" {
            return Err(Error::Auth(format!(
                "`{provider_key}` entry in {} is `{kind}`, not OAuth",
                self.path.display()
            )));
        }

        let missing = |what: &str| {
            Error::Auth(format!(
                "`{provider_key}` entry in {} is missing its {what}",
                self.path.display()
            ))
        };
        let nodejs_schema = entry.contains_key("access");
        let access =
            string_field(entry, "access", "access_token").ok_or_else(|| missing("access token"))?;
        let refresh = string_field(entry, "refresh", "refresh_token")
            .ok_or_else(|| missing("refresh token"))?;
        let expires_ms = entry
            .get("expires")
            .and_then(Value::as_u64)
            .ok_or_else(|| missing("expiry"))?;

        Ok(OAuthEntry {
            provider_key: provider_key.to_string(),
            access,
            refresh,
            expires_ms,
            nodejs_schema,
            raw: entry.clone(),
        })
    }

    /// Writes a (refreshed) entry back, preserving every other provider's
    /// entry and the entry's own unknown fields and schema spelling.
    pub fn write_oauth(&self, entry: &OAuthEntry) -> Result<()> {
        let mut root = match std::fs::read_to_string(&self.path) {
            Ok(raw) => {
                serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| Value::Object(Map::new()))
            }
            Err(_) => Value::Object(Map::new()),
        };
        let object = root.as_object_mut().ok_or_else(|| {
            Error::Auth(format!(
                "auth file {} is not a JSON object",
                self.path.display()
            ))
        })?;
        object.insert(
            entry.provider_key.clone(),
            Value::Object(entry.raw_synced()),
        );
        let serialized = serde_json::to_string_pretty(&root)
            .map_err(|error| Error::Auth(format!("serializing auth file failed: {error}")))?;
        std::fs::write(&self.path, serialized)
            .map_err(|error| Error::Auth(format!("writing {}: {error}", self.path.display())))
    }
}

/// One provider's OAuth credential, plus the schema it was read in.
#[derive(Clone, Debug)]
pub struct OAuthEntry {
    provider_key: String,
    pub access: String,
    pub refresh: String,
    pub expires_ms: u64,
    nodejs_schema: bool,
    raw: Map<String, Value>,
}

impl OAuthEntry {
    /// True when the token is at or within the refresh window of expiry.
    pub fn is_expiring(&self, now_ms: u64) -> bool {
        self.expires_ms <= now_ms.saturating_add(REFRESH_WINDOW_MS)
    }

    /// Applies a token-endpoint response, with the standard safety margin.
    pub fn apply(&mut self, refreshed: TokenRefresh) {
        self.access = refreshed.access_token;
        if let Some(refresh) = refreshed.refresh_token {
            self.refresh = refresh;
        }
        self.expires_ms = now_ms()
            .saturating_add(refreshed.expires_in.saturating_mul(1000))
            .saturating_sub(EXPIRY_SAFETY_MS);
    }

    /// The raw entry with the live fields mirrored in, in the original
    /// schema's spelling.
    fn raw_synced(&self) -> Map<String, Value> {
        let mut raw = self.raw.clone();
        let (access_key, refresh_key) = if self.nodejs_schema {
            ("access", "refresh")
        } else {
            ("access_token", "refresh_token")
        };
        raw.insert(access_key.to_string(), Value::String(self.access.clone()));
        raw.insert(refresh_key.to_string(), Value::String(self.refresh.clone()));
        raw.insert("expires".to_string(), Value::Number(self.expires_ms.into()));
        raw
    }
}

/// A token-endpoint refresh response (the subset consumed).
#[derive(Debug, Deserialize)]
pub struct TokenRefresh {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Seconds.
    #[serde(default)]
    pub expires_in: u64,
}

/// Exchanges a refresh token at an OAuth token endpoint.
async fn refresh_token(
    client: &Client,
    token_url: &str,
    payload: &Value,
    label: &str,
) -> Result<TokenRefresh> {
    let request = client.post(token_url).json(payload)?;
    let response = request.send().await.map_err(|_| {
        // Never surface transport detail — it may echo the refresh token.
        Error::Auth(format!("{label} token refresh request failed"))
    })?;
    let status = response.status();
    if !response.is_success() {
        return Err(Error::Auth(format!(
            "{label} token refresh failed (HTTP {status})"
        )));
    }
    response
        .json::<TokenRefresh>()
        .map_err(|error| Error::Auth(format!("invalid {label} token refresh response: {error}")))
}

/// Refreshes a ChatGPT (OpenAI Codex) OAuth token.
pub async fn refresh_chatgpt(
    client: &Client,
    refresh: &str,
    token_url: Option<&str>,
) -> Result<TokenRefresh> {
    refresh_token(
        client,
        token_url.unwrap_or(CHATGPT_TOKEN_URL),
        &serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": CHATGPT_CLIENT_ID,
            "refresh_token": refresh,
            "scope": CHATGPT_OAUTH_SCOPES,
        }),
        "codex",
    )
    .await
}

/// Refreshes an Anthropic OAuth token.
pub async fn refresh_anthropic(
    client: &Client,
    refresh: &str,
    token_url: Option<&str>,
) -> Result<TokenRefresh> {
    refresh_token(
        client,
        token_url.unwrap_or(ANTHROPIC_TOKEN_URL),
        &serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": ANTHROPIC_CLIENT_ID,
            "refresh_token": refresh,
        }),
        "anthropic",
    )
    .await
}

/// Resolves a fresh ChatGPT bearer from the auth file, refreshing (and
/// writing back) when near expiry.
pub async fn resolve_chatgpt_bearer(client: &Client, auth_file: &AuthFile) -> Result<String> {
    let mut entry = auth_file.read_oauth(CHATGPT_PROVIDER_KEY)?;
    if entry.is_expiring(now_ms()) {
        let refreshed = refresh_chatgpt(client, &entry.refresh, None).await?;
        entry.apply(refreshed);
        auth_file.write_oauth(&entry)?;
    }
    Ok(entry.access)
}

/// Resolves a fresh Anthropic bearer from the auth file, refreshing (and
/// writing back) when near expiry.
pub async fn resolve_anthropic_bearer(client: &Client, auth_file: &AuthFile) -> Result<String> {
    let mut entry = auth_file.read_oauth(ANTHROPIC_PROVIDER_KEY)?;
    if entry.is_expiring(now_ms()) {
        let refreshed = refresh_anthropic(client, &entry.refresh, None).await?;
        entry.apply(refreshed);
        auth_file.write_oauth(&entry)?;
    }
    Ok(entry.access)
}

fn string_field(entry: &Map<String, Value>, primary: &str, alternate: &str) -> Option<String> {
    entry
        .get(primary)
        .or_else(|| entry.get(alternate))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        file: AuthFile,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.file.path());
        }
    }

    fn write_fixture(name: &str, contents: &str) -> Fixture {
        let path = std::env::temp_dir().join(format!(
            "tongs-auth-test-{}-{name}.json",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("write fixture");
        Fixture {
            file: AuthFile::new(path),
        }
    }

    fn far_future() -> u64 {
        now_ms() + 60 * 60 * 1000
    }

    #[test]
    fn reads_both_schemas() {
        let nodejs = write_fixture(
            "nodejs",
            &serde_json::json!({
                "openai-codex": {
                    "type": "oauth",
                    "access": "node-access",
                    "refresh": "node-refresh",
                    "accountId": "acct",
                    "expires": far_future(),
                }
            })
            .to_string(),
        );
        let entry = nodejs.file.read_oauth("openai-codex").unwrap();
        assert_eq!(entry.access, "node-access");
        assert!(entry.nodejs_schema);
        assert!(!entry.is_expiring(now_ms()));

        let rust = write_fixture(
            "rust",
            &serde_json::json!({
                "openai-codex": {
                    "type": "o_auth",
                    "access_token": "rust-access",
                    "refresh_token": "rust-refresh",
                    "expires": far_future(),
                }
            })
            .to_string(),
        );
        let entry = rust.file.read_oauth("openai-codex").unwrap();
        assert_eq!(entry.access, "rust-access");
        assert!(!entry.nodejs_schema);
    }

    #[test]
    fn write_back_preserves_schema_and_unknown_fields() {
        let fixture = write_fixture(
            "writeback",
            &serde_json::json!({
                "openai-codex": {
                    "type": "oauth",
                    "access": "old",
                    "refresh": "old-r",
                    "accountId": "acct-1",
                    "expires": 0,
                },
                "anthropic": { "type": "oauth", "access": "keep" }
            })
            .to_string(),
        );
        let mut entry = fixture.file.read_oauth("openai-codex").unwrap();
        entry.apply(TokenRefresh {
            access_token: "new".to_string(),
            refresh_token: Some("new-r".to_string()),
            expires_in: 3600,
        });
        fixture.file.write_oauth(&entry).unwrap();

        let reread: Value =
            serde_json::from_str(&std::fs::read_to_string(fixture.file.path()).unwrap()).unwrap();
        assert_eq!(reread["openai-codex"]["access"], "new");
        assert_eq!(reread["openai-codex"]["refresh"], "new-r");
        assert_eq!(reread["openai-codex"]["accountId"], "acct-1");
        assert!(reread["openai-codex"].get("access_token").is_none());
        assert_eq!(reread["anthropic"]["access"], "keep");
        assert!(reread["openai-codex"]["expires"].as_u64().unwrap() > 0);
    }

    #[test]
    fn rejects_non_oauth_entries_without_leaking() {
        let fixture = write_fixture(
            "apikey",
            &serde_json::json!({
                "openai-codex": { "type": "api_key", "key": "sk-secret" }
            })
            .to_string(),
        );
        let error = fixture.file.read_oauth("openai-codex").unwrap_err();
        let rendered = format!("{error}");
        assert!(rendered.contains("not OAuth"));
        assert!(!rendered.contains("sk-secret"));
    }

    #[test]
    fn expiring_token_is_detected() {
        let fixture = write_fixture(
            "expiry",
            &serde_json::json!({
                "openai-codex": {
                    "type": "oauth",
                    "access": "a",
                    "refresh": "r",
                    "expires": now_ms() + 60_000,
                }
            })
            .to_string(),
        );
        let entry = fixture.file.read_oauth("openai-codex").unwrap();
        // Within the 5-minute refresh window.
        assert!(entry.is_expiring(now_ms()));
    }
}
