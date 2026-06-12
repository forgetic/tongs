//! Configuration paths.

use std::path::PathBuf;

pub struct Config;

impl Config {
    /// The shared credentials file both pi CLIs write (`~/.pi/agent/auth.json`).
    /// tongs keeps reading this path so existing operator logins keep working.
    pub fn auth_path() -> PathBuf {
        Self::agent_dir().join("auth.json")
    }

    fn agent_dir() -> PathBuf {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        home.join(".pi").join("agent")
    }
}
