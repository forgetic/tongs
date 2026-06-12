//! End-to-end tool-shell tests on a real skein runtime: filesystem tools via
//! the blocking pool, bash via the async child API.

use serde_json::json;
use tongs::model::ContentBlock;
use tongs::tools::{
    ToolOutput, create_bash_tool, create_edit_tool, create_find_tool, create_grep_tool,
    create_ls_tool, create_read_tool, create_write_tool,
};

fn output_text(output: &ToolOutput) -> String {
    output
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

struct Workspace {
    root: std::path::PathBuf,
}

impl Workspace {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tongs-tools-it-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create workspace");
        Self { root }
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn write_read_edit_round_trip() {
    let workspace = Workspace::new("wre");
    let root = workspace.root.clone();
    tongs::runtime::block_on(async move {
        let write = create_write_tool(&root);
        let output = write
            .execute("c1", json!({"path": "src/main.rs", "content": "fn main() {}\n"}), None)
            .await
            .expect("write");
        assert!(!output.is_error);

        let edit = create_edit_tool(&root);
        let output = edit
            .execute(
                "c2",
                json!({"path": "src/main.rs", "edits": [
                    {"oldText": "fn main() {}", "newText": "fn main() { run(); }"}
                ]}),
                None,
            )
            .await
            .expect("edit");
        assert!(!output.is_error, "edit failed: {}", output_text(&output));

        let read = create_read_tool(&root);
        let output = read
            .execute("c3", json!({"path": "src/main.rs"}), None)
            .await
            .expect("read");
        assert_eq!(output_text(&output), "fn main() { run(); }\n");
    });
}

#[test]
fn ls_grep_find_see_the_tree() {
    let workspace = Workspace::new("lgf");
    let root = workspace.root.clone();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn alpha() {}\npub fn beta() {}\n").unwrap();
    std::fs::write(root.join("README.md"), "# readme\n").unwrap();

    tongs::runtime::block_on(async move {
        let ls = create_ls_tool(&root);
        let output = ls.execute("c1", json!({}), None).await.expect("ls");
        let text = output_text(&output);
        assert!(text.contains("src/"));
        assert!(text.contains("README.md"));

        let grep = create_grep_tool(&root);
        let output = grep
            .execute("c2", json!({"pattern": "fn alpha", "glob": "*.rs"}), None)
            .await
            .expect("grep");
        let text = output_text(&output);
        assert!(text.contains("src/lib.rs:1: pub fn alpha() {}"), "grep output: {text}");

        let output = grep
            .execute("c3", json!({"pattern": "nothing-matches-this"}), None)
            .await
            .expect("grep no match");
        assert_eq!(output_text(&output), "No matches found.");

        let find = create_find_tool(&root);
        let output = find
            .execute("c4", json!({"pattern": "*.rs"}), None)
            .await
            .expect("find");
        assert_eq!(output_text(&output), "src/lib.rs");
    });
}

#[test]
fn bash_runs_merges_and_reports_exit() {
    let workspace = Workspace::new("bash");
    let root = workspace.root.clone();
    tongs::runtime::block_on(async move {
        let bash = create_bash_tool(&root);

        let output = bash
            .execute("c1", json!({"command": "echo out; echo err >&2"}), None)
            .await
            .expect("bash");
        assert!(!output.is_error);
        let text = output_text(&output);
        assert!(text.contains("out"));
        assert!(text.contains("err"), "stderr should be merged: {text}");

        let output = bash
            .execute("c2", json!({"command": "echo boom; exit 3"}), None)
            .await
            .expect("bash exit code");
        assert!(output.is_error);
        let text = output_text(&output);
        assert!(text.contains("boom"));
        assert!(text.contains("Exit code: 3"));

        let output = bash
            .execute("c3", json!({"command": "pwd"}), None)
            .await
            .expect("bash pwd");
        let text = output_text(&output);
        assert!(
            text.trim_end().ends_with(
                root.file_name().unwrap().to_str().unwrap()
            ),
            "cwd should be the workspace: {text}"
        );
    });
}

#[test]
fn bash_timeout_kills_the_command() {
    let workspace = Workspace::new("timeout");
    let root = workspace.root.clone();
    tongs::runtime::block_on(async move {
        let bash = create_bash_tool(&root);
        let started = std::time::Instant::now();
        let output = bash
            .execute(
                "c1",
                json!({"command": "echo started-marker; sleep 30; echo finished-marker", "timeout": 1}),
                None,
            )
            .await
            .expect("bash timeout");
        assert!(output.is_error);
        let text = output_text(&output);
        assert!(text.contains("timed out after 1 seconds"), "got: {text}");
        assert!(text.contains("started-marker"));
        assert!(!text.contains("finished-marker"));
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "timeout should fire promptly"
        );
    });
}
