//! Offline **subject** conformance — T3 (request-grammar validation) and T4
//! (cross-driver response consistency) for tongs against jig's authoritative
//! contract.
//!
//! Data-driven over every committed
//! `tests/fixtures/<dialect>/<scenario>/recordings/tongs/` recording captured
//! from the **real** backends by the (manual, online) harness in
//! `tests/subject_record.rs`. These tests are **offline** — no network, no
//! credentials — and run in the default `cargo test`: they read committed
//! bytes and compare structures against the **authoritative** templates that
//! jig derived from official-client recordings (Claude Code, Codex, the
//! OpenAI/DeepSeek SDK), read from the sibling jig checkout via
//! [`jig_core::fixtures_root`].
//!
//! - **T3 — request validation.** Reduce the subject `request.json` body to
//!   its request *grammar* and assert it is **conformant** with the
//!   authoritative `request.template.json` grammar: every JSON key /
//!   value-type / array-element shape tongs sends must appear in the official
//!   client's request. The two requests are not equal — the official client
//!   sends its whole prompt and tool catalogue — so T3 compares the **wire
//!   grammar**, not content or size. An unreviewed divergence is a candidate
//!   tongs bug and fails the test with the offending JSON path.
//! - **T4 — cross-driver response consistency (best-effort).** Parse the
//!   subject `response.sse` under the dialect and mask it the way a response
//!   template is derived, then assert its canonical `reply` grammar matches
//!   the authoritative `response.template.json`'s `reply`. Both drivers, fed
//!   the same scenario, should yield the same masked reply skeleton. Scenarios
//!   whose reply shape legitimately differs (the model is free to answer
//!   differently) are reported, not failed; only `single-text` is a hard gate.
//!
//! On top of the data-driven checks, [`subject_matrix_is_complete`] is a
//! **full-matrix guard**: every required `(dialect, scenario)` cell must be
//! captured, or be an explicitly reviewed-unavailable entry — a missing
//! subject recording *fails* the build instead of being silently tolerated.
//!
//! A failure prints the readable JSON path that diverged, never two large
//! blobs.

use std::path::{Path, PathBuf};

use jig_core::Dialect;
use jig_core::conform::{
    ResponseTemplate, derive_response_template, grammar_findings, request_grammar, structural_diff,
};
use serde_json::Value;

/// The committed tongs subject recordings, resolved from this crate's manifest
/// dir so the test works regardless of the cwd `cargo test` runs from.
fn subject_fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// jig's authoritative fixture tree (templates derived from official-client
/// recordings), via the path dependency on the sibling checkout.
fn jig_fixtures_root() -> PathBuf {
    jig_core::fixtures_root()
        .canonicalize()
        .expect("fixtures/ exists in the jig checkout")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&read(path)).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// One tongs subject recording to check: which dialect/scenario it is, where
/// its files live, and where the scenario's authoritative templates live in
/// jig's tree.
struct SubjectCase {
    dialect: Dialect,
    label: String,
    jig_scenario_root: PathBuf,
    subject_dir: PathBuf,
}

/// Every committed `recordings/tongs/` recording across all dialects, sorted.
/// Driving the tests off the committed tree means a new subject recording
/// needs no test edit; an empty tree (nothing captured yet) yields no cases —
/// required coverage is enforced by [`subject_matrix_is_complete`], not here.
fn subject_cases() -> Vec<SubjectCase> {
    let mut cases = Vec::new();
    let root = subject_fixtures_root();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return cases;
    };
    let mut dialect_dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dialect_dirs.sort();

    for dialect_dir in dialect_dirs {
        let Some(dialect) = Dialect::for_path(dialect_route(&dialect_dir)) else {
            continue;
        };
        let mut scenario_dirs: Vec<PathBuf> = std::fs::read_dir(&dialect_dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", dialect_dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        scenario_dirs.sort();

        for scenario_root in scenario_dirs {
            let subject_dir = scenario_root.join("recordings/tongs");
            if !subject_dir.join("request.json").exists() {
                continue;
            }
            let scenario = scenario_root
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let slug = dialect_slug(&dialect_dir);
            cases.push(SubjectCase {
                dialect,
                label: format!("{slug}/{scenario}"),
                jig_scenario_root: jig_fixtures_root().join(&slug).join(&scenario),
                subject_dir,
            });
        }
    }
    cases
}

/// The **expected** tongs subject matrix: every `(dialect, scenario)` cell for
/// which a `recordings/tongs/` recording is *required* to exist.
///
/// It is every cell of jig's authoritative template matrix the online harness
/// can drive deterministically:
///
/// - `thinking-text` is deliberately **not** here even though the `anthropic`
///   and `codex` dialects have authoritative templates: the scenario is
///   steered only via the prompt and forcing a reasoning turn out of the SDK
///   is not reliable (see the `Scenario` doc in `tests/support/subject.rs`).
/// - `codex/parallel-tool-calls` is **not** here because jig has no
///   authoritative `request.template.json` for it — there is nothing to
///   anchor T3 against. When that template lands in jig, adding the cell here
///   is a one-line change.
const EXPECTED_SUBJECT_MATRIX: &[(&str, &str)] = &[
    ("openai", "single-text"),
    ("openai", "tool-call"),
    ("openai", "tool-result-final"),
    ("openai", "parallel-tool-calls"),
    ("anthropic", "single-text"),
    ("anthropic", "tool-call"),
    ("anthropic", "tool-result-final"),
    ("anthropic", "parallel-tool-calls"),
    ("codex", "single-text"),
    ("codex", "tool-call"),
    ("codex", "tool-result-final"),
];

/// **Reviewed missing subject cells**: required cells from
/// [`EXPECTED_SUBJECT_MATRIX`] that are *currently uncaptured* for a reviewed,
/// external reason, plus that reason. The matrix-completeness guard treats
/// these as a known, accepted gap rather than a hard failure, while every
/// *other* missing cell fails the build.
///
/// The list is self-cleaning: [`subject_matrix_is_complete`] fails if an entry
/// here is actually *present* on disk (a stale skip), and fails if an entry
/// names a cell not in [`EXPECTED_SUBJECT_MATRIX`] (a typo or drift). So a gap
/// cannot rot silently.
///
/// Each entry is `((dialect, scenario), why-unavailable)`.
const REVIEWED_MISSING_SUBJECTS: &[((&str, &str), &str)] = &[];

/// Whether a tongs subject recording exists on disk for `(dialect, scenario)`:
/// the presence of its `request.json` is the same liveness signal
/// [`subject_cases`] uses to admit a case.
fn subject_recording_exists(dialect: &str, scenario: &str) -> bool {
    subject_fixtures_root()
        .join(dialect)
        .join(scenario)
        .join("recordings/tongs/request.json")
        .exists()
}

/// The fixture-tree dialect-dir name (`openai`/`anthropic`/`codex`).
fn dialect_slug(dialect_dir: &Path) -> String {
    dialect_dir
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

/// Map a fixtures dialect directory to the route its dialect parses on, so
/// `Dialect::for_path` can resolve it (the tree slug and the route differ).
fn dialect_route(dialect_dir: &Path) -> &'static str {
    match dialect_slug(dialect_dir).as_str() {
        "openai" => "/chat/completions",
        "anthropic" => "/v1/messages",
        "codex" => "/backend-api/codex/responses",
        _ => "/unknown",
    }
}

/// The captured request body of a recording's `request.json`, parsed from the
/// `body` string field.
fn request_body(dir: &Path) -> Value {
    let request = read_json(&dir.join("request.json"));
    let body = request["body"].as_str().expect("request body string");
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{}: body not JSON: {e}", dir.display()))
}

/// **Full-matrix guard.** Every required subject cell in
/// [`EXPECTED_SUBJECT_MATRIX`] must either be present on disk or be an
/// explicitly reviewed-unavailable cell in [`REVIEWED_MISSING_SUBJECTS`].
///
/// Three failure modes, each with an actionable message:
///   1. a required cell is **missing** and **not reviewed** → capture it (or,
///      if genuinely unavailable, add a reviewed entry with the reason);
///   2. a reviewed-missing cell is **actually present** → delete the now-stale
///      skip entry so the cell rejoins the real T3/T4 gate; and
///   3. a reviewed-missing entry names a cell **outside** the expected matrix
///      → a typo/drift; fix the entry.
#[test]
fn subject_matrix_is_complete() {
    let is_reviewed_missing =
        |cell: (&str, &str)| REVIEWED_MISSING_SUBJECTS.iter().any(|(c, _)| *c == cell);

    // (3) The reviewed-missing list must only name cells that are actually
    // required; otherwise the skip list silently drifts from the matrix.
    for ((dialect, scenario), _why) in REVIEWED_MISSING_SUBJECTS {
        assert!(
            EXPECTED_SUBJECT_MATRIX.contains(&(*dialect, *scenario)),
            "REVIEWED_MISSING_SUBJECTS names {dialect}/{scenario}, which is not in \
             EXPECTED_SUBJECT_MATRIX — remove the stale entry or add the cell to the matrix"
        );
    }

    // (2) A reviewed-missing cell that is now present is a stale skip: drop it
    // so the cell is held to the real T3/T4 conformance gate again.
    let stale: Vec<String> = REVIEWED_MISSING_SUBJECTS
        .iter()
        .filter(|((dialect, scenario), _)| subject_recording_exists(dialect, scenario))
        .map(|((dialect, scenario), _)| format!("{dialect}/{scenario}"))
        .collect();
    assert!(
        stale.is_empty(),
        "these cells are now captured but still listed in REVIEWED_MISSING_SUBJECTS — \
         delete the stale skip entries so they rejoin the T3/T4 gate:\n  {}",
        stale.join("\n  ")
    );

    // (1) The core guard: every required cell is present, or reviewed-unavailable.
    let missing: Vec<String> = EXPECTED_SUBJECT_MATRIX
        .iter()
        .filter(|(dialect, scenario)| {
            !subject_recording_exists(dialect, scenario)
                && !is_reviewed_missing((dialect, scenario))
        })
        .map(|(dialect, scenario)| format!("{dialect}/{scenario}"))
        .collect();
    assert!(
        missing.is_empty(),
        "required tongs subject recordings are missing and not reviewed (full-matrix guard). \
         Capture each with `JIG_DIALECT=<d> JIG_SCENARIO=<s> cargo test -p tongs \
         --test subject_record record_one_subject_fixture -- --ignored --exact`, or add a \
         reviewed entry to REVIEWED_MISSING_SUBJECTS if the cell is genuinely unavailable:\n  {}",
        missing.join("\n  ")
    );

    // Surface the accepted gaps in test output so they stay visible rather
    // than silently skipped.
    let reviewed_gaps: Vec<String> = REVIEWED_MISSING_SUBJECTS
        .iter()
        .map(|((dialect, scenario), why)| format!("{dialect}/{scenario}: {why}"))
        .collect();
    if !reviewed_gaps.is_empty() {
        eprintln!("subject matrix reviewed-unavailable cells:");
        for gap in &reviewed_gaps {
            eprintln!("  - {gap}");
        }
    }
}

/// **Reviewed T3 findings**: request-grammar divergences a human has inspected
/// and judged **benign** — a spec-valid field tongs sends that the *one*
/// official-client capture happened not to, **not** a tongs bug. Each entry is
/// `(label-suffix-or-"*", json-path, why-benign)`. T3 stays a real gate (an
/// *unreviewed* divergence fails the build) while not flagging known-good
/// optional fields.
///
/// A label of `"*"` applies to every dialect/scenario; otherwise it must equal
/// the case label (`"openai/single-text"`).
const REVIEWED_T3_FINDINGS: &[(&str, &str, &str)] = &[
    // tongs always sets an explicit `max_tokens` on the chat-completions
    // request; it is a valid, optional OpenAI field that the recorded
    // DeepSeek-SDK sample simply omitted. Spec-compliant, not a bug. (Fires on
    // the four openai cells only — the anthropic request carries max_tokens in
    // the authoritative template too, and codex does not use the field.)
    (
        "*",
        "max_tokens",
        "valid optional chat-completions field tongs always sets; the authoritative sample omitted it",
    ),
    // tongs encodes a text-only user turn with the Anthropic messages API's
    // documented *string shorthand* (`"content": "..."`); Claude Code always
    // sends an array of content blocks because it attaches per-block
    // `cache_control`. Semantically identical, both accepted (HTTP 200).
    (
        "anthropic/single-text",
        "messages[0].content",
        "user content string shorthand (tongs) vs block array (official); both documented and accepted",
    ),
    (
        "anthropic/tool-call",
        "messages[0].content",
        "user content string shorthand (tongs) vs block array (official); both documented and accepted",
    ),
    (
        "anthropic/tool-result-final",
        "messages[0].content",
        "user content string shorthand (tongs) vs block array (official); both documented and accepted",
    ),
    (
        "anthropic/parallel-tool-calls",
        "messages[0].content",
        "user content string shorthand (tongs) vs block array (official); both documented and accepted",
    ),
    // A `tool_result` block's `content` accepts a plain string or an array of
    // blocks; tongs always sends the block array, the official capture sent
    // the string form. Both documented, both accepted. The path is the
    // grammar-collapsed distinct-element index, not the literal position.
    (
        "anthropic/tool-result-final",
        "messages[1].content[0].content",
        "tool_result content block array (tongs) vs string shorthand (official); both documented and accepted",
    ),
    // tongs always serializes `is_error` on a tool_result block; the official
    // client omits the (documented, optional) field when false.
    (
        "anthropic/tool-result-final",
        "messages[1].content[0].is_error",
        "documented optional tool_result field tongs always sets; the official client omits it when false",
    ),
    // tongs replays the Responses item id (`fc_…`) on an echoed function_call
    // input item — a deliberate, unit-tested design for item/reasoning replay
    // continuity (see `normalize_tool_call_id` in providers/openai_responses).
    // The codex CLI strips item ids when replaying; the backend accepts both
    // (HTTP 200).
    (
        "codex/tool-result-final",
        "input[0].id",
        "Responses item id replayed for continuity (deliberate tongs design); the codex CLI omits it",
    ),
];

/// Whether a finding at `path` for `label` is a reviewed-benign divergence.
fn is_reviewed(label: &str, path: &str) -> bool {
    REVIEWED_T3_FINDINGS
        .iter()
        .any(|(scope, p, _)| (*scope == "*" || *scope == label) && *p == path)
}

#[test]
fn t3_subject_request_grammar_conforms_to_authoritative() {
    let mut reviewed_seen = Vec::new();
    for case in subject_cases() {
        let template_path = case.jig_scenario_root.join("request.template.json");
        if !template_path.exists() {
            // No authoritative request template for this scenario → nothing to
            // validate the subject grammar against. (Should not happen for the
            // matrix cells, but skip rather than panic.)
            continue;
        }

        let subject = request_body(&case.subject_dir);
        let authoritative = read_json(&template_path)["body"].clone();

        // Partition findings into reviewed-benign (allowlisted) and unexpected.
        let (reviewed, unexpected): (Vec<_>, Vec<_>) = grammar_findings(&subject, &authoritative)
            .into_iter()
            .partition(|f| is_reviewed(&case.label, &f.path));
        for f in &reviewed {
            reviewed_seen.push(format!("{}: {}", case.label, f.path));
        }

        assert!(
            unexpected.is_empty(),
            "T3 {}: tongs request grammar diverges from the authoritative contract \
             (unreviewed — a candidate tongs bug; add to REVIEWED_T3_FINDINGS only after \
             review):\n  {}",
            case.label,
            unexpected
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }
    // Surface the reviewed findings so they stay visible in test output rather
    // than silently suppressed.
    if !reviewed_seen.is_empty() {
        eprintln!("T3 reviewed (benign) findings:");
        for f in &reviewed_seen {
            eprintln!("  - {f}");
        }
    }
}

#[test]
fn t4_subject_response_reply_matches_authoritative() {
    for case in subject_cases() {
        let template_path = case.jig_scenario_root.join("response.template.json");
        if !template_path.exists() {
            continue;
        }

        // Skip a subject response that did not complete (a finding captured by
        // the harness as a non-2xx).
        let status = read_json(&case.subject_dir.join("response.headers"))["status"]
            .as_u64()
            .unwrap_or(0);
        if !(200..300).contains(&status) {
            continue;
        }

        let sse = std::fs::read(case.subject_dir.join("response.sse")).unwrap();
        let subject_template = match derive_response_template(case.dialect, &sse, &[]) {
            Ok(t) => t,
            // A subject stream that does not parse is a finding surfaced
            // elsewhere (the raw bytes are committed); T4 is best-effort, so
            // skip it here.
            Err(_) => continue,
        };

        let authoritative: ResponseTemplate = serde_json::from_value(read_json(&template_path))
            .expect("authoritative template shape");

        // Compare the canonical reply *grammar* (turn kinds + masked content +
        // stop), not the headers (different capture paths) — cross-driver
        // consistency.
        let subj_reply = request_grammar(&subject_template.reply);
        let auth_reply = request_grammar(&authoritative.reply);
        let diff = structural_diff(&auth_reply, &subj_reply);

        // Best-effort: a reply-shape difference is reported but only fails for
        // the single-text scenario, where both drivers must yield one masked
        // Text turn.
        let scenario_is_single_text = case.label.ends_with("/single-text");
        if scenario_is_single_text {
            assert!(
                diff.is_empty(),
                "T4 {}: subject reply grammar differs from authoritative:\n  {}",
                case.label,
                diff.join("\n  ")
            );
        } else if !diff.is_empty() {
            // Non-fatal cross-driver note for the tool scenarios (the model is
            // free to answer differently); printed for the operator, not a
            // failure.
            eprintln!(
                "T4 {} (best-effort, non-fatal): reply grammar differs:\n  {}",
                case.label,
                diff.join("\n  ")
            );
        }
    }
}

/// Backstop the redaction invariant from the test side: no bearer/secret-shaped
/// string under any committed tongs subject recording.
#[test]
fn no_secret_material_under_subject_recordings() {
    for case in subject_cases() {
        for name in [
            "request.json",
            "response.headers",
            "meta.json",
            "response.sse",
        ] {
            let path = case.subject_dir.join(name);
            if !path.exists() {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let text = String::from_utf8_lossy(&bytes);
            assert!(
                !text.contains("Bearer sk-")
                    && !text.contains("sk-ant-oat")
                    && !text.contains("sk-live"),
                "possible secret in {}",
                path.display()
            );
            for line in text.lines() {
                if line.to_ascii_lowercase().contains("\"authorization\"") {
                    assert!(
                        !line.contains("Bearer ") || line.contains("REDACTED"),
                        "authorization not redacted in {}",
                        path.display()
                    );
                }
            }
        }
    }
}
