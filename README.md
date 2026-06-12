# tongs

A minimal, sans-IO agent SDK for Rust, built on
[skein](../skein): unified model/message types, streaming LLM providers
(OpenAI Responses, Anthropic Messages, OpenAI-compatible completions), OAuth
token handling, and the seven classic coding tools (read, ls, grep, find,
bash, edit, write).

tongs is a clean-room port of the semantics of Mario Zechner's MIT TypeScript
[Pi](https://github.com/earendil-works/pi) — see
[FORK_NOTES.md](FORK_NOTES.md) for the provenance and the rule that keeps this
repository clean.

## Workspace layout

| crate | purpose |
|-------|---------|
| `crates/tongs` | the SDK: `model`, `sse`, `provider`/`providers`, `tools`, `auth`, `http` |

## Design

Sans-IO throughout: pure-function-testable state machines for core logic
(the SSE decoder is fed bytes and emits events; provider adapters are pure
request-builders and response-parsers; edit matching and search filtering are
pure), with a thin shell doing async I/O on skein (`http/h1` + `tls` for
streaming, `fs`, `process` for tools).

## Using it

```toml
[dependencies]
tongs = { path = "../tongs/crates/tongs" }
```

## License

MIT — see [LICENSE](LICENSE). Ported semantics copyright Mario Zechner;
the port copyright Free Ekanayaka.
