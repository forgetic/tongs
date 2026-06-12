# Provenance and clean-room rule

tongs is a clean-room Rust port of the agent-SDK semantics of **Pi**, Mario
Zechner's MIT-licensed TypeScript agent toolkit (`earendil-works/pi`, formerly
`badlogic/pi-mono`). It exists to replace the `pi_agent_rust` crate in our
projects, because no clean-MIT version of that crate exists: every published
release (0.1.7, 0.1.13, 0.1.18) is licensed "MIT + OpenAI/Anthropic Rider",
and the rider was present in that repository's first LICENSE commit.

## The rule

**Code from `pi_agent_rust` must never be copied into this repository.**
Not snippets, not type definitions, not comments. The permitted reference
material is, exhaustively:

1. The MIT TypeScript Pi source (`earendil-works/pi`) — semantics, wire
   formats, tool behaviour, auth-file schema.
2. Public provider API documentation (OpenAI, Anthropic, DeepSeek).
3. Our own MIT projects (skein, temper, smith, anvil) for runtime and
   architectural patterns.

API-shape compatibility with `pi_agent_rust` (module paths, type names,
function signatures that our consumers already call) is fine — interfaces are
not copyrightable expression and the surface is documented by our consumers'
own call sites — but every implementation line here is written fresh against
the TypeScript source and the provider docs.

## Licensing

MIT, dual copyright: Mario Zechner for the ported semantics, Free Ekanayaka
for the port. See LICENSE.

## Design posture (differs from upstream by intent)

Unlike both Pi implementations, tongs is **sans-IO**: core logic (SSE
decoding, provider request-building/response-parsing, edit matching, output
truncation) lives in pure, synchronously-testable state machines and
functions; a thin shell does the async I/O on
[skein](../skein) (h1 HTTP + TLS, fs, child processes). This mirrors
`temper-io-engine` and anvil's AgentMachine/AgentShell split.
