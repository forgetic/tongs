//! A pure Server-Sent-Events decoder.
//!
//! Sans-IO: feed it raw transport bytes, collect decoded events. No socket,
//! no clock, no allocation beyond the line buffer — unit-testable by feeding
//! byte slices in any chunking and asserting on the events out.
//!
//! Semantics ported from the TypeScript Pi decoder
//! (`packages/ai/src/providers/anthropic.ts`): CRLF/LF/CR line breaks, `:`
//! comment lines skipped, `event:`/`data:` fields collected (multi-line data
//! joined with `\n`), an empty line flushes the pending event, and a final
//! `finish()` flushes whatever remains at end of stream.

/// One decoded SSE event.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SseEvent {
    /// The `event:` field, when present.
    pub event: Option<String>,
    /// The joined `data:` lines.
    pub data: String,
}

/// The decoder state machine.
#[derive(Debug, Default)]
pub struct SseDecoder {
    /// Bytes carried over until a complete line arrives.
    buffer: Vec<u8>,
    /// True when the previous chunk ended exactly on a `\r`, so a leading
    /// `\n` in the next chunk belongs to that break.
    pending_cr: bool,
    event: Option<String>,
    data: Vec<String>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consumes a chunk of transport bytes, returning every event completed
    /// by it.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for &byte in bytes {
            match byte {
                b'\n' if self.pending_cr => {
                    // The `\n` of a `\r\n` split across chunks; the line was
                    // already taken at the `\r`.
                    self.pending_cr = false;
                }
                b'\n' => {
                    self.take_line(&mut events);
                }
                b'\r' => {
                    self.take_line(&mut events);
                    self.pending_cr = true;
                }
                _ => {
                    self.pending_cr = false;
                    self.buffer.push(byte);
                }
            }
        }
        events
    }

    /// Ends the stream: decodes an unterminated final line and flushes the
    /// pending event, if any.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            self.take_line(&mut events);
        }
        if let Some(event) = self.flush() {
            events.push(event);
        }
        events
    }

    fn take_line(&mut self, events: &mut Vec<SseEvent>) {
        self.pending_cr = false;
        let line = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        if let Some(event) = self.decode_line(&line) {
            events.push(event);
        }
    }

    /// One line of the protocol; a blank line completes the pending event.
    fn decode_line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return self.flush();
        }
        if line.starts_with(':') {
            return None;
        }

        let (field, value) = match line.find(':') {
            Some(index) => (&line[..index], &line[index + 1..]),
            None => (line, ""),
        };
        let value = value.strip_prefix(' ').unwrap_or(value);

        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data.push(value.to_string()),
            _ => {}
        }
        None
    }

    fn flush(&mut self) -> Option<SseEvent> {
        if self.event.is_none() && self.data.is_empty() {
            return None;
        }
        Some(SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(chunks: &[&str]) -> Vec<SseEvent> {
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.feed(chunk.as_bytes()));
        }
        events.extend(decoder.finish());
        events
    }

    #[test]
    fn decodes_simple_data_event() {
        let events = decode_all(&["data: {\"x\":1}\n\n"]);
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "{\"x\":1}".to_string(),
            }]
        );
    }

    #[test]
    fn decodes_named_event() {
        let events = decode_all(&["event: message_start\ndata: {}\n\n"]);
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("message_start".to_string()),
                data: "{}".to_string(),
            }]
        );
    }

    #[test]
    fn joins_multi_line_data() {
        let events = decode_all(&["data: a\ndata: b\n\n"]);
        assert_eq!(events[0].data, "a\nb");
    }

    #[test]
    fn skips_comment_lines() {
        let events = decode_all(&[": keep-alive\n\ndata: x\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn handles_crlf_and_bare_cr() {
        let events = decode_all(&["data: a\r\n\r\n", "data: b\r\rdata: c\n\n"]);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
        assert_eq!(events[2].data, "c");
    }

    #[test]
    fn crlf_split_across_chunks() {
        let events = decode_all(&["data: a\r", "\n\r", "\ndata: b\n\n"]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
    }

    #[test]
    fn byte_at_a_time_chunking() {
        let raw = "event: e\ndata: hello\n\ndata: [DONE]\n\n";
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        for byte in raw.as_bytes() {
            events.extend(decoder.feed(std::slice::from_ref(byte)));
        }
        events.extend(decoder.finish());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("e"));
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[1].data, "[DONE]");
    }

    #[test]
    fn finish_flushes_unterminated_event() {
        let mut decoder = SseDecoder::new();
        assert!(decoder.feed(b"data: tail").is_empty());
        let events = decoder.finish();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "tail");
    }

    #[test]
    fn no_space_after_colon() {
        let events = decode_all(&["data:x\n\n"]);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn field_without_colon_is_field_name_only() {
        // Per the SSE spec a line without ":" is a field with empty value;
        // a bare `data` line contributes an empty data line.
        let events = decode_all(&["data\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }
}
