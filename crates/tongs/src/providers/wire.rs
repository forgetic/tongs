//! The shared SSE wire driver.
//!
//! [`WireAdapter`] is the pure half a provider implements: SSE events in,
//! unified stream events out, no I/O. [`sse_event_stream`] is the shell half:
//! it lazily pulls the HTTP body as the consumer polls, feeds bytes through
//! the [`SseDecoder`](crate::sse::SseDecoder), and hands decoded events to
//! the adapter. Dropping the stream closes the connection.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::http::client::IncomingBody;
use crate::model::StreamEvent;
use crate::provider::EventStream;
use crate::sse::{SseDecoder, SseEvent};
use crate::{Error, Result};

/// The pure provider-side half of a streaming response: folds decoded SSE
/// events into unified stream events. Implementations are state machines,
/// unit-testable with synthetic [`SseEvent`]s and no transport.
pub(crate) trait WireAdapter: Send + Unpin + 'static {
    /// Consumes one SSE event.
    fn on_sse(&mut self, event: SseEvent) -> Result<Vec<StreamEvent>>;

    /// Transport end-of-stream. Returns trailing events; an adapter that has
    /// already emitted its terminal `Done`/`Error` returns nothing, one that
    /// has not should error (the stream ended mid-response).
    fn on_eof(&mut self) -> Result<Vec<StreamEvent>>;
}

/// Builds the public [`EventStream`] from a streaming HTTP body and a pure
/// adapter.
pub(crate) fn sse_event_stream(body: IncomingBody, adapter: impl WireAdapter) -> EventStream {
    EventStream::new(SseDriven {
        body,
        decoder: SseDecoder::new(),
        adapter,
        queue: VecDeque::new(),
        done: false,
    })
}

struct SseDriven<A> {
    body: IncomingBody,
    decoder: SseDecoder,
    adapter: A,
    queue: VecDeque<Result<StreamEvent>>,
    done: bool,
}

impl<A: WireAdapter> SseDriven<A> {
    /// Feeds decoded SSE events through the adapter into the queue. An
    /// adapter error terminates the stream after being yielded.
    fn dispatch(&mut self, events: Vec<SseEvent>) {
        for event in events {
            if self.done {
                return;
            }
            match self.adapter.on_sse(event) {
                Ok(out) => self.queue.extend(out.into_iter().map(Ok)),
                Err(error) => {
                    self.queue.push_back(Err(error));
                    self.done = true;
                }
            }
        }
    }
}

impl<A: WireAdapter> futures_core::Stream for SseDriven<A> {
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(item) = this.queue.pop_front() {
                return Poll::Ready(Some(item));
            }
            if this.done {
                return Poll::Ready(None);
            }
            match Pin::new(&mut this.body).poll_chunk(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(Some(bytes))) => {
                    let events = this.decoder.feed(&bytes);
                    this.dispatch(events);
                }
                Poll::Ready(Ok(None)) => {
                    let trailing = this.decoder.finish();
                    this.dispatch(trailing);
                    if !this.done {
                        match this.adapter.on_eof() {
                            Ok(out) => this.queue.extend(out.into_iter().map(Ok)),
                            Err(error) => this.queue.push_back(Err(error)),
                        }
                    }
                    this.done = true;
                }
                Poll::Ready(Err(error)) => {
                    this.queue.push_back(Err(Error::Http(error.to_string())));
                    this.done = true;
                }
            }
        }
    }
}
