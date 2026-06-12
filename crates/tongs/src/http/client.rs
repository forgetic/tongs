//! A small HTTP client on skein's h1 stack.
//!
//! Buffered requests (`send`) cover OAuth refresh and other token flows;
//! streaming requests (`send_streaming`, crate-internal) carry provider SSE.
//! TLS, Host, and Content-Length are handled by skein's `HttpClient`.

use std::future::poll_fn;
use std::pin::Pin;
use std::sync::Arc;

use skein::bytes::Buf;
use skein::http::body::{Body, Frame};
use skein::http::h1::client::{ClientIncomingBody, ClientStreamingResponse};
use skein::http::h1::http_client::{ClientIo, HttpClient, HttpClientConfig, RedirectPolicy};
use skein::http::h1::types::Method;

use crate::{Error, Result};

/// A reusable HTTP client (connection setup per request; cheap to clone).
#[derive(Clone)]
pub struct Client {
    inner: Arc<HttpClient>,
}

impl Client {
    pub fn new() -> Self {
        let config = HttpClientConfig {
            redirect_policy: RedirectPolicy::Limited(10),
            user_agent: Some("tongs/0.1".to_string()),
            ..HttpClientConfig::default()
        };
        Self {
            inner: Arc::new(HttpClient::with_config(config)),
        }
    }

    pub fn get(&self, url: &str) -> RequestBuilder {
        self.builder(Method::Get, url)
    }

    pub fn post(&self, url: &str) -> RequestBuilder {
        self.builder(Method::Post, url)
    }

    fn builder(&self, method: Method, url: &str) -> RequestBuilder {
        RequestBuilder {
            client: self.inner.clone(),
            method,
            url: url.to_string(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Client { .. }")
    }
}

/// One request being assembled.
pub struct RequestBuilder {
    client: Arc<HttpClient>,
    method: Method,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl RequestBuilder {
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Sets a JSON body (and `Content-Type: application/json`).
    pub fn json(mut self, value: &impl serde::Serialize) -> Result<Self> {
        self.body = serde_json::to_vec(value)?;
        self.headers
            .push(("Content-Type".to_string(), "application/json".to_string()));
        Ok(self)
    }

    /// Sends the request and buffers the full response.
    pub async fn send(self) -> Result<Response> {
        let response = self
            .client
            .request(self.method, &self.url, self.headers, self.body)
            .await
            .map_err(|error| Error::Http(error.to_string()))?;
        Ok(Response {
            status: response.status,
            headers: response.headers,
            body: response.body,
        })
    }

    /// Sends the request and returns the head plus a streaming body.
    pub(crate) async fn send_streaming(self) -> Result<StreamingResponse> {
        let response = self
            .client
            .request_streaming(self.method, &self.url, self.headers, self.body)
            .await
            .map_err(|error| Error::Http(error.to_string()))?;
        Ok(StreamingResponse { inner: response })
    }
}

/// A buffered HTTP response.
#[derive(Clone, Debug)]
pub struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Response {
    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// The first value of a header, case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The response body as text. Async to keep call sites
    /// transport-agnostic (`response.text().await`).
    pub async fn text(self) -> Result<String> {
        String::from_utf8(self.body).map_err(|error| Error::Decode(error.to_string()))
    }

    /// Parses the body as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        Ok(serde_json::from_slice(&self.body)?)
    }
}

/// A response with the body still on the wire.
pub(crate) struct StreamingResponse {
    inner: ClientStreamingResponse<ClientIo>,
}

impl StreamingResponse {
    pub(crate) fn status(&self) -> u16 {
        self.inner.head.status
    }

    pub(crate) fn into_body(self) -> IncomingBody {
        IncomingBody {
            inner: self.inner.body,
        }
    }

    /// Drains the streaming body into memory (small non-2xx error bodies).
    pub(crate) async fn read_to_end(self) -> Result<Vec<u8>> {
        let mut body = self.into_body();
        let mut collected = Vec::new();
        while let Some(chunk) = body.next_chunk().await? {
            collected.extend_from_slice(&chunk);
        }
        Ok(collected)
    }
}

/// A streaming response body yielding raw byte chunks.
pub(crate) struct IncomingBody {
    inner: ClientIncomingBody<ClientIo>,
}

impl IncomingBody {
    /// The next data chunk, or `None` at end of body.
    pub(crate) async fn next_chunk(&mut self) -> Result<Option<Vec<u8>>> {
        poll_fn(|cx| Pin::new(&mut *self).poll_chunk(cx)).await
    }

    /// Poll-level chunk read, used by the SSE event stream.
    pub(crate) fn poll_chunk(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<Option<Vec<u8>>>> {
        use std::task::Poll;
        loop {
            match Pin::new(&mut self.inner).poll_frame(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(Ok(None)),
                Poll::Ready(Some(Err(error))) => {
                    return Poll::Ready(Err(Error::Http(error.to_string())));
                }
                Poll::Ready(Some(Ok(frame))) => {
                    match frame {
                        Frame::Data(mut cursor) => {
                            let mut bytes = Vec::with_capacity(cursor.remaining());
                            while cursor.has_remaining() {
                                let chunk = cursor.chunk();
                                bytes.extend_from_slice(chunk);
                                let advanced = chunk.len();
                                cursor.advance(advanced);
                            }
                            if bytes.is_empty() {
                                continue;
                            }
                            return Poll::Ready(Ok(Some(bytes)));
                        }
                        // Trailers carry nothing we consume.
                        Frame::Trailers(_) => continue,
                    }
                }
            }
        }
    }
}
