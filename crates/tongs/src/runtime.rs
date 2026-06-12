//! Runtime bootstrap for binaries, examples, and async tests.
//!
//! Mirrors the engine-runtime pattern shared by our projects: a
//! single-threaded skein runtime with the I/O reactor attached and a small
//! blocking pool, plus a [`block_on`] that runs the future **as a spawned
//! task** so the body has an ambient `Cx` (skein's sockets and timers need
//! one; a bare `Runtime::block_on` future has none).

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use skein::runtime::reactor::create_reactor;
use skein::runtime::{Runtime, RuntimeBuilder, RuntimeHandle};

/// A skein runtime configured for tongs: reactor attached, small blocking
/// pool for filesystem helpers.
pub struct TongsRuntime {
    runtime: Runtime,
}

impl TongsRuntime {
    pub fn handle(&self) -> RuntimeHandle {
        self.runtime.handle()
    }

    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }
}

/// Builds the runtime (single-threaded, libuv-shaped).
pub fn build_runtime() -> Result<TongsRuntime, String> {
    let reactor =
        create_reactor().map_err(|error| format!("creating skein reactor failed: {error}"))?;
    let runtime = RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .with_reactor(reactor)
        .build()
        .map_err(|error| format!("building skein runtime failed: {error}"))?;
    Ok(TongsRuntime { runtime })
}

/// Builds a runtime and runs one future to completion as a task with an
/// ambient `Cx`. Panics from the future are propagated.
pub fn block_on<F>(future: F) -> F::Output
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let runtime = build_runtime().expect("build skein runtime");
    block_on_runtime(&runtime, future)
}

/// [`block_on`] on an already-built runtime.
pub fn block_on_runtime<F>(runtime: &TongsRuntime, future: F) -> F::Output
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let (sender, receiver) = result_slot();
    runtime.handle().spawn_with_cx(move |_cx| async move {
        let mut future = Box::pin(future);
        let outcome = std::future::poll_fn(move |task_cx| {
            let poll = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                future.as_mut().poll(task_cx)
            }));
            match poll {
                Ok(Poll::Ready(value)) => Poll::Ready(Ok(value)),
                Ok(Poll::Pending) => Poll::Pending,
                Err(payload) => Poll::Ready(Err(payload)),
            }
        })
        .await;
        sender.send(outcome);
    });
    match runtime.block_on(receiver.recv()) {
        Some(Ok(value)) => value,
        Some(Err(payload)) => std::panic::resume_unwind(payload),
        None => panic!("tongs task vanished without a result"),
    }
}

/// The current time on the clock that fires timers: the ambient timer-driver
/// clock inside a task, the process wall clock otherwise (the same base
/// driverless sleeps are checked against).
pub fn engine_now() -> skein::types::Time {
    skein::cx::Cx::current().map_or_else(skein::time::wall_now, |cx| {
        cx.timer_driver()
            .map_or_else(skein::time::wall_now, |driver| driver.now())
    })
}

/// A minimal runtime-agnostic oneshot (plain mutex + waker), so the result
/// can be awaited from the raw `Runtime::block_on` context which has no `Cx`.
fn result_slot<T>() -> (SlotSender<T>, SlotReceiver<T>) {
    let shared = Arc::new(Mutex::new(Slot {
        value: None,
        waker: None,
        closed: false,
    }));
    (
        SlotSender {
            shared: Arc::clone(&shared),
        },
        SlotReceiver { shared },
    )
}

struct Slot<T> {
    value: Option<T>,
    waker: Option<Waker>,
    closed: bool,
}

struct SlotSender<T> {
    shared: Arc<Mutex<Slot<T>>>,
}

impl<T> SlotSender<T> {
    fn send(self, value: T) {
        let mut slot = self.shared.lock().expect("slot lock");
        slot.value = Some(value);
        if let Some(waker) = slot.waker.take() {
            waker.wake();
        }
    }
}

impl<T> Drop for SlotSender<T> {
    fn drop(&mut self) {
        let mut slot = self.shared.lock().expect("slot lock");
        slot.closed = true;
        if let Some(waker) = slot.waker.take() {
            waker.wake();
        }
    }
}

struct SlotReceiver<T> {
    shared: Arc<Mutex<Slot<T>>>,
}

impl<T> SlotReceiver<T> {
    async fn recv(self) -> Option<T> {
        std::future::poll_fn(move |task_cx| {
            let mut slot = self.shared.lock().expect("slot lock");
            if let Some(value) = slot.value.take() {
                return Poll::Ready(Some(value));
            }
            if slot.closed {
                return Poll::Ready(None);
            }
            slot.waker = Some(task_cx.waker().clone());
            Poll::Pending
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn block_on_returns_value() {
        assert_eq!(super::block_on(async { 41 + 1 }), 42);
    }

    #[test]
    #[should_panic(expected = "boom")]
    fn block_on_propagates_panics() {
        super::block_on(async { panic!("boom") });
    }
}
