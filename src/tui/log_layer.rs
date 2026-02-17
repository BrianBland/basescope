use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

const MAX_LOG_LINES: usize = 100;

/// Shared ring buffer of formatted log lines for TUI display.
#[derive(Clone)]
pub struct LogBuffer {
    inner: Arc<Mutex<VecDeque<LogLine>>>,
}

pub struct LogLine {
    pub level: Level,
    pub message: String,
}

impl LogBuffer {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(MAX_LOG_LINES))),
        }
    }

    fn push(&self, line: LogLine) {
        let mut buf = self.inner.lock();
        if buf.len() >= MAX_LOG_LINES {
            buf.pop_front();
        }
        buf.push_back(line);
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Returns the most recent `n` log lines.
    pub fn recent(&self, n: usize) -> Vec<(Level, String)> {
        let buf = self.inner.lock();
        buf.iter()
            .rev()
            .take(n)
            .map(|l| (l.level, l.message.clone()))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}

/// A tracing layer that captures log events into a shared ring buffer.
pub struct TuiLogLayer {
    buffer: LogBuffer,
}

impl TuiLogLayer {
    pub fn new(buffer: LogBuffer) -> Self {
        Self { buffer }
    }
}

struct MessageVisitor {
    message: String,
    fields: Vec<(String, String)>,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            self.fields
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields
                .push((field.name().to_string(), value.to_string()));
        }
    }
}

impl<S: Subscriber> Layer<S> for TuiLogLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor {
            message: String::new(),
            fields: Vec::new(),
        };
        event.record(&mut visitor);

        let target = event.metadata().target();
        let short_target = target.rsplit("::").next().unwrap_or(target);

        let mut msg = format!("{short_target}: {}", visitor.message);
        for (k, v) in &visitor.fields {
            msg.push_str(&format!(" {k}={v}"));
        }

        self.buffer.push(LogLine {
            level: *event.metadata().level(),
            message: msg,
        });
    }
}
