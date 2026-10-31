//! The `webhook` target: each record sent as a JSON `POST`.
use std::{
    collections::BTreeMap,
    sync::mpsc,
    thread::JoinHandle,
    time::Duration,
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};

use super::{
    Cx,
    Event,
    Opts,
    Sink,
    record::json_line,
};

/// How long one POST may take.
const TIMEOUT: Duration = Duration::from_secs(10);

/// POSTs each record to a URL, `Content-Type: application/json`, with the
/// configured headers, in order, from a thread of its own: a slow endpoint
/// never holds the audio up. A failed POST is logged, and the next is
/// tried; [`finish`](Sink::finish) waits for every queued record.
pub struct WebhookSink {
    tx: mpsc::Sender<String>,
    worker: JoinHandle<()>,
}

impl WebhookSink {
    /// Starts posting to `url` with `headers`.
    pub fn open(
        url: String,
        headers: BTreeMap<String, String>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<String>();
        let worker = std::thread::spawn(move || {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_global(Some(TIMEOUT))
                .build()
                .into();
            let mut failing = false;
            for body in rx {
                let mut request = agent.post(&url).header("Content-Type", "application/json");
                for (name, value) in &headers {
                    request = request.header(name, value);
                }
                match request.send(&body) {
                    Ok(_) if failing => {
                        log::warn!("webhook {url}: delivering again");
                        failing = false;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        // Every failure at DEBUG, the first of a run at WARN.
                        if !failing {
                            log::warn!("webhook {url}: {e}");
                            failing = true;
                        }
                        log::debug!("webhook {url}: a record lost: {e}");
                    }
                }
            }
        });
        Self { tx, worker }
    }
}

impl Sink for WebhookSink {
    fn push(
        &mut self,
        event: &Event,
        cx: &Cx,
        opts: &Opts,
    ) -> BunsenResult<()> {
        // A send fails only once the worker has gone, which `finish` reports.
        let _ = self.tx.send(json_line(event, opts, cx));
        Ok(())
    }

    fn finish(self: Box<Self>) -> BunsenResult<()> {
        let Self { tx, worker } = *self;
        drop(tx);
        worker
            .join()
            .map_err(|_| BunsenError::sys("the webhook worker panicked"))
    }
}
