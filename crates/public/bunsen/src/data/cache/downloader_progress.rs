//! Bridges the `downloader` crate's progress reporter onto the observer
//! stack.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
        MutexGuard,
        PoisonError,
    },
};

use downloader::progress::Reporter;

use super::{
    TransferDesc,
    TransferObserver,
    TransferObservers,
    TransferOutcome,
    TransferProgress,
    TransferProgressStack,
};

/// A `downloader` reporter that opens one transfer on the observer stack
/// per attempt.
///
/// `downloader` calls `setup` at the start of every attempt (it retries a
/// failed URL), `progress` per chunk, `set_message` with the HTTP status
/// once the body is through, and `done` only after a successful download.
/// So a second `setup` finishes the open transfer as failed with the last
/// status message, and the caller finishes a transfer that never saw `done`
/// through [`fail_open`](Self::fail_open) once `download` returns.
pub(crate) struct DownloaderReporter {
    observers: TransferObservers,
    source: String,
    dest: PathBuf,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// The transfer of the attempt in flight, if any.
    open: Option<TransferProgressStack>,

    /// The last message `downloader` set for it: `<file> <try>/<tries> -
    /// <status>` after the body.
    last_message: Option<String>,
}

impl DownloaderReporter {
    /// Reports transfers of `source` (the URLs, for the observers) landing
    /// at `dest` to `observers`.
    pub(crate) fn new(
        observers: &[Arc<dyn TransferObserver>],
        source: String,
        dest: PathBuf,
    ) -> Self {
        Self {
            observers: observers.to_vec(),
            source,
            dest,
            state: Mutex::new(State::default()),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Finishes the open transfer, if any, with `outcome`.
    pub(crate) fn finish(
        &self,
        outcome: TransferOutcome<'_>,
    ) {
        if let Some(open) = self.state().open.take() {
            open.finish(outcome);
        }
    }

    /// Finishes the open transfer, if any, as failed: with the last message
    /// `downloader` set for it, or `fallback` when it set none.
    pub(crate) fn fail_open(
        &self,
        fallback: &str,
    ) {
        let mut state = self.state();
        if let Some(open) = state.open.take() {
            let message = state.last_message.as_deref().unwrap_or(fallback);
            open.finish(TransferOutcome::Failed(message));
        }
    }
}

impl Reporter for DownloaderReporter {
    fn setup(
        &self,
        max_progress: Option<u64>,
        _message: &str,
    ) {
        self.fail_open("retried");
        let open = TransferProgressStack::begin(
            &self.observers,
            &TransferDesc {
                source: &self.source,
                dest: &self.dest,
                total: max_progress,
            },
        );
        let mut state = self.state();
        state.open = Some(open);
        state.last_message = None;
    }

    fn progress(
        &self,
        current: u64,
    ) {
        if let Some(open) = &self.state().open {
            open.position(current);
        }
    }

    fn set_message(
        &self,
        message: &str,
    ) {
        self.state().last_message = Some(message.to_string());
    }

    fn done(&self) {
        self.finish(TransferOutcome::Complete);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::Arc,
    };

    use downloader::progress::Reporter;

    use super::*;
    use crate::data::cache::testing::{
        CacheProgressEvent,
        RecordingObserver,
    };

    fn reporter(observer: &Arc<RecordingObserver>) -> DownloaderReporter {
        let observers: Vec<Arc<dyn TransferObserver>> = vec![observer.clone()];
        DownloaderReporter::new(
            &observers,
            "https://example.invalid/file.bin".to_string(),
            PathBuf::from("/cache/file.bin"),
        )
    }

    fn begin(total: Option<u64>) -> CacheProgressEvent {
        CacheProgressEvent::Begin {
            source: "https://example.invalid/file.bin".to_string(),
            dest: PathBuf::from("/cache/file.bin"),
            total,
        }
    }

    /// One attempt: `setup` opens the transfer, `progress` advances it, and
    /// `done` completes it. A second `done` is a no-op.
    #[test]
    fn test_one_attempt_completes_on_done() {
        let observer = Arc::new(RecordingObserver::default());
        let reporter = reporter(&observer);

        reporter.setup(Some(10), "file.bin 1/3");
        reporter.progress(4);
        reporter.progress(10);
        reporter.set_message("file.bin 1/3 - 200");
        reporter.done();
        reporter.done();

        assert_eq!(
            observer.events(),
            vec![
                begin(Some(10)),
                CacheProgressEvent::Position(4),
                CacheProgressEvent::Position(10),
                CacheProgressEvent::Finish(Ok(())),
            ]
        );
    }

    /// A retry: the second `setup` fails the first transfer with the status
    /// `downloader` reported for it, then opens a fresh one.
    #[test]
    fn test_retry_fails_the_open_transfer_with_its_status() {
        let observer = Arc::new(RecordingObserver::default());
        let reporter = reporter(&observer);

        reporter.setup(None, "file.bin 1/3");
        reporter.progress(1);
        reporter.set_message("file.bin 1/3 - 404");
        reporter.setup(Some(5), "file.bin 2/3");
        reporter.progress(5);
        reporter.done();

        assert_eq!(
            observer.events(),
            vec![
                begin(None),
                CacheProgressEvent::Position(1),
                CacheProgressEvent::Finish(Err("file.bin 1/3 - 404".to_string())),
                begin(Some(5)),
                CacheProgressEvent::Position(5),
                CacheProgressEvent::Finish(Ok(())),
            ]
        );
    }

    /// A download that never reached `done` is failed by the caller, with
    /// the fallback when `downloader` set no message; afterwards there is
    /// nothing open for `fail_open` or `finish` to act on.
    #[test]
    fn test_fail_open_uses_the_fallback_without_a_message() {
        let observer = Arc::new(RecordingObserver::default());
        let reporter = reporter(&observer);

        reporter.fail_open("nothing to fail");
        assert!(observer.events().is_empty());

        reporter.setup(Some(3), "file.bin 1/3");
        reporter.fail_open("gave up");
        reporter.fail_open("again");
        reporter.finish(TransferOutcome::Complete);

        assert_eq!(
            observer.events(),
            vec![
                begin(Some(3)),
                CacheProgressEvent::Finish(Err("gave up".to_string())),
            ]
        );
    }
}
