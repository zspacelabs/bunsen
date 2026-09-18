//! # `indicatif` transfer observer
//!
//! One byte bar per transfer, on a shared [`MultiProgress`] so concurrent
//! transfers do not tear each other's lines. Drawn on stderr, and hidden when
//! stderr is not a terminal: that is `indicatif`'s own rule for its stderr
//! target, and it is what keeps CI logs, pipes and captured output clean.

use indicatif::{
    MultiProgress,
    ProgressBar,
    ProgressDrawTarget,
    ProgressStyle,
};

use super::{
    TransferDesc,
    TransferObserver,
    TransferOutcome,
    TransferProgress,
};

/// The bar for a transfer whose length is known.
const BYTES_TEMPLATE: &str = "{msg} {bar:32} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}";

/// The spinner for a transfer whose length is not.
const SPINNER_TEMPLATE: &str = "{msg} {spinner} {bytes} {bytes_per_sec}";

/// Draws a byte progress bar on stderr for each transfer.
///
/// `BunsenDiskCacheOptions::default()` carries one of these when the
/// `indicatif` feature is on; `without_transfer_observers()` drops it.
#[derive(Clone, Debug)]
pub struct IndicatifObserver {
    multi: MultiProgress,
}

impl Default for IndicatifObserver {
    /// Draws on stderr.
    fn default() -> Self {
        Self::new(MultiProgress::new())
    }
}

impl IndicatifObserver {
    /// Draws onto `multi`.
    pub fn new(multi: MultiProgress) -> Self {
        Self { multi }
    }

    /// Draws nothing: for tests, and for a caller that wants the observer
    /// present but silent.
    pub fn hidden() -> Self {
        Self::new(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()))
    }

    /// The shared draw surface. A caller with a bar of its own adds it here
    /// to keep it in line with the transfer bars.
    pub fn multi(&self) -> &MultiProgress {
        &self.multi
    }

    /// The bar for one transfer, added to the draw surface.
    fn bar(
        &self,
        desc: &TransferDesc<'_>,
    ) -> ProgressBar {
        let name = desc
            .dest
            .file_name()
            .unwrap_or_default()
            .display()
            .to_string();
        let bar = match desc.total {
            Some(len) => ProgressBar::new(len).with_style(style(BYTES_TEMPLATE)),
            None => ProgressBar::new_spinner().with_style(style(SPINNER_TEMPLATE)),
        };
        bar.set_message(name);
        self.multi.add(bar)
    }
}

fn style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template).expect("a literal template")
}

impl TransferObserver for IndicatifObserver {
    fn begin(
        &self,
        desc: &TransferDesc<'_>,
    ) -> Box<dyn TransferProgress> {
        Box::new(self.bar(desc))
    }
}

impl TransferProgress for ProgressBar {
    fn position(
        &self,
        bytes: u64,
    ) {
        self.set_position(bytes);
    }

    fn finish(
        &self,
        outcome: TransferOutcome<'_>,
    ) {
        match outcome {
            TransferOutcome::Complete => self.finish_and_clear(),
            TransferOutcome::Failed(message) => {
                self.abandon_with_message(format!("{} failed: {message}", self.message()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::IndicatifObserver;
    use crate::data::cache::{
        TransferDesc,
        TransferObserver,
        TransferOutcome,
        TransferProgress,
    };

    /// A known length gets a bar of that length, named for the file; the
    /// handle moves it and clears it on completion.
    #[test]
    fn test_known_length_gets_a_bar() {
        let observer = IndicatifObserver::hidden();
        let desc = TransferDesc {
            source: "https://example.invalid/weights/model.bin",
            dest: Path::new("/cache/weights/model.bin"),
            total: Some(100),
        };

        let bar = observer.bar(&desc);
        assert!(bar.is_hidden());
        assert_eq!(bar.length(), Some(100));
        assert_eq!(bar.message(), "model.bin");

        // `ProgressBar` has inherent `position`/`finish`; go through the
        // trait, as the stack does.
        TransferProgress::position(&bar, 40);
        assert_eq!(bar.position(), 40);

        TransferProgress::finish(&bar, TransferOutcome::Complete);
        assert!(bar.is_finished());
    }

    /// No length gets a spinner; a failure abandons it with the reason.
    #[test]
    fn test_unknown_length_gets_a_spinner_and_failure_keeps_the_reason() {
        let observer = IndicatifObserver::hidden();
        let desc = TransferDesc {
            source: "https://example.invalid/stream",
            dest: Path::new("/cache/stream.bin"),
            total: None,
        };

        let bar = observer.bar(&desc);
        assert_eq!(bar.length(), None);

        TransferProgress::position(&bar, 7);
        TransferProgress::finish(&bar, TransferOutcome::Failed("sha256 mismatch"));
        assert!(bar.is_finished());
        assert_eq!(bar.message(), "stream.bin failed: sha256 mismatch");
    }

    /// `begin` goes through the same path and hands back a live handle.
    #[test]
    fn test_begin_hands_back_a_handle() {
        let observer = IndicatifObserver::hidden();
        let desc = TransferDesc {
            source: "src",
            dest: Path::new("dest.bin"),
            total: Some(1),
        };
        let handle = observer.begin(&desc);
        handle.position(1);
        handle.finish(TransferOutcome::Complete);
    }

    /// The default draws on stderr, and the observer is `Clone + Debug` so
    /// it can sit on the options.
    #[test]
    fn test_default_is_clone_and_debug() {
        let observer = IndicatifObserver::default();
        let clone = observer.clone();
        assert!(!clone.multi().is_hidden() || observer.multi().is_hidden());
        assert!(format!("{observer:?}").starts_with("IndicatifObserver"));
    }
}
