//! # Transfer observers
//!
//! A transfer is one file moving into the cache: a download today; a copy or
//! a verification pass later. Observers watch transfers without taking part
//! in them. The disk cache carries a stack of them
//! ([`BunsenDiskCacheOptions::transfer_observers`]) and tells every observer
//! about every transfer, in registration order.
//!
//! The shape is a factory plus a per-transfer handle, because a progress bar
//! has a lifetime: it is created with a length, advanced, and then finished
//! or abandoned. [`TransferObserver::begin`] hands out one
//! [`TransferProgress`] per transfer; [`TransferProgressStack`] holds the
//! handles one transfer got from every observer, and fans out to them.
//!
//! [`BunsenDiskCacheOptions::transfer_observers`]: super::BunsenDiskCacheOptions::transfer_observers

use std::{
    fmt,
    path::Path,
    sync::Arc,
};

/// One transfer, as an observer first sees it.
#[derive(Clone, Copy, Debug)]
pub struct TransferDesc<'a> {
    /// What is being read: a URL, or a path for a local copy.
    pub source: &'a str,

    /// Where it lands.
    pub dest: &'a Path,

    /// Bytes expected, when the source says.
    pub total: Option<u64>,
}

/// How a transfer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOutcome<'a> {
    /// The file is at its destination.
    Complete,

    /// The transfer stopped short; the message says why.
    Failed(&'a str),
}

/// One transfer's sink.
///
/// Every method takes `&self`: the sinks this wraps (an `indicatif` bar, a
/// `downloader` reporter) are interior-mutable, and a handle may be shared
/// across threads. An implementation that needs `&mut` owns its own lock.
pub trait TransferProgress: Send + Sync {
    /// Bytes landed so far, as an absolute count.
    fn position(
        &self,
        bytes: u64,
    );

    /// The transfer is over. Called once.
    fn finish(
        &self,
        outcome: TransferOutcome<'_>,
    );
}

/// Watches transfers; lives on [`BunsenDiskCacheOptions`].
///
/// [`BunsenDiskCacheOptions`]: super::BunsenDiskCacheOptions
pub trait TransferObserver: Send + Sync + fmt::Debug {
    /// Called as a transfer starts; returns the sink for that transfer.
    fn begin(
        &self,
        desc: &TransferDesc<'_>,
    ) -> Box<dyn TransferProgress>;
}

/// The observer stack: every observer sees every transfer, in this order.
pub type TransferObservers = Vec<Arc<dyn TransferObserver>>;

/// The handles one transfer got from every observer in a stack.
///
/// [`position`](TransferProgress::position) and
/// [`finish`](TransferProgress::finish) fan out to each handle, in the order
/// the observers were registered.
pub struct TransferProgressStack {
    handles: Vec<Box<dyn TransferProgress>>,
}

impl TransferProgressStack {
    /// Begins one transfer on every observer.
    pub fn begin(
        observers: &[Arc<dyn TransferObserver>],
        desc: &TransferDesc<'_>,
    ) -> Self {
        Self {
            handles: observers.iter().map(|o| o.begin(desc)).collect(),
        }
    }

    /// How many handles this transfer has: one per observer.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// `true` when no observer is watching.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}

impl fmt::Debug for TransferProgressStack {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("TransferProgressStack")
            .field("handles", &self.handles.len())
            .finish()
    }
}

impl TransferProgress for TransferProgressStack {
    fn position(
        &self,
        bytes: u64,
    ) {
        for handle in &self.handles {
            handle.position(bytes);
        }
    }

    fn finish(
        &self,
        outcome: TransferOutcome<'_>,
    ) {
        for handle in &self.handles {
            handle.finish(outcome);
        }
    }
}

/// Test support: an observer that records what it sees.
#[cfg(test)]
pub(crate) mod testing {
    use std::{
        path::PathBuf,
        sync::{
            Arc,
            Mutex,
            PoisonError,
        },
    };

    use super::{
        TransferDesc,
        TransferObserver,
        TransferOutcome,
        TransferProgress,
    };

    /// One thing a [`RecordingObserver`] saw.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum Event {
        /// A transfer began.
        Begin {
            source: String,
            dest: PathBuf,
            total: Option<u64>,
        },
        /// Bytes landed so far.
        Position(u64),
        /// The transfer ended: `Ok` for complete, `Err` with the message.
        Finish(Result<(), String>),
    }

    type Log = Arc<Mutex<Vec<Event>>>;

    /// Records every event, across every transfer it is told about.
    #[derive(Debug, Default)]
    pub(crate) struct RecordingObserver {
        log: Log,
    }

    impl RecordingObserver {
        /// Everything seen so far, in order.
        pub(crate) fn events(&self) -> Vec<Event> {
            self.log
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    impl TransferObserver for RecordingObserver {
        fn begin(
            &self,
            desc: &TransferDesc<'_>,
        ) -> Box<dyn TransferProgress> {
            push(
                &self.log,
                Event::Begin {
                    source: desc.source.to_string(),
                    dest: desc.dest.to_path_buf(),
                    total: desc.total,
                },
            );
            Box::new(RecordingHandle(self.log.clone()))
        }
    }

    struct RecordingHandle(Log);

    impl TransferProgress for RecordingHandle {
        fn position(
            &self,
            bytes: u64,
        ) {
            push(&self.0, Event::Position(bytes));
        }

        fn finish(
            &self,
            outcome: TransferOutcome<'_>,
        ) {
            let outcome = match outcome {
                TransferOutcome::Complete => Ok(()),
                TransferOutcome::Failed(message) => Err(message.to_string()),
            };
            push(&self.0, Event::Finish(outcome));
        }
    }

    fn push(
        log: &Log,
        event: Event,
    ) {
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::{
            Arc,
            Mutex,
        },
    };

    use super::{
        TransferDesc,
        TransferObserver,
        TransferOutcome,
        TransferProgress,
        TransferProgressStack,
        testing::{
            Event,
            RecordingObserver,
        },
    };

    /// Every observer gets its own handle, and every handle sees every
    /// event.
    #[test]
    fn test_stack_fans_out_to_every_observer() {
        let a = Arc::new(RecordingObserver::default());
        let b = Arc::new(RecordingObserver::default());
        let observers: Vec<Arc<dyn TransferObserver>> = vec![a.clone(), b.clone()];

        let desc = TransferDesc {
            source: "https://example.invalid/file.bin",
            dest: Path::new("/cache/file.bin"),
            total: Some(10),
        };
        let stack = TransferProgressStack::begin(&observers, &desc);
        assert_eq!(stack.len(), 2);
        assert!(!stack.is_empty());
        assert_eq!(format!("{stack:?}"), "TransferProgressStack { handles: 2 }");

        stack.position(3);
        stack.position(10);
        stack.finish(TransferOutcome::Complete);

        let expected = vec![
            Event::Begin {
                source: "https://example.invalid/file.bin".to_string(),
                dest: Path::new("/cache/file.bin").to_path_buf(),
                total: Some(10),
            },
            Event::Position(3),
            Event::Position(10),
            Event::Finish(Ok(())),
        ];
        assert_eq!(a.events(), expected);
        assert_eq!(b.events(), expected);
    }

    /// Observers are reached in registration order, on `begin` and on each
    /// event after it.
    #[test]
    fn test_stack_keeps_registration_order() {
        /// Writes its tag into a shared log on every call.
        #[derive(Debug)]
        struct Tagged {
            tag: &'static str,
            log: Arc<Mutex<Vec<String>>>,
        }
        struct TaggedHandle {
            tag: &'static str,
            log: Arc<Mutex<Vec<String>>>,
        }
        impl TransferObserver for Tagged {
            fn begin(
                &self,
                _: &TransferDesc<'_>,
            ) -> Box<dyn TransferProgress> {
                self.log.lock().unwrap().push(format!("{}:begin", self.tag));
                Box::new(TaggedHandle {
                    tag: self.tag,
                    log: self.log.clone(),
                })
            }
        }
        impl TransferProgress for TaggedHandle {
            fn position(
                &self,
                bytes: u64,
            ) {
                self.log
                    .lock()
                    .unwrap()
                    .push(format!("{}:position {bytes}", self.tag));
            }

            fn finish(
                &self,
                outcome: TransferOutcome<'_>,
            ) {
                self.log
                    .lock()
                    .unwrap()
                    .push(format!("{}:finish {outcome:?}", self.tag));
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let observers: Vec<Arc<dyn TransferObserver>> = vec![
            Arc::new(Tagged {
                tag: "first",
                log: log.clone(),
            }),
            Arc::new(Tagged {
                tag: "second",
                log: log.clone(),
            }),
        ];

        let desc = TransferDesc {
            source: "src",
            dest: Path::new("dest"),
            total: None,
        };
        let stack = TransferProgressStack::begin(&observers, &desc);
        stack.position(1);
        stack.finish(TransferOutcome::Failed("boom"));

        assert_eq!(
            *log.lock().unwrap(),
            vec![
                "first:begin",
                "second:begin",
                "first:position 1",
                "second:position 1",
                "first:finish Failed(\"boom\")",
                "second:finish Failed(\"boom\")",
            ]
        );
    }

    /// No observers: the stack is empty and every call is a no-op.
    #[test]
    fn test_empty_stack_is_inert() {
        let desc = TransferDesc {
            source: "src",
            dest: Path::new("dest"),
            total: None,
        };
        let stack = TransferProgressStack::begin(&[], &desc);
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        stack.position(5);
        stack.finish(TransferOutcome::Complete);
    }
}
