//! Generic iterator adapters used to compose the streaming data pipeline.
//!
//! * [`IterWatcher`](crate::iterators::IterWatcher) runs a side-effecting
//!   callback for each yielded item, used to wire up shared atomic counters for
//!   [`EpochStats`](crate::dataloader::EpochStats).
//! * [`ShuffleIter`](crate::iterators::ShuffleIter) performs a bounded
//!   reservoir shuffle over an inner iterator, decoupling on-disk locality from
//!   training-step ordering.

mod iter_watcher;
mod shuffle_iter;

pub use iter_watcher::*;
pub use shuffle_iter::*;
