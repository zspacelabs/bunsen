//! Transfer observers, `bunsen_fetch`'s: who is told as a file moves in.

/// Test support: an observer that records what it sees, loopback HTTP
/// servers, and a known digest.
#[cfg(test)]
pub(crate) use bunsen_fetch::testing;
pub use bunsen_fetch::{
    TransferDesc,
    TransferObserver,
    TransferObservers,
    TransferOutcome,
    TransferProgress,
    TransferProgressStack,
};
