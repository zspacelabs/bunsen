//! `Detailed`: causes with more to say than one line.

use alloc::borrow::Cow;

/// A cause type with more to say than its one-line `Display`: a diff table, a
/// dump, a list.
///
/// [`BunsenError::from_detailed`](crate::errors::BunsenError::from_detailed)
/// copies the details into the error, where the
/// [report](crate::errors::BunsenError::report) prints them under the
/// message. The error's one-line `Display` leaves them out.
pub trait Detailed {
    /// The multi-line details, if there are any.
    fn details(&self) -> Option<Cow<'_, str>>;
}
