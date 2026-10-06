//! `Frame` and `Remark`: what an error passed through.

use alloc::borrow::Cow;
use core::panic::Location;

use crate::errors::BunsenErrorKind;

/// One line of context on a [`BunsenError`](crate::errors::BunsenError): what
/// the code was doing when the error passed through it.
///
/// Frames are narrative. They print in the error's `Display` and in its
/// [report](crate::errors::BunsenError::report), and nothing should branch on
/// them: what went wrong is the error's kind and cause. A frame may carry
/// multi-line details, which only the report prints.
///
/// Frames are added with [`ResultContext`](crate::errors::ResultContext) or
/// [`BunsenError::context`](crate::errors::BunsenError::context), and record
/// the source location of that call.
#[derive(Clone, Debug)]
pub struct Frame {
    message: Cow<'static, str>,
    details: Option<Cow<'static, str>>,
    location: &'static Location<'static>,
}

impl Frame {
    pub(crate) fn new(
        message: Cow<'static, str>,
        details: Option<Cow<'static, str>>,
        location: &'static Location<'static>,
    ) -> Self {
        Self {
            message,
            details,
            location,
        }
    }

    /// The one-line message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The multi-line details, if any.
    pub fn details(&self) -> Option<&str> {
        self.details.as_deref()
    }

    /// Where the frame was added.
    pub fn location(&self) -> &'static Location<'static> {
        self.location
    }
}

/// A change of an error's kind by re-marking: an
/// [`as_illegal`](crate::errors::BunsenError::as_illegal),
/// [`as_policy`](crate::errors::BunsenError::as_policy) or
/// [`as_internal`](crate::errors::BunsenError::as_internal) that changed it.
///
/// The report prints remarks; `Display` does not.
#[derive(Clone, Copy, Debug)]
pub struct Remark {
    from: BunsenErrorKind,
    to: BunsenErrorKind,
    location: &'static Location<'static>,
}

impl Remark {
    pub(crate) fn new(
        from: BunsenErrorKind,
        to: BunsenErrorKind,
        location: &'static Location<'static>,
    ) -> Self {
        Self { from, to, location }
    }

    /// The kind before the re-mark.
    pub fn from(&self) -> BunsenErrorKind {
        self.from
    }

    /// The kind after the re-mark.
    pub fn to(&self) -> BunsenErrorKind {
        self.to
    }

    /// Where the re-mark happened.
    pub fn location(&self) -> &'static Location<'static> {
        self.location
    }
}
