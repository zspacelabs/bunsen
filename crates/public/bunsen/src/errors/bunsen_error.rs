//! `BunsenError` and `BunsenResult`.

use alloc::{
    borrow::Cow,
    string::ToString,
    sync::Arc,
    vec::Vec,
};
use core::{
    error::Error,
    fmt,
    panic::Location,
};
use std::backtrace::Backtrace;

use crate::errors::{
    BunsenErrorKind,
    Detailed,
    Frame,
    LookupError,
    Remark,
    Report,
};

/// The result of a fallible bunsen operation: the `try_x` half of the
/// [convention](crate::errors#convention-try_x-and-x).
pub type BunsenResult<T> = core::result::Result<T, BunsenError>;

/// A shared, type-erased cause.
pub(crate) type DynCause = Arc<dyn Error + Send + Sync + 'static>;

/// The crate-wide error.
///
/// A `BunsenError` is a [kind](BunsenErrorKind), a one-line message, optional
/// multi-line [details](Self::details), the [frames](Self::frames) of context
/// it gathered on its way up, and an optional cause. The
/// [module docs](crate::errors) say which kind fits which failure, how to
/// build one, and how to handle one.
///
/// - `Display` (`{}`) is one line: the frames, outermost first, then the
///   message, joined by `": "`.
/// - `{:#}`, `Debug` and [`report`](Self::report) print the full report: the
///   kind, the message and where it was built, the details, the re-marks, each
///   frame with its location and details, the cause chain, and, for a bug, the
///   backtrace.
///
/// Cloning is cheap: the error is shared behind an `Arc`, and adding a frame to
/// a shared error copies only its frame list.
#[derive(Clone)]
pub struct BunsenError(Arc<Inner>);

#[derive(Clone)]
struct Inner {
    kind: BunsenErrorKind,
    message: Cow<'static, str>,
    details: Option<Cow<'static, str>>,
    location: &'static Location<'static>,
    context: Vec<Frame>,
    cause: Option<DynCause>,
    /// The message already says what the cause says: set when the message is
    /// built from the cause, so the cause is not printed twice.
    message_is_cause: bool,
    backtrace: Option<Arc<Backtrace>>,
    remarks: Vec<Remark>,
}

impl BunsenError {
    /// An error of `kind` with a one-line `message`.
    ///
    /// The per-kind constructors ([`illegal`](Self::illegal),
    /// [`policy`](Self::policy), ...) are shorthand for this.
    #[track_caller]
    pub fn new(
        kind: BunsenErrorKind,
        message: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::build(kind, message.into(), None, false, Location::caller())
    }

    pub(crate) fn build(
        kind: BunsenErrorKind,
        message: Cow<'static, str>,
        cause: Option<DynCause>,
        message_is_cause: bool,
        location: &'static Location<'static>,
    ) -> Self {
        let backtrace = kind.is_bug().then(|| Arc::new(Backtrace::capture()));
        Self(Arc::new(Inner {
            kind,
            message,
            details: None,
            location,
            context: Vec::new(),
            cause,
            message_is_cause,
            backtrace,
            remarks: Vec::new(),
        }))
    }

    /// An [`Illegal`](BunsenErrorKind::Illegal) error: the caller broke a rule
    /// of the API.
    #[track_caller]
    pub fn illegal(message: impl Into<Cow<'static, str>>) -> Self {
        Self::build(
            BunsenErrorKind::Illegal,
            message.into(),
            None,
            false,
            Location::caller(),
        )
    }

    /// An [`Internal`](BunsenErrorKind::Internal) error: one of bunsen's own
    /// invariants broke.
    #[track_caller]
    pub fn internal(message: impl Into<Cow<'static, str>>) -> Self {
        Self::build(
            BunsenErrorKind::Internal,
            message.into(),
            None,
            false,
            Location::caller(),
        )
    }

    /// A [`Policy`](BunsenErrorKind::Policy) error: the operation is
    /// misconfigured, and bunsen refuses to proceed.
    #[track_caller]
    pub fn policy(message: impl Into<Cow<'static, str>>) -> Self {
        Self::build(
            BunsenErrorKind::Policy,
            message.into(),
            None,
            false,
            Location::caller(),
        )
    }

    /// A [`Lookup`](BunsenErrorKind::Lookup) error from a [`LookupError`],
    /// whose `Display` is the message.
    #[track_caller]
    pub fn lookup(error: LookupError) -> Self {
        Self::from_cause(BunsenErrorKind::Lookup, error)
    }

    /// An [`Unavailable`](BunsenErrorKind::Unavailable) error: not reachable
    /// now.
    #[track_caller]
    pub fn unavailable(message: impl Into<Cow<'static, str>>) -> Self {
        Self::build(
            BunsenErrorKind::Unavailable,
            message.into(),
            None,
            false,
            Location::caller(),
        )
    }

    /// An [`InvalidResource`](BunsenErrorKind::InvalidResource) error:
    /// something from outside the program is not what was expected.
    #[track_caller]
    pub fn invalid_resource(message: impl Into<Cow<'static, str>>) -> Self {
        Self::build(
            BunsenErrorKind::InvalidResource,
            message.into(),
            None,
            false,
            Location::caller(),
        )
    }

    /// An [`Unsupported`](BunsenErrorKind::Unsupported) error: a valid
    /// request this code, or this build, does not do.
    #[track_caller]
    pub fn unsupported(message: impl Into<Cow<'static, str>>) -> Self {
        Self::build(
            BunsenErrorKind::Unsupported,
            message.into(),
            None,
            false,
            Location::caller(),
        )
    }

    /// A [`Sys`](BunsenErrorKind::Sys) error: the host failed.
    ///
    /// For an `io::Error`, [`sys_at`](crate::errors::sys_at) and
    /// [`sys_op`](crate::errors::sys_op) sort it by its kind instead.
    #[track_caller]
    pub fn sys(message: impl Into<Cow<'static, str>>) -> Self {
        Self::build(
            BunsenErrorKind::Sys,
            message.into(),
            None,
            false,
            Location::caller(),
        )
    }

    /// An [`Other`](BunsenErrorKind::Other) error wrapping a foreign error
    /// that has no better kind; its `Display` is the message.
    ///
    /// Shaped for `map_err`: `result.map_err(BunsenError::other)`.
    #[track_caller]
    pub fn other<E>(error: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::from_cause(BunsenErrorKind::Other, error)
    }

    /// An error of `kind` built from `cause`: the cause's `Display` is the
    /// message.
    ///
    /// The cause stays reachable with [`find`](Self::find). Since the
    /// message already says what the cause says,
    /// [`source`](Error::source) skips the cause and returns the cause's own
    /// source, and the report prints the cause's text once.
    #[track_caller]
    pub fn from_cause<E>(
        kind: BunsenErrorKind,
        cause: E,
    ) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        let message = cause.to_string();
        Self::build(
            kind,
            message.into(),
            Some(Arc::new(cause)),
            true,
            Location::caller(),
        )
    }

    /// [`from_cause`](Self::from_cause) for a cause that implements
    /// [`Detailed`]: its details become the error's details.
    #[track_caller]
    pub fn from_detailed<C>(
        kind: BunsenErrorKind,
        cause: C,
    ) -> Self
    where
        C: Error + Detailed + Send + Sync + 'static,
    {
        let details = cause.details().map(|d| Cow::Owned(d.into_owned()));
        let mut error = Self::from_cause(kind, cause);
        error.inner_mut().details = details;
        error
    }

    /// Attaches `cause`, replacing any cause the error had.
    ///
    /// The message stays the error's own, so the report prints the cause
    /// under it as "caused by".
    pub fn with_cause<E>(
        mut self,
        cause: E,
    ) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        let inner = self.inner_mut();
        inner.cause = Some(Arc::new(cause));
        inner.message_is_cause = false;
        self
    }

    /// Sets the multi-line details, which only the report prints.
    pub fn with_details(
        mut self,
        details: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.inner_mut().details = Some(details.into());
        self
    }

    /// Adds a frame of context: what the code was doing.
    ///
    /// [`ResultContext::context`](crate::errors::ResultContext::context) is
    /// the form for a `BunsenResult`.
    #[track_caller]
    pub fn context(
        mut self,
        frame: impl fmt::Display,
    ) -> Self {
        let frame = Frame::new(frame.to_string().into(), None, Location::caller());
        self.inner_mut().context.push(frame);
        self
    }

    /// Adds a frame of context that carries multi-line details.
    #[track_caller]
    pub fn context_details(
        mut self,
        frame: impl fmt::Display,
        details: impl fmt::Display,
    ) -> Self {
        let frame = Frame::new(
            frame.to_string().into(),
            Some(details.to_string().into()),
            Location::caller(),
        );
        self.inner_mut().context.push(frame);
        self
    }

    /// Re-marks a runtime error as [`Illegal`](BunsenErrorKind::Illegal): a
    /// higher level declares that the call should never have happened.
    ///
    /// Every kind but [`Internal`](BunsenErrorKind::Internal) changes; an
    /// `Illegal` error passes through unchanged.
    #[track_caller]
    // `as_*` by value: re-marking consumes the error and returns it, like a
    // builder; the name says what it becomes.
    #[allow(clippy::wrong_self_convention)]
    pub fn as_illegal(self) -> Self {
        match self.kind() {
            BunsenErrorKind::Illegal | BunsenErrorKind::Internal => self,
            _ => self.remark(BunsenErrorKind::Illegal, Location::caller()),
        }
    }

    /// Re-marks an [`Illegal`](BunsenErrorKind::Illegal) error as
    /// [`Policy`](BunsenErrorKind::Policy), at an input boundary: the values
    /// that broke the rule came from outside the program.
    ///
    /// Every other kind passes through unchanged.
    #[track_caller]
    // `as_*` by value: re-marking consumes the error and returns it, like a
    // builder; the name says what it becomes.
    #[allow(clippy::wrong_self_convention)]
    pub fn as_policy(self) -> Self {
        match self.kind() {
            BunsenErrorKind::Illegal => self.remark(BunsenErrorKind::Policy, Location::caller()),
            _ => self,
        }
    }

    /// Re-marks an [`Illegal`](BunsenErrorKind::Illegal) error as
    /// [`Internal`](BunsenErrorKind::Internal), where bunsen passed values it
    /// built itself.
    ///
    /// Every other kind passes through unchanged.
    #[track_caller]
    // `as_*` by value: re-marking consumes the error and returns it, like a
    // builder; the name says what it becomes.
    #[allow(clippy::wrong_self_convention)]
    pub fn as_internal(self) -> Self {
        match self.kind() {
            BunsenErrorKind::Illegal => self.remark(BunsenErrorKind::Internal, Location::caller()),
            _ => self,
        }
    }

    fn remark(
        mut self,
        to: BunsenErrorKind,
        location: &'static Location<'static>,
    ) -> Self {
        let inner = self.inner_mut();
        inner.remarks.push(Remark::new(inner.kind, to, location));
        inner.kind = to;
        if to.is_bug() && inner.backtrace.is_none() {
            inner.backtrace = Some(Arc::new(Backtrace::capture()));
        }
        self
    }

    fn inner_mut(&mut self) -> &mut Inner {
        Arc::make_mut(&mut self.0)
    }

    /// The kind.
    pub fn kind(&self) -> BunsenErrorKind {
        self.0.kind
    }

    /// The one-line message, without the frames.
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// The multi-line details, if any.
    pub fn details(&self) -> Option<&str> {
        self.0.details.as_deref()
    }

    /// Where the error was built.
    pub fn location(&self) -> &'static Location<'static> {
        self.0.location
    }

    /// The frames of context, innermost first.
    pub fn frames(&self) -> &[Frame] {
        &self.0.context
    }

    /// The re-marks that changed the kind, oldest first.
    pub fn remarks(&self) -> &[Remark] {
        &self.0.remarks
    }

    /// The cause, if any.
    pub fn cause(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        self.0.cause.as_deref()
    }

    /// The backtrace captured when the error became a bug
    /// ([`Illegal`](BunsenErrorKind::Illegal) or
    /// [`Internal`](BunsenErrorKind::Internal)), if any. Whether it holds
    /// frames depends on `RUST_BACKTRACE`.
    pub fn backtrace(&self) -> Option<&Backtrace> {
        self.0.backtrace.as_deref()
    }

    /// The first error of type `E` in the cause chain: the cause, then the
    /// cause's source, and so on.
    ///
    /// This is the way to reach a typed cause. It starts at the stored cause,
    /// so it finds a cause whose `Display` is the message (see
    /// [`from_cause`](Self::from_cause)), which [`source`](Error::source)
    /// skips. A downcast through the standard source chain, such as
    /// `anyhow::Error::downcast_ref`, does not see that cause.
    ///
    /// `find` matches on type identity: a cause type from a second,
    /// semver-incompatible copy of bunsen in the same build is a different
    /// type.
    pub fn find<E>(&self) -> Option<&E>
    where
        E: Error + 'static,
    {
        let mut current: Option<&(dyn Error + 'static)> =
            self.0.cause.as_deref().map(|c| c as &(dyn Error + 'static));
        while let Some(error) = current {
            if let Some(found) = error.downcast_ref::<E>() {
                return Some(found);
            }
            current = error.source();
        }
        None
    }

    /// The full report, for printing: the kind, the message and where it was
    /// built, the details, the re-marks, the frames, the cause chain, and,
    /// for a bug, the backtrace.
    ///
    /// `{:#}` and `Debug` print the same.
    pub fn report(&self) -> Report<'_> {
        Report::new(self)
    }
}

impl Error for BunsenError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        let cause = self.0.cause.as_deref()?;
        if self.0.message_is_cause {
            cause.source()
        } else {
            Some(cause)
        }
    }
}

impl fmt::Display for BunsenError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if f.alternate() {
            return fmt::Display::fmt(&self.report(), f);
        }
        for frame in self.0.context.iter().rev() {
            write!(f, "{}: ", frame.message())?;
        }
        f.write_str(&self.0.message)
    }
}

impl fmt::Debug for BunsenError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Display::fmt(&self.report(), f)
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use std::io;

    use super::*;
    use crate::errors::{
        LookupProblem,
        ResultContext,
    };

    #[test]
    fn test_display_is_frames_then_message() {
        let e = BunsenError::policy("downloads are off")
            .context("fetching shard 3")
            .context("loading set \"large\"");
        assert_eq!(e.kind(), BunsenErrorKind::Policy);
        assert_eq!(e.message(), "downloads are off");
        assert_eq!(
            e.to_string(),
            "loading set \"large\": fetching shard 3: downloads are off"
        );
        assert_eq!(e.frames().len(), 2);
        assert_eq!(e.frames()[0].message(), "fetching shard 3");
    }

    #[test]
    fn test_location_is_the_caller() {
        let line = line!() + 1;
        let e = BunsenError::illegal("x");
        assert_eq!(e.location().line(), line);
        assert!(e.location().file().ends_with("bunsen_error.rs"));
    }

    #[test]
    fn test_from_cause_prints_the_cause_once() {
        let io = io::Error::new(io::ErrorKind::Other, "disk on fire");
        let e = BunsenError::from_cause(BunsenErrorKind::Sys, io);
        assert_eq!(e.to_string(), "disk on fire");
        assert!(e.0.message_is_cause);
        assert!(e.source().is_none());
        assert!(e.find::<io::Error>().is_some());

        let report = format!("{e:#}");
        assert_eq!(report.matches("disk on fire").count(), 1, "{report}");
    }

    #[test]
    fn test_with_cause_is_caused_by() {
        let io = io::Error::new(io::ErrorKind::Other, "disk on fire");
        let e = BunsenError::sys("writing the cache").with_cause(io);
        assert_eq!(e.to_string(), "writing the cache");
        assert!(!e.0.message_is_cause);
        assert!(e.source().is_some());
        let report = format!("{e:#}");
        assert!(report.contains("caused by: disk on fire"), "{report}");
    }

    #[test]
    fn test_find_walks_the_chain() {
        let io = io::Error::new(io::ErrorKind::NotFound, "no such file");
        let lookup = LookupError::path("/x/y", LookupProblem::Missing).with_source(io);
        let e = BunsenError::lookup(lookup).context("reading the vocab");
        assert_eq!(e.kind(), BunsenErrorKind::Lookup);
        assert_eq!(
            e.find::<LookupError>().unwrap().problem,
            LookupProblem::Missing
        );
        assert_eq!(
            e.find::<io::Error>().unwrap().kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    fn test_remarks() {
        let e = BunsenError::lookup(LookupError::missing("column", "x"));
        let e = e.as_policy();
        assert_eq!(e.kind(), BunsenErrorKind::Lookup);
        assert!(e.remarks().is_empty());

        let e = e.as_illegal();
        assert_eq!(e.kind(), BunsenErrorKind::Illegal);
        assert_eq!(e.remarks().len(), 1);
        assert_eq!(e.remarks()[0].from(), BunsenErrorKind::Lookup);
        assert!(e.backtrace().is_some());

        let e = e.as_policy();
        assert_eq!(e.kind(), BunsenErrorKind::Policy);
        assert_eq!(e.remarks().len(), 2);

        let internal = BunsenError::internal("x").as_illegal().as_policy();
        assert_eq!(internal.kind(), BunsenErrorKind::Internal);
    }

    #[test]
    fn test_clone_then_context_does_not_touch_the_original() {
        let a = BunsenError::unavailable("timed out");
        let b = a.clone().context("pumping");
        assert!(a.frames().is_empty());
        assert_eq!(b.frames().len(), 1);
    }

    #[test]
    fn test_result_context() {
        let r: BunsenResult<()> = Err(BunsenError::illegal("stride is zero"));
        let e = r
            .context("building conv")
            .context_details(|| ("block 3", "stride: 0\nkernel: 3"))
            .as_policy()
            .unwrap_err();
        assert_eq!(e.kind(), BunsenErrorKind::Policy);
        assert_eq!(e.to_string(), "block 3: building conv: stride is zero");
        assert_eq!(e.frames()[1].details(), Some("stride: 0\nkernel: 3"));
    }

    #[test]
    fn test_error_bounds() {
        fn assert_bounds<T: Error + Send + Sync + Clone + 'static>() {}
        assert_bounds::<BunsenError>();
    }
}
