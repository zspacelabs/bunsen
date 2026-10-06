//! `sys_at` and `sys_op`: sorting `io::Error`s into kinds.

use alloc::{
    format,
    sync::Arc,
};
use core::panic::Location;
use std::{
    io,
    path::{
        Path,
        PathBuf,
    },
};

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    LookupError,
    LookupProblem,
};

/// The kind an `io::Error` sorts into, by its `io::ErrorKind`.
///
/// - `NotFound`, `PermissionDenied`: [`Lookup`](BunsenErrorKind::Lookup);
/// - `TimedOut`, `ConnectionReset`, `ConnectionAborted`, `ConnectionRefused`,
///   `Interrupted`, `WouldBlock`:
///   [`Unavailable`](BunsenErrorKind::Unavailable);
/// - `InvalidData`, `UnexpectedEof`:
///   [`InvalidResource`](BunsenErrorKind::InvalidResource);
/// - anything else: [`Sys`](BunsenErrorKind::Sys).
pub fn io_error_kind(error: &io::Error) -> BunsenErrorKind {
    match error.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied => BunsenErrorKind::Lookup,
        io::ErrorKind::TimedOut
        | io::ErrorKind::ConnectionReset
        | io::ErrorKind::ConnectionAborted
        | io::ErrorKind::ConnectionRefused
        | io::ErrorKind::Interrupted
        | io::ErrorKind::WouldBlock => BunsenErrorKind::Unavailable,
        io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof => {
            BunsenErrorKind::InvalidResource
        }
        _ => BunsenErrorKind::Sys,
    }
}

fn sort_io_error(
    op: &str,
    path: Option<&Path>,
    error: io::Error,
    location: &'static Location<'static>,
) -> BunsenError {
    let problem = match error.kind() {
        io::ErrorKind::NotFound => Some(LookupProblem::Missing),
        io::ErrorKind::PermissionDenied => Some(LookupProblem::Denied),
        _ => None,
    };
    match (problem, path) {
        (Some(problem), Some(path)) => {
            let lookup = LookupError::path(path, problem).with_source(error);
            let message = lookup.to_string();
            BunsenError::build(
                BunsenErrorKind::Lookup,
                message.into(),
                Some(Arc::new(lookup)),
                true,
                location,
            )
        }
        _ => {
            let kind = io_error_kind(&error);
            let message = match path {
                Some(path) => format!("{op} {}: {error}", path.display()),
                None => format!("{op}: {error}"),
            };
            // The message shows the io::Error's text; the error is kept as
            // the cause for `find`, and not printed again.
            BunsenError::build(kind, message.into(), Some(Arc::new(error)), true, location)
        }
    }
}

/// Sorts an `io::Error` from operation `op` on `path` into a [`BunsenError`]
/// of the [kind it means](io_error_kind), keeping the `io::Error` as the cause
/// and naming the path. Shaped for `map_err`:
///
/// ```
/// use bunsen::errors::{
///     BunsenErrorKind,
///     BunsenResult,
///     sys_at,
/// };
///
/// fn read(path: &str) -> BunsenResult<String> {
///     std::fs::read_to_string(path).map_err(sys_at("read", path))
/// }
///
/// let e = read("/no/such/file").unwrap_err();
/// assert_eq!(e.kind(), BunsenErrorKind::Lookup);
/// assert_eq!(e.to_string(), "/no/such/file: not found");
/// ```
///
/// A missing or forbidden path is a [`Lookup`](BunsenErrorKind::Lookup) whose
/// cause is a [`LookupError`] for the path, wrapping the `io::Error`. Anything
/// else names the operation and the path: `"write /tmp/x: Read-only file
/// system (os error 30)"`.
///
/// There is no `From<io::Error> for BunsenError`: a bare `?` could not say
/// which path failed.
#[track_caller]
pub fn sys_at(
    op: &'static str,
    path: impl AsRef<Path>,
) -> impl FnOnce(io::Error) -> BunsenError {
    let location = Location::caller();
    let path: PathBuf = path.as_ref().to_path_buf();
    move |error| sort_io_error(op, Some(&path), error, location)
}

/// [`sys_at`] for an `io::Error` with no path: a thread spawn, a socket, a
/// device. The message names the operation.
#[track_caller]
pub fn sys_op(op: &'static str) -> impl FnOnce(io::Error) -> BunsenError {
    let location = Location::caller();
    move |error| sort_io_error(op, None, error, location)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorting() {
        let missing = io::Error::new(io::ErrorKind::NotFound, "gone");
        let e = sys_at("open", "/a/b")(missing);
        assert_eq!(e.kind(), BunsenErrorKind::Lookup);
        assert_eq!(e.to_string(), "/a/b: not found");
        assert_eq!(
            e.find::<LookupError>().unwrap().problem,
            LookupProblem::Missing
        );
        assert!(e.find::<io::Error>().is_some());

        let full = io::Error::new(io::ErrorKind::StorageFull, "disk full");
        let e = sys_at("write", "/a/b")(full);
        assert_eq!(e.kind(), BunsenErrorKind::Sys);
        assert_eq!(e.to_string(), "write /a/b: disk full");

        let timeout = io::Error::new(io::ErrorKind::TimedOut, "slow");
        let e = sys_op("connect")(timeout);
        assert_eq!(e.kind(), BunsenErrorKind::Unavailable);
        assert_eq!(e.to_string(), "connect: slow");
    }
}
