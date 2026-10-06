//! `Report`: the full, multi-line form of a `BunsenError`.

use core::{
    error::Error,
    fmt,
};
use std::backtrace::BacktraceStatus;

use crate::errors::BunsenError;

/// The full, multi-line form of a [`BunsenError`], from
/// [`BunsenError::report`]; `{:#}` and `Debug` on the error print the same.
///
/// ```text
/// error[InvalidResource]: 51864 is not a Whisper vocabulary size
///   at src/kits/speech/whisper/driver/whisper_token_layout.rs:228:13
///   | <the error's details>
///   re-marked Illegal -> Policy at src/data/pretrained/deferred.rs:122:9
///   while token layout (src/kits/speech/whisper/pretrained/construct.rs:308:17)
///   | <the frame's details>
///   while loading whisper "openai/base" (src/data/pretrained/deferred.rs:122:9)
/// caused by: <the source chain, one line each>
/// backtrace:
/// <for Illegal and Internal, when RUST_BACKTRACE captured one>
/// ```
///
/// The frames read innermost first: each `while` line is one layer further
/// out. The report is a logical traceback: it names the layers that added
/// context, with their source locations, works in release builds, and
/// travels with the error across threads.
pub struct Report<'a> {
    error: &'a BunsenError,
}

impl<'a> Report<'a> {
    pub(crate) fn new(error: &'a BunsenError) -> Self {
        Self { error }
    }
}

fn write_details(
    f: &mut fmt::Formatter<'_>,
    details: &str,
) -> fmt::Result {
    for line in details.lines() {
        if line.is_empty() {
            writeln!(f)?;
            write!(f, "  |")?;
        } else {
            writeln!(f)?;
            write!(f, "  | {line}")?;
        }
    }
    Ok(())
}

impl fmt::Display for Report<'_> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let e = self.error;
        write!(f, "error[{}]: {}", e.kind(), e.message())?;
        writeln!(f)?;
        write!(f, "  at {}", e.location())?;
        if let Some(details) = e.details() {
            write_details(f, details)?;
        }
        for remark in e.remarks() {
            writeln!(f)?;
            write!(
                f,
                "  re-marked {} -> {} at {}",
                remark.from(),
                remark.to(),
                remark.location()
            )?;
        }
        for frame in e.frames() {
            writeln!(f)?;
            write!(f, "  while {} ({})", frame.message(), frame.location())?;
            if let Some(details) = frame.details() {
                write_details(f, details)?;
            }
        }
        let mut source = e.source();
        while let Some(cause) = source {
            writeln!(f)?;
            write!(f, "caused by: {cause}")?;
            source = cause.source();
        }
        if let Some(backtrace) = e.backtrace()
            && backtrace.status() == BacktraceStatus::Captured
        {
            writeln!(f)?;
            write!(f, "backtrace:\n{backtrace}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Report<'_> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;
    use crate::errors::BunsenErrorKind;

    #[test]
    fn test_report_layout() {
        let e = BunsenError::illegal("stride is zero")
            .with_details("stride: 0\nkernel: 3")
            .context("building block 3")
            .context_details("building the stem", "blocks: 4")
            .as_policy();
        let report = format!("{}", e.report());
        let lines: alloc::vec::Vec<&str> = report.lines().collect();
        assert_eq!(lines[0], "error[Policy]: stride is zero");
        assert!(lines[1].starts_with("  at "), "{report}");
        assert_eq!(lines[2], "  | stride: 0");
        assert_eq!(lines[3], "  | kernel: 3");
        assert!(
            lines[4].starts_with("  re-marked Illegal -> Policy at "),
            "{report}"
        );
        assert!(
            lines[5].starts_with("  while building block 3 ("),
            "{report}"
        );
        assert!(
            lines[6].starts_with("  while building the stem ("),
            "{report}"
        );
        assert_eq!(lines[7], "  | blocks: 4");

        assert_eq!(format!("{e:#}"), report);
        assert_eq!(format!("{e:?}"), report);
        assert_eq!(e.kind(), BunsenErrorKind::Policy);
    }
}
