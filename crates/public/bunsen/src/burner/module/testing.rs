//! # Testing rekeys
//!
//! [`assert_rekeys`]: a [`Rekey`] checked against a table of names, every
//! row at once.
//!
//! This module exists with the `testing` feature, and in bunsen's own unit
//! tests.

use alloc::{
    format,
    string::String,
    vec::Vec,
};

use crate::burner::module::Rekey;

/// Asserts that `rekey` maps each `from` to its `to`, a module path, or
/// `None` for a name it drops.
///
/// The table's targets are `&str` or `Option<&str>`. Every row is checked
/// before the assertion fails, so one failure lists every row that missed.
///
/// ```
/// use bunsen::burner::module::{
///     Rekey,
///     testing::assert_rekeys,
/// };
///
/// let rekey = Rekey::keep_rest()
///     .drop_suffix(".num_batches_tracked")
///     .rename("fc.", "head.fc.");
/// assert_rekeys(
///     &rekey,
///     [
///         ("fc.weight", Some("head.fc.weight")),
///         ("bn1.num_batches_tracked", None),
///     ],
/// );
/// ```
///
/// # Panics
///
/// When any row maps elsewhere; the message lists each such row as
/// `from -> got (want)`.
#[track_caller]
pub fn assert_rekeys<'a, T: Into<Option<&'a str>>>(
    rekey: &Rekey,
    table: impl IntoIterator<Item = (&'a str, T)>,
) {
    let show = |to: Option<&str>| match to {
        Some(path) => format!("{path:?}"),
        None => String::from("dropped"),
    };
    let misses: Vec<String> = table
        .into_iter()
        .filter_map(|(from, want)| {
            let want = want.into();
            let got = rekey.apply(from);
            (got.as_deref() != want).then(|| {
                format!(
                    "  {from:?} -> {} (want {})",
                    show(got.as_deref()),
                    show(want)
                )
            })
        })
        .collect();
    assert!(
        misses.is_empty(),
        "the rekey maps {} rows elsewhere:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rekey() -> Rekey {
        Rekey::keep_rest()
            .drop_suffix(".num_batches_tracked")
            .rename("fc.", "head.fc.")
    }

    /// Both table shapes, mapped and dropped rows.
    #[test]
    fn test_rows_match() {
        assert_rekeys(
            &rekey(),
            [
                ("fc.weight", "head.fc.weight"),
                ("conv.weight", "conv.weight"),
            ],
        );
        assert_rekeys(
            &rekey(),
            [
                ("fc.bias", Some("head.fc.bias")),
                ("bn.num_batches_tracked", None),
            ],
        );
    }

    /// Every missed row is listed, not only the first.
    #[test]
    #[should_panic(expected = "the rekey maps 2 rows elsewhere:\n  \"fc.weight\" -> \
                               \"head.fc.weight\" (want \"fc.weight\")\n  \
                               \"bn.num_batches_tracked\" -> dropped (want \"bn\")")]
    fn test_every_miss_is_listed() {
        assert_rekeys(
            &rekey(),
            [
                ("fc.weight", Some("fc.weight")),
                ("conv.weight", Some("conv.weight")),
                ("bn.num_batches_tracked", Some("bn")),
            ],
        );
    }
}
