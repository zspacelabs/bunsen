/// A [`Lookup`](crate::errors::BunsenErrorKind::Lookup) error for a name a
/// table does not have: a [`LookupError`](crate::errors::LookupError) for
/// `name` among the `kind`s, with the names the table has as candidates, under
/// a frame naming the table when there is one: `"<table>: no <kind> "x";
/// there are: a, b"`.
#[track_caller]
pub(crate) fn not_found(
    table: Option<&str>,
    kind: &'static str,
    name: &str,
    names: &[&str],
) -> crate::errors::BunsenError {
    let error = crate::errors::BunsenError::lookup(
        crate::errors::LookupError::missing(kind, name).with_candidates(names.iter().copied()),
    );
    match table {
        Some(table) => error.context(table),
        None => error,
    }
}
