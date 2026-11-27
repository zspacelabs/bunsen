//! The refusal of a resource of a kind a kit has no reader for.

use alloc::format;

use crate::errors::BunsenError;

/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported): `kit` has
/// no reader for a `slot` resource of `kind` (`None`, a resource that names
/// no kind). The one wording every kit's
/// [`for_resource_set`](crate::data::builds::ConstructInit::for_resource_set)
/// refuses with.
pub fn unsupported_kind(
    kit: &str,
    slot: &str,
    kind: Option<&str>,
) -> BunsenError {
    BunsenError::unsupported(format!("no {kit} reader for a `{slot}` of kind {kind:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_kind() {
        crate::errors::testing::ErrorMatcher::unsupported()
            .message_eq("no Voxtral reader for a `checkpoint` of kind Some(\"gguf\")")
            .assert(&unsupported_kind("Voxtral", "checkpoint", Some("gguf")));
    }
}
