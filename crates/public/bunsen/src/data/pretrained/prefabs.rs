//! Config prefabs: the geometries a kit knows by name.

use alloc::vec::Vec;

use crate::{
    data::not_found,
    errors::BunsenResult,
};

/// A prefab table: each prefab's name, and the constructor that builds it.
pub type PrefabTable<C> = &'static [(&'static str, fn() -> C)];

/// A config with prefabs: the geometries a kit knows by name, each a
/// constructor of the config documented where it is defined
/// (`WhisperApiConfig::tiny_en`).
///
/// A prefab is what a name means as a shape before any bytes are fetched. A
/// build states the geometry it is built in
/// ([`BuildRef::stated`](crate::data::builds::BuildRef::stated)), which the
/// kit's tests check is a prefab's; [`find_prefab`](Self::find_prefab) is the
/// reverse lookup, for a config that arrived without a name.
///
/// ```
/// use bunsen::data::pretrained::{
///     PrefabTable,
///     Prefabs,
/// };
///
/// #[derive(Debug, PartialEq)]
/// struct Toy {
///     width: usize,
/// }
///
/// impl Toy {
///     /// Width 1.
///     fn narrow() -> Self {
///         Self { width: 1 }
///     }
///
///     /// Width 2.
///     fn wide() -> Self {
///         Self { width: 2 }
///     }
/// }
///
/// impl Prefabs for Toy {
///     const PREFABS: PrefabTable<Self> =
///         &[("narrow", Self::narrow), ("wide", Self::wide)];
///     const PREFAB_TABLE: &'static str = "toys";
/// }
///
/// assert_eq!(Toy::prefab_names(), ["narrow", "wide"]);
/// assert_eq!(Toy::prefab("wide"), Some(Toy { width: 2 }));
/// assert_eq!(Toy::find_prefab(|t| t.width == 1), Some("narrow"));
/// ```
pub trait Prefabs: Sized + 'static {
    /// What the prefabs are called together, in errors: `whisper`.
    const PREFAB_TABLE: &'static str;

    /// The prefabs, by name, in listing order: `("tiny.en", Self::tiny_en)`.
    const PREFABS: PrefabTable<Self>;

    /// Every prefab name, in listing order.
    fn prefab_names() -> Vec<&'static str> {
        Self::PREFABS.iter().map(|(name, _)| *name).collect()
    }

    /// The prefab `name`, built.
    fn prefab(name: &str) -> Option<Self> {
        Self::PREFABS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, build)| build())
    }

    /// The prefab `name`, built.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup), with a
    /// [`LookupError`](crate::errors::LookupError) naming the prefabs there
    /// are, under a frame naming the table.
    fn try_prefab(name: &str) -> BunsenResult<Self> {
        Self::prefab(name).ok_or_else(|| {
            not_found(
                Some(Self::PREFAB_TABLE),
                "prefab",
                name,
                &Self::prefab_names(),
            )
        })
    }

    /// The prefab `name`, built, for a name the kit compiles in.
    ///
    /// # Panics
    /// If there is no such prefab.
    fn expect_prefab(name: &str) -> Self {
        Self::try_prefab(name).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The name of the first prefab whose config satisfies `pred`: the
    /// reverse lookup, for a config that arrived without a name.
    fn find_prefab(mut pred: impl FnMut(&Self) -> bool) -> Option<&'static str> {
        Self::PREFABS
            .iter()
            .find(|(_, build)| pred(&build()))
            .map(|(name, _)| *name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        LookupError,
        testing::{
            ErrorMatcher,
            predicate,
            text,
        },
    };

    #[derive(Debug, PartialEq)]
    struct Toy {
        width: usize,
    }

    impl Prefabs for Toy {
        const PREFABS: PrefabTable<Self> = &[
            ("narrow", || Toy { width: 1 }),
            ("wide", || Toy { width: 2 }),
        ];
        const PREFAB_TABLE: &'static str = "toys";
    }

    #[test]
    fn test_names_lookups_and_find() {
        assert_eq!(Toy::prefab_names(), ["narrow", "wide"]);
        assert_eq!(Toy::prefab("wide"), Some(Toy { width: 2 }));
        assert_eq!(Toy::prefab("huge"), None);
        assert_eq!(Toy::expect_prefab("narrow"), Toy { width: 1 });
        assert_eq!(Toy::find_prefab(|c| c.width == 2), Some("wide"));
        assert!(Toy::find_prefab(|c| c.width == 3).is_none());
    }

    /// A miss names the table and what it holds.
    #[test]
    fn test_a_miss_names_what_there_is() {
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::eq(
                "toys: no prefab \"huge\"; there are: narrow, wide",
            ))
            .cause(predicate("candidates narrow, wide", |c: &LookupError| {
                c.key == "huge" && c.candidates == ["narrow", "wide"]
            }))
            .assert_err(&Toy::try_prefab("huge"));
    }
}
