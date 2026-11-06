//! A module's canonical weight names.

use super::Rekey;

/// A module's canonical weight names: the scheme its loaders map a
/// checkpoint onto, and the map from that scheme onto its record paths.
///
/// A loader maps a checkpoint in two stages
/// ([`Rekey::then`]): its family's names onto the module's canonical ones
/// (the kit's map), then those onto the module's record paths
/// (`canonical_rekey`). A block owns its canonical names; a stack or a
/// model mounts its blocks' maps where it holds them. So a block's record
/// layout can change without any kit's map changing, and a rule written
/// against canonical names (a load transform, a scan of a checkpoint's
/// layers) keeps finding what it names.
///
/// A block's canonical names are the paths its record had when the scheme
/// was set; a block whose record moves maps those names onto the new paths.
///
/// ```rust
/// use bunsen::burner::module::{
///     CanonicalKeys,
///     Rekey,
/// };
///
/// /// A block whose norm is recorded as `norms.pre`, canonically `norm`.
/// struct Block;
///
/// impl CanonicalKeys for Block {
///     fn canonical_rekey(&self) -> Rekey {
///         Rekey::keep_rest().rename("norm.", "norms.pre.")
///     }
/// }
///
/// let layers = vec![Block, Block];
/// let rekey = Rekey::keep_rest().mount_canonical("layers.", &layers);
/// assert_eq!(
///     rekey.apply("layers.1.norm.weight").as_deref(),
///     Some("layers.1.norms.pre.weight")
/// );
/// ```
pub trait CanonicalKeys {
    /// The map from this module's canonical names onto its record paths.
    fn canonical_rekey(&self) -> Rekey;
}

impl Rekey {
    /// This map, with `module`'s canonical map mounted at the path `at`
    /// (`"stack."`), where the module holds it.
    pub fn mount_canonical(
        self,
        at: impl Into<String>,
        module: &impl CanonicalKeys,
    ) -> Self {
        let at = at.into();
        self.mount(at.clone(), at, module.canonical_rekey())
    }
}

/// Each element's map, mounted at its index (`{i}.`): exact for a stack
/// whose layers differ.
impl<T: CanonicalKeys> CanonicalKeys for Vec<T> {
    fn canonical_rekey(&self) -> Rekey {
        self.iter()
            .enumerate()
            .fold(Rekey::keep_rest(), |rekey, (i, item)| {
                rekey.mount_canonical(format!("{i}."), item)
            })
    }
}

/// The inner module's map, when there is one; names kept otherwise.
impl<T: CanonicalKeys> CanonicalKeys for Option<T> {
    fn canonical_rekey(&self) -> Rekey {
        self.as_ref()
            .map_or_else(Rekey::keep_rest, T::canonical_rekey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Renames(&'static str, &'static str);

    impl CanonicalKeys for Renames {
        fn canonical_rekey(&self) -> Rekey {
            Rekey::keep_rest().rename(self.0, self.1)
        }
    }

    /// Each layer's map applies under its own index only.
    #[test]
    fn test_vec_mounts_each_layer() {
        let layers = vec![Renames("a.", "x."), Renames("a.", "y.")];
        let rekey = layers.canonical_rekey();
        assert_eq!(rekey.apply("0.a.w").as_deref(), Some("0.x.w"));
        assert_eq!(rekey.apply("1.a.w").as_deref(), Some("1.y.w"));
        assert_eq!(rekey.apply("2.a.w").as_deref(), Some("2.a.w"));
    }

    #[test]
    fn test_option_is_the_inner_map_or_kept() {
        assert_eq!(
            Some(Renames("a.", "b."))
                .canonical_rekey()
                .apply("a.w")
                .as_deref(),
            Some("b.w")
        );
        assert_eq!(
            None::<Renames>.canonical_rekey().apply("a.w").as_deref(),
            Some("a.w")
        );
    }
}
