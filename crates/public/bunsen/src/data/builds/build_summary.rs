//! Build summaries: which build a load ran, in a line.

use alloc::{
    string::{
        String,
        ToString,
    },
    vec::Vec,
};
use core::fmt;

use crate::data::{
    builds::BuildRef,
    catalog::presets::PresetSelection,
};

/// Which build a load ran, and through which presets: the build's identity
/// without its resolved config, for a log line or a published record.
///
/// Made from a ref ([`BuildRef::summary`]) and completed by the load
/// ([`Loaded::summary`](crate::data::builds::Loaded::summary)), which knows
/// what the kit's hook applied
/// ([`ConstructInit::applied_presets`](crate::data::builds::ConstructInit::applied_presets)).
/// Its `Display` is one line:
/// `catalog:openai/small [multilingual,translate]{beam_size: 1} (groups:
/// openai, whisper)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildSummary {
    /// The qualified id, `provider:ref`, or a given resource set's name.
    pub id: String,

    /// Every group the build is in, by full name; empty for a given set.
    pub groups: Vec<String>,

    /// What the lookup asked of the build's presets, its box and patch;
    /// `None` for a given set.
    pub requested: Option<PresetSelection>,

    /// What the kit's hook applied: the default presets no selection
    /// displaced, then the selected, and the patch. `None` before the load,
    /// or for a kit that resolves no presets.
    pub applied: Option<PresetSelection>,
}

impl BuildSummary {
    /// The summary with `applied` as what the hook applied.
    pub fn with_applied(
        self,
        applied: Option<PresetSelection>,
    ) -> Self {
        Self { applied, ..self }
    }
}

impl fmt::Display for BuildSummary {
    /// `id`, then the applied selection (or, before a load, the requested
    /// one) when it is not empty, then the selection asked for when the
    /// hook applied more, then the groups.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(&self.id)?;
        let shown = self.applied.as_ref().or(self.requested.as_ref());
        if let Some(shown) = shown
            && !shown.is_empty()
        {
            write!(f, " {shown}")?;
        }
        if let (Some(applied), Some(requested)) = (&self.applied, &self.requested)
            && applied != requested
            && !requested.is_empty()
        {
            write!(f, " (asked {requested})")?;
        }
        if !self.groups.is_empty() {
            write!(f, " (groups: {})", self.groups.join(", "))?;
        }
        Ok(())
    }
}

impl BuildRef {
    /// What the ref names, as a [`BuildSummary`]: its id, its build's
    /// groups and the lookup's selection. Nothing is applied yet.
    pub fn summary(&self) -> BuildSummary {
        let (groups, requested) = match self.build() {
            Some((_, build)) => (
                build.memberships.iter().map(ToString::to_string).collect(),
                self.selection().cloned(),
            ),
            None => (Vec::new(), None),
        };
        BuildSummary {
            id: self.id(),
            groups,
            requested,
            applied: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;

    fn summary() -> BuildSummary {
        BuildSummary {
            id: "catalog:openai/small".to_string(),
            groups: Vec::from(["openai".to_string(), "whisper".to_string()]),
            requested: Some(PresetSelection::of(["translate"])),
            applied: None,
        }
    }

    #[test]
    fn test_display_before_a_load_shows_the_request() {
        assert_eq!(
            format!("{}", summary()),
            "catalog:openai/small [translate] (groups: openai, whisper)"
        );
    }

    #[test]
    fn test_display_after_a_load_shows_what_was_applied() {
        let s = summary().with_applied(Some(PresetSelection::of(["multilingual", "translate"])));
        assert_eq!(
            format!("{s}"),
            "catalog:openai/small [multilingual,translate] (asked [translate]) (groups: openai, whisper)"
        );
    }

    #[test]
    fn test_display_of_a_given_set_is_its_name() {
        let s = BuildSummary {
            id: "model.bin".to_string(),
            ..BuildSummary::default()
        };
        assert_eq!(format!("{s}"), "model.bin");
    }
}
