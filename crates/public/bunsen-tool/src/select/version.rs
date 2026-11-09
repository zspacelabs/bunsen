//! Release order, and the `latest` select step.
use std::cmp::Ordering;

use bunsen::data::pretrained::{
    selection::{
        SelectOp,
        SelectionCandidate,
    },
    tags::VERSION,
};
use serde_json::Value;

/// A candidate's release, as its `version` tag says: the sequence, and the
/// version in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Release {
    /// The release sequence.
    pub group: String,

    /// The version in it.
    pub version: semver::Version,
}

impl Release {
    /// `candidate`'s release; `None` when it has no `version` tag, or one
    /// that is not `{group, version}` with a semver version (the catalog
    /// build refuses such a tag).
    pub fn of(candidate: &SelectionCandidate) -> Option<Release> {
        let tag = candidate.get(VERSION)?;
        let group = tag.get("group").and_then(Value::as_str)?;
        let version = tag.get("version").and_then(Value::as_str)?;
        Some(Release {
            group: group.to_string(),
            version: semver::Version::parse(version).ok()?,
        })
    }
}

/// Two versions of one sequence, in release order: semver precedence, then
/// the `+tag`, which orders here (semver's own precedence ignores it): an
/// untagged version before a tagged one, two tags split on `.`, each part
/// compared as a number when both are numbers, as text otherwise, a number
/// before text.
pub fn release_cmp(
    a: &semver::Version,
    b: &semver::Version,
) -> Ordering {
    a.cmp_precedence(b)
        .then_with(|| tag_cmp(a.build.as_str(), b.build.as_str()))
}

/// Two `+tag`s, in release order.
fn tag_cmp(
    a: &str,
    b: &str,
) -> Ordering {
    match (a.is_empty(), b.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        (false, false) => {}
    }
    let mut a = a.split('.');
    let mut b = b.split('.');
    loop {
        let ordering = match (a.next(), b.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => match (x.parse::<u64>(), y.parse::<u64>()) {
                (Ok(m), Ok(n)) => m.cmp(&n),
                (Ok(_), Err(_)) => Ordering::Less,
                (Err(_), Ok(_)) => Ordering::Greater,
                (Err(_), Err(_)) => x.cmp(y),
            },
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
}

/// The `latest` select step: the most recent release of each sequence.
///
/// A candidate in no release sequence scores 1. One in a sequence scores
/// `max(floor, 1 - step * rank)`, its rank the number of newer releases of
/// its sequence among the current candidates (0 for the newest; equal
/// versions share a rank; a release dropped by an earlier step no longer
/// counts). A `floor` of 0 drops a release `1 / step` or more behind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Latest {
    /// The weight lost per newer release.
    pub step: f64,

    /// The least score, 0 or more.
    pub floor: f64,
}

impl Latest {
    /// The default step.
    pub const STEP: f64 = 0.05;
}

impl Default for Latest {
    fn default() -> Self {
        Self {
            step: Self::STEP,
            floor: 0.0,
        }
    }
}

impl SelectOp for Latest {
    fn scores(
        &self,
        items: &[(SelectionCandidate, f64)],
    ) -> Vec<f64> {
        let releases: Vec<Option<Release>> = items.iter().map(|(c, _)| Release::of(c)).collect();
        releases
            .iter()
            .map(|release| {
                let Some(release) = release else {
                    return 1.0;
                };
                let mut newer: Vec<&semver::Version> = releases
                    .iter()
                    .flatten()
                    .filter(|other| {
                        other.group == release.group
                            && release_cmp(&other.version, &release.version) == Ordering::Greater
                    })
                    .map(|other| &other.version)
                    .collect();
                newer.sort_by(|a, b| release_cmp(a, b));
                newer.dedup_by(|a, b| release_cmp(a, b) == Ordering::Equal);
                let score = 1.0 - self.step * newer.len() as f64;
                score.max(self.floor).max(0.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use bunsen::data::pretrained::selection::Selection;
    use serde_json::json;

    use super::*;
    use crate::select::testing::candidate;

    fn v(s: &str) -> semver::Version {
        semver::Version::parse(s).unwrap()
    }

    #[test]
    fn test_release_order() {
        let ordered = [
            "1.0.0",
            "2.0.0",
            "3.0.0",
            "3.0.0+2507",
            "3.0.0+2507.1",
            "3.0.0+250801",
            "3.0.0+a",
            "3.1.0",
        ];
        for pair in ordered.windows(2) {
            assert_eq!(
                release_cmp(&v(pair[0]), &v(pair[1])),
                Ordering::Less,
                "{} < {}",
                pair[0],
                pair[1]
            );
        }
        assert_eq!(
            release_cmp(&v("3.0.0+2507"), &v("3.0.0+2507")),
            Ordering::Equal
        );
        assert_eq!(release_cmp(&v("1.0.0-rc1"), &v("1.0.0")), Ordering::Less);
    }

    fn release(
        name: &str,
        group: &str,
        version: &str,
    ) -> SelectionCandidate {
        candidate(
            name,
            json!({"version": {"group": group, "version": version}}),
        )
    }

    /// Each sequence's newest release scores 1, the next `1 - step`; a
    /// candidate in no sequence scores 1; a release `1 / step` behind drops
    /// at the default floor, and stays at a raised one.
    #[test]
    fn test_latest() {
        let items = || {
            Selection::new([
                release("v1", "large", "1.0.0"),
                release("v2", "large", "2.0.0"),
                release("v3", "large", "3.0.0"),
                release("v3-again", "large", "3.0.0"),
                candidate("turbo", json!({})),
                release("q", "qwen3", "3.0.0+2507"),
            ])
        };
        let weights = |s: Selection| -> Vec<(String, f64)> {
            s.iter()
                .map(|(c, w)| (c.pretrained.name.clone(), w))
                .collect()
        };
        let got = weights(items().select(&Latest::default()));
        let want = [
            ("v1", 0.9),
            ("v2", 0.95),
            ("v3", 1.0),
            ("v3-again", 1.0),
            ("turbo", 1.0),
            ("q", 1.0),
        ];
        assert_eq!(got.len(), want.len());
        for ((name, w), (want_name, want_w)) in got.iter().zip(want) {
            assert_eq!(name, want_name);
            assert!((w - want_w).abs() < 1e-12, "{name}: {w}");
        }

        let steep = Latest {
            step: 0.5,
            floor: 0.0,
        };
        let names: Vec<String> = weights(items().select(&steep))
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(
            names,
            ["v2", "v3", "v3-again", "turbo", "q"],
            "v1 is 2 behind"
        );
        let floored = Latest {
            step: 0.5,
            floor: 0.1,
        };
        let got = weights(items().select(&floored));
        assert_eq!(got[0], ("v1".to_string(), 0.1));
    }
}
