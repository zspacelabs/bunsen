//! Release order, and the `latest` select step.

use bunsen::data::pretrained::{
    build_catalog::release_time::ReleaseTime,
    selection::{
        SelectOp,
        SelectionCandidate,
    },
};
use serde_json::Value;

/// The tag a row carries its release in: `{group, released}`, the release
/// group's full name and when the build was released, from the build
/// catalog.
pub const RELEASE: &str = "release";

/// A candidate's release, as its `release` tag says: the group, and when.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Release {
    /// The release group.
    pub group: String,

    /// When it was released.
    pub released: ReleaseTime,
}

impl Release {
    /// `candidate`'s release; `None` when it has no `release` tag, or one
    /// that is not `{group, released}` with a release time.
    pub fn of(candidate: &SelectionCandidate) -> Option<Release> {
        let tag = candidate.get(RELEASE)?;
        let group = tag.get("group").and_then(Value::as_str)?;
        let released = tag.get("released").and_then(Value::as_str)?;
        Some(Release {
            group: group.to_string(),
            released: ReleaseTime::parse(released).ok()?,
        })
    }

    /// Whether `other` is a later release of the same group: its time
    /// starts after this one's ends.
    pub fn is_before(
        &self,
        other: &Release,
    ) -> bool {
        self.group == other.group && self.released.precedes(&other.released)
    }
}

/// The `latest` select step: the most recent release of each group.
///
/// A candidate in no release group scores 1. One in a group scores
/// `max(floor, 1 - step * rank)`, its rank the number of newer releases of
/// its group among the current candidates (0 for the newest; releases at
/// one time share a rank; a release dropped by an earlier step no longer
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
                // The newer releases, those released at one time counted once.
                let mut newer: Vec<&ReleaseTime> = Vec::new();
                for other in releases.iter().flatten() {
                    if release.is_before(other)
                        && !newer.iter().any(|t| t.overlaps(&other.released))
                    {
                        newer.push(&other.released);
                    }
                }
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

    fn release(
        name: &str,
        group: &str,
        released: &str,
    ) -> SelectionCandidate {
        candidate(
            name,
            json!({RELEASE: {"group": group, "released": released}}),
        )
    }

    /// A release precedes another of its group when its time ends first.
    #[test]
    fn test_release_order() {
        let r = |group: &str, released: &str| Release {
            group: group.to_string(),
            released: ReleaseTime::parse(released).unwrap(),
        };
        assert!(r("large", "2022-09").is_before(&r("large", "2022-12")));
        assert!(!r("large", "2022-12").is_before(&r("large", "2022-09")));
        assert!(!r("large", "2023").is_before(&r("large", "2023-11-06")));
        assert!(!r("large", "2022-09").is_before(&r("qwen3", "2025-07")));
    }

    /// Each group's newest release scores 1, the next `1 - step`; a
    /// candidate in no group scores 1; a release `1 / step` behind drops
    /// at the default floor, and stays at a raised one.
    #[test]
    fn test_latest() {
        let items = || {
            Selection::new([
                release("v1", "large", "2022-09"),
                release("v2", "large", "2022-12"),
                release("v3", "large", "2023-11-06"),
                release("v3-again", "large", "2023-11-06"),
                candidate("turbo", json!({})),
                release("q", "qwen3", "2025-07"),
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
