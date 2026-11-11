//! The `latest` weighing step: each release group's newest release first.

use crate::select::core::{
    SelectionCandidate,
    WeighOp,
};

/// The most recent release of each release group.
///
/// A candidate in no release group scores 1. One in a group scores
/// `max(floor, 1 - step * rank)`, its rank the number of newer releases of
/// its group among the current candidates (0 for the newest; a release an
/// earlier step dropped no longer counts). Releases of one group never
/// overlap in time, so the order is strict. A `floor` of 0 drops a release
/// `1 / step` or more behind.
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

impl WeighOp for Latest {
    fn scores(
        &self,
        items: &[(SelectionCandidate, f64)],
    ) -> Vec<f64> {
        items
            .iter()
            .map(|(candidate, _)| {
                let Some(release) = &candidate.build.release else {
                    return 1.0;
                };
                let newer = items
                    .iter()
                    .filter_map(|(other, _)| other.build.release.as_ref())
                    .filter(|other| {
                        other.group == release.group && release.released.precedes(&other.released)
                    })
                    .count();
                let score = 1.0 - self.step * newer as f64;
                score.max(self.floor).max(0.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::{
        core::Order,
        testing::{
            selection,
            shown,
        },
    };

    const LARGE: [&str; 4] = ["large-v1", "large-v2", "large-v3", "large-v3-turbo"];

    #[test]
    fn test_newest_of_each_group_scores_highest() {
        let s = selection(&LARGE).weigh(&Latest {
            step: 0.25,
            floor: 0.0,
        });
        let weights: Vec<(String, f64)> = s
            .iter()
            .map(|(c, w)| (c.name().leaf().to_string(), w))
            .collect();
        assert_eq!(
            weights,
            [
                ("large-v1".to_string(), 0.5),
                ("large-v2".to_string(), 0.75),
                ("large-v3".to_string(), 1.0),
                // In no release group.
                ("large-v3-turbo".to_string(), 1.0),
            ]
        );
        let ordered = s.order(&[Order::Descending], |_, w| [Some(w)]);
        assert_eq!(shown(&ordered)[..2], ["large-v3", "large-v3-turbo"]);
    }

    #[test]
    fn test_floor_and_drop() {
        let dropped = selection(&LARGE).weigh(&Latest {
            step: 0.5,
            floor: 0.0,
        });
        assert_eq!(shown(&dropped), ["large-v2", "large-v3", "large-v3-turbo"]);
        let floored = selection(&LARGE).weigh(&Latest {
            step: 0.5,
            floor: 0.1,
        });
        assert_eq!(floored.len(), 4);
    }

    #[test]
    fn test_rank_counts_only_current_candidates() {
        // Without v3, v2 is the newest.
        let s = selection(&["large-v1", "large-v2"]).weigh(&Latest {
            step: 0.25,
            floor: 0.0,
        });
        let weights: Vec<f64> = s.iter().map(|(_, w)| w).collect();
        assert_eq!(weights, [0.75, 1.0]);
    }
}
