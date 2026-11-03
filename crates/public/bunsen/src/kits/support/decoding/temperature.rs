//! The temperature ladder.
use burn::config::Config;

use super::SearchConfig;

/// The temperatures a decode climbs, and how many trajectories its
/// sampling rungs draw.
///
/// The first rung is the search proper (temperature zero, say, with its
/// beam); each later rung samples ([`SearchConfig::at_temperature`]).
/// [`climb`](Self::climb) decodes rung after rung until a judge passes a
/// result: what passes is the model's (Whisper's thresholds on compression
/// and log probability are one judge), the ladder only orchestrates.
#[derive(Config, Debug, PartialEq)]
pub struct TemperatureLadder {
    /// The temperatures tried in order; the first is the search proper,
    /// the rest are sampling. Never empty.
    #[config(default = "vec![0.0]")]
    pub temperatures: Vec<f64>,

    /// Sample trajectories per group above temperature zero; `None` is one.
    #[config(default = "None")]
    pub best_of: Option<usize>,
}

impl TemperatureLadder {
    /// `0, 0.2, ..., 1.0`: upstream Whisper's ladder.
    pub fn whisper_upstream() -> Self {
        Self::new().with_temperatures(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    }

    /// The search of the rung at `temperature`, derived from `base`.
    pub fn rung(
        &self,
        base: &SearchConfig,
        temperature: f64,
    ) -> SearchConfig {
        base.at_temperature(temperature, self.best_of)
    }

    /// Decodes up the ladder until a rung passes.
    ///
    /// # Arguments
    /// * `base` - the search at the first rung; each rung derives from it.
    /// * `first` - the first rung's result, when the caller already has it (a
    ///   batched first decode, say); `None` decodes it.
    /// * `decode` - runs one rung, given its search.
    /// * `needs_fallback` - the judge: whether a result fails, and the next
    ///   rung should be tried.
    ///
    /// # Returns
    /// The last result: the first that passed, or the top rung's.
    ///
    /// # Panics
    /// If the ladder has no rung.
    pub fn climb<R>(
        &self,
        base: &SearchConfig,
        first: Option<R>,
        mut decode: impl FnMut(&SearchConfig) -> R,
        mut needs_fallback: impl FnMut(&R) -> bool,
    ) -> R {
        assert!(
            !self.temperatures.is_empty(),
            "the ladder has at least one rung"
        );
        let mut first = first;
        let mut result = None;
        for &temperature in &self.temperatures {
            let decoded = match first.take() {
                Some(given) => given,
                None => decode(&self.rung(base, temperature)),
            };
            let needs = needs_fallback(&decoded);
            result = Some(decoded);
            if !needs {
                break;
            }
        }
        result.expect("at least one rung ran")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first rung that passes wins; a given first result stands in for
    /// the first rung; when every rung fails, the top one comes back.
    #[test]
    fn test_climb() {
        let ladder = TemperatureLadder::whisper_upstream().with_best_of(Some(2));
        let base = SearchConfig::new().with_beam_size(4);

        let mut seen = Vec::new();
        let out = ladder.climb(
            &base,
            None,
            |search| {
                seen.push((search.temperature, search.beam_size, search.best_of));
                search.temperature
            },
            |&t| t < 0.4,
        );
        assert_eq!(out, 0.4);
        assert_eq!(seen, [(0.0, 4, None), (0.2, 1, Some(2)), (0.4, 1, Some(2))]);

        let out = ladder.climb(&base, Some(-1.0), |s| s.temperature, |_| false);
        assert_eq!(out, -1.0, "a given first result is the first rung");

        let out = ladder.climb(&base, None, |s| s.temperature, |_| true);
        assert_eq!(out, 1.0, "every rung fails: the top one");
    }
}
