//! Stop strings, matched across text deltas.

/// Matches stop strings across a stream of text deltas.
///
/// [`push`](Self::push) returns the text that is safe to show: everything
/// before the first stop string, less any tail that could still grow into
/// one, which is held back until the next delta decides it. When a stop
/// string completes, the text before it is released and the stream is over.
#[derive(Clone, Debug, Default)]
pub struct StopStrings {
    stops: Vec<String>,
    held: String,
    stopped: Option<String>,
}

impl StopStrings {
    /// Matches `stops`; empty strings are ignored.
    pub fn new(stops: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            stops: stops
                .into_iter()
                .map(Into::into)
                .filter(|s: &String| !s.is_empty())
                .collect(),
            held: String::new(),
            stopped: None,
        }
    }

    /// The stop string that ended the stream, once one has.
    pub fn stopped(&self) -> Option<&str> {
        self.stopped.as_deref()
    }

    /// Takes a delta, and returns the text now safe to show.
    pub fn push(
        &mut self,
        delta: &str,
    ) -> String {
        if self.stopped.is_some() {
            return String::new();
        }
        self.held.push_str(delta);
        let first = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|at| (at, s)))
            .min_by_key(|&(at, _)| at);
        if let Some((at, stop)) = first {
            self.stopped = Some(stop.clone());
            let out = self.held[..at].to_string();
            self.held.clear();
            return out;
        }
        let keep = self.partial_tail();
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        out
    }

    /// The text still held back, now final: the stream ended without its
    /// tail becoming a stop string.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// The length of the longest tail of the held text that a stop string
    /// begins with.
    fn partial_tail(&self) -> usize {
        let held = self.held.as_str();
        self.stops
            .iter()
            .flat_map(|stop| {
                (1..stop.len().min(held.len() + 1))
                    .rev()
                    .filter(|&n| stop.is_char_boundary(n))
                    .find(|&n| held.ends_with(&stop[..n]))
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_stop_split_across_deltas() {
        let mut stops = StopStrings::new(["<end>"]);
        assert_eq!(stops.push("hello <e"), "hello ");
        assert_eq!(stops.push("n"), "");
        assert_eq!(stops.push("d> world"), "");
        assert_eq!(stops.stopped(), Some("<end>"));
        assert_eq!(stops.push("more"), "");
    }

    #[test]
    fn test_a_false_start_is_released() {
        let mut stops = StopStrings::new(["<end>", "STOP"]);
        assert_eq!(stops.push("a <e"), "a ");
        assert_eq!(stops.push("x"), "<ex");
        assert_eq!(stops.push(" S"), " ");
        assert_eq!(stops.finish(), "S");
        assert_eq!(stops.stopped(), None);
    }

    #[test]
    fn test_the_first_stop_wins() {
        let mut stops = StopStrings::new(["b", "a"]);
        assert_eq!(stops.push("xab"), "x");
        assert_eq!(stops.stopped(), Some("a"));
    }

    #[test]
    fn test_no_stops_hold_nothing() {
        let mut stops = StopStrings::new(Vec::<String>::new());
        assert_eq!(stops.push("é<"), "é<");
        assert_eq!(stops.finish(), "");
    }
}
