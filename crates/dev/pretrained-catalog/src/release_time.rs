//! Release timestamps: a canonical spelling at some precision, read as the
//! span of time it covers.

use std::{
    cmp::Ordering,
    fmt,
};

use jiff::{
    Span,
    Timestamp,
    civil::DateTime,
    fmt::strtime::BrokenDownTime,
    tz::TimeZone,
};

use crate::error::{
    CatalogError,
    CatalogErrorKind,
    CatalogResult,
};

/// How precisely a release timestamp is pinned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReleasePrecision {
    /// `2025`.
    Year,
    /// `2025-07`.
    Month,
    /// `2025-07-15`.
    Day,
    /// `2025-07-15T09:30Z`.
    Minute,
    /// `2025-07-15T09:30:00Z`.
    Second,
}

impl ReleasePrecision {
    fn format(self) -> &'static str {
        match self {
            Self::Year => "%Y",
            Self::Month => "%Y-%m",
            Self::Day => "%Y-%m-%d",
            Self::Minute => "%Y-%m-%dT%H:%MZ",
            Self::Second => "%Y-%m-%dT%H:%M:%SZ",
        }
    }

    fn unit(self) -> Span {
        match self {
            Self::Year => Span::new().years(1),
            Self::Month => Span::new().months(1),
            Self::Day => Span::new().days(1),
            Self::Minute => Span::new().minutes(1),
            Self::Second => Span::new().seconds(1),
        }
    }

    /// The precision a canonical spelling is written at, judged by its
    /// shape alone.
    fn of(spelling: &str) -> Option<Self> {
        match spelling.len() {
            4 => Some(Self::Year),
            7 => Some(Self::Month),
            10 => Some(Self::Day),
            17 => Some(Self::Minute),
            20 => Some(Self::Second),
            _ => None,
        }
    }
}

/// A release timestamp: the span `[start, end)` that its spelling covers,
/// in UTC. `2025-07` covers all of July 2025.
///
/// Ordering compares starts, then precision; [`precedes`](Self::precedes)
/// is the stricter "ends no later than the other starts".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReleaseTime {
    start: DateTime,
    precision: ReleasePrecision,
}

impl ReleaseTime {
    /// Parses a canonical spelling: `YYYY`, `YYYY-MM`, `YYYY-MM-DD`,
    /// `YYYY-MM-DDTHH:MMZ` or `YYYY-MM-DDTHH:MM:SSZ`.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Release`] when it isn't one, or names no real
    /// time.
    pub fn parse(spelling: &str) -> CatalogResult<Self> {
        let bad = |why: &str| {
            CatalogError::new(
                CatalogErrorKind::Release,
                format!(
                    "`{spelling}` is not a release timestamp ({why}); write `YYYY`, `YYYY-MM`, \
                     `YYYY-MM-DD`, `YYYY-MM-DDTHH:MMZ` or `YYYY-MM-DDTHH:MM:SSZ`, in UTC"
                ),
            )
        };
        let precision = ReleasePrecision::of(spelling).ok_or_else(|| bad("its length"))?;
        let fields =
            BrokenDownTime::parse(precision.format(), spelling).map_err(|e| bad(&e.to_string()))?;
        let start = DateTime::new(
            fields.year().ok_or_else(|| bad("no year"))?,
            fields.month().unwrap_or(1),
            fields.day().unwrap_or(1),
            fields.hour().unwrap_or(0),
            fields.minute().unwrap_or(0),
            fields.second().unwrap_or(0),
            0,
        )
        .map_err(|e| bad(&e.to_string()))?;
        let time = Self { start, precision };
        if time.canonical() != spelling {
            return Err(bad("not its canonical spelling"));
        }
        Ok(time)
    }

    /// How precisely it is pinned.
    pub fn precision(&self) -> ReleasePrecision {
        self.precision
    }

    /// The span's first instant.
    pub fn start(&self) -> Timestamp {
        utc(self.start)
    }

    /// The instant after the span's last.
    pub fn end(&self) -> Timestamp {
        utc(self
            .start
            .checked_add(self.precision.unit())
            .expect("a parsed year is far from the end of time"))
    }

    /// Whether the whole span ends no later than `other` starts.
    pub fn precedes(
        &self,
        other: &ReleaseTime,
    ) -> bool {
        self.end() <= other.start()
    }

    /// Whether neither span precedes the other.
    pub fn overlaps(
        &self,
        other: &ReleaseTime,
    ) -> bool {
        !self.precedes(other) && !other.precedes(self)
    }

    /// The canonical spelling.
    pub fn canonical(&self) -> String {
        self.start.strftime(self.precision.format()).to_string()
    }
}

impl PartialOrd for ReleaseTime {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReleaseTime {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.start
            .cmp(&other.start)
            .then(self.precision.cmp(&other.precision))
    }
}

impl fmt::Display for ReleaseTime {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(&self.canonical())
    }
}

fn utc(time: DateTime) -> Timestamp {
    time.to_zoned(TimeZone::UTC)
        .expect("every civil time is a UTC instant")
        .timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> ReleaseTime {
        ReleaseTime::parse(s).unwrap()
    }

    #[test]
    fn test_each_precision_parses_and_round_trips() {
        for (s, p) in [
            ("2025", ReleasePrecision::Year),
            ("2025-07", ReleasePrecision::Month),
            ("2025-07-15", ReleasePrecision::Day),
            ("2025-07-15T09:30Z", ReleasePrecision::Minute),
            ("2025-07-15T09:30:05Z", ReleasePrecision::Second),
        ] {
            let time = t(s);
            assert_eq!(time.precision(), p, "{s}");
            assert_eq!(time.canonical(), s);
        }
    }

    #[test]
    fn test_spans_cover_one_unit() {
        let july = t("2025-07");
        assert_eq!(july.start().to_string(), "2025-07-01T00:00:00Z");
        assert_eq!(july.end().to_string(), "2025-08-01T00:00:00Z");
        let year = t("2024");
        assert_eq!(year.end().to_string(), "2025-01-01T00:00:00Z");
    }

    #[test]
    fn test_precedes_and_overlaps() {
        assert!(t("2022-09").precedes(&t("2022-12")));
        assert!(t("2025-06").precedes(&t("2025-07-01")));
        assert!(t("2025").overlaps(&t("2025-07")));
        assert!(t("2025-07").overlaps(&t("2025-07-15T09:30Z")));
        assert!(!t("2025-07-14").overlaps(&t("2025-07-15")));
    }

    #[test]
    fn test_non_canonical_spellings_are_refused() {
        for bad in [
            "25",
            "2025-7",
            "2025-13",
            "2025-02-30",
            "2025/07",
            "2025-07-15 09:30Z",
            "2025-07-15T09:30+01:00",
            "July 2025",
        ] {
            let err = ReleaseTime::parse(bad).unwrap_err();
            assert_eq!(err.kind, CatalogErrorKind::Release, "{bad}");
        }
    }
}
