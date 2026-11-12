//! Release timestamps: a canonical spelling at some precision, read as the
//! span of time it covers.

use core::{
    cmp::Ordering,
    fmt,
};

use time::{
    Date,
    Duration,
    Month,
    OffsetDateTime,
    PrimitiveDateTime,
    Time,
};

use crate::data::catalog::error::{
    CatalogResult,
    CatalogRule,
    catalog_error,
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
    start: PrimitiveDateTime,
    precision: ReleasePrecision,
}

impl ReleaseTime {
    /// Parses a canonical spelling: `YYYY`, `YYYY-MM`, `YYYY-MM-DD`,
    /// `YYYY-MM-DDTHH:MMZ` or `YYYY-MM-DDTHH:MM:SSZ`.
    ///
    /// # Errors
    /// [`CatalogRule::Release`] when it isn't one, or names no real
    /// time.
    pub fn parse(spelling: &str) -> CatalogResult<Self> {
        let bad = |why: &str| {
            catalog_error(
                CatalogRule::Release,
                format!(
                    "`{spelling}` is not a release timestamp ({why}); write `YYYY`, `YYYY-MM`, \
                     `YYYY-MM-DD`, `YYYY-MM-DDTHH:MMZ` or `YYYY-MM-DDTHH:MM:SSZ`, in UTC"
                ),
            )
        };
        let precision = ReleasePrecision::of(spelling).ok_or_else(|| bad("its length"))?;
        let field = |range: core::ops::Range<usize>| -> CatalogResult<u32> {
            let digits = spelling.get(range).ok_or_else(|| bad("its shape"))?;
            if !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad("its shape"));
            }
            digits.parse().map_err(|_| bad("its shape"))
        };
        let separators: &[(usize, u8)] = match precision {
            ReleasePrecision::Year => &[],
            ReleasePrecision::Month => &[(4, b'-')],
            ReleasePrecision::Day => &[(4, b'-'), (7, b'-')],
            ReleasePrecision::Minute => &[(4, b'-'), (7, b'-'), (10, b'T'), (13, b':'), (16, b'Z')],
            ReleasePrecision::Second => &[
                (4, b'-'),
                (7, b'-'),
                (10, b'T'),
                (13, b':'),
                (16, b':'),
                (19, b'Z'),
            ],
        };
        if separators
            .iter()
            .any(|&(at, byte)| spelling.as_bytes()[at] != byte)
        {
            return Err(bad("its shape"));
        }
        let at_least = |p: ReleasePrecision| precision >= p;
        let year = field(0..4)? as i32;
        let month = if at_least(ReleasePrecision::Month) {
            field(5..7)?
        } else {
            1
        };
        let day = if at_least(ReleasePrecision::Day) {
            field(8..10)?
        } else {
            1
        };
        let (hour, minute) = if at_least(ReleasePrecision::Minute) {
            (field(11..13)?, field(14..16)?)
        } else {
            (0, 0)
        };
        let second = if at_least(ReleasePrecision::Second) {
            field(17..19)?
        } else {
            0
        };
        let month = u8::try_from(month)
            .ok()
            .and_then(|m| Month::try_from(m).ok())
            .ok_or_else(|| bad("no such month"))?;
        let date =
            Date::from_calendar_date(year, month, day as u8).map_err(|e| bad(&e.to_string()))?;
        let clock = Time::from_hms(hour as u8, minute as u8, second as u8)
            .map_err(|e| bad(&e.to_string()))?;
        let start = PrimitiveDateTime::new(date, clock);
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
    pub fn start(&self) -> OffsetDateTime {
        self.start.assume_utc()
    }

    /// The instant after the span's last.
    pub fn end(&self) -> OffsetDateTime {
        let start = self.start;
        let first_of = |year: i32, month: Month| {
            Date::from_calendar_date(year, month, 1).expect("the first of a month is a date")
        };
        let end = match self.precision {
            ReleasePrecision::Year => {
                PrimitiveDateTime::new(first_of(start.year() + 1, Month::January), Time::MIDNIGHT)
            }
            ReleasePrecision::Month => {
                let year = if start.month() == Month::December {
                    start.year() + 1
                } else {
                    start.year()
                };
                PrimitiveDateTime::new(first_of(year, start.month().next()), Time::MIDNIGHT)
            }
            ReleasePrecision::Day => start + Duration::days(1),
            ReleasePrecision::Minute => start + Duration::minutes(1),
            ReleasePrecision::Second => start + Duration::seconds(1),
        };
        end.assume_utc()
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
        let t = self.start;
        let (year, month, day) = (t.year(), u8::from(t.month()), t.day());
        let (hour, minute, second) = (t.hour(), t.minute(), t.second());
        match self.precision {
            ReleasePrecision::Year => format!("{year:04}"),
            ReleasePrecision::Month => format!("{year:04}-{month:02}"),
            ReleasePrecision::Day => format!("{year:04}-{month:02}-{day:02}"),
            ReleasePrecision::Minute => {
                format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}Z")
            }
            ReleasePrecision::Second => {
                format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::catalog::error::CatalogErrorExt;

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
        assert_eq!(july.start(), time::macros::datetime!(2025-07-01 0:00 UTC));
        assert_eq!(july.end(), time::macros::datetime!(2025-08-01 0:00 UTC));
        let december = t("2024-12");
        assert_eq!(december.end(), time::macros::datetime!(2025-01-01 0:00 UTC));
        let year = t("2024");
        assert_eq!(year.end(), time::macros::datetime!(2025-01-01 0:00 UTC));
        let day = t("2024-02-29");
        assert_eq!(day.end(), time::macros::datetime!(2024-03-01 0:00 UTC));
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
            assert_eq!(err.rule(), Some(CatalogRule::Release), "{bad}");
        }
    }
}
