use std::panic::Location;

use serde::{
    Deserialize,
    Serialize,
};

/// Serializable [`Location`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LocationDesc {
    filename: String,
    line: u32,
    col: u32,
}

impl From<&Location<'_>> for LocationDesc {
    fn from(loc: &Location) -> Self {
        Self {
            filename: loc.file().to_string(),
            line: loc.line(),
            col: loc.column(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn who_called_me() -> &'static Location<'static> {
        Location::caller()
    }

    #[test]
    fn test_location_desc() {
        let loc = who_called_me();
        let loc_desc = LocationDesc::from(loc);
        assert_eq!(loc_desc.filename, loc.file().to_string());
        assert_eq!(loc_desc.line, loc.line());
        assert_eq!(loc_desc.col, loc.column());
    }
}
