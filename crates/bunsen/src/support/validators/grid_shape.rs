//! # Config Parsers

/// Parse a shape string into a ``(H, W)`` tuple.
///
/// Accepts:
/// - ``SHAPE``: ``(SHAPE, SHAPE)``.
/// - ``H,W``: ``(H, W)``.
///
/// # Returns
///
/// a result, or error message.
pub fn parse_grid_shape(s: &str) -> Result<[usize; 2], String> {
    if s.contains(",") {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 2 {
            return Err("Shape must be in the format WIDTH,HEIGHT".to_string());
        }
        let width = parts[0]
            .parse::<usize>()
            .map_err(|_| "Invalid width".to_string())?;
        let height = parts[1]
            .parse::<usize>()
            .map_err(|_| "Invalid height".to_string())?;
        Ok([width, height])
    } else {
        let size = s.parse::<usize>().map_err(|_| "Invalid size".to_string())?;
        Ok([size, size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shape() {
        assert_eq!(parse_grid_shape("10,10"), Ok([10, 10]));
        assert_eq!(parse_grid_shape("10"), Ok([10, 10]));

        assert_eq!(
            parse_grid_shape("10,10,10"),
            Err("Shape must be in the format WIDTH,HEIGHT".to_string())
        );
    }
}
