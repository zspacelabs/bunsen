use alloc::sync::Arc;
use core::fmt::Debug;

use super::*;
use crate::errors::BunsenResult;

/// A detokenizer with no assets: renders each id as its number.
#[derive(Debug)]
struct Numbers;

impl Detokenizer for Numbers {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
    }
}

/// The shape a kit holds: shareable, dynamic, and debuggable.
#[test]
fn test_trait_object() {
    let detokenizer: Option<Arc<dyn Detokenizer>> = Some(Arc::new(Numbers));

    assert_eq!(format!("{detokenizer:?}"), "Some(Numbers)");
    assert_eq!(
        detokenizer
            .as_ref()
            .unwrap()
            .detokenize(&[1, 2, 3])
            .unwrap(),
        "1 2 3"
    );
}
