use super::*;
#[cfg(doc)]
use crate::blocks::transformers::decoders::causal::CausalDecoderBlock;
use crate::errors::testing::{
    ErrorMatcher,
    text,
};

/// The text helper parses and converts; a conversion's error passes
/// through.
#[test]
fn test_config_from_str() {
    let n = config_from_str(r#"{"vocab_size": 7}"#, |t| t.usize("vocab_size")).unwrap();
    assert_eq!(n, 7);
    ErrorMatcher::invalid_resource()
        .display(text::contains("missing_field"))
        .assert_err(&config_from_str(r#"{}"#, |t| t.usize("missing_field")));
}

/// The file helper names the file in a conversion's error, and in a
/// read's.
#[test]
fn test_config_from_path_names_the_file() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let missing = dir.join("no-such-config.json");
    assert!(
        config_from_path(&missing, |t| t.usize("vocab_size"))
            .unwrap_err()
            .to_string()
            .contains("no-such-config.json")
    );
}
