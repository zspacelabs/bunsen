use bunsen::errors::{
    BunsenResult,
    LookupError,
    LookupProblem,
    testing::{
        ErrorMatcher,
        predicate,
    },
};
use serde_json::json;

use super::*;
use crate::augmentation::control::{
    noop::NOOP_STAGE,
    sequence::STAGE_SEQUENCE,
};

#[test]
fn test_plugin_builder_errors() {
    let builder = GlobalRegistryBuilder {};

    ErrorMatcher::lookup()
        .cause(predicate("a missing \"nope\"", |l: &LookupError| {
            l.key == "nope"
                && l.problem == LookupProblem::Missing
                && l.candidates.iter().any(|c| c == NOOP_STAGE)
        }))
        .assert_err(
            &builder
                .build_stage(&AugmentationStageConfig {
                    name: "nope".to_string(),
                    body: serde_json::Value::Null,
                })
                .map(|_| ()),
        );

    ErrorMatcher::illegal()
        .frame_contains(STAGE_SEQUENCE)
        .has_cause::<serde_json::Error>()
        .assert_err(
            &builder
                .build_stage(&AugmentationStageConfig {
                    name: STAGE_SEQUENCE.to_string(),
                    body: json!({ "stages": 3 }),
                })
                .map(|_| ()),
        );
}

#[test]
fn test_plugin_builder() -> BunsenResult<()> {
    let cfg_json = json! {
        {
            "name": STAGE_SEQUENCE,
            "body": {
              "stages": [
                {
                  "name": NOOP_STAGE,
                }
              ]
            }
        }
    };
    let pretty = serde_json::to_string_pretty(&cfg_json).unwrap();
    println!("{pretty}");

    let cfg = serde_json::from_value::<AugmentationStageConfig>(cfg_json.clone()).unwrap();

    let builder = GlobalRegistryBuilder {};

    let plugin = builder.build_stage(&cfg)?;

    let new_cfg = plugin.as_config();
    let new_pretty = serde_json::to_string_pretty(&new_cfg).unwrap();
    println!("{new_pretty}");

    // assert!(false);

    Ok(())
}
