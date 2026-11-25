//! The fixture plan: each row resolved through bunsen, its files and the
//! validation shard fetched into the cache, written as JSON for
//! `validation/tools/gen_timm_fixtures.py`.
//!
//! `cargo run -p timm-model-validation --features checkpoints --bin
//! timm-fixture-plan [ROW...] > plan.json`; no rows is every row with a
//! fixture ([`ROWS`]).

use bunsen::{
    data::builds::CONFIG,
    kits::images::{
        classification::resnet::pretrained::default_resnet_factory,
        support::{
            card::ImageModelCard,
            interop::timm::timm_model_config_from_path,
        },
    },
};
use timm_model_validation::{
    IMAGES,
    ROWS,
    fixture_path,
    input_key,
    inputs_path,
    validation_shard,
    weights_cache,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rows: Vec<String> = if args.is_empty() {
        ROWS.iter().map(|r| r.to_string()).collect()
    } else {
        args
    };
    let cache = weights_cache();
    let factory = default_resnet_factory().expect("the resnet factory");
    let plan_rows: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            let deferred = factory
                .resolve(row, &cache)
                .unwrap_or_else(|e| panic!("resolve {row}: {e:#}"));
            let loaded = cache
                .load(&deferred.model.to_resource_set())
                .unwrap_or_else(|e| panic!("fetch {row}: {e:#}"));
            let card = ImageModelCard::try_from_build(&deferred.model, &loaded)
                .unwrap_or_else(|e| panic!("{row}'s card: {e:#}"));
            // timm's own architecture name: a hub repo's config says it; a
            // torchvision row is the timm architecture of its name.
            let architecture = match loaded.get(CONFIG) {
                Some(config) => {
                    timm_model_config_from_path(&config.path)
                        .unwrap_or_else(|e| panic!("{row}'s config: {e:#}"))
                        .architecture
                }
                None => row.rsplit('/').next().unwrap().to_string(),
            };
            let key = input_key(&card.transform);
            serde_json::json!({
                "row": row,
                "architecture": architecture,
                "checkpoint": loaded.expect("checkpoint").unwrap(),
                "transform": card.transform,
                "input_key": key,
                "inputs": inputs_path(&cache, &key),
                "fixture": fixture_path(row),
            })
        })
        .collect();
    let plan = serde_json::json!({
        "images": IMAGES,
        "shard": validation_shard(&cache),
        "rows": plan_rows,
    });
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
