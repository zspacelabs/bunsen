//! The `--think` / `--no-think` shortcut as a product of tags and presets,
//! with no per-family code: each flag names a member of the registered bool
//! axis `thinking`, and selecting it is an ordinary preset selection. A
//! build that fixes the mode explains why the flag doesn't apply.
//!
//! `cargo run -p pretrained-catalog --example think_flags`

use pretrained_catalog::{
    CatalogName,
    fixtures,
    presets::PresetSelection,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stack = fixtures::builtin_stack()?;
    let thinking = CatalogName::parse("thinking")?;
    let chat = CatalogName::parse("qwen3-lm")?;

    for build in stack.members_of(&chat) {
        let resolved = stack.resolve_build(&build)?;
        let mode = resolved
            .tag("thinking")
            .expect("every qwen3 build uses the axis");
        println!("{} (thinking: {mode})", stack.display_name(&build).short);
        for (flag, on) in [("--think", true), ("--no-think", false)] {
            match resolved.presets.bool_member(&thinking, on) {
                Ok(preset) => {
                    let selection = PresetSelection::of([preset.as_str()]);
                    let applied = resolved.presets.select(json!({}), &selection)?;
                    println!(
                        "  {flag:<11} implies --preset {preset}: {}",
                        applied.document
                    );
                }
                Err(e) => println!("  {flag:<11} doesn't apply: {}", e.message),
            }
        }
    }
    Ok(())
}
