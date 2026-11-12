//! The `--think` / `--no-think` shortcut as a product of tags and presets,
//! with no per-family code: each flag names a member of the registered bool
//! axis `thinking`, and selecting it is an ordinary preset selection.
//!
//! - **On one build**, the flag is that build's preset, or an explained refusal
//!   when the build fixes the other mode.
//! - **When choosing**, the flag is the preset the tool applies, and the prefix
//!   that keeps only the builds that can take it: the chosen build always can.
//!
//! `cargo run -p pretrained-catalog --example think_flags`

use bunsen::data::catalog::{
    CatalogErrorExt,
    CatalogName,
    fixtures,
    presets::PresetSelection,
    select::{
        choose::{
            SelectRequest,
            SelectionContext,
            bool_axis_member,
            preset_prefix,
        },
        state::{
            SelectEnv,
            StageOrigin,
        },
    },
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stack = fixtures::example_stack()?;
    let thinking = CatalogName::parse("thinking")?;
    let chat = CatalogName::parse("$contracts/qwen3-lm")?;

    println!("On each build:");
    for build in stack.builds_of_contract(&chat) {
        let resolved = stack.resolve_build(&build)?;
        let mode = resolved
            .tag("thinking")
            .expect("every qwen3 build uses the axis");
        println!("  {} (thinking: {mode})", stack.display_name(&build).short);
        for (flag, on) in [("--think", true), ("--no-think", false)] {
            match resolved.presets.bool_member(&thinking, on) {
                Ok(preset) => {
                    let selection = PresetSelection::of([preset.as_str()]);
                    let applied = resolved.presets.select(json!({}), &selection)?;
                    println!(
                        "    {flag:<11} implies --preset {preset}: {}",
                        applied.document
                    );
                }
                Err(e) => println!("    {flag:<11} doesn't apply: {}", e.problem_message()),
            }
        }
    }

    println!("Choosing among the qwen3-lm builds, largest first:");
    let env = SelectEnv::new(&stack);
    let context = SelectionContext {
        defaults: serde_saphyr::from_str("[{order: {desc: parameters}}]")?,
        named: Vec::new(),
    };
    let request = SelectRequest::default().stage(
        serde_saphyr::from_str("{keep: {$contract: qwen3-lm}}")?,
        StageOrigin::Query,
    );
    for (flag, on) in [("--think", true), ("--no-think", false)] {
        let member = bool_axis_member(&stack, "thinking", on)?;
        let presets = PresetSelection::of([member.as_str()]);
        let prefix = preset_prefix(&presets, vec![flag.to_string()]);
        let (build, run) = context.choose(&env, None, prefix, &request)?;
        let applied = build.build.presets.select(json!({}), &presets)?;
        println!(
            "  {flag:<11} chooses {} (the prefix dropped {}): {}",
            stack.display_name(&build.build.name).short,
            run.trace.stages[0].dropped,
            applied.document
        );
    }
    Ok(())
}
