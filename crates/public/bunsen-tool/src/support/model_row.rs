//! A checkpoint's row, as the `list` commands print it.

use bunsen::data::{
    catalog::ResolvedBuild,
    resources::ResourceCache,
};
use bunsen_app::cache::human_bytes;

use crate::support::preset_lines;

/// One row of a checkpoint `list` (`chat`'s, `decide`'s): its id, aliases,
/// description and origin, and each file's state in `cache` and path in its
/// repo; under `urls`, its first source too.
///
/// ```text
/// catalog:nanochat-students/rl-d20 (default)
///   aliases: nanochat-d20-rl
///   nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat
///   origin: https://huggingface.co/nanochat-students/rl-d20
///   checkpoint: [cached]
///     path: model_000650.pt
///     url: https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a8.../model_000650.pt
/// ```
pub fn print_model_row(
    cache: &ResourceCache,
    id: &str,
    row: &ResolvedBuild,
    default: &str,
    urls: bool,
) {
    let default = if row.name == default {
        " (default)"
    } else {
        ""
    };
    println!();
    println!("{id}{default}");
    if !row.aliases.is_empty() {
        let aliases: Vec<&str> = row.aliases.iter().map(|a| a.leaf()).collect();
        println!("  aliases: {}", aliases.join(", "));
    }
    println!("  {}", row.description);
    let resources = row.resource_set();
    if let Some(size) = resources.size() {
        println!("  size: {}", human_bytes(size));
    }
    let dtypes = resources.dtypes();
    if !dtypes.is_empty() {
        println!("  dtype: {}", dtypes.join(", "));
    }
    if let Some(origin) = &row.origin {
        println!("  origin: {origin}");
    }
    for line in preset_lines(&row.presets) {
        println!("{line}");
    }
    for (key, r) in &resources.resources {
        println!("  {key}: [{}]", cache.status(r));
        println!("    path: {}", r.file);
        if urls && !r.sources.is_empty() {
            println!("    {}", r.sources[0]);
        }
    }
}
