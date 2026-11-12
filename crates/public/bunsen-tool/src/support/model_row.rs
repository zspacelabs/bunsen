//! A checkpoint's row, as the `list` commands print it.

use bunsen::data::{
    builds::Pretrained,
    resources::PretrainedCache,
};
use bunsen_app::cache::human_bytes;

use crate::support::preset_lines;

/// One row of a checkpoint `list` (`chat`'s, `decide`'s): its id, aliases,
/// description and origin, and each file's state in `cache` and path in its
/// repo; under `urls`, its first source too.
///
/// ```text
/// well-known:nanochat-students/rl-d20 (default)
///   aliases: nanochat-d20-rl
///   nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat
///   origin: https://huggingface.co/nanochat-students/rl-d20
///   checkpoint: [cached]
///     path: model_000650.pt
///     url: https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a8.../model_000650.pt
/// ```
pub fn print_model_row(
    cache: &PretrainedCache,
    id: &str,
    row: &Pretrained,
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
        println!("  aliases: {}", row.aliases.join(", "));
    }
    println!("  {}", row.description);
    if let Some(size) = row.resources.size() {
        println!("  size: {}", human_bytes(size));
    }
    let dtypes = row.resources.dtypes();
    if !dtypes.is_empty() {
        println!("  dtype: {}", dtypes.join(", "));
    }
    if let Some(origin) = &row.origin {
        println!("  origin: {origin}");
    }
    for line in preset_lines(&row.presets) {
        println!("{line}");
    }
    for (key, r) in &row.resources.resources {
        println!("  {key}: [{}]", cache.status(r));
        println!("    path: {}", r.file);
        if urls && !r.sources.is_empty() {
            println!("    {}", r.sources[0]);
        }
    }
}
