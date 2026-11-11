//! Loads bunsen's example catalogs (`build_catalog/fixtures/`) from their
//! files as a stack, and walks the queries: layers, facets ("all families"
//! against "things in families"), builds by their shortest names, and a
//! release sequence.
//!
//! `cargo run -p pretrained-catalog --example catalog_tour`

use std::path::PathBuf;

use bunsen::data::pretrained::build_catalog::{
    CatalogName,
    PretrainedCatalogStack,
    document::EntryKind,
    lookup::LeafMatch,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../public/bunsen/src/data/pretrained/build_catalog/fixtures");
    let mut stack = PretrainedCatalogStack::new();
    for file in ["prelude.yml", "commons.yml", "qwen3.yml", "whisper.yml"] {
        stack.push_file(dir.join(file))?;
    }

    println!("layers, bottom first:");
    for layer in stack.layers() {
        println!("  {:<8} {} entries", layer.label(), layer.entries().len());
    }

    let family = CatalogName::parse("family")?;
    println!("\nall families (the groups of the facet `family`):");
    for group in stack.facet_groups(&family) {
        println!("  {}", group.name());
    }
    println!("\nthings in families (builds in some `family` group):");
    for build in stack.builds_in_facet(&family) {
        let shown = stack.display_name(&build);
        let resolved = stack.resolve_build(&build)?;
        println!(
            "  {:<24} {:<32} family {}",
            shown.short,
            shown.full.to_string(),
            resolved.facets[&family]
        );
    }

    println!("\nresources, by their shortest unique names:");
    for resource in stack.of_kind(EntryKind::Resource) {
        println!("  {}", stack.display_name(resource.name()).short);
    }

    let large = stack.find("large/release", &[EntryKind::Release], LeafMatch::Exact)?;
    println!("\nthe release sequence `{large}`, oldest first:");
    for (build, released) in stack.releases(&large)? {
        println!("  {released:<12} {}", build.leaf());
    }
    println!("latest: {}", stack.latest(&large)?.expect("has members"));
    Ok(())
}
