//! A supplemental catalog on top of the built-in stack: what its extending
//! occurrences assume, how lookups fall through to the declarations below,
//! and what is refused.
//!
//! `cargo run -p pretrained-catalog --example supplemental`

use pretrained_catalog::{
    CatalogName,
    fixtures,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stack = fixtures::builtin_stack()?;
    stack.push_yaml("supplemental", fixtures::SUPPLEMENTAL)?;

    let qwen3 = CatalogName::parse("qwen/qwen3")?;
    println!("occurrences of `{qwen3}`:");
    for occurrence in stack.occurrences(&qwen3) {
        let what = match occurrence.entry.merge() {
            pretrained_catalog::document::MergeMode::Deny => "declaring",
            pretrained_catalog::document::MergeMode::Update => "extending",
            pretrained_catalog::document::MergeMode::Replace => "replacing",
        };
        println!(
            "  {what:<9} in {}",
            stack.layers()[occurrence.layer].label()
        );
    }
    println!("its children, collected from every layer:");
    for child in stack.children(&qwen3) {
        println!("  {} ({})", child.name().leaf(), child.kind());
    }

    let added = stack.find_build("qwen3-1.7b")?;
    let resolved = stack.resolve_build(&added)?;
    println!(
        "\n`{added}` inherits {:?}, and is in {:?}",
        resolved
            .template_chain
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        resolved
            .memberships
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    println!("\nrefused supplements:");
    let bad = [
        (
            "redeclares a publisher",
            "entries:\n  - { kind: namespace, name: qwen, description: \"Mine.\" }\n",
        ),
        (
            "expects the wrong facet",
            "entries:\n  - { kind: namespace, name: qwen, merge: update, facet: family }\n",
        ),
    ];
    for (why, yaml) in bad {
        let errors = stack.push_yaml("bad", yaml).unwrap_err();
        println!("  {why}:");
        for error in &errors.0 {
            println!("    {error}");
        }
    }
    println!("\nthe stack still has {} layers", stack.layers().len());
    Ok(())
}
