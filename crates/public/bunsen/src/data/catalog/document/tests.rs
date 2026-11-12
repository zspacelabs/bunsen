use serde_json::Value;

use super::*;

const SAMPLE: &str = r#"
family: { $kind: facet, $description: "A line of related architectures." }
thinking: { $kind: axis, $axis: bool, $description: "Whether the model thinks first." }
qwen:
  $kind: namespace
  $description: "Qwen."
  qwen3-4b:
    $kind: build
    $description: "Qwen3-4B."
    $join: [qwen/qwen3-line, { $group: qwen/qwen3-releases, $released: "2025-04" }]
    $resources: { config: qwen/qwen3-4b/config }
    parameters: 4000000000
    $presets:
      $axes:
        thinking: { $offers: [thinking], $default: thinking }
      thinking: { $merge: replace, $description: "Think.", $patch: { a: 1 } }
    config:
      $kind: resource
      $file: config.json
      $format: json
      $sources: [{ $hub: { $repo: Qwen/Qwen3-4B, $revision: abc } }, { $url: "https://x/y" }]
"#;

#[test]
fn test_yaml_parses_every_kind_written() {
    let doc = BunsenCatalog::from_yaml(SAMPLE).unwrap();
    let kinds: Vec<EntryKind> = doc.entries.iter().map(CatalogEntry::kind).collect();
    assert_eq!(
        kinds,
        [EntryKind::Facet, EntryKind::Axis, EntryKind::Namespace]
    );
    let CatalogEntry::Build(build) = &doc.entries[2].children()[0] else {
        panic!("a build")
    };
    assert_eq!(build.join.len(), 2);
    assert!(matches!(build.join[1], Membership::Release(_)));
    assert_eq!(build.presets.axes[0].offers, ["thinking"]);
    assert_eq!(build.presets.presets[0].merge, MergeMode::Replace);
    let CatalogEntry::Resource(res) = &build.children[0] else {
        panic!("a resource")
    };
    assert_eq!(res.sources.len(), 2);
}

#[test]
fn test_yaml_and_json_round_trip() {
    let doc = BunsenCatalog::from_yaml(SAMPLE).unwrap();
    let json = doc.to_json().unwrap();
    assert_eq!(BunsenCatalog::from_json(&json).unwrap(), doc);
    let yaml = doc.to_yaml().unwrap();
    assert_eq!(BunsenCatalog::from_yaml(&yaml).unwrap(), doc);
}

/// A templated resource's fields survive the keyed form.
#[test]
fn test_a_templated_resource_round_trips() {
    let yaml = r#"
shards:
  $kind: resource
  $file: "shard_{index}.parquet"
  $format: parquet
  $index_width: 5
  $count: 2
  $unpinned: true
  $sources: [{ $hub: { $repo: o/set, $revision: abc, $dataset: true } }]
"#;
    let doc = BunsenCatalog::from_yaml(yaml).unwrap();
    let written = doc.to_node().to_yaml().unwrap();
    assert!(written.contains("$index_width: 5"), "{written}");
    assert!(written.contains("$count: 2"), "{written}");
    assert!(written.contains("$unpinned: true"), "{written}");
    assert!(written.contains("$dataset: true"), "{written}");
    assert_eq!(BunsenCatalog::from_yaml(&written).unwrap(), doc);
}

#[test]
fn test_unknown_fields_and_kinds_are_refused() {
    let unknown_field = "f: { $kind: facet, $description: \"d\", $extra: 1 }\n";
    assert!(BunsenCatalog::from_yaml(unknown_field).is_err());
    let unknown_kind = "f: { $kind: family, $description: \"d\" }\n";
    assert!(BunsenCatalog::from_yaml(unknown_kind).is_err());
}

fn refused(yaml: &str) -> String {
    BunsenCatalog::from_yaml(yaml).unwrap_err().to_string()
}

#[test]
fn test_a_bare_key_is_a_child_or_a_tag_by_kind_and_shape() {
    let doc = BunsenCatalog::from_yaml(
        r#"
g:
  $kind: namespace
  $description: "G."
  b:
    $kind: build
    $description: "B."
    parameters: 7
    scores: { mmlu: 0.5 }                 # a map without `$kind`: a tag
    weights: { $kind: resource, $file: w, $format: f, $sources: [] }
  t:
    $kind: template_class
    $description: "T."
    language: [en]
"#,
    )
    .unwrap();
    let g = &doc.entries[0];
    assert_eq!(g.children().len(), 2);
    let CatalogEntry::Build(b) = &g.children()[0] else {
        panic!("a build")
    };
    let tags: Vec<String> = b.tags.keys().map(ToString::to_string).collect();
    assert_eq!(tags, ["parameters", "scores"]);
    assert_eq!(b.children.len(), 1);
    assert_eq!(b.children[0].name(), "weights");
    let CatalogEntry::TemplateClass(t) = &g.children()[1] else {
        panic!("a template class")
    };
    assert_eq!(t.tags.len(), 1);
}

#[test]
fn test_the_keyed_form_refuses_what_it_cant_read() {
    assert!(refused("g: { $description: \"No kind.\" }\n").contains("no `$kind`"));
    assert!(
        refused("g:\n  $kind: namespace\n  $description: \"G.\"\n  child: { $file: x }\n")
            .contains("no `$kind`")
    );
    assert!(
        refused("t: { $kind: tag, $type: $bool, $description: \"T.\", extra: 1 }\n")
            .contains("a tag has no bare keys")
    );
    assert!(refused("$entries: []\n").contains("the catalog has no `$entries`"));
    assert!(
        refused("b: { $kind: build, $description: \"B.\", $presets: { $base: {} } }\n")
            .contains("`$presets` has no `$base`")
    );
    let repeated = refused("a: { $kind: facet, $description: \"A.\" }\na: { $kind: facet }\n");
    assert!(
        repeated.contains("repeated") || repeated.contains("uplicate"),
        "{repeated}"
    );
}

#[test]
fn test_the_file_order_is_kept() {
    let doc = BunsenCatalog::from_yaml(
        r#"
zeta: { $kind: facet, $description: "Z." }
alpha:
  $kind: namespace
  $description: "A."
  y: { $kind: namespace, $description: "Y." }
  b: { $kind: namespace, $description: "B." }
c:
  $kind: template_class
  $description: "C."
  $presets:
    $axes:
      zz: { $offers: [zz], $default: zz }
      aa: { $offers: [aa], $default: aa }
    second: {}
    first: {}
"#,
    )
    .unwrap();
    let names: Vec<&str> = doc.entries.iter().map(CatalogEntry::name).collect();
    assert_eq!(names, ["zeta", "alpha", "c"]);
    let kids: Vec<&str> = doc.entries[1]
        .children()
        .iter()
        .map(CatalogEntry::name)
        .collect();
    assert_eq!(kids, ["y", "b"]);
    let CatalogEntry::TemplateClass(c) = &doc.entries[2] else {
        panic!("a template class")
    };
    let axes: Vec<String> = c.presets.axes.iter().map(|a| a.axis.to_string()).collect();
    assert_eq!(axes, ["zz", "aa"]);
    let presets: Vec<&str> = c.presets.presets.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(presets, ["second", "first"]);
    // Written back, the order holds.
    let again = BunsenCatalog::from_yaml(&doc.to_yaml().unwrap()).unwrap();
    assert_eq!(again, doc);
    assert!(doc.to_yaml().unwrap().starts_with("zeta:"));
}

#[test]
fn test_an_editor_reads_changes_and_writes_a_tree() {
    let mut node = OrderedNode::from_yaml(SAMPLE).unwrap();
    let OrderedNode::Map(entries) = &mut node else {
        panic!("a map")
    };
    // Move the namespace first; the catalog reads it in the new order.
    let qwen = entries.pop().unwrap();
    entries.insert(0, qwen);
    let doc = BunsenCatalog::from_node(node.clone()).unwrap();
    assert_eq!(doc.entries[0].name(), "qwen");
    // Written back, the entries keep their order; each entry's fields
    // come in their declared order.
    assert_eq!(BunsenCatalog::from_node(doc.to_node()).unwrap(), doc);
    assert_eq!(
        doc.entries[0].to_node().get("$kind"),
        Some(&OrderedNode::Scalar(Value::String("namespace".into())))
    );
    assert!(node.get("qwen").is_some_and(OrderedNode::is_entry));
    let yaml = node.to_yaml().unwrap();
    assert_eq!(OrderedNode::from_yaml(&yaml).unwrap(), node);
    assert!(OrderedNode::from_yaml("a: 1\na: 2\n").is_err());
}

#[test]
fn test_bool_axis_members_and_negation() {
    let axis = AxisEntry {
        name: "thinking".into(),
        merge: MergeMode::Deny,
        description: "d".into(),
        axis: AxisKind::Bool,
        members: vec![],
    };
    assert_eq!(axis.member_names(), ["thinking", "no-thinking"]);
    assert_eq!(axis.negate("thinking").as_deref(), Some("no-thinking"));
    assert_eq!(axis.negate("no-thinking").as_deref(), Some("thinking"));
    assert_eq!(axis.negate("other"), None);

    let task = AxisEntry {
        name: "task".into(),
        merge: MergeMode::Deny,
        description: "d".into(),
        axis: AxisKind::Categorical,
        members: vec!["transcribe".into(), "translate".into()],
    };
    assert_eq!(task.negate("transcribe"), None);
}
