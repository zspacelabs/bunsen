//! Catalog resources in the loader's form: `ResolvedBuild::resource_set`.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use crate::data::{
    catalog::{
        ResolvedBuild,
        document::{
            ResourceEntry,
            ResourceSource,
        },
    },
    resources::{
        Resource,
        ResourceSet,
        Source,
        hub::HubRepo,
        resolve_dir_template,
    },
};

impl ResolvedBuild {
    /// The build's resources in the loader's form, a [`ResourceSet`] named
    /// for the build: each bound slot's resource under its full name, a
    /// sharded one as `{slot}.index` and `{slot}.{n:05}` (numbered from 1),
    /// its sources made concrete (a base URL or a Hub repo joined to the
    /// file's path, a local dir resolved from its templates, now).
    pub fn resource_set(&self) -> ResourceSet {
        let mut set = ResourceSet::new(self.name.to_string());
        set.description = self.description.clone();
        set.license = self.license.clone();
        set.origin = self.origin.clone();
        for (slot, bound) in &self.resources {
            for resource in slot_resources(slot, &bound.entry, &bound.resource.to_string()) {
                set = set.with_resource(resource);
            }
        }
        set
    }
}

/// The resource `entry`, named `name`, as a set named `name` that holds it in
/// `slot`.
pub(crate) fn entry_resource_set(
    name: &str,
    entry: &ResourceEntry,
    slot: &str,
) -> ResourceSet {
    let mut set = ResourceSet::new(name);
    if let Some(description) = &entry.description {
        set.description = description.clone();
    }
    for r in slot_resources(slot, entry, name) {
        set = set.with_resource(r);
    }
    set
}

/// The resources a slot bound to `entry` is: one, or a sharded resource's
/// index and shards.
fn slot_resources(
    slot: &str,
    entry: &ResourceEntry,
    name: &str,
) -> Vec<Resource> {
    let resource =
        |slot: String, file: &str, sha256: Option<&String>, size, format: &str, sources| Resource {
            slot,
            file: file.to_string(),
            sha256: sha256.cloned(),
            format: Some(format.to_string()),
            size,
            dtype: entry.dtype.clone(),
            name: name.to_string(),
            sources,
        };
    let sources_of = |sources: &mut dyn Iterator<Item = &ResourceSource>, file: &str| {
        sources.map(|s| to_source(s, file)).collect::<Vec<_>>()
    };
    if let (Some(width), Some(count)) = (entry.index_width, entry.count) {
        // A templated resource: its files, each in a slot numbered by its
        // index, pinned by `$shards` when the catalog lists them.
        return (0..count)
            .map(|index| {
                let pinned = entry.shards.get(index);
                let file = entry.templated_file(index);
                let sources = match pinned {
                    Some(shard) => sources_of(&mut entry.shard_sources(shard), &file),
                    None => sources_of(&mut entry.sources.iter().filter(|s| s.is_place()), &file),
                };
                resource(
                    format!("{slot}.{index:0width$}"),
                    &file,
                    pinned.map(|s| &s.sha256),
                    pinned.map(|s| s.size),
                    &entry.format,
                    sources,
                )
            })
            .collect();
    }
    if entry.shards.is_empty() {
        return alloc::vec![resource(
            slot.to_string(),
            &entry.file,
            entry.sha256.as_ref(),
            entry.size,
            &entry.format,
            sources_of(&mut entry.sources.iter(), &entry.file),
        )];
    }
    let shard_format = entry.format.strip_suffix("-index").unwrap_or(&entry.format);
    let mut out = alloc::vec![resource(
        format!("{slot}.index"),
        &entry.file,
        entry.sha256.as_ref(),
        entry.size,
        &entry.format,
        sources_of(&mut entry.sources.iter(), &entry.file),
    )];
    for (n, shard) in entry.shards.iter().enumerate() {
        out.push(resource(
            format!("{slot}.{:05}", n + 1),
            &shard.file,
            Some(&shard.sha256),
            Some(shard.size),
            shard_format,
            sources_of(&mut entry.shard_sources(shard), &shard.file),
        ));
    }
    out
}

/// A catalog source as a source of `file`: a URL as it is; a base URL or a
/// Hub repo with the file's path appended; a local dir resolved from its
/// templates.
fn to_source(
    source: &ResourceSource,
    file: &str,
) -> Source {
    match source {
        ResourceSource::Url(url) => Source::Url(url.clone()),
        ResourceSource::BaseUrl(base) => {
            Source::Url(format!("{}/{file}", base.trim_end_matches('/')))
        }
        ResourceSource::Hub {
            repo,
            revision,
            dataset,
        } => {
            let hub = HubRepo::new(repo.as_str(), revision.as_str());
            let hub = if *dataset { hub.dataset() } else { hub };
            Source::Url(hub.file_url(file))
        }
        ResourceSource::LocalDir { name, default } => Source::LocalDir {
            name: name.clone(),
            dir: default.iter().find_map(|t| resolve_dir_template(t)),
        },
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::data::catalog::{
        BunsenCatalogStack,
        CatalogName,
    };

    /// A templated resource is a slot per file, numbered by its index: pinned
    /// from `$shards`, or, unpinned, each had from the resource's places.
    #[test]
    fn test_a_templated_resource_is_a_slot_per_file() {
        let mut stack = BunsenCatalogStack::new();
        stack
            .push_yaml(
                "prelude",
                r#"
$contracts:
  data-set:
    $kind: contract
    $description: "Numbered files."
    $resources: { shards: { $format: [parquet] } }
template_class_root: { $kind: template_class, $description: "The root." }
"#,
            )
            .unwrap();
        stack
            .push_yaml(
                "data",
                &format!(
                    r#"
data:
  $kind: namespace
  $description: "Data."
  set:
    $kind: build
    $description: "A set."
    $contract: data-set
    $resources: {{ shards: shards }}
    shards:
      $kind: resource
      $file: "shard_{{index}}.parquet"
      $format: parquet
      $index_width: 3
      $count: 2
      $sources: [{{ $hub: {{ $repo: o/set, $revision: abc, $dataset: true }} }}]
      $shards:
        - {{ $file: shard_000.parquet, $sha256: "{a}", $size: 9 }}
        - {{ $file: shard_001.parquet, $sha256: "{b}", $size: 8 }}
  loose:
    $kind: build
    $description: "An unpinned set."
    $contract: data-set
    $resources: {{ shards: shards }}
    shards:
      $kind: resource
      $file: "shard_{{index}}.parquet"
      $format: parquet
      $index_width: 3
      $count: 2
      $unpinned: true
      $sources: [{{ $base_url: "https://data.example/set" }}]
"#,
                    a = "a".repeat(64),
                    b = "b".repeat(64)
                ),
            )
            .unwrap();
        let n = |name: &str| CatalogName::parse(name).unwrap();

        let set = stack.resolve_build(&n("data/set")).unwrap().resource_set();
        assert_eq!(set.keys(), ["shards.000", "shards.001"]);
        let first = set.get("shards.000").unwrap();
        assert_eq!(first.file, "shard_000.parquet");
        assert_eq!(first.sha256, Some("a".repeat(64)));
        assert_eq!(first.size, Some(9));
        assert_eq!(first.name, "data/set/shards");
        assert_eq!(
            first.urls(),
            vec!["https://huggingface.co/datasets/o/set/resolve/abc/shard_000.parquet"]
        );

        let loose = stack
            .resolve_build(&n("data/loose"))
            .unwrap()
            .resource_set();
        let second = loose.get("shards.001").unwrap();
        assert_eq!(second.sha256, None);
        assert_eq!(
            second.urls(),
            vec!["https://data.example/set/shard_001.parquet"]
        );
    }
}
