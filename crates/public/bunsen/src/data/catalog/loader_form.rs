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
    /// its sources made concrete (a base URL joined to the file's path, a Hub
    /// repo's file at each of its [`hub_origins`](Self::hub_origins) in order,
    /// a local dir resolved from its templates, now).
    pub fn resource_set(&self) -> ResourceSet {
        let mut set = ResourceSet::new(self.name.to_string());
        set.description = self.description.clone();
        set.license = self.license.clone();
        set.origin = self.origin.clone();
        for (slot, bound) in &self.resources {
            let name = bound.resource.to_string();
            for resource in bound.entry.to_resources(slot, &name, &self.hub_origins) {
                set = set.with_resource(resource);
            }
        }
        set
    }
}

impl ResourceEntry {
    /// This resource, named `name`, as a set named `name` that holds it in
    /// `slot`, its Hub sources at each of `origins`.
    pub(crate) fn to_slot_set(
        &self,
        name: &str,
        slot: &str,
        origins: &[String],
    ) -> ResourceSet {
        let mut set = ResourceSet::new(name);
        if let Some(description) = &self.description {
            set.description = description.clone();
        }
        for r in self.to_resources(slot, name, origins) {
            set = set.with_resource(r);
        }
        set
    }
}

impl ResourceEntry {
    /// The resources a slot bound to this entry is: one, or a sharded
    /// resource's index and shards.
    fn to_resources(
        &self,
        slot: &str,
        name: &str,
        origins: &[String],
    ) -> Vec<Resource> {
        let resource =
            |slot: String, file: &str, sha256: Option<&String>, size, format: &str, sources| {
                Resource {
                    slot,
                    file: file.to_string(),
                    sha256: sha256.cloned(),
                    format: Some(format.to_string()),
                    size,
                    dtype: self.dtype.clone(),
                    name: name.to_string(),
                    sources,
                }
            };
        let sources_of = |sources: &mut dyn Iterator<Item = &ResourceSource>, file: &str| {
            sources
                .flat_map(|s| to_sources(s, file, origins))
                .collect::<Vec<_>>()
        };
        if let (Some(width), Some(count)) = (self.index_width, self.count) {
            // A templated resource: its files, each in a slot numbered by its
            // index, pinned by `$shards` when the catalog lists them.
            return (0..count)
                .map(|index| {
                    let pinned = self.shards.get(index);
                    let file = self.templated_file(index);
                    let sources = match pinned {
                        Some(shard) => sources_of(&mut self.shard_sources(shard), &file),
                        None => {
                            sources_of(&mut self.sources.iter().filter(|s| s.is_place()), &file)
                        }
                    };
                    resource(
                        format!("{slot}.{index:0width$}"),
                        &file,
                        pinned.map(|s| &s.sha256),
                        pinned.map(|s| s.size),
                        &self.format,
                        sources,
                    )
                })
                .collect();
        }
        if self.shards.is_empty() {
            return alloc::vec![resource(
                slot.to_string(),
                &self.file,
                self.sha256.as_ref(),
                self.size,
                &self.format,
                sources_of(&mut self.sources.iter(), &self.file),
            )];
        }
        let shard_format = self.format.strip_suffix("-index").unwrap_or(&self.format);
        let mut out = alloc::vec![resource(
            format!("{slot}.index"),
            &self.file,
            self.sha256.as_ref(),
            self.size,
            &self.format,
            sources_of(&mut self.sources.iter(), &self.file),
        )];
        for (n, shard) in self.shards.iter().enumerate() {
            out.push(resource(
                format!("{slot}.{:05}", n + 1),
                &shard.file,
                Some(&shard.sha256),
                Some(shard.size),
                shard_format,
                sources_of(&mut self.shard_sources(shard), &shard.file),
            ));
        }
        out
    }
}

/// A catalog source as the sources of `file`: a URL as it is; a base URL
/// with the file's path appended; a Hub repo's file at each of `origins` (the
/// Hub mirror set), in order, or at the Hub's own origin when there are none;
/// a local dir resolved from its templates.
fn to_sources(
    source: &ResourceSource,
    file: &str,
    origins: &[String],
) -> Vec<Source> {
    match source {
        ResourceSource::Url(url) => alloc::vec![Source::Url(url.clone())],
        ResourceSource::BaseUrl(base) => {
            alloc::vec![Source::Url(format!(
                "{}/{file}",
                base.trim_end_matches('/')
            ))]
        }
        ResourceSource::Hub {
            repo,
            revision,
            dataset,
        } => {
            let hub = HubRepo::new(repo.as_str(), revision.as_str());
            let hub = if *dataset { hub.dataset() } else { hub };
            if origins.is_empty() {
                alloc::vec![Source::Url(hub.file_url(file))]
            } else {
                origins
                    .iter()
                    .map(|origin| Source::Url(hub.file_url_at(origin, file)))
                    .collect()
            }
        }
        ResourceSource::LocalDir { name, default } => alloc::vec![Source::LocalDir {
            name: name.clone(),
            dir: default.iter().find_map(|t| resolve_dir_template(t)),
        }],
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

    /// A Hub source is had from each mirror the catalog declares, in order:
    /// a layer above adds a mirror after the ones below.
    #[test]
    fn test_a_hub_source_is_had_from_each_mirror() {
        let mut stack = BunsenCatalogStack::new();
        stack
            .push_yaml(
                "prelude",
                &format!(
                    r#"
$contracts:
{}
  one-file:
    $kind: contract
    $description: "One file."
    $resources: {{ file: {{ $format: [json] }} }}
template_class_root: {{ $kind: template_class, $description: "The root." }}
"#,
                    crate::data::resources::hub::HF_MIRROR_CONTRACT
                        .lines()
                        .filter(|l| !l.starts_with('#'))
                        .map(|l| format!("  {l}\n"))
                        .collect::<String>()
                ),
            )
            .unwrap();
        let mirror = |name: &str, origin: &str| {
            format!(
                "{name}:\n  $kind: build\n  $description: \"A mirror.\"\n  $contract: hf-mirror\n  \
                 $args: {{ origin: \"{origin}\" }}\n"
            )
        };
        stack
            .push_yaml(
                "hub",
                &format!(
                    "{}m:\n  $kind: build\n  $description: \"M.\"\n  $contract: one-file\n  \
                     $resources: {{ file: file }}\n  file:\n    $kind: resource\n    $file: \
                     config.json\n    $format: json\n    $sources: [{{ $hub: {{ $repo: o/m, \
                     $revision: abc }} }}]\n",
                    mirror("hub", "https://huggingface.co")
                ),
            )
            .unwrap();
        stack
            .push_yaml("mirror", &mirror("other", "https://mirror.example"))
            .unwrap();
        assert_eq!(
            stack.hub_origins(),
            ["https://huggingface.co", "https://mirror.example"]
        );
        let set = stack
            .resolve_build(&CatalogName::parse("m").unwrap())
            .unwrap()
            .resource_set();
        assert_eq!(
            set.get("file").unwrap().urls(),
            vec![
                "https://huggingface.co/o/m/resolve/abc/config.json",
                "https://mirror.example/o/m/resolve/abc/config.json",
            ]
        );
    }

    /// The built-in mirror set is the Hub's own origin first.
    #[test]
    fn test_the_builtin_mirror_set_starts_at_the_hub() {
        let catalog = crate::data::catalog::BuiltinCatalog::get().unwrap();
        assert_eq!(
            catalog.hub_origins().first().map(String::as_str),
            Some(crate::data::resources::hub::HUB_ORIGIN)
        );
    }
}
