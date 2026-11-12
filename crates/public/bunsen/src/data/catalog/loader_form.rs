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
        ResourceSource::Hub { repo, revision } => Source::Url(format!(
            "https://huggingface.co/{repo}/resolve/{revision}/{file}"
        )),
        ResourceSource::LocalDir { name, default } => Source::LocalDir {
            name: name.clone(),
            dir: default.iter().find_map(|t| resolve_dir_template(t)),
        },
    }
}
