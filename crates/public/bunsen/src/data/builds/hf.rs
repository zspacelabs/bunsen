//! Hugging Face as a pretrained provider: `hf:org/repo`.

use alloc::{
    collections::BTreeSet,
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

#[cfg(feature = "cache")]
use crate::data::resources::ResourceCache;
use crate::{
    data::{
        builds::BuildProvider,
        catalog::{
            CatalogName,
            ResolvedBuild,
            document::{
                ResourceEntry,
                ResourceShard,
                ResourceSource,
            },
        },
        resources::{
            Resource,
            SAFETENSORS,
            SAFETENSORS_INDEX,
            Source,
            hub::{
                HUB_MAIN,
                HUB_ORIGIN,
                HubEntry,
                HubRepo,
            },
        },
    },
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        ParseError,
    },
};

/// The provider's default name: the `hf` of `hf:openai/whisper-tiny`, and
/// the root its files are cached under.
pub const HF: &str = "hf";

/// The file a `transformers` repo keeps its weights in, when they fit one.
pub const HF_SINGLE_FILE: &str = "model.safetensors";

/// The index of a sharded checkpoint: the shard each tensor is in.
pub const HF_INDEX_FILE: &str = "model.safetensors.index.json";

/// The model's config, as `transformers` writes it.
pub const HF_CONFIG_FILE: &str = "config.json";

/// The key `config.json` rides under.
pub const CONFIG: &str = "config";

/// The key the checkpoint is under unless the kit names another.
pub const DEFAULT_CHECKPOINT_KEY: &str = "checkpoint";

/// The shard number and count of `model-00001-of-00002.safetensors`.
fn shard_numbers(file: &str) -> Option<(usize, usize)> {
    let rest = file.strip_prefix("model-")?.strip_suffix(".safetensors")?;
    let (n, of) = rest.split_once("-of-")?;
    Some((n.parse().ok()?, of.parse().ok()?))
}

/// `; missing model-00003-of-00003.safetensors`: the shards of `1..=of`
/// that `shards` lacks, at most eight of them by name, or nothing when it
/// lacks none.
fn missing_shards(
    of: usize,
    shards: &[(usize, usize, &HubEntry)],
) -> String {
    let present: BTreeSet<usize> = shards
        .iter()
        .map(|(n, _, _)| *n)
        .filter(|n| (1..=of).contains(n))
        .collect();
    let count = of - present.len();
    if count == 0 {
        return String::new();
    }
    let named: Vec<String> = (1..=of)
        .filter(|i| !present.contains(i))
        .take(8)
        .map(|i| format!("model-{i:05}-of-{of:05}.safetensors"))
        .collect();
    let more = if count > named.len() {
        format!(", and {} more", count - named.len())
    } else {
        String::new()
    };
    format!("; missing {}{more}", named.join(", "))
}

/// Hugging Face repos, by ref, as safetensors checkpoints: the `hf:`
/// provider.
///
/// `hf:{org}/{repo}` is a Hugging Face repo: `hf:openai/whisper-large-v3`
/// is <https://huggingface.co/openai/whisper-large-v3>. The provider
/// answers such a ref with a row whose resource set is the repo's
/// safetensors checkpoint, and it lists nothing: the hub is not enumerable, and
/// a bare name never reaches it
/// ([`answers_bare_names`](BuildProvider::answers_bare_names) is
/// `false`). A kit registers one in its default factory, after its own
/// tables and under its own checkpoint key
/// ([`with_checkpoint_key`](Self::with_checkpoint_key)); Whisper's does.
/// The kit's hook then reads the row with
/// [`SafetensorsCheckpoint`](crate::data::formats::SafetensorsCheckpoint).
///
/// # Resolving a ref
///
/// What a repo holds comes from the hub's file listing (its tree API, at
/// [`tree_url`](Self::tree_url)). The listing is fetched once through the
/// [`ResourceCache`](crate::data::resources::ResourceCache) and kept there,
/// so a ref resolved once resolves again offline. The listing names the files
/// and pins each LFS file by its SHA-256, so the checkpoint is
/// digest-addressed in the cache like a catalog row's:
/// `<cache>/resources/hf/<org>/<repo>/<slot>/<sha256>/<file>`, under the
/// repo and the slot its file fills ([`resource_name`](Self::resource_name)),
/// the listing beside them. Fetching the listing
/// needs the `fetch` feature, unless the cache has it already.
///
/// Resolving a ref without a cache is an error, and
/// [`lookup`](BuildProvider::lookup) always refuses: the provider
/// guesses nothing about a repo. A repo that is not there, or a gated one,
/// answers 401, and the error says so.
///
/// # Sharding
///
/// `transformers` saves a model as one `model.safetensors` until it passes
/// a shard-size limit, then as `model-00001-of-0000N.safetensors` and so
/// on, with `model.safetensors.index.json` naming the shard each tensor is
/// in. The row ([`build_from_listing`](Self::build_from_listing)) is one
/// resource, or the index and every shard, under the kit's checkpoint key:
/// `checkpoint`, or `checkpoint.index` and `checkpoint.00001`, and so on.
/// Each is labelled [`SAFETENSORS`] or [`SAFETENSORS_INDEX`] for the kit's
/// reader. A kit whose checkpoints may also come as one archive
/// ([`with_archive_checkpoint`](Self::with_archive_checkpoint): a `.nemo`)
/// gets that file instead from a repo with no safetensors checkpoint,
/// labelled as it asks. `config.json` rides along as `config` when the repo
/// has one.
/// Anything else in the repo (other frameworks' weights, tokenizer files)
/// is not a resource: a kit reads what it knows.
///
/// The shards must be the whole set their names count: every one of
/// `model-00001-of-0000N` to `model-0000N-of-0000N`, with the one `N`. A
/// listing that lacks any is refused, naming them, before anything is
/// fetched. The listing says nothing of what the index holds; that the
/// index names exactly these shards is checked when the checkpoint is read
/// ([`SafetensorsCheckpoint::from_loaded`](crate::data::formats::SafetensorsCheckpoint::from_loaded)).
///
/// # Revisions
///
/// The listing is at a revision: [`HUB_MAIN`] unless another is named
/// ([`with_revision`](Self::with_revision)). A repo whose `main` moves is
/// seen again only when the cache is cleared or the revision is named.
/// [`with_origin`](Self::with_origin) points the provider at a mirror, and
/// [`with_name`](Self::with_name) lets two providers, at two revisions,
/// share a factory.
#[derive(Clone, Debug)]
pub struct HfProvider {
    /// The provider's name, and the root its files are cached under.
    pub name: String,

    /// Where the hub is: [`HUB_ORIGIN`], or a mirror.
    pub origin: String,

    /// The revision every ref resolves at: a branch, a tag, or a commit.
    pub revision: String,

    /// The key the checkpoint goes under: the kit's.
    pub checkpoint_key: String,

    /// The other files a row carries when the repo has them, beyond the
    /// checkpoint and `config.json`: a kit's vocabulary, say.
    pub extra_files: Vec<HfExtraFile>,

    /// The one-file checkpoints a repo without a safetensors checkpoint may
    /// hold instead, in preference order: a `.nemo`, say.
    pub archive_checkpoints: Vec<HfArchiveCheckpoint>,
}

/// A one-file checkpoint an [`HfProvider`] row takes when the repo has no
/// safetensors checkpoint: the repo's one file ending in `extension`,
/// labelled `kind` for the kit's reader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfArchiveCheckpoint {
    /// The file name's ending: `.nemo`.
    pub extension: String,

    /// The resource's `kind`.
    pub kind: String,
}

/// A file an [`HfProvider`] row carries when the repo has it: under `key`,
/// labelled `kind` for the kit's reader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfExtraFile {
    /// The resource key.
    pub key: String,

    /// The file's path in the repo: `tokenizer.json`.
    pub path: String,

    /// The resource's `kind`.
    pub kind: String,
}

impl Default for HfProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl HfProvider {
    /// `hf`, at [`HUB_MAIN`], the checkpoint under `checkpoint`.
    pub fn new() -> Self {
        Self {
            name: HF.to_string(),
            origin: HUB_ORIGIN.to_string(),
            revision: HUB_MAIN.to_string(),
            checkpoint_key: DEFAULT_CHECKPOINT_KEY.to_string(),
            extra_files: Vec::new(),
            archive_checkpoints: Vec::new(),
        }
    }

    /// Repos at a mirror of the hub, which serves the same paths.
    pub fn with_origin(
        mut self,
        origin: impl Into<String>,
    ) -> Self {
        self.origin = origin.into();
        self
    }

    /// Another name: two of these in one factory, at two revisions, say.
    pub fn with_name(
        mut self,
        name: impl Into<String>,
    ) -> Self {
        self.name = name.into();
        self
    }

    /// Repos at `revision`.
    pub fn with_revision(
        mut self,
        revision: impl Into<String>,
    ) -> Self {
        self.revision = revision.into();
        self
    }

    /// The checkpoint under `key`.
    pub fn with_checkpoint_key(
        mut self,
        key: impl Into<String>,
    ) -> Self {
        self.checkpoint_key = key.into();
        self
    }

    /// Rows carry `path` under `key`, labelled `kind`, when the repo has
    /// it: `with_extra_file("vocabulary", "tokenizer.json", "tokenizer.json")`.
    pub fn with_extra_file(
        mut self,
        key: impl Into<String>,
        path: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        self.extra_files.push(HfExtraFile {
            key: key.into(),
            path: path.into(),
            kind: kind.into(),
        });
        self
    }

    /// A repo without a safetensors checkpoint may hold one file ending in
    /// `extension` instead, its checkpoint, labelled `kind`:
    /// `with_archive_checkpoint(".nemo", "nemo")`.
    pub fn with_archive_checkpoint(
        mut self,
        extension: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        self.archive_checkpoints.push(HfArchiveCheckpoint {
            extension: extension.into(),
            kind: kind.into(),
        });
        self
    }

    /// The URL `file` in `org/repo` is served at, at this provider's
    /// revision.
    pub fn url(
        &self,
        org: &str,
        repo: &str,
        file: &str,
    ) -> String {
        HubRepo::new(format!("{org}/{repo}"), &self.revision).file_url_at(&self.origin, file)
    }

    /// The URL of `org/repo`'s file listing at this provider's revision.
    pub fn tree_url(
        &self,
        org: &str,
        repo: &str,
    ) -> String {
        HubRepo::new(format!("{org}/{repo}"), &self.revision).tree_url_at(&self.origin)
    }

    /// The `org` and `repo` of a ref, which is exactly `org/repo`.
    ///
    /// # Errors
    /// [`Illegal`](BunsenErrorKind::Illegal), with a [`ParseError`], for
    /// anything else. A caller that took the ref from outside the program
    /// re-marks it [`as_policy`](BunsenError::as_policy), as
    /// [`BuildFactory::resolve`](crate::data::builds::BuildFactory::resolve)
    /// does.
    pub fn split<'a>(
        &self,
        name: &'a str,
    ) -> BunsenResult<(&'a str, &'a str)> {
        match name.split_once('/') {
            Some((org, repo)) if !org.is_empty() && !repo.is_empty() && !repo.contains('/') => {
                Ok((org, repo))
            }
            _ => Err(BunsenError::from_cause(
                BunsenErrorKind::Illegal,
                ParseError::new("Hugging Face ref")
                    .input(format!("{}:{name}", self.name))
                    .because("a ref is org/repo"),
            )),
        }
    }

    /// The listing resource for `org/repo`: the tree API's JSON, unpinned,
    /// named `{provider}/{org}/{repo}/listing`.
    pub fn listing_resource(
        &self,
        org: &str,
        repo: &str,
    ) -> Resource {
        Resource {
            size: None,
            dtype: None,
            slot: "listing".to_string(),
            file: format!("{org}--{repo}--{}.json", self.revision.replace('/', "-")),
            sha256: None,
            format: Some("json".to_string()),
            name: self.resource_name(org, repo, "listing"),
            sources: vec![Source::Url(self.tree_url(org, repo))],
        }
    }

    /// The full name a resource of `org/repo` in `slot` is cached under:
    /// `{provider}/{org}/{repo}/{slot}` (`hf/openai/whisper-tiny/checkpoint`),
    /// a sharded checkpoint's index and shards under their one slot.
    pub fn resource_name(
        &self,
        org: &str,
        repo: &str,
        slot: &str,
    ) -> String {
        format!("{}/{org}/{repo}/{slot}", self.name)
    }

    /// One file of `org/repo`, as the catalog declares a resource: had from
    /// its URL at this provider's revision, pinned when the listing pins it.
    fn entry(
        &self,
        org: &str,
        repo: &str,
        entry: &HubEntry,
        format: &str,
    ) -> ResourceEntry {
        let mut declared = ResourceEntry::new(
            entry.path.clone(),
            format,
            vec![ResourceSource::Url(self.url(org, repo, &entry.path))],
        );
        declared.sha256 = entry.sha256();
        declared.size = Some(entry.size);
        declared
    }

    /// The catalog name of `org/repo`'s build, or of its resource in
    /// `slot` ([`resource_name`](Self::resource_name)).
    ///
    /// # Errors
    /// [`CatalogRule::Name`](crate::data::catalog::CatalogRule::Name) for an
    /// org or repo that is no name segment under this provider's root.
    fn catalog_name(
        &self,
        org: &str,
        repo: &str,
        slot: Option<&str>,
    ) -> BunsenResult<CatalogName> {
        CatalogName::parse(&match slot {
            Some(slot) => self.resource_name(org, repo, slot),
            None => format!("{}/{org}/{repo}", self.name),
        })
    }

    /// The build for `name` from the repo's file listing, named
    /// `{provider}/{org}/{repo}` (`hf/openai/whisper-tiny`): the safetensors
    /// checkpoint, one file or the index and every shard, or else the first
    /// of the [`archive_checkpoints`](Self::archive_checkpoints) the repo
    /// holds, pinned by the hub's LFS digests; `config.json` when there is
    /// one, and each of the [`extra_files`](Self::extra_files) the repo has.
    ///
    /// # Errors
    /// As [`split`](Self::split) for a malformed ref.
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource), under a frame
    /// naming the ref, when the listing has neither [`HF_SINGLE_FILE`] nor
    /// [`HF_INDEX_FILE`] with a complete set of shards, nor an archive
    /// checkpoint, naming what it has instead, and the shards a set of one
    /// `N` lacks: the repo is not the checkpoint a ref names. So too when it
    /// holds two files of an archive checkpoint's ending.
    pub fn build_from_listing(
        &self,
        name: &str,
        listing: &[HubEntry],
    ) -> BunsenResult<ResolvedBuild> {
        let (org, repo) = self.split(name)?;
        let files: Vec<&HubEntry> = listing.iter().filter(|e| e.is_file()).collect();
        let file = |path: &str| files.iter().copied().find(|e| e.path == path);
        let origin = format!("{}/{org}/{repo}", self.origin);
        let key = self.checkpoint_key.as_str();

        let mut resources: Vec<(&str, ResourceEntry)> = Vec::new();
        let what;
        if let Some(single) = file(HF_SINGLE_FILE) {
            resources.push((key, self.entry(org, repo, single, SAFETENSORS)));
            what = format!("{HF_SINGLE_FILE}, {}", megabytes(single.size));
        } else if let Some(index) = file(HF_INDEX_FILE) {
            let mut shards: Vec<(usize, usize, &HubEntry)> = files
                .iter()
                .filter_map(|e| shard_numbers(&e.path).map(|(n, of)| (n, of, *e)))
                .collect();
            shards.sort_by_key(|(n, _, _)| *n);
            let of = shards.first().map(|(_, of, _)| *of).unwrap_or(0);
            let one_set = shards.iter().all(|(_, o, _)| *o == of);
            let complete = !shards.is_empty()
                && one_set
                && shards.len() == of
                && shards.iter().enumerate().all(|(i, (n, _, _))| *n == i + 1);
            if !complete {
                let has: Vec<&str> = shards.iter().map(|(_, _, e)| e.path.as_str()).collect();
                let mut message = format!(
                    "{HF_INDEX_FILE} with an incomplete set of shards: {}",
                    if has.is_empty() {
                        "none".to_string()
                    } else {
                        has.join(", ")
                    }
                );
                if one_set {
                    message.push_str(&missing_shards(of, &shards));
                }
                return Err(
                    BunsenError::invalid_resource(message).context(format!("{}:{name}", self.name))
                );
            }
            let mut checkpoint = self.entry(org, repo, index, SAFETENSORS_INDEX);
            let mut total = 0;
            for (_, _, entry) in &shards {
                total += entry.size;
                let sha256 = entry.sha256().ok_or_else(|| {
                    BunsenError::invalid_resource(format!(
                        "the shard {} is not in LFS, so nothing pins it",
                        entry.path
                    ))
                    .context(format!("{}:{name}", self.name))
                })?;
                checkpoint.shards.push(ResourceShard {
                    file: entry.path.clone(),
                    sha256,
                    size: entry.size,
                    sources: vec![ResourceSource::Url(self.url(org, repo, &entry.path))],
                });
            }
            resources.push((key, checkpoint));
            what = format!("{} shards, {}", shards.len(), megabytes(total));
        } else if let Some((archive, kind)) = self.archive_checkpoint(name, &files)? {
            resources.push((key, self.entry(org, repo, archive, kind)));
            what = format!("{}, {}", archive.path, megabytes(archive.size));
        } else {
            let weights: Vec<&str> = files
                .iter()
                .map(|e| e.path.as_str())
                .filter(|p| {
                    p.ends_with(".safetensors")
                        || p.ends_with(".bin")
                        || p.ends_with(".h5")
                        || p.ends_with(".msgpack")
                        || p.ends_with(".pt")
                        || p.ends_with(".gguf")
                        || p.ends_with(".nemo")
                })
                .collect();
            let archives: String = self
                .archive_checkpoints
                .iter()
                .map(|a| format!(", or one *{}", a.extension))
                .collect();
            return Err(BunsenError::invalid_resource(format!(
                "no {HF_SINGLE_FILE} or {HF_INDEX_FILE}{archives} at {}; the repo has: {}",
                self.revision,
                if weights.is_empty() {
                    "no weights".to_string()
                } else {
                    weights.join(", ")
                }
            ))
            .context(format!("{}:{name}", self.name)));
        }
        let mut with: Vec<&str> = Vec::new();
        if let Some(config) = file(HF_CONFIG_FILE) {
            resources.push((CONFIG, self.entry(org, repo, config, "json")));
            with.push(HF_CONFIG_FILE);
        }
        for extra in &self.extra_files {
            if let Some(entry) = file(&extra.path) {
                resources.push((&extra.key, self.entry(org, repo, entry, &extra.kind)));
                with.push(&extra.path);
            }
        }
        let with = if with.is_empty() {
            String::new()
        } else {
            format!(", with {}", with.join(" and "))
        };
        let description = format!(
            "{org}/{repo} on Hugging Face at {}: {what}{with}",
            self.revision
        );
        let mut build = ResolvedBuild::new(self.catalog_name(org, repo, None)?, description);
        build.origin = Some(origin);
        for (slot, entry) in resources {
            build = build.with_resource(slot, self.catalog_name(org, repo, Some(slot))?, entry);
        }
        Ok(build)
    }

    /// The first of [`archive_checkpoints`](Self::archive_checkpoints) the
    /// repo holds, with its kind.
    ///
    /// # Errors
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource), under a frame
    /// naming the ref, naming them, when the repo holds more than one file
    /// of that ending: which is the checkpoint is not the provider's guess.
    fn archive_checkpoint<'e>(
        &'e self,
        name: &str,
        files: &[&'e HubEntry],
    ) -> BunsenResult<Option<(&'e HubEntry, &'e str)>> {
        for archive in &self.archive_checkpoints {
            let found: Vec<&HubEntry> = files
                .iter()
                .copied()
                .filter(|e| e.path.ends_with(&archive.extension))
                .collect();
            match found.as_slice() {
                [] => continue,
                [one] => return Ok(Some((one, archive.kind.as_str()))),
                many => {
                    let names: Vec<&str> = many.iter().map(|e| e.path.as_str()).collect();
                    return Err(BunsenError::invalid_resource(format!(
                        "{} {} files, and no way to tell the checkpoint: {}",
                        many.len(),
                        archive.extension,
                        names.join(", ")
                    ))
                    .context(format!("{}:{name}", self.name)));
                }
            }
        }
        Ok(None)
    }

    /// The repo's file listing, through the cache: fetched once, read from
    /// there after.
    #[cfg(feature = "cache")]
    fn listing(
        &self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<Vec<HubEntry>> {
        use crate::errors::{
            LookupError,
            LookupProblem,
            ResultContext,
            sys_at,
        };

        let (org, repo) = self.split(name)?;
        let resource = self.listing_resource(org, repo);
        let resolved = cache.resolve(&resource);
        // The hub answers 401 to an anonymous caller both for a repo that is
        // not there and for a gated one.
        let unauthorized = resolved
            .as_ref()
            .err()
            .and_then(|e| e.find::<LookupError>())
            .is_some_and(|l| l.problem == LookupProblem::Unauthorized);
        let resolved = resolved.with_context(|| {
            if unauthorized {
                format!(
                    "{}:{name}: the repo does not exist, or is gated (bunsen fetches anonymously, so a gated repo is unreachable)",
                    self.name
                )
            } else {
                format!("{}:{name}: file listing", self.name)
            }
        })?;
        let path = &resolved.path;
        let file = std::fs::File::open(path)
            .map_err(sys_at("open", path))
            .with_context(|| format!("{}:{name}: file listing", self.name))?;
        serde_json::from_reader(std::io::BufReader::new(file)).map_err(|e| {
            BunsenError::from_cause(
                BunsenErrorKind::InvalidResource,
                ParseError::new("Hugging Face file listing")
                    .at(path.display())
                    .with_source(e),
            )
            .with_details(format!(
                "the listing is cached unpinned; remove {} to fetch it again",
                path.display()
            ))
            .context(format!("{}:{name}: file listing", self.name))
        })
    }
}

impl BuildProvider for HfProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Hugging Face repos by ref: `hf:org/repo` is its safetensors checkpoint at a revision, pinned by the hub's listing; lists nothing"
    }

    fn origin(&self) -> Option<&str> {
        Some(HUB_ORIGIN)
    }

    /// Nothing: the hub is not enumerable.
    fn list(&self) -> Vec<ResolvedBuild> {
        Vec::new()
    }

    /// An error: what a repo holds is the hub's to say, through a cache.
    ///
    /// # Errors
    /// As [`split`](Self::split) for a malformed ref;
    /// [`Policy`](BunsenErrorKind::Policy) otherwise, always: the request
    /// needs a cache.
    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<Option<ResolvedBuild>> {
        self.split(name)?;
        Err(BunsenError::policy(
            "a Hugging Face ref is resolved through a cache, which asks the hub what the repo holds",
        )
        .context(format!("{}:{name}", self.name)))
    }

    /// The row for `org/repo` from its file listing, fetched once into
    /// `cache` under `kit` and read from there after.
    ///
    /// # Errors
    /// As [`split`](Self::split) and
    /// [`build_from_listing`](Self::build_from_listing). For a listing that
    /// cannot be had, the cache's error under a frame naming the ref: a repo
    /// that is not there, or a gated one, answers 401, a
    /// [`Lookup`](BunsenErrorKind::Lookup) whose
    /// [`LookupError`](crate::errors::LookupError) problem is
    /// [`Unauthorized`](crate::errors::LookupProblem::Unauthorized), and an
    /// offline cache that never saw the repo is a
    /// [`Policy`](BunsenErrorKind::Policy) error. A cached listing that does
    /// not parse is
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource), with a
    /// [`ParseError`].
    #[cfg(feature = "cache")]
    fn resolve(
        &self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<Option<ResolvedBuild>> {
        let listing = self.listing(name, cache)?;
        self.build_from_listing(name, &listing).map(Some)
    }

    /// Never: a bare name means a row bunsen knows.
    fn answers_bare_names(&self) -> bool {
        false
    }
}

/// `151.1 MB` or `6.17 GB`.
fn megabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.1} MB", bytes as f64 / 1e6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::resources::hub::HubLfs,
        errors::testing::{
            ErrorMatcher,
            text,
        },
    };

    fn entry(
        path: &str,
        size: u64,
        sha256: Option<&str>,
    ) -> HubEntry {
        HubEntry {
            kind: "file".to_string(),
            path: path.to_string(),
            size,
            lfs: sha256.map(|oid| HubLfs {
                oid: oid.to_string(),
                size,
            }),
        }
    }

    const TINY_SHA: &str = "7ebd0e69e78190ff5b1b2e6c7c0a6a9b1a4e2f0f0b3d2c1a9f8e7d6c5b4a3f2e";
    const SHARD_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const SHARD_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    /// A single-file repo: the checkpoint pinned by its LFS digest, the
    /// config along, nothing else of the repo.
    #[test]
    fn test_a_single_file_repo_is_one_pinned_checkpoint() {
        let hf = HfProvider::new();
        let listing = vec![
            entry(".gitattributes", 1438, None),
            entry("config.json", 1983, None),
            entry("model.safetensors", 151_061_672, Some(TINY_SHA)),
            entry("pytorch_model.bin", 151_095_027, Some(SHARD_1)),
            entry("tokenizer.json", 2_480_466, None),
        ];
        let row = hf
            .build_from_listing("openai/whisper-tiny", &listing)
            .unwrap();
        assert_eq!(row.name, "hf/openai/whisper-tiny");
        assert_eq!(hf.id(&row), "hf:openai/whisper-tiny");
        assert_eq!(
            row.origin.as_deref(),
            Some("https://huggingface.co/openai/whisper-tiny")
        );
        assert_eq!(
            row.description,
            "openai/whisper-tiny on Hugging Face at main: model.safetensors, 151.1 MB, with config.json"
        );
        row.resource_set().validate().unwrap();
        assert_eq!(row.resource_set().keys(), ["checkpoint", "config"]);

        let set_checkpoint = row.resource_set();

        let checkpoint = set_checkpoint.get("checkpoint").unwrap();
        assert_eq!(checkpoint.file, "model.safetensors");
        assert_eq!(checkpoint.sha256.as_deref(), Some(TINY_SHA));
        assert_eq!(checkpoint.format.as_deref(), Some("safetensors"));
        assert_eq!(checkpoint.name, "hf/openai/whisper-tiny/checkpoint");
        assert_eq!(
            checkpoint.sources,
            [Source::Url(
                "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors"
                    .to_string()
            )]
        );
        let set_config = row.resource_set();
        let config = set_config.get("config").unwrap();
        assert_eq!(config.file, "config.json");
        assert_eq!(config.sha256, None);
        assert_eq!(config.format.as_deref(), Some("json"));
    }

    /// An extra file rides along under its key and kind, pinned when the
    /// hub pins it; one the repo lacks is simply absent.
    #[test]
    fn test_extra_files_ride_along() {
        let hf = HfProvider::new()
            .with_extra_file("vocabulary", "tokenizer.json", "tokenizer.json")
            .with_extra_file("merges", "merges.txt", "text");
        let listing = vec![
            entry("config.json", 1983, None),
            entry("model.safetensors", 151_061_672, Some(TINY_SHA)),
            entry("tokenizer.json", 2_480_466, Some(SHARD_1)),
        ];
        let row = hf.build_from_listing("org/asr", &listing).unwrap();
        assert_eq!(
            row.resource_set().keys(),
            ["checkpoint", "config", "vocabulary"]
        );
        let set_vocab = row.resource_set();
        let vocab = set_vocab.get("vocabulary").unwrap();
        assert_eq!(vocab.file, "tokenizer.json");
        assert_eq!(vocab.format.as_deref(), Some("tokenizer.json"));
        assert_eq!(vocab.sha256.as_deref(), Some(SHARD_1));
        assert_eq!(
            row.description,
            "org/asr on Hugging Face at main: model.safetensors, 151.1 MB, with config.json and tokenizer.json"
        );
    }

    /// A sharded repo: the index and every shard, in shard order, under
    /// the checkpoint key; an incomplete set is refused.
    #[test]
    fn test_a_sharded_repo_is_the_index_and_its_shards() {
        let hf = HfProvider::new().with_checkpoint_key("weights");
        let listing = vec![
            entry(
                "model-00002-of-00002.safetensors",
                1_170_000_000,
                Some(SHARD_2),
            ),
            entry(
                "model-00001-of-00002.safetensors",
                5_000_000_000,
                Some(SHARD_1),
            ),
            entry("model.safetensors.index.json", 71_000, None),
        ];
        let row = hf
            .build_from_listing("ivrit-ai/whisper-large-v3", &listing)
            .unwrap();
        assert_eq!(
            row.resource_set().keys(),
            ["weights.00001", "weights.00002", "weights.index"]
        );
        assert_eq!(
            row.description,
            "ivrit-ai/whisper-large-v3 on Hugging Face at main: 2 shards, 6.17 GB"
        );
        let set_index = row.resource_set();
        let index = set_index.get("weights.index").unwrap();
        assert_eq!(index.file, "model.safetensors.index.json");
        assert_eq!(index.format.as_deref(), Some("safetensors-index"));
        assert_eq!(index.sha256, None);
        let set_first = row.resource_set();
        let first = set_first.get("weights.00001").unwrap();
        assert_eq!(first.file, "model-00001-of-00002.safetensors");
        assert_eq!(first.sha256.as_deref(), Some(SHARD_1));
        assert_eq!(first.format.as_deref(), Some("safetensors"));

        let incomplete = vec![
            entry("model-00002-of-00002.safetensors", 1, Some(SHARD_2)),
            entry("model.safetensors.index.json", 71_000, None),
        ];
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("incomplete set of shards")
            .frame_contains("hf:ivrit-ai/whisper-large-v3")
            .assert_err(&hf.build_from_listing("ivrit-ai/whisper-large-v3", &incomplete));
    }

    /// A set numbered from 1 with no gap is still incomplete when its last
    /// shards are missing: the `-of-N` total says how many there are, and
    /// the error names the ones the listing lacks.
    #[test]
    fn test_a_sharded_repo_missing_its_last_shards_is_refused() {
        let hf = HfProvider::new();
        let listing = vec![
            entry("model.safetensors.index.json", 71_000, None),
            entry("model-00001-of-00003.safetensors", 1, Some(SHARD_1)),
            entry("model-00002-of-00003.safetensors", 1, Some(SHARD_2)),
        ];
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("incomplete set of shards")
            .message_contains("missing model-00003-of-00003.safetensors")
            .assert_err(&hf.build_from_listing("org/repo", &listing));

        // A long tail is named eight shards at most, then counted.
        let listing = vec![
            entry("model.safetensors.index.json", 71_000, None),
            entry("model-00001-of-00012.safetensors", 1, Some(SHARD_1)),
        ];
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("missing model-00002-of-00012.safetensors, ")
            .message(text::ends_with(
                "model-00009-of-00012.safetensors, and 3 more",
            ))
            .assert_err(&hf.build_from_listing("org/repo", &listing));
    }

    /// A repo with weights in another format only is refused naming them;
    /// a malformed ref is refused before any listing is read; a lookup
    /// without a cache is refused.
    #[test]
    fn test_what_is_refused() {
        let hf = HfProvider::new();
        let listing = vec![
            entry("pytorch_model.bin", 10, Some(SHARD_1)),
            entry("tf_model.h5", 10, Some(SHARD_2)),
            entry("README.md", 10, None),
        ];
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("no model.safetensors")
            .message_contains("pytorch_model.bin, tf_model.h5")
            .assert_err(&hf.build_from_listing("some/repo", &listing));

        for bad in [
            "whisper-tiny",
            "openai/",
            "/whisper-tiny",
            "openai/whisper/tiny",
            "",
        ] {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains("org/repo")
                .has_cause::<ParseError>()
                .assert_err(&hf.build_from_listing(bad, &[]));
        }
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("through a cache")
            .assert_err(&hf.lookup("openai/whisper-tiny"));
        assert!(hf.list().is_empty());
        assert!(hf.ids().is_empty());
        assert!(!hf.answers_bare_names());
        assert_eq!(hf.origin(), Some("https://huggingface.co"));
    }

    /// A repo with no safetensors checkpoint gives its one archive
    /// checkpoint, under the checkpoint key and the archive's kind; two of
    /// one ending are refused, naming them; a provider that asks for none
    /// refuses the repo, naming the archive among its weights.
    #[test]
    fn test_an_archive_checkpoint_stands_in() {
        let hf = HfProvider::new().with_archive_checkpoint(".nemo", "nemo");
        let listing = vec![
            entry(".gitattributes", 1438, None),
            entry("README.md", 10, None),
            entry(
                "diar_streaming_sortformer_4spk-v2.nemo",
                471_367_680,
                Some(TINY_SHA),
            ),
        ];
        let row = hf
            .build_from_listing("nvidia/diar_streaming_sortformer_4spk-v2", &listing)
            .unwrap();
        row.resource_set().validate().unwrap();
        assert_eq!(row.resource_set().keys(), ["checkpoint"]);
        let set_checkpoint = row.resource_set();
        let checkpoint = set_checkpoint.get("checkpoint").unwrap();
        assert_eq!(checkpoint.file, "diar_streaming_sortformer_4spk-v2.nemo");
        assert_eq!(checkpoint.format.as_deref(), Some("nemo"));
        assert_eq!(checkpoint.sha256.as_deref(), Some(TINY_SHA));
        assert!(
            row.description
                .ends_with(": diar_streaming_sortformer_4spk-v2.nemo, 471.4 MB"),
            "{}",
            row.description
        );

        // A safetensors checkpoint is preferred.
        let mut both = listing.clone();
        both.push(entry("model.safetensors", 10, Some(SHARD_1)));
        let row = hf.build_from_listing("org/repo", &both).unwrap();
        assert_eq!(
            row.resource_set()
                .get("checkpoint")
                .unwrap()
                .format
                .as_deref(),
            Some("safetensors")
        );

        let mut two = listing.clone();
        two.push(entry("other.nemo", 10, Some(SHARD_1)));
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("2 .nemo files")
            .message_contains("other.nemo")
            .frame_contains("hf:org/repo")
            .assert_err(&hf.build_from_listing("org/repo", &two));

        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("the repo has: diar_streaming_sortformer_4spk-v2.nemo")
            .assert_err(&HfProvider::new().build_from_listing("org/repo", &listing));
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("or one *.nemo at main; the repo has: no weights")
            .assert_err(&hf.build_from_listing("org/repo", &listing[..2]));
    }

    /// The URLs: a file at a revision, and the listing.
    #[test]
    fn test_urls_carry_the_revision() {
        let hf = HfProvider::new().with_revision("06f233fe06e710322aca913c1bc4249a0d71fce1");
        assert_eq!(
            hf.url("openai", "whisper-large-v3", "model.safetensors"),
            "https://huggingface.co/openai/whisper-large-v3/resolve/06f233fe06e710322aca913c1bc4249a0d71fce1/model.safetensors"
        );
        assert_eq!(
            hf.tree_url("openai", "whisper-large-v3"),
            "https://huggingface.co/api/models/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1"
        );
        let listing = hf.listing_resource("openai", "whisper-large-v3");
        assert_eq!(
            listing.file,
            "openai--whisper-large-v3--06f233fe06e710322aca913c1bc4249a0d71fce1.json"
        );
        assert_eq!(listing.name, "hf/openai/whisper-large-v3/listing");
        listing.validate().unwrap();

        let named = HfProvider::new().with_name("hf-dev").with_revision("dev");
        assert_eq!(named.name(), "hf-dev");
        assert_eq!(named.listing_resource("o", "r").name, "hf-dev/o/r/listing");
    }

    /// The listing as the hub serves it: LFS files carry a digest, the
    /// rest do not.
    #[test]
    fn test_the_listing_parses() {
        let json = r#"[
          {"type":"file","oid":"abc","size":1983,"path":"config.json"},
          {"type":"file","oid":"def","size":151061672,"path":"model.safetensors",
           "lfs":{"oid":"7ebd0e69e78190ff5b1b2e6c7c0a6a9b1a4e2f0f0b3d2c1a9f8e7d6c5b4a3f2e","size":151061672,"pointerSize":135}},
          {"type":"directory","oid":"ghi","size":0,"path":"onnx"}
        ]"#;
        let listing: Vec<HubEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(listing.len(), 3);
        assert_eq!(listing[0].sha256(), None);
        assert_eq!(listing[1].sha256().as_deref(), Some(TINY_SHA));
        assert_eq!(listing[2].kind, "directory");
        let row = HfProvider::new()
            .build_from_listing("openai/whisper-tiny", &listing)
            .unwrap();
        assert_eq!(row.resource_set().keys(), ["checkpoint", "config"]);
    }

    /// A digest's hex case carries nothing, and the cache's pins, paths and
    /// checks are all lowercase: a listing that spells a digest in
    /// uppercase pins the file by the same digest, in lowercase.
    #[test]
    fn test_an_uppercase_digest_pins_in_lowercase() {
        let json = format!(
            r#"[{{"type":"file","size":3,"path":"model.safetensors","lfs":{{"oid":"{}","size":3}}}}]"#,
            TINY_SHA.to_uppercase()
        );
        let listing: Vec<HubEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(listing[0].sha256().as_deref(), Some(TINY_SHA));
        let row = HfProvider::new()
            .build_from_listing("org/repo", &listing)
            .unwrap();
        row.resource_set().validate().unwrap();
        assert_eq!(
            row.resource_set()
                .get("checkpoint")
                .unwrap()
                .sha256
                .as_deref(),
            Some(TINY_SHA)
        );
    }

    /// Through a cache, against a loopback hub: the listing is fetched
    /// once, the row is pinned by it, and an offline cache that has the
    /// listing resolves the ref again without the network; a hub that
    /// answers 401 is reported with the ref and what 401 means.
    #[cfg(feature = "fetch")]
    #[test]
    fn test_resolve_fetches_the_listing_once_and_keeps_it() {
        use crate::data::{
            cache::{
                BunsenDiskCacheOptions,
                testing::{
                    serve_once,
                    serve_status,
                },
            },
            resources::{
                CacheStatus,
                ResourceCacheOptions,
            },
        };

        static LISTING: &[u8] = br#"[
          {"type":"file","oid":"abc","size":1983,"path":"config.json"},
          {"type":"file","oid":"def","size":3,"path":"model.safetensors",
           "lfs":{"oid":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","size":3,"pointerSize":135}}
        ]"#;
        let dir = tempfile::tempdir().unwrap();
        let cache_at = |offline: bool| {
            ResourceCache::new(
                ResourceCacheOptions::default()
                    .with_disk(
                        BunsenDiskCacheOptions::default()
                            .with_cache_dir(Some(dir.path().join("cache")))
                            .without_transfer_observers(),
                    )
                    .with_offline(offline),
            )
            .unwrap()
        };
        // The server answers one request, whatever the path: the listing.
        let origin_of = |url: String| url.rsplit_once('/').unwrap().0.to_string();
        let hf = HfProvider::new().with_origin(origin_of(serve_once("any", LISTING)));

        let online = cache_at(false);
        let row = hf.resolve("org/repo", &online).unwrap().unwrap();
        assert_eq!(row.name, "hf/org/repo");
        let set_checkpoint = row.resource_set();
        let checkpoint = set_checkpoint.get("checkpoint").unwrap();
        assert_eq!(
            checkpoint.sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(
            checkpoint.sources,
            [Source::Url(format!(
                "{}/org/repo/resolve/main/model.safetensors",
                hf.origin
            ))]
        );
        assert_eq!(
            online.status(&hf.listing_resource("org", "repo")),
            CacheStatus::Cached,
            "the listing stays in the cache"
        );

        // Offline, the same ref resolves from the cached listing; the
        // server took one request and is gone.
        let offline = cache_at(true);
        let again = hf.resolve("org/repo", &offline).unwrap().unwrap();
        assert_eq!(again, row);
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .display(text::starts_with("hf:org/other: file listing: "))
            .message_contains("the cache is offline")
            .assert_err(&hf.resolve("org/other", &offline));

        let gone = HfProvider::new().with_origin(origin_of(serve_status("any", 401)));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::starts_with(
                "hf:nobody/nothing: the repo does not exist, or is gated",
            ))
            .cause(crate::errors::testing::predicate(
                "an unauthorized lookup",
                |c: &crate::errors::LookupError| {
                    c.problem == crate::errors::LookupProblem::Unauthorized
                },
            ))
            .assert_err(&gone.resolve("nobody/nothing", &online));
    }
}
