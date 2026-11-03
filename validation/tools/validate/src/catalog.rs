//! `cargo validate catalog`: builds kits' pretrained catalogs from sources.

use std::{
    io::Read,
    path::{
        Path,
        PathBuf,
    },
};

use bunsen::data::pretrained::{
    CatalogBase,
    CatalogBundle,
    CatalogFamily,
    CatalogGroup,
    CatalogResource,
    PretrainedCatalog,
};
use serde::Deserialize;
use sha2::{
    Digest,
    Sha256,
};

/// The version of the catalog the tool writes: part of the build cache's
/// key, so a tool that writes differently rebuilds every catalog.
const FORMAT: &str = concat!(env!("CARGO_PKG_VERSION"), "/catalog-1");

/// The kinds whose files are checkpoints: their `dtype` is the one the
/// bundle's `config.json` names.
const CHECKPOINT_KINDS: &[&str] = &["safetensors", "safetensors-index"];

/// A kit's catalog source, `validation/catalogs/{kit}.yml`: what is written
/// by hand. The tool adds what is derived: digests, sizes, dtypes, and the
/// Hub base URLs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogSource {
    /// The kit.
    pub kit: String,

    /// Where the generated catalog is checked in, from the repository root.
    pub out: PathBuf,

    /// The table's name.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The table's groups, in listing order, each with its rows.
    pub groups: Vec<CatalogGroup>,

    /// The resource bundles.
    pub bundles: Vec<SourceBundle>,

    /// The model families whose presets rows inherit.
    #[serde(default)]
    pub families: Vec<CatalogFamily>,
}

/// One resource bundle of a source: on the Hub at a pinned revision, or
/// at explicit bases.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBundle {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub origin: Option<String>,
    pub namespace: String,

    /// A Hub repo at a pinned revision: the first base, and where digests
    /// and sizes are read from.
    #[serde(default)]
    pub hub: Option<Hub>,

    /// Bases after the Hub's, or every base when there is no Hub repo.
    #[serde(default)]
    pub bases: Vec<CatalogBase>,

    pub resources: Vec<SourceResource>,
}

/// A Hub repo at a pinned revision.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hub {
    /// `org/name`.
    pub repo: String,

    /// A commit.
    pub revision: String,
}

/// One file of a source bundle. A `sha256` given here is authoritative: it
/// must agree with the one the Hub lists, when the Hub lists one, and a file
/// not on the Hub must have one.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceResource {
    pub key: String,
    pub file: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub sources: Vec<CatalogBase>,
}

/// What the tool reads from the network.
pub trait Remote {
    /// The bytes at `url`.
    fn get(
        &self,
        url: &str,
    ) -> Result<Vec<u8>, String>;

    /// The bytes at `url`, and the URL of the next page when the response
    /// links one (`Link: <…>; rel="next"`).
    fn get_page(
        &self,
        url: &str,
    ) -> Result<(Vec<u8>, Option<String>), String>;

    /// The size of what is at `url`, from a `HEAD`'s `Content-Length`.
    fn head_size(
        &self,
        url: &str,
    ) -> Result<Option<u64>, String>;

    /// The SHA-256 of what is at `url`, streamed.
    fn sha256(
        &self,
        url: &str,
    ) -> Result<String, String>;
}

/// The network, through `ureq`.
pub struct Http(ureq::Agent);

impl Http {
    pub fn new() -> Self {
        Self(ureq::Agent::new_with_defaults())
    }
}

fn next_link(link: &str) -> Option<String> {
    link.split(',').find_map(|part| {
        let (target, params) = part.split_once(';')?;
        params.contains("rel=\"next\"").then(|| {
            target
                .trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

impl Remote for Http {
    fn get(
        &self,
        url: &str,
    ) -> Result<Vec<u8>, String> {
        Ok(self.get_page(url)?.0)
    }

    fn get_page(
        &self,
        url: &str,
    ) -> Result<(Vec<u8>, Option<String>), String> {
        let mut response = self
            .0
            .get(url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let next = response
            .headers()
            .get("link")
            .and_then(|l| l.to_str().ok())
            .and_then(next_link);
        let body = response
            .body_mut()
            .with_config()
            .limit(64 << 20)
            .read_to_vec()
            .map_err(|e| format!("GET {url}: {e}"))?;
        Ok((body, next))
    }

    fn head_size(
        &self,
        url: &str,
    ) -> Result<Option<u64>, String> {
        // A compressed response's length is not the file's.
        let response = self
            .0
            .head(url)
            .header("Accept-Encoding", "identity")
            .call()
            .map_err(|e| format!("HEAD {url}: {e}"))?;
        Ok(response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok()))
    }

    fn sha256(
        &self,
        url: &str,
    ) -> Result<String, String> {
        let mut response = self
            .0
            .get(url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let mut reader = response.body_mut().as_reader();
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| format!("GET {url}: {e}"))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hex(&hasher.finalize()))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_of(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// One file in a Hub repo's tree listing.
#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Debug, Deserialize)]
struct Lfs {
    oid: String,
    size: u64,
}

/// Every file of `repo` at `revision`, by path, across the listing's pages.
fn hub_listing(
    remote: &dyn Remote,
    hub: &Hub,
) -> Result<Vec<TreeEntry>, String> {
    let mut url = Some(format!(
        "https://huggingface.co/api/models/{}/tree/{}?recursive=true",
        hub.repo, hub.revision
    ));
    let mut entries = Vec::new();
    while let Some(page) = url {
        let (body, next) = remote.get_page(&page)?;
        let mut more: Vec<TreeEntry> = serde_json::from_slice(&body)
            .map_err(|e| format!("{page}: not a tree listing: {e}"))?;
        entries.append(&mut more);
        url = next;
    }
    Ok(entries.into_iter().filter(|e| e.kind == "file").collect())
}

impl Hub {
    fn base(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}",
            self.repo, self.revision
        )
    }
}

/// The two digests of a file reconciled: at least one, and equal when both.
fn reconcile(
    what: &str,
    explicit: Option<&str>,
    derived: Option<&str>,
) -> Result<String, String> {
    match (explicit, derived) {
        (Some(e), Some(d)) if e != d => Err(format!(
            "{what}: the source's sha256 {e} disagrees with the derived {d}"
        )),
        (Some(e), _) => Ok(e.to_string()),
        (None, Some(d)) => Ok(d.to_string()),
        (None, None) => Err(format!(
            "{what}: no sha256: it is not on the Hub, so the source must give one"
        )),
    }
}

/// The dtype a `config.json` names: `torch_dtype`, `dtype`, or either under
/// `text_config`.
fn config_dtype(config: &[u8]) -> Option<String> {
    let json: serde_json::Value = serde_json::from_slice(config).ok()?;
    let find = |v: &serde_json::Value| {
        ["torch_dtype", "dtype"]
            .iter()
            .find_map(|k| v.get(k).and_then(|d| d.as_str()).map(str::to_string))
    };
    find(&json).or_else(|| json.get("text_config").and_then(find))
}

/// The URL `resource` is fetched from first: its own first URL source, else
/// its file under the bundle's first URL base.
fn first_url(
    resource: &SourceResource,
    bases: &[CatalogBase],
) -> Option<String> {
    resource
        .sources
        .iter()
        .find_map(|s| match s {
            CatalogBase::Url(u) => Some(u.clone()),
            CatalogBase::LocalDir(_) => None,
        })
        .or_else(|| {
            bases.iter().find_map(|b| match b {
                CatalogBase::Url(u) => {
                    Some(format!("{}/{}", u.trim_end_matches('/'), resource.file))
                }
                CatalogBase::LocalDir(_) => None,
            })
        })
}

/// Builds one bundle: its bases, and every file's digest, size and dtype.
fn build_bundle(
    remote: &dyn Remote,
    bundle: &SourceBundle,
    verify_remote: bool,
) -> Result<CatalogBundle, String> {
    let listing = match &bundle.hub {
        Some(hub) => Some(hub_listing(remote, hub)?),
        None => None,
    };
    let mut bases = Vec::new();
    if let Some(hub) = &bundle.hub {
        bases.push(CatalogBase::Url(hub.base()));
    }
    bases.extend(bundle.bases.iter().cloned());

    let mut config_dtype_found = None;
    let mut resources = Vec::new();
    for r in &bundle.resources {
        let what = format!("{}: {}", bundle.name, r.file);
        let entry = listing
            .as_ref()
            .map(|l| {
                l.iter()
                    .find(|e| e.path == r.file)
                    .ok_or_else(|| format!("{what}: not in the Hub listing"))
            })
            .transpose()?;
        let url = first_url(r, &bases);
        let (derived, size) = match entry {
            Some(e) => match &e.lfs {
                Some(lfs) => (Some(lfs.oid.clone()), Some(lfs.size)),
                None => {
                    // A small metadata file: fetched, so its digest is
                    // derived and a probe can read it.
                    let url = url.clone().ok_or_else(|| format!("{what}: no URL"))?;
                    let bytes = remote.get(&url)?;
                    if r.file == "config.json" {
                        config_dtype_found = config_dtype(&bytes);
                    }
                    (Some(sha256_of(&bytes)), Some(e.size))
                }
            },
            None => {
                let size = match &url {
                    Some(u) => remote.head_size(u)?,
                    None => None,
                };
                (None, size)
            }
        };
        let sha256 = reconcile(&what, r.sha256.as_deref(), derived.as_deref())?;
        if let Some(u) = &url
            && u.contains("openaipublic.azureedge.net")
            && !u.contains(&sha256)
        {
            return Err(format!(
                "{what}: the URL's digest is not its sha256 {sha256}"
            ));
        }
        if verify_remote {
            let u = url
                .clone()
                .ok_or_else(|| format!("{what}: no URL to verify"))?;
            let fetched = remote.sha256(&u)?;
            if fetched != sha256 {
                return Err(format!("{what}: {u} hashes to {fetched}, not {sha256}"));
            }
        }
        resources.push(CatalogResource {
            key: r.key.clone(),
            file: r.file.clone(),
            sha256: Some(sha256),
            kind: r.kind.clone(),
            size: r.size.or(size),
            dtype: r.dtype.clone(),
            sources: r.sources.clone(),
        });
    }
    for r in &mut resources {
        if r.dtype.is_none()
            && r.kind
                .as_deref()
                .is_some_and(|k| CHECKPOINT_KINDS.contains(&k))
        {
            r.dtype = config_dtype_found.clone();
        }
    }
    Ok(CatalogBundle {
        name: bundle.name.clone(),
        description: bundle.description.clone(),
        license: bundle.license.clone(),
        origin: bundle.origin.clone(),
        namespace: bundle.namespace.clone(),
        bases,
        resources,
    })
}

/// Builds a source's catalog.
pub fn build(
    remote: &dyn Remote,
    source: &CatalogSource,
    verify_remote: bool,
) -> Result<PretrainedCatalog, String> {
    let bundles = source
        .bundles
        .iter()
        .map(|b| build_bundle(remote, b, verify_remote))
        .collect::<Result<_, _>>()?;
    let catalog = PretrainedCatalog {
        kit: source.kit.clone(),
        name: source.name.clone(),
        description: source.description.clone(),
        groups: source.groups.clone(),
        bundles,
        families: source.families.clone(),
    };
    catalog
        .to_table()
        .map_err(|e| format!("{}: the catalog does not load: {e}", source.kit))?;
    Ok(catalog)
}

/// The catalog as it is checked in: pretty JSON with a trailing newline.
pub fn render(catalog: &PretrainedCatalog) -> String {
    let mut json = serde_json::to_string_pretty(catalog).expect("a catalog serializes");
    json.push('\n');
    json
}

/// Where a build of `source_text` is kept in the validation workspace's
/// `target/`: keyed by the source and the tool's format, so a changed
/// source, or a tool that writes differently, builds again.
fn cache_path(
    validation: &Path,
    kit: &str,
    source_text: &str,
) -> PathBuf {
    let key = sha256_of(format!("{FORMAT}\n{source_text}").as_bytes());
    validation
        .join("target")
        .join("catalog")
        .join(format!("{kit}-{}.json", &key[..16]))
}

/// What `cargo validate catalog` does with each built catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Build into `target/` when it is not there already.
    Build,

    /// Build, then fail when the checked-in catalog differs.
    Check,

    /// Build, then overwrite the checked-in catalog.
    Update,
}

/// The sources in `validation/catalogs/`, or those of `kits`.
pub fn sources(
    validation: &Path,
    kits: &[String],
) -> Result<Vec<PathBuf>, String> {
    let dir = validation.join("catalogs");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yml"))
        .collect();
    paths.sort();
    if !kits.is_empty() {
        for kit in kits {
            if !paths
                .iter()
                .any(|p| p.file_stem().is_some_and(|s| s == kit.as_str()))
            {
                return Err(format!(
                    "no catalog source for kit {kit:?} in {}",
                    dir.display()
                ));
            }
        }
        paths.retain(|p| {
            p.file_stem()
                .is_some_and(|s| kits.iter().any(|k| s == k.as_str()))
        });
    }
    Ok(paths)
}

/// Runs `mode` on the source at `path`; `Ok(false)` when `--check` finds the
/// checked-in catalog out of date.
pub fn run(
    remote: &dyn Remote,
    validation: &Path,
    path: &Path,
    mode: Mode,
    verify_remote: bool,
) -> Result<bool, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let source: CatalogSource =
        serde_saphyr::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let cached = cache_path(validation, &source.kit, &text);
    let rendered = if cached.exists() && !verify_remote {
        std::fs::read_to_string(&cached).map_err(|e| format!("read {}: {e}", cached.display()))?
    } else {
        let rendered = render(&build(remote, &source, verify_remote)?);
        if let Some(dir) = cached.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(&cached, &rendered).map_err(|e| format!("{}: {e}", cached.display()))?;
        rendered
    };
    let out = validation.join("..").join(&source.out);
    match mode {
        Mode::Build => {
            eprintln!("built {} catalog: {}", source.kit, cached.display());
            Ok(true)
        }
        Mode::Check => {
            let current = std::fs::read_to_string(&out).unwrap_or_default();
            if current == rendered {
                eprintln!("{}: {} is current", source.kit, source.out.display());
                Ok(true)
            } else {
                eprintln!(
                    "{}: {} is out of date: `cargo validate catalog --update --kit {}`",
                    source.kit,
                    source.out.display(),
                    source.kit
                );
                Ok(false)
            }
        }
        Mode::Update => {
            std::fs::write(&out, &rendered).map_err(|e| format!("{}: {e}", out.display()))?;
            eprintln!("{}: wrote {}", source.kit, source.out.display());
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A network of fixed answers.
    #[derive(Default)]
    struct Fake {
        bodies: BTreeMap<String, Vec<u8>>,
        sizes: BTreeMap<String, u64>,
    }

    impl Remote for Fake {
        fn get(
            &self,
            url: &str,
        ) -> Result<Vec<u8>, String> {
            self.bodies
                .get(url)
                .cloned()
                .ok_or_else(|| format!("no {url}"))
        }

        fn get_page(
            &self,
            url: &str,
        ) -> Result<(Vec<u8>, Option<String>), String> {
            Ok((self.get(url)?, None))
        }

        fn head_size(
            &self,
            url: &str,
        ) -> Result<Option<u64>, String> {
            Ok(self.sizes.get(url).copied())
        }

        fn sha256(
            &self,
            url: &str,
        ) -> Result<String, String> {
            Ok(sha256_of(&self.get(url)?))
        }
    }

    const LISTING: &str = "https://huggingface.co/api/models/o/m/tree/abc?recursive=true";
    const CONFIG: &[u8] = br#"{"torch_dtype": "bfloat16"}"#;
    const WEIGHTS: &[u8] = b"weights";

    fn fake() -> Fake {
        let mut fake = Fake::default();
        let listing = format!(
            r#"[{{"type":"file","path":"model.safetensors","size":1,"lfs":{{"oid":"{}","size":1000}}}},
                {{"type":"file","path":"config.json","size":{}}},
                {{"type":"directory","path":"sub"}}]"#,
            sha256_of(WEIGHTS),
            CONFIG.len()
        );
        fake.bodies.insert(LISTING.into(), listing.into_bytes());
        fake.bodies.insert(
            "https://huggingface.co/o/m/resolve/abc/config.json".into(),
            CONFIG.to_vec(),
        );
        fake.bodies.insert(
            "https://huggingface.co/o/m/resolve/abc/model.safetensors".into(),
            WEIGHTS.to_vec(),
        );
        fake.bodies
            .insert("https://x.example/v.txt".into(), b"vocab".to_vec());
        fake.sizes.insert("https://x.example/v.txt".into(), 5);
        fake
    }

    fn source(extra: &str) -> CatalogSource {
        let text = format!(
            r#"
kit: toy
out: x/catalog.json
name: well-known
description: the toy table
groups:
  - name: o
    description: o's
    rows: [ {{ name: m, description: a model, bundles: [o/m, x/vocab] }} ]
bundles:
  - name: o/m
    description: the model
    namespace: o
    hub: {{ repo: o/m, revision: abc }}
    resources:
      - {{ key: checkpoint, file: model.safetensors, kind: safetensors }}
      - {{ key: config, file: config.json, kind: json }}
  - name: x/vocab
    description: the vocabulary
    namespace: x
    bases: [ {{ url: "https://x.example" }} ]
    resources:
      - {{ key: vocabulary, file: v.txt{extra} }}
"#
        );
        serde_saphyr::from_str(&text).unwrap()
    }

    /// Hub files take their digest and size from the listing (an LFS file)
    /// or from their bytes (a small file); a checkpoint takes the config's
    /// dtype; a file off the Hub takes the source's digest and a `HEAD`'s
    /// size.
    #[test]
    fn test_build() {
        let vocab_sha = sha256_of(b"vocab");
        let catalog = build(&fake(), &source(&format!(", sha256: {vocab_sha}")), false).unwrap();
        let model = &catalog.bundles[0];
        assert_eq!(
            model.bases,
            [CatalogBase::Url(
                "https://huggingface.co/o/m/resolve/abc".into()
            )]
        );
        let checkpoint = &model.resources[0];
        assert_eq!(
            checkpoint.sha256.as_deref(),
            Some(sha256_of(WEIGHTS).as_str())
        );
        assert_eq!(checkpoint.size, Some(1000));
        assert_eq!(checkpoint.dtype.as_deref(), Some("bfloat16"));
        let config = &model.resources[1];
        assert_eq!(config.sha256.as_deref(), Some(sha256_of(CONFIG).as_str()));
        assert_eq!(config.dtype, None);
        let vocab = &catalog.bundles[1].resources[0];
        assert_eq!(vocab.sha256.as_deref(), Some(vocab_sha.as_str()));
        assert_eq!(vocab.size, Some(5));
        assert!(render(&catalog).ends_with("}\n"));
    }

    /// A file off the Hub with no digest is refused; a digest that disagrees
    /// with the Hub's is refused; `--verify-remote` refuses bytes that do
    /// not hash to the pin.
    #[test]
    fn test_refusals() {
        let error = build(&fake(), &source(""), false).unwrap_err();
        assert!(error.contains("must give one"), "{error}");

        let mut wrong = source(&format!(", sha256: {}", sha256_of(b"vocab")));
        wrong.bundles[0].resources[0].sha256 = Some("b".repeat(64));
        let error = build(&fake(), &wrong, false).unwrap_err();
        assert!(error.contains("disagrees"), "{error}");

        let stale = source(&format!(", sha256: {}", "c".repeat(64)));
        assert!(build(&fake(), &stale, false).is_ok());
        let error = build(&fake(), &stale, true).unwrap_err();
        assert!(error.contains("hashes to"), "{error}");
    }

    /// The build cache's key follows the source text.
    #[test]
    fn test_cache_key() {
        let dir = Path::new("/v");
        assert_eq!(cache_path(dir, "k", "a"), cache_path(dir, "k", "a"));
        assert_ne!(cache_path(dir, "k", "a"), cache_path(dir, "k", "b"));
    }

    #[test]
    fn test_next_link() {
        assert_eq!(
            next_link(r#"<https://h/x?cursor=2>; rel="next""#).as_deref(),
            Some("https://h/x?cursor=2")
        );
        assert_eq!(next_link(r#"<https://h/x>; rel="prev""#), None);
    }

    #[test]
    fn test_config_dtype() {
        assert_eq!(
            config_dtype(br#"{"dtype":"float32"}"#).as_deref(),
            Some("float32")
        );
        assert_eq!(
            config_dtype(br#"{"text_config":{"torch_dtype":"bfloat16"}}"#).as_deref(),
            Some("bfloat16")
        );
        assert_eq!(config_dtype(b"{}"), None);
    }

    /// A source row's presets pass through to the catalog; a preset without
    /// a description is refused at build.
    #[test]
    fn test_build_row_presets() {
        let vocab_sha = sha256_of(b"vocab");
        let with_presets = |description: &str| {
            let mut source = source(&format!(", sha256: {vocab_sha}"));
            source.groups[0].rows[0].presets = serde_saphyr::from_str(&format!(
                r#"
presets:
  - {{ name: fast, description: "{description}", patch: {{ beam: 1 }} }}
  - {{ name: good, description: a wide search, patch: {{ beam: 5 }} }}
axes:
  - {{ name: quality, description: how hard the search tries, members: [fast, good] }}
default: [good]
"#
            ))
            .unwrap();
            source
        };

        let source = with_presets("a greedy search");
        let catalog = build(&fake(), &source, false).unwrap();
        let presets = &catalog.groups[0].rows[0].presets;
        assert_eq!(presets, &source.groups[0].rows[0].presets);
        assert_eq!(presets.default, ["good"]);
        assert_eq!(
            presets.get("fast").unwrap().patch,
            serde_json::json!({"beam": 1})
        );

        let error = build(&fake(), &with_presets(""), false).unwrap_err();
        assert!(
            error.contains("the preset `fast` has no description"),
            "{error}"
        );
    }
}
