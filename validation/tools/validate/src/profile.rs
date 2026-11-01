//! Profiles, `validation/profiles/*.yml`: which models a machine validates.

use std::path::{
    Path,
    PathBuf,
};

use indexmap::IndexMap;
use serde::Deserialize;

use crate::catalog::{
    Catalog,
    Dtype,
    ModelSpec,
    Size,
};

/// One value or a list of them.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T: PartialEq> OneOrMany<T> {
    fn contains(
        &self,
        x: &T,
    ) -> bool {
        match self {
            OneOrMany::One(v) => v == x,
            OneOrMany::Many(vs) => vs.contains(x),
        }
    }

    fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(v) => vec![v],
            OneOrMany::Many(vs) => vs,
        }
    }
}

/// A selector: matches a model when every key it has does.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    #[serde(rename = "crate")]
    krate: Option<OneOrMany<String>>,
    model: Option<OneOrMany<String>>,
    size: Option<OneOrMany<Size>>,
    core: Option<bool>,
}

impl Selector {
    fn matches(
        &self,
        krate: &str,
        model: &str,
        spec: &ModelSpec,
    ) -> bool {
        self.krate
            .as_ref()
            .is_none_or(|k| k.contains(&krate.to_string()))
            && self
                .model
                .as_ref()
                .is_none_or(|m| m.contains(&model.to_string()))
            && self.size.as_ref().is_none_or(|s| s.contains(&spec.size))
            && self.core.is_none_or(|c| c == spec.core)
    }
}

/// A kind of check: a `driver` check is a test with a `driver` segment in
/// its path, a stream driver held to the same model's one-shot decode; a
/// `model` check is any other test of a model, against its reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Check {
    Driver,
    Model,
}

/// The backend a run is on; `flex` is the CPU, which every build has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Wgpu,
    Vulkan,
    Cuda,
    Metal,
    Flex,
}

impl Backend {
    /// The backend's feature on a validation crate; the CPU's is none.
    pub fn feature(self) -> Option<&'static str> {
        match self {
            Backend::Wgpu => Some("wgpu"),
            Backend::Vulkan => Some("vulkan"),
            Backend::Cuda => Some("cuda"),
            Backend::Metal => Some("metal"),
            Backend::Flex => None,
        }
    }
}

/// A profile file, as written.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    extends: Option<OneOrMany<String>>,
    #[serde(default)]
    include: Vec<Selector>,
    #[serde(default)]
    deny: Vec<Selector>,
    dtypes: Option<Vec<Dtype>>,
    checks: Option<Vec<Check>>,
    backend: Option<Backend>,
}

/// A profile, its `extends` folded in.
#[derive(Debug, Default)]
pub struct Profile {
    pub include: Vec<Selector>,
    pub deny: Vec<Selector>,
    pub dtypes: Option<Vec<Dtype>>,
    pub checks: Option<Vec<Check>>,
    pub backend: Option<Backend>,
}

impl Profile {
    /// The dtypes it checks in: its own, or `f32`.
    pub fn dtypes(&self) -> Vec<Dtype> {
        self.dtypes.clone().unwrap_or_else(|| vec![Dtype::F32])
    }

    /// The checks it runs: its own, or both kinds.
    pub fn checks(&self) -> Vec<Check> {
        self.checks
            .clone()
            .unwrap_or_else(|| vec![Check::Model, Check::Driver])
    }
}

/// Where profile `name` is: a path to a `.yml`, or a name (or a file name)
/// in `dir`.
fn profile_path(
    name: &str,
    dir: &Path,
) -> Result<PathBuf, String> {
    let path = Path::new(name);
    let found = if name.ends_with(".yml") || name.ends_with(".yaml") {
        if path.exists() {
            path.to_path_buf()
        } else {
            dir.join(path.file_name().unwrap_or_default())
        }
    } else {
        dir.join(format!("{name}.yml"))
    };
    if found.exists() {
        return Ok(found);
    }
    let mut known: Vec<String> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    e.path()
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                })
                .collect()
        })
        .unwrap_or_default();
    known.sort();
    Err(format!(
        "no profile {name:?}: a path to a .yml, or one of {known:?}"
    ))
}

/// Reads profile `name` (a name in `dir`, or a path), folding in what it
/// extends.
pub fn load(
    name: &str,
    dir: &Path,
) -> Result<Profile, String> {
    let read = |name: &str| {
        let path = profile_path(name, dir)?;
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))
    };
    load_with(name, &read, &mut Vec::new())
}

/// Reads profile `name` through `read`, which gives a profile's text by its
/// name, folding in what it extends.
fn load_with(
    name: &str,
    read: &dyn Fn(&str) -> Result<String, String>,
    seen: &mut Vec<String>,
) -> Result<Profile, String> {
    if seen.iter().any(|s| s == name) {
        seen.push(name.to_string());
        return Err(format!("profiles extend in a cycle: {}", seen.join(" -> ")));
    }
    seen.push(name.to_string());
    let text = read(name)?;
    let raw: RawProfile = if text.trim().is_empty() {
        RawProfile::default()
    } else {
        serde_saphyr::from_str(&text).map_err(|e| format!("profile {name}: {e}"))?
    };
    let mut profile = Profile::default();
    for parent in raw.extends.map(OneOrMany::into_vec).unwrap_or_default() {
        let base = load_with(&parent, read, seen)?;
        profile.include.extend(base.include);
        profile.deny.extend(base.deny);
        profile.dtypes = profile.dtypes.or(base.dtypes);
        profile.checks = profile.checks.or(base.checks);
        profile.backend = profile.backend.or(base.backend);
    }
    seen.pop();
    profile.include.extend(raw.include);
    profile.deny.extend(raw.deny);
    profile.dtypes = raw.dtypes.or(profile.dtypes);
    profile.checks = raw.checks.or(profile.checks);
    profile.backend = raw.backend.or(profile.backend);
    Ok(profile)
}

/// Per crate (those in `crates`, or every one when it is empty), each
/// model's choice: what an `include` selector matches, less what a `deny`
/// matches.
pub fn choose(
    catalog: &Catalog,
    profile: &Profile,
    crates: &[String],
) -> IndexMap<String, IndexMap<String, bool>> {
    catalog
        .crates
        .iter()
        .filter(|(name, _)| crates.is_empty() || crates.contains(name))
        .map(|(name, spec)| {
            let models = spec
                .models
                .iter()
                .map(|(model, m)| {
                    let keep = profile.include.iter().any(|s| s.matches(name, model, m))
                        && !profile.deny.iter().any(|s| s.matches(name, model, m));
                    (model.clone(), keep)
                })
                .collect();
            (name.clone(), models)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Catalog {
        Catalog::load(&crate::validation_dir().join("models.yml")).unwrap()
    }

    fn profiles() -> PathBuf {
        crate::validation_dir().join("profiles")
    }

    fn chosen(
        name: &str,
        krate: &str,
    ) -> Vec<String> {
        let profile = load(name, &profiles()).unwrap();
        choose(&catalog(), &profile, &[])[krate]
            .iter()
            .filter(|(_, keep)| **keep)
            .map(|(m, _)| m.clone())
            .collect()
    }

    /// The shipped profiles are cumulative, and `-core` is part of `-all`.
    #[test]
    fn test_shipped_profiles() {
        assert_eq!(chosen("medium-core", "qwen3"), ["qwen3-0.6b"]);
        assert_eq!(chosen("small-core", "qwen3"), Vec::<String>::new());
        assert_eq!(chosen("medium-all", "qwen3").len(), 4);
        for (core, all) in [
            ("small-core", "small-all"),
            ("medium-core", "medium-all"),
            ("large-core", "large-all"),
        ] {
            for krate in catalog().crates.keys() {
                let all = chosen(all, krate);
                for model in chosen(core, krate) {
                    assert!(all.contains(&model), "{core} ⊄ {krate} {model}");
                }
            }
        }
        let ci = load("ci", &profiles()).unwrap();
        assert_eq!(ci.checks(), [Check::Driver]);
        assert_eq!(ci.backend, Some(Backend::Flex));
        assert_eq!(chosen("ci", "whisper"), ["openai/tiny"]);
    }

    /// A deny removes from what an extended profile chose; selectors take one
    /// value or a list; an unknown key, a cycle and an unknown profile are
    /// refused.
    #[test]
    fn test_selectors_and_deny() {
        let files = [
            ("base", "include:\n  - {size: [small, medium]}\n"),
            (
                "mine",
                "extends: base\ndeny:\n  - {crate: qwen3, model: qwen3-4b}\ndtypes: [f32, bf16]\n",
            ),
            ("bad", "include:\n  - {colour: red}\n"),
            ("loop", "extends: loop\n"),
            ("empty", ""),
        ];
        let read = |name: &str| {
            files
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, text)| text.to_string())
                .ok_or_else(|| format!("no profile {name:?}"))
        };
        let load = |name: &str| load_with(name, &read, &mut Vec::new());
        let profile = load("mine").unwrap();
        let picked = &choose(&catalog(), &profile, &[])["qwen3"];
        assert!(picked["qwen3-1.7b"] && !picked["qwen3-4b"]);
        assert_eq!(profile.dtypes(), [Dtype::F32, Dtype::Bf16]);
        assert_eq!(profile.checks(), [Check::Model, Check::Driver]);
        assert!(load("bad").is_err());
        assert!(load("loop").unwrap_err().contains("cycle"));
        assert!(load("nowhere").is_err());
        assert!(load("empty").unwrap().include.is_empty());
    }
}
