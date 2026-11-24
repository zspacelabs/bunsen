//! A classifier's class labels: `ImageNet`'s `WordNet` synsets, and their
//! lemmas.

use std::{
    collections::HashMap,
    path::Path,
};

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    sys_at,
};

/// The class labels of an image classifier, as timm's `ImageNetInfo` reads
/// them: one `WordNet` synset id per class, in class order (`n01440764`), and
/// optionally each synset's lemmas, its readable name (`tench, Tinca
/// tinca`).
///
/// The label set of a model follows from its class count, as timm's
/// `infer_imagenet_subset` infers it ([`imagenet_subset_for`]).
///
/// ```rust
/// use bunsen::kits::images::support::labels::ImageLabelSet;
///
/// let labels = ImageLabelSet::try_parse(
///     "n01440764\nn01443537\n",
///     Some("n01440764\ttench, Tinca tinca\nn01443537\tgoldfish, Carassius auratus\n"),
/// )?;
/// assert_eq!(labels.len(), 2);
/// assert_eq!(labels.synset(1), "n01443537");
/// assert_eq!(labels.name(0), "tench, Tinca tinca");
/// # Ok::<(), bunsen::errors::BunsenError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageLabelSet {
    synsets: Vec<String>,
    names: Option<Vec<String>>,
}

impl ImageLabelSet {
    /// Parses a synset list, one id per line, and optionally a lemma table,
    /// one `synset<TAB>lemmas` line per synset (timm's
    /// `imagenet_synset_to_lemma.txt`, which covers every subset).
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    /// [`InvalidResource`](BunsenErrorKind::InvalidResource) for an empty
    /// synset list, a blank synset line, a lemma line without a tab, or a
    /// synset the lemma table does not name.
    pub fn try_parse(
        synsets: &str,
        lemmas: Option<&str>,
    ) -> BunsenResult<Self> {
        let invalid = |message: String| BunsenError::new(BunsenErrorKind::InvalidResource, message);
        let synsets: Vec<String> = synsets.lines().map(str::to_string).collect();
        if synsets.is_empty() {
            return Err(invalid("an empty synset list".to_string()));
        }
        if let Some(line) = synsets.iter().position(|s| s.trim().is_empty()) {
            return Err(invalid(format!("a blank synset at line {}", line + 1)));
        }
        let names = lemmas
            .map(|lemmas| {
                let table = lemmas
                    .lines()
                    .enumerate()
                    .map(|(i, line)| {
                        line.split_once('\t').ok_or_else(|| {
                            invalid(format!("a lemma line without a tab at line {}", i + 1))
                        })
                    })
                    .collect::<BunsenResult<HashMap<_, _>>>()?;
                synsets
                    .iter()
                    .map(|synset| {
                        table
                            .get(synset.as_str())
                            .map(|name| name.to_string())
                            .ok_or_else(|| invalid(format!("no lemmas for synset {synset}")))
                    })
                    .collect::<BunsenResult<Vec<_>>>()
            })
            .transpose()?;
        Ok(Self { synsets, names })
    }

    /// Reads and parses a synset file, and optionally a lemma file
    /// ([`try_parse`](Self::try_parse)).
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    /// The `io::Error` of a file that cannot be read, through
    /// [`sys_at`]; [`try_parse`](Self::try_parse)'s, framed with the file.
    pub fn try_from_files(
        synsets: impl AsRef<Path>,
        lemmas: Option<impl AsRef<Path>>,
    ) -> BunsenResult<Self> {
        let read = |path: &Path| std::fs::read_to_string(path).map_err(sys_at("read labels", path));
        let synsets = synsets.as_ref();
        let synset_text = read(synsets)?;
        let lemma_text = lemmas.as_ref().map(|p| read(p.as_ref())).transpose()?;
        Self::try_parse(&synset_text, lemma_text.as_deref())
            .map_err(|e| e.context(format!("the label set {}", synsets.display())))
    }

    /// The number of classes.
    pub fn len(&self) -> usize {
        self.synsets.len()
    }

    /// Whether the set has no classes; never, for a parsed set.
    pub fn is_empty(&self) -> bool {
        self.synsets.is_empty()
    }

    /// Class `index`'s synset id.
    ///
    /// # Panics
    /// When `index` is not a class.
    pub fn synset(
        &self,
        index: usize,
    ) -> &str {
        &self.synsets[index]
    }

    /// Class `index`'s readable name: its lemmas, or its synset id when the
    /// set has none.
    ///
    /// # Panics
    /// When `index` is not a class.
    pub fn name(
        &self,
        index: usize,
    ) -> &str {
        match &self.names {
            Some(names) => &names[index],
            None => self.synset(index),
        }
    }

    /// Whether the set has lemmas.
    pub fn has_names(&self) -> bool {
        self.names.is_some()
    }
}

/// The `ImageNet` subset a classifier of `num_classes` classes predicts, as
/// timm's `infer_imagenet_subset` infers it, or `None` for a count no subset
/// has.
///
/// | classes | subset |
/// |---|---|
/// | 1000 | `imagenet-1k` |
/// | 11221 | `imagenet-21k-miil` |
/// | 11821 | `imagenet-12k` |
/// | 21841 | `imagenet-22k` |
/// | 21842 | `imagenet-22k-ms` |
/// | 21843 | `imagenet-21k-goog` |
pub fn imagenet_subset_for(num_classes: usize) -> Option<&'static str> {
    match num_classes {
        1000 => Some("imagenet-1k"),
        11221 => Some("imagenet-21k-miil"),
        11821 => Some("imagenet-12k"),
        21841 => Some("imagenet-22k"),
        21842 => Some("imagenet-22k-ms"),
        21843 => Some("imagenet-21k-goog"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::ErrorMatcher;

    #[test]
    fn test_names_fall_back_to_the_synset() {
        let labels = ImageLabelSet::try_parse("n1\nn2", None).unwrap();
        assert!(!labels.has_names());
        assert_eq!(labels.name(1), "n2");
    }

    /// A lemma table covers more synsets than a subset has; the extra lines
    /// are ignored.
    #[test]
    fn test_lemmas_cover_a_superset() {
        let labels =
            ImageLabelSet::try_parse("n2\nn1\n", Some("n1\tone\nn2\ttwo\nn3\tthree\n")).unwrap();
        assert_eq!(labels.name(0), "two");
        assert_eq!(labels.name(1), "one");
    }

    #[test]
    fn test_try_parse_rejects_bad_files() {
        let invalid = || ErrorMatcher::kind(BunsenErrorKind::InvalidResource);
        invalid()
            .message_contains("an empty synset list")
            .assert_err(&ImageLabelSet::try_parse("", None));
        invalid()
            .message_contains("blank synset at line 2")
            .assert_err(&ImageLabelSet::try_parse("n1\n\nn2", None));
        invalid()
            .message_contains("without a tab at line 1")
            .assert_err(&ImageLabelSet::try_parse("n1", Some("n1 one")));
        invalid()
            .message_contains("no lemmas for synset n2")
            .assert_err(&ImageLabelSet::try_parse("n1\nn2", Some("n1\tone")));
    }

    #[test]
    fn test_try_from_files_frames_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let synsets = dir.path().join("synsets.txt");
        std::fs::write(&synsets, "n1\n").unwrap();
        let lemmas = dir.path().join("lemmas.txt");
        std::fs::write(&lemmas, "n2\ttwo\n").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .frame_contains("synsets.txt")
            .assert_err(&ImageLabelSet::try_from_files(&synsets, Some(&lemmas)));
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&ImageLabelSet::try_from_files(
            dir.path().join("missing"),
            None::<&Path>,
        ));
    }

    /// The built-in catalog pins timm's `ImageNet`-1k synsets and its lemma
    /// table, by those names.
    #[test]
    fn test_the_builtin_catalog_has_the_imagenet_label_files() {
        let catalog = crate::data::catalog::BuiltinCatalog::get().unwrap();
        for (name, file) in [
            ("imagenet/synsets-1k", "imagenet_synsets.txt"),
            ("imagenet/lemmas", "imagenet_synset_to_lemma.txt"),
        ] {
            let set = catalog.resource(name, "labels").unwrap();
            assert_eq!(set.get("labels").unwrap().file, file, "{name}");
        }
    }

    #[test]
    fn test_imagenet_subset_for() {
        assert_eq!(imagenet_subset_for(1000), Some("imagenet-1k"));
        assert_eq!(imagenet_subset_for(21843), Some("imagenet-21k-goog"));
        assert_eq!(imagenet_subset_for(10), None);
    }
}
