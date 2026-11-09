//! The shared tag vocabulary, as data.
use alloc::string::String;

/// The shape a tag's value takes: the convention the catalog tool holds a
/// row's value to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagShape {
    /// `true` or `false`.
    Bool,

    /// A whole number, zero or more.
    Count,

    /// Any text.
    Text,

    /// One of these strings.
    OneOf(&'static [&'static str]),

    /// A list of strings.
    TextList,

    /// A list of canonical BCP 47 language tags, `"*"` among them for
    /// "unspecified other languages".
    Languages,

    /// `{group, version}`: the release sequence's name, and a semver version
    /// with an optional `+tag`.
    Version,

    /// An object of numbers, by name.
    Scores,
}

/// Who writes a tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagAuthor {
    /// The row's source, by hand.
    Row,

    /// The catalog, from what it knows; a row never writes it.
    Catalog,

    /// The catalog, from the row's resources where they say; the row's
    /// source where they do not, and never against them.
    CatalogElseRow,
}

/// One key of the vocabulary: its name, what it means, its value's shape,
/// and who writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TagDef {
    /// The key.
    pub key: &'static str,

    /// What it says about a model.
    pub description: &'static str,

    /// Its value's shape.
    pub shape: TagShape,

    /// Who writes it.
    pub author: TagAuthor,
}

impl AsRef<str> for TagDef {
    fn as_ref(&self) -> &str {
        self.key
    }
}

impl From<TagDef> for String {
    fn from(def: TagDef) -> Self {
        def.key.into()
    }
}

/// What a kit adds to the vocabulary: its own keys (each prefixed with the
/// kit's name and a dot), and the keys every one of its rows must carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KitTags {
    /// The kit's own keys.
    pub keys: &'static [TagDef],

    /// The keys, shared or its own, every row of the kit carries.
    pub required: &'static [&'static str],
}

impl KitTags {
    /// No keys of its own, none required.
    pub const NONE: KitTags = KitTags {
        keys: &[],
        required: &[],
    };
}

/// The row's kit: `"qwen3"`.
pub const KIT: TagDef = TagDef {
    key: "kit",
    description: "the kit the row is a model of",
    shape: TagShape::Text,
    author: TagAuthor::Catalog,
};

/// The model variation groups the model is part of: `["qwen3", "qwen3-2507"]`.
pub const GROUPS: TagDef = TagDef {
    key: "groups",
    description: "the model variation groups the model is part of",
    shape: TagShape::TextList,
    author: TagAuthor::Row,
};

/// A release in a sequence of releases: `{group: "whisper-large", version:
/// "3.0.0"}`. Only where the change between releases is quality or training,
/// not architecture; all else equal, the most recent is wanted. Releases
/// order by semver precedence, then by the `+tag`: an untagged version
/// before a tagged one, two tags dot-split, a numeric part as a number.
pub const VERSION: TagDef = TagDef {
    key: "version",
    description: "a release in a sequence: the sequence, and a semver version (with an optional +tag)",
    shape: TagShape::Version,
    author: TagAuthor::Row,
};

/// The languages the model handles, as canonical BCP 47 tags; `"*"` stands for
/// unspecified others: `["en"]`, `["en", "zh", "*"]`, `["*"]`.
pub const LANGUAGE: TagDef = TagDef {
    key: "language",
    description: "the languages it handles, as BCP 47 tags; \"*\" for unspecified others",
    shape: TagShape::Languages,
    author: TagAuthor::Row,
};

/// The total parameter count, tied weights counted once, as the publisher
/// states it (often rounded, as a model's name does): `600000000`.
pub const PARAMETERS: TagDef = TagDef {
    key: "parameters",
    description: "the total parameter count, tied weights counted once, as the publisher states it",
    shape: TagShape::Count,
    author: TagAuthor::Row,
};

/// A mixture of experts' per-token active parameter count.
pub const ACTIVE_PARAMETERS: TagDef = TagDef {
    key: "active_parameters",
    description: "the parameters active per token, for a mixture of experts",
    shape: TagShape::Count,
    author: TagAuthor::Row,
};

/// The element type the stored weights are in: `"bf16"`. The bytes a model
/// takes once loaded depend on the backend as well, so they are no tag.
pub const DTYPE: TagDef = TagDef {
    key: "dtype",
    description: "the element type the stored weights are in",
    shape: TagShape::OneOf(&["f32", "f16", "bf16"]),
    author: TagAuthor::CatalogElseRow,
};

/// Scores, by name: `{"wer.librispeech-clean": 1.9}`.
pub const SCORES: TagDef = TagDef {
    key: "scores",
    description: "measured scores, by name",
    shape: TagShape::Scores,
    author: TagAuthor::Row,
};

/// Whether the model was trained to reason (reinforcement learning, or
/// reasoning fine-tuning), with or without a thinking block.
pub const REASONING: TagDef = TagDef {
    key: "reasoning",
    description: "it was trained to reason",
    shape: TagShape::Bool,
    author: TagAuthor::Row,
};

/// Whether the model writes a thinking block (`<think>`): `"always"`,
/// `"switch"` (turned on and off, its default its presets'), or `"never"`.
pub const THINKING: TagDef = TagDef {
    key: "thinking",
    description: "whether it writes a thinking block: always, switch, or never",
    shape: TagShape::OneOf(&["always", "switch", "never"]),
    author: TagAuthor::Row,
};

/// Whether the model is a base model: pretrained, not tuned to follow
/// instructions or chat.
pub const BASE: TagDef = TagDef {
    key: "base",
    description: "a pretrained model, not tuned to follow instructions",
    shape: TagShape::Bool,
    author: TagAuthor::Row,
};

/// Whether the model can translate (a speech model's `[translate]`, say).
pub const TRANSLATE: TagDef = TagDef {
    key: "translate",
    description: "it can translate",
    shape: TagShape::Bool,
    author: TagAuthor::Row,
};

/// The shared vocabulary, every key a bare name.
pub const VOCABULARY: &[TagDef] = &[
    KIT,
    GROUPS,
    VERSION,
    LANGUAGE,
    PARAMETERS,
    ACTIVE_PARAMETERS,
    DTYPE,
    SCORES,
    REASONING,
    THINKING,
    BASE,
    TRANSLATE,
];

/// The shared key called `key`.
pub fn shared_tag(key: &str) -> Option<&'static TagDef> {
    VOCABULARY.iter().find(|def| def.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary() {
        for (i, def) in VOCABULARY.iter().enumerate() {
            assert!(!def.key.contains('.'), "{}: a shared key is bare", def.key);
            assert!(
                VOCABULARY[..i].iter().all(|d| d.key != def.key),
                "{} twice",
                def.key
            );
        }
        assert_eq!(shared_tag("dtype"), Some(&DTYPE));
        assert_eq!(shared_tag("size"), None);
        assert_eq!(KIT.as_ref(), "kit");
        assert_eq!(String::from(KIT), "kit");
        assert_eq!(KitTags::NONE.required, &[] as &[&str]);
    }
}
