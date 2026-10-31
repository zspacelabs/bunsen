//! Fetches the nanochat checkpoints.
//!
//! One set per row, each behind its own feature, laid out as a bunsen
//! pretrained cache under the paths the `nanochat` kit's rows give their
//! files. Fetching, pinning and the overrides (one per file, named for the
//! row and the file: `NANOCHAT_RL_D20_MODEL_000650_PT`,
//! `NANOCHAT_D18_SFT_TOKENIZER_PKL`, ...) follow `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
    pretrained_cache_path,
};

/// The nanochat kit's segment of the pretrained cache layout:
/// `NANOCHAT_KIT` in bunsen.
const NANOCHAT_KIT: &str = "nanochat";

/// One row's files: the cache namespace they are kept under, and each file's
/// path in its repository with the asset that fetches it.
struct Row {
    namespace: &'static str,
    files: &'static [(&'static str, Asset<'static>)],
}

/// `nanochat-students/base-d20`.
const BASE_D20: Row = Row {
    namespace: "nanochat-students",
    files: &[
        (
            "model_021400.pt",
            Asset {
                override_var: "NANOCHAT_BASE_D20_MODEL_021400_PT",
                url: "https://huggingface.co/nanochat-students/base-d20/resolve/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61/model_021400.pt",
                sha256: "574670de45c667a98fe72c2d145f24007ecc1778721e2481e41500842dd3ba13",
            },
        ),
        (
            "meta_021400.json",
            Asset {
                override_var: "NANOCHAT_BASE_D20_META_021400_JSON",
                url: "https://huggingface.co/nanochat-students/base-d20/resolve/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61/meta_021400.json",
                sha256: "a34a9b144a30d2a8066e64a91b7becc8579ea355c075e20427c28afb83ef106c",
            },
        ),
        (
            "tokenizer.pkl",
            Asset {
                override_var: "NANOCHAT_BASE_D20_TOKENIZER_PKL",
                url: "https://huggingface.co/nanochat-students/base-d20/resolve/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61/tokenizer.pkl",
                sha256: "8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b",
            },
        ),
        (
            "token_bytes.pt",
            Asset {
                override_var: "NANOCHAT_BASE_D20_TOKEN_BYTES_PT",
                url: "https://huggingface.co/nanochat-students/base-d20/resolve/4c6e4e8d8468a6b66302a472cb950bb73a1e2a61/token_bytes.pt",
                sha256: "e1b6cdee5d02fe1018b2b1d2ae5b736be665f9c0e7d10c81dcf935e7efaf8cb5",
            },
        ),
    ],
};

/// `nanochat-students/mid-d20`.
const MID_D20: Row = Row {
    namespace: "nanochat-students",
    files: &[
        (
            "model_000765.pt",
            Asset {
                override_var: "NANOCHAT_MID_D20_MODEL_000765_PT",
                url: "https://huggingface.co/nanochat-students/mid-d20/resolve/8db9593636d2b4dd8546cb211f3586f90da49269/model_000765.pt",
                sha256: "cc0560b6adf23e663191ba3e73707381b648e94a6489493c2589baf87eb1863a",
            },
        ),
        (
            "meta_000765.json",
            Asset {
                override_var: "NANOCHAT_MID_D20_META_000765_JSON",
                url: "https://huggingface.co/nanochat-students/mid-d20/resolve/8db9593636d2b4dd8546cb211f3586f90da49269/meta_000765.json",
                sha256: "1282220593eaf3e120b3e6be0eb3894e054638d46021511ede8d60b074ece58d",
            },
        ),
        (
            "tokenizer.pkl",
            Asset {
                override_var: "NANOCHAT_MID_D20_TOKENIZER_PKL",
                url: "https://huggingface.co/nanochat-students/mid-d20/resolve/8db9593636d2b4dd8546cb211f3586f90da49269/tokenizer.pkl",
                sha256: "8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b",
            },
        ),
        (
            "token_bytes.pt",
            Asset {
                override_var: "NANOCHAT_MID_D20_TOKEN_BYTES_PT",
                url: "https://huggingface.co/nanochat-students/mid-d20/resolve/8db9593636d2b4dd8546cb211f3586f90da49269/token_bytes.pt",
                sha256: "e1b6cdee5d02fe1018b2b1d2ae5b736be665f9c0e7d10c81dcf935e7efaf8cb5",
            },
        ),
    ],
};

/// `nanochat-students/rl-d20`, its final checkpoint.
const RL_D20: Row = Row {
    namespace: "nanochat-students",
    files: &[
        (
            "model_000650.pt",
            Asset {
                override_var: "NANOCHAT_RL_D20_MODEL_000650_PT",
                url: "https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a82b381a7b0b4f04f3bc5f3580a996745e/model_000650.pt",
                sha256: "ff2eee182e2aa396615d3b481ddf17884a7dbabb3caa47f66eede343135accff",
            },
        ),
        (
            "meta_000650.json",
            Asset {
                override_var: "NANOCHAT_RL_D20_META_000650_JSON",
                url: "https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a82b381a7b0b4f04f3bc5f3580a996745e/meta_000650.json",
                sha256: "353d774b8548a317a5e71762b775ee8ea506332f9b2acf8dff4fbfd9e813131a",
            },
        ),
        (
            "tokenizer.pkl",
            Asset {
                override_var: "NANOCHAT_RL_D20_TOKENIZER_PKL",
                url: "https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a82b381a7b0b4f04f3bc5f3580a996745e/tokenizer.pkl",
                sha256: "8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b",
            },
        ),
        (
            "token_bytes.pt",
            Asset {
                override_var: "NANOCHAT_RL_D20_TOKEN_BYTES_PT",
                url: "https://huggingface.co/nanochat-students/rl-d20/resolve/22fbb6a82b381a7b0b4f04f3bc5f3580a996745e/token_bytes.pt",
                sha256: "e1b6cdee5d02fe1018b2b1d2ae5b736be665f9c0e7d10c81dcf935e7efaf8cb5",
            },
        ),
    ],
};

/// `nitishpandey04/nanochat-d18-base`.
const D18_BASE: Row = Row {
    namespace: "nitishpandey04",
    files: &[
        (
            "base_checkpoints/d18/model_003712.pt",
            Asset {
                override_var: "NANOCHAT_D18_BASE_MODEL_003712_PT",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-base/resolve/bd4999f3186942207bac6b4b41ff743381e5fc8b/base_checkpoints/d18/model_003712.pt",
                sha256: "979a5ac9ea785da98d38dd9b58bd07d39d06230a6644d1f4e157f045499989b3",
            },
        ),
        (
            "base_checkpoints/d18/meta_003712.json",
            Asset {
                override_var: "NANOCHAT_D18_BASE_META_003712_JSON",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-base/resolve/bd4999f3186942207bac6b4b41ff743381e5fc8b/base_checkpoints/d18/meta_003712.json",
                sha256: "7ecc0e6f4f0315d106d4fb213fec593879456475a21de34b71c329a4a8a4e7b4",
            },
        ),
        (
            "tokenizer/tokenizer.pkl",
            Asset {
                override_var: "NANOCHAT_D18_BASE_TOKENIZER_PKL",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-base/resolve/bd4999f3186942207bac6b4b41ff743381e5fc8b/tokenizer/tokenizer.pkl",
                sha256: "ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2",
            },
        ),
        (
            "tokenizer/token_bytes.pt",
            Asset {
                override_var: "NANOCHAT_D18_BASE_TOKEN_BYTES_PT",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-base/resolve/bd4999f3186942207bac6b4b41ff743381e5fc8b/tokenizer/token_bytes.pt",
                sha256: "009ef93d20dd4684497c19c4b7fed29278b53f20022d2dc39264e27b9eefa2a8",
            },
        ),
    ],
};

/// `nitishpandey04/nanochat-d18-sft`.
const D18_SFT: Row = Row {
    namespace: "nitishpandey04",
    files: &[
        (
            "chatsft_checkpoints/d18/model_000478.pt",
            Asset {
                override_var: "NANOCHAT_D18_SFT_MODEL_000478_PT",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-sft/resolve/e2c7c119fee2a0caf6a0c69704ccbe57bad87e63/chatsft_checkpoints/d18/model_000478.pt",
                sha256: "951589ed950a12358d69ab3180c39a237cf24fea80f01a4b6691f90e3bacf17a",
            },
        ),
        (
            "chatsft_checkpoints/d18/meta_000478.json",
            Asset {
                override_var: "NANOCHAT_D18_SFT_META_000478_JSON",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-sft/resolve/e2c7c119fee2a0caf6a0c69704ccbe57bad87e63/chatsft_checkpoints/d18/meta_000478.json",
                sha256: "e0acf64f9e195a42d5de9dc284f5d9a814a0b29d7a48e89f5f96ebb891967028",
            },
        ),
        (
            "tokenizer/tokenizer.pkl",
            Asset {
                override_var: "NANOCHAT_D18_SFT_TOKENIZER_PKL",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-sft/resolve/e2c7c119fee2a0caf6a0c69704ccbe57bad87e63/tokenizer/tokenizer.pkl",
                sha256: "ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2",
            },
        ),
        (
            "tokenizer/token_bytes.pt",
            Asset {
                override_var: "NANOCHAT_D18_SFT_TOKEN_BYTES_PT",
                url: "https://huggingface.co/nitishpandey04/nanochat-d18-sft/resolve/e2c7c119fee2a0caf6a0c69704ccbe57bad87e63/tokenizer/token_bytes.pt",
                sha256: "009ef93d20dd4684497c19c4b7fed29278b53f20022d2dc39264e27b9eefa2a8",
            },
        ),
    ],
};

/// Fetches `row`'s files into their pretrained-cache paths under `OUT_DIR`.
fn fetch(row: &Row) {
    for (file, asset) in row.files {
        let dest =
            pretrained_cache_path(&out_dir(), NANOCHAT_KIT, row.namespace, asset.sha256, file);
        asset.resolve(&dest);
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=NANOCHAT_CACHE_DIR={}", out_dir().display());

    #[cfg(feature = "base_d20")]
    fetch(&BASE_D20);
    #[cfg(feature = "mid_d20")]
    fetch(&MID_D20);
    #[cfg(feature = "rl_d20")]
    fetch(&RL_D20);
    #[cfg(feature = "d18_base")]
    fetch(&D18_BASE);
    #[cfg(feature = "d18_sft")]
    fetch(&D18_SFT);
}
