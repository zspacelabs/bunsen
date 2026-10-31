//! Fetches the Qwen3 assets.
//!
//! Three sets, each behind its own feature: `checkpoint` is Qwen's
//! `Qwen3-0.6B` and `large_checkpoints` its `Qwen3-1.7B`, `Qwen3-4B` and
//! `Qwen3-4B-Instruct-2507`, each as `transformers` ships it and laid out as a
//! bunsen pretrained cache; `onnx_gen` is `onnx-community`'s fp32 ONNX export
//! of `Qwen3-0.6B`, which `qwen3-model-validation` compares bunsen against.
//!
//! `burn-onnx` 0.22 cannot import the export as published, so the graph is
//! rewritten first, by `rewrite_genai_onnx::rewrite_for_burn_onnx`.
//!
//! Fetching, pinning and the overrides (one per file, named for the row and
//! the file: `QWEN3_0_6B_MODEL_SAFETENSORS`, `QWEN3_ONNX_MODEL`, ...) follow
//! `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
    place_file,
    pretrained_cache_path,
};

/// The export's graph: an onnxruntime-genai export of `Qwen3-0.6B`, fp32,
/// opset 14 with `com.microsoft` ops, a 28-layer KV cache; part of its
/// weights inline, the rest in [`ONNX_DATA`].
const ONNX_MODEL: Asset = Asset {
    override_var: "QWEN3_ONNX_MODEL",
    url: "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/58ca03a5e7b354b480b1a82b34bde8dc7b854cc0/onnx/model.onnx",
    sha256: "0a8f1cfe531fc7b15d26db55090f0667863f0c62c952dff128cf23cd421e3fbc",
};

/// The graph's external weights, under the name the graph gives them.
const ONNX_DATA: Asset = Asset {
    override_var: "QWEN3_ONNX_MODEL_DATA",
    url: "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/58ca03a5e7b354b480b1a82b34bde8dc7b854cc0/onnx/model.onnx_data",
    sha256: "e0927a7de2fe7c2a9b50a121905cf2edda78fb2900c85c1790db98188bbfe4be",
};

/// The Qwen3 kit's segment of the pretrained cache layout: `QWEN3_KIT` in
/// bunsen.
const QWEN3_KIT: &str = "qwen3";

/// The namespace of Qwen's rows.
const QWEN_NAMESPACE: &str = "qwen";

/// One file of a pretrained row.
struct Pretrained {
    asset: Asset<'static>,
    file: &'static str,
}

/// `Qwen/Qwen3-0.6B`'s files, at the revision bunsen's `qwen/qwen3-0.6b` row
/// pins.
const QWEN3_0_6B: &[Pretrained] = &[
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_0_6B_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/config.json",
            sha256: "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        },
        file: "config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_0_6B_GENERATION_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/generation_config.json",
            sha256: "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        },
        file: "generation_config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_0_6B_MODEL_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors",
            sha256: "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        },
        file: "model.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_0_6B_TOKENIZER_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json",
            sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
        file: "tokenizer.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_0_6B_TOKENIZER_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json",
            sha256: "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        },
        file: "tokenizer_config.json",
    },
];

/// `Qwen/Qwen3-1.7B`'s files, at the revision bunsen's `qwen/qwen3-1.7b` row
/// pins.
const QWEN3_1_7B: &[Pretrained] = &[
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_1_7B_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json",
            sha256: "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
        },
        file: "config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_1_7B_GENERATION_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/generation_config.json",
            sha256: "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        },
        file: "generation_config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_1_7B_MODEL_00001_OF_00002_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model-00001-of-00002.safetensors",
            sha256: "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        },
        file: "model-00001-of-00002.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_1_7B_MODEL_00002_OF_00002_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model-00002-of-00002.safetensors",
            sha256: "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
        },
        file: "model-00002-of-00002.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_1_7B_MODEL_SAFETENSORS_INDEX_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model.safetensors.index.json",
            sha256: "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
        },
        file: "model.safetensors.index.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_1_7B_TOKENIZER_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer.json",
            sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
        file: "tokenizer.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_1_7B_TOKENIZER_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer_config.json",
            sha256: "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        },
        file: "tokenizer_config.json",
    },
];

/// `Qwen/Qwen3-4B`'s files, at the revision bunsen's `qwen/qwen3-4b` row pins.
const QWEN3_4B: &[Pretrained] = &[
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/config.json",
            sha256: "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
        },
        file: "config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_GENERATION_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/generation_config.json",
            sha256: "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        },
        file: "generation_config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_MODEL_00001_OF_00003_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model-00001-of-00003.safetensors",
            sha256: "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
        },
        file: "model-00001-of-00003.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_MODEL_00002_OF_00003_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model-00002-of-00003.safetensors",
            sha256: "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
        },
        file: "model-00002-of-00003.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_MODEL_00003_OF_00003_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model-00003-of-00003.safetensors",
            sha256: "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
        },
        file: "model-00003-of-00003.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_MODEL_SAFETENSORS_INDEX_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/model.safetensors.index.json",
            sha256: "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
        },
        file: "model.safetensors.index.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_TOKENIZER_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer.json",
            sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
        file: "tokenizer.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_TOKENIZER_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer_config.json",
            sha256: "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        },
        file: "tokenizer_config.json",
    },
];

/// `Qwen/Qwen3-4B-Instruct-2507`'s files, at the revision bunsen's
/// `qwen/qwen3-4b-instruct-2507` row pins.
const QWEN3_4B_INSTRUCT_2507: &[Pretrained] = &[
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/config.json",
            sha256: "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
        },
        file: "config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_GENERATION_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/generation_config.json",
            sha256: "835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48",
        },
        file: "generation_config.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_MODEL_00001_OF_00003_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/model-00001-of-00003.safetensors",
            sha256: "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6",
        },
        file: "model-00001-of-00003.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_MODEL_00002_OF_00003_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/model-00002-of-00003.safetensors",
            sha256: "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1",
        },
        file: "model-00002-of-00003.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_MODEL_00003_OF_00003_SAFETENSORS",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/model-00003-of-00003.safetensors",
            sha256: "7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d",
        },
        file: "model-00003-of-00003.safetensors",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_MODEL_SAFETENSORS_INDEX_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/model.safetensors.index.json",
            sha256: "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c",
        },
        file: "model.safetensors.index.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_TOKENIZER_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer.json",
            sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
        file: "tokenizer.json",
    },
    Pretrained {
        asset: Asset {
            override_var: "QWEN3_4B_INSTRUCT_2507_TOKENIZER_CONFIG_JSON",
            url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json",
            sha256: "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
        },
        file: "tokenizer_config.json",
    },
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=QWEN3_CACHE_DIR={}", out_dir().display());

    #[cfg(feature = "checkpoint")]
    fetch_row(QWEN3_0_6B);

    #[cfg(feature = "large_checkpoints")]
    for row in [QWEN3_1_7B, QWEN3_4B, QWEN3_4B_INSTRUCT_2507] {
        fetch_row(row);
    }

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
}

/// Fetches every file of a row into the cache layout.
fn fetch_row(row: &[Pretrained]) {
    for p in row {
        let dest = pretrained_cache_path(
            &out_dir(),
            QWEN3_KIT,
            QWEN_NAMESPACE,
            p.asset.sha256,
            p.file,
        );
        p.asset.resolve(&dest);
    }
}

/// Fetches the export, rewrites it, and generates the Rust model from it.
#[cfg(feature = "onnx_gen")]
fn generate_reference() {
    use rewrite_genai_onnx::onnx_ir::{
        Message,
        ModelProto,
    };

    let out = out_dir();
    // The generated weights are loaded from `OUT_DIR` at run time rather than
    // embedded.
    println!("cargo:rustc-env=QWEN3_ONNX_OUT_DIR={}", out.display());

    let fetched = ONNX_MODEL.resolve(&out.join("model.onnx"));
    println!("cargo:rerun-if-changed={}", fetched.display());
    // The rewritten graph reads its external weights from beside itself,
    // under their remote name, which `onnx-ir` refuses through a symlink.
    let data = out.join("model.onnx_data");
    let found = ONNX_DATA.resolve(&data);
    place_file(&found, &data).unwrap_or_else(|e| panic!("{e}"));

    let bytes =
        std::fs::read(&fetched).unwrap_or_else(|e| panic!("read {}: {e}", fetched.display()));
    let mut model = ModelProto::parse_from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", fetched.display()));
    drop(bytes);
    rewrite_genai_onnx::rewrite_for_burn_onnx(
        &mut model,
        &[
            ("GroupQueryAttention", 28),
            ("RotaryEmbedding", 56),
            ("SimplifiedLayerNormalization", 57),
            ("SkipSimplifiedLayerNormalization", 56),
        ],
    );

    // `ModelGen` names the generated module after the file.
    let patched = out.join("qwen3.onnx");
    std::fs::write(
        &patched,
        model.write_to_bytes().expect("serialize the graph"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", patched.display()));
    drop(model);

    burn_onnx::ModelGen::new()
        .input(patched.to_str().expect("asset path is UTF-8"))
        .out_dir("./")
        .run_from_script();
}
