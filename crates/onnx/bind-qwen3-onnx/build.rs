//! Fetches the Qwen3 assets.
//!
//! Three sets, each behind its own feature: `checkpoint` is Qwen's
//! `Qwen3-0.6B` and `large_checkpoints` its `Qwen3-1.7B`, `Qwen3-4B` and
//! `Qwen3-4B-Instruct-2507`, each as `transformers` ships it and laid out as a
//! bunsen pretrained cache; `onnx_gen` is `onnx-community`'s fp32 ONNX export
//! of `Qwen3-0.6B`, which `qwen3-model-validation` compares bunsen against.
//!
//! `burn-onnx` 0.22 cannot import the export as published, so the graph is
//! rewritten first ([`rewrite_for_burn_onnx`]).
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
    use onnx_ir::{
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
    rewrite_for_burn_onnx(&mut model);

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

/// The width of one attention head in the pinned export.
#[cfg(feature = "onnx_gen")]
const HEAD_DIM: i64 = 128;

/// Rewrites the export's four onnxruntime ops, which `burn-onnx` 0.22 cannot
/// import, to standard ONNX at opset 23:
///
/// 1. `SimplifiedLayerNormalization(x, w)`, an RMS norm, becomes `x /
///    Sqrt(ReduceMean(x * x, -1) + eps) * w`.
/// 2. `com.microsoft::SkipSimplifiedLayerNormalization(a, b, w)` becomes `s = a
///    + b`, its first output the RMS norm of `s`, its fourth `s`.
/// 3. `com.microsoft::RotaryEmbedding(x, positions, cos, sin)`
///    (non-interleaved, over the whole head) becomes the half-split rotation
///    `(x1 cos - x2 sin, x2 cos + x1 sin)`, the tables gathered at the
///    positions once and shared by every layer.
/// 4. `com.microsoft::GroupQueryAttention(q, k, v, past_k, past_v, ...)`
///    becomes opset 23's `Attention` with its query and key-value head counts
///    and scale, over the cache, under an additive bias built once: key `j` is
///    visible to query `i` when `j <= past + i` and `attention_mask` keeps it.
///    The nodes that fed its own mask inputs are then dead, and dropped.
///
/// The default opset is declared 23 (`Attention`'s), and `com.microsoft`
/// is no longer imported; no other op in the graph reads differently at 23.
/// The rewrite is checked against onnxruntime's run of the published graph
/// by `qwen3-model-validation`'s fixtures.
#[cfg(feature = "onnx_gen")]
fn rewrite_for_burn_onnx(model: &mut onnx_ir::ModelProto) {
    use std::collections::{
        BTreeMap,
        HashSet,
    };

    use onnx_ir::{
        AttributeProto,
        AttributeType,
        NodeProto,
        TensorProto,
    };

    const FLOAT: i32 = 1;
    const INT64: i32 = 7;
    const BOOL: i32 = 9;

    enum Attr {
        Int(i64),
        Float(f32),
    }

    fn attribute(
        name: &str,
        value: Attr,
    ) -> AttributeProto {
        let mut a = AttributeProto::new();
        a.name = name.into();
        match value {
            Attr::Int(i) => {
                a.type_ = AttributeType::INT.into();
                a.i = i;
            }
            Attr::Float(f) => {
                a.type_ = AttributeType::FLOAT.into();
                a.f = f;
            }
        }
        a
    }

    fn read_int(
        n: &NodeProto,
        name: &str,
    ) -> Option<i64> {
        n.attribute.iter().find(|a| a.name == name).map(|a| a.i)
    }

    fn read_float(
        n: &NodeProto,
        name: &str,
    ) -> Option<f32> {
        n.attribute.iter().find(|a| a.name == name).map(|a| a.f)
    }

    /// The graph being written: its nodes, and the initializers it adds.
    struct Out {
        nodes: Vec<NodeProto>,
        initializers: Vec<TensorProto>,
        eps: BTreeMap<u32, String>,
        rope: Option<(String, String, String, String)>,
        bias: Option<String>,
    }

    impl Out {
        fn node(
            &mut self,
            op: &str,
            name: &str,
            inputs: &[&str],
            outputs: &[&str],
            attrs: Vec<AttributeProto>,
        ) -> String {
            let mut n = NodeProto::new();
            n.op_type = op.into();
            n.name = name.into();
            n.input = inputs.iter().map(|s| s.to_string()).collect();
            n.output = outputs.iter().map(|s| s.to_string()).collect();
            n.attribute = attrs;
            self.nodes.push(n);
            outputs[0].to_string()
        }

        fn i64s(
            &mut self,
            name: &str,
            values: &[i64],
            scalar: bool,
        ) -> String {
            let mut t = TensorProto::new();
            t.name = name.into();
            t.data_type = INT64;
            t.dims = if scalar {
                vec![]
            } else {
                vec![values.len() as i64]
            };
            t.int64_data = values.to_vec();
            self.initializers.push(t);
            name.into()
        }

        fn f32_scalar(
            &mut self,
            name: &str,
            value: f32,
        ) -> String {
            let mut t = TensorProto::new();
            t.name = name.into();
            t.data_type = FLOAT;
            t.float_data = vec![value];
            self.initializers.push(t);
            name.into()
        }

        /// `x` RMS-normalized over its last axis, then scaled by `w`, into
        /// `out`.
        fn rms_norm(
            &mut self,
            x: &str,
            w: &str,
            eps: f32,
            out: &str,
            p: &str,
        ) {
            let eps = match self.eps.get(&eps.to_bits()) {
                Some(name) => name.clone(),
                None => {
                    let name = self.f32_scalar(&format!("rw/eps{}", self.eps.len()), eps);
                    self.eps.insert(eps.to_bits(), name.clone());
                    name
                }
            };
            let sq = self.node(
                "Mul",
                &format!("{p}/sq"),
                &[x, x],
                &[&format!("{p}/sq")],
                vec![],
            );
            let ms = self.node(
                "ReduceMean",
                &format!("{p}/ms"),
                &[&sq, "rw/axes_last"],
                &[&format!("{p}/ms")],
                vec![attribute("keepdims", Attr::Int(1))],
            );
            let ad = self.node(
                "Add",
                &format!("{p}/ad"),
                &[&ms, &eps],
                &[&format!("{p}/ad")],
                vec![],
            );
            let sd = self.node(
                "Sqrt",
                &format!("{p}/sd"),
                &[&ad],
                &[&format!("{p}/sd")],
                vec![],
            );
            let nx = self.node(
                "Div",
                &format!("{p}/nx"),
                &[x, &sd],
                &[&format!("{p}/nx")],
                vec![],
            );
            self.node("Mul", &format!("{p}/w"), &[&nx, w], &[out], vec![]);
        }

        /// The `[batch, seq, 1, half]` cosine and sine tables at the positions,
        /// gathered once.
        fn rope_tables(
            &mut self,
            positions: &str,
            cos_cache: &str,
            sin_cache: &str,
        ) -> (String, String) {
            if let Some((pos, cc, _, _)) = &self.rope {
                assert!(
                    pos == positions && cc == cos_cache,
                    "every rotary node shares one table"
                );
            } else {
                let p = "rw/rope";
                let axis0 = vec![attribute("axis", Attr::Int(0))];
                let c3 = self.node(
                    "Gather",
                    &format!("{p}/cos3"),
                    &[cos_cache, positions],
                    &[&format!("{p}/cos3")],
                    axis0.clone(),
                );
                let s3 = self.node(
                    "Gather",
                    &format!("{p}/sin3"),
                    &[sin_cache, positions],
                    &[&format!("{p}/sin3")],
                    axis0,
                );
                let ax2 = self.i64s(&format!("{p}/ax2"), &[2], false);
                let cos = self.node(
                    "Unsqueeze",
                    &format!("{p}/cos"),
                    &[&c3, &ax2],
                    &[&format!("{p}/cos")],
                    vec![],
                );
                let sin = self.node(
                    "Unsqueeze",
                    &format!("{p}/sin"),
                    &[&s3, &ax2],
                    &[&format!("{p}/sin")],
                    vec![],
                );
                self.rope = Some((positions.into(), cos_cache.into(), cos, sin));
            }
            let (_, _, cos, sin) = self.rope.as_ref().unwrap();
            (cos.clone(), sin.clone())
        }

        /// The `[batch, 1, seq, total]` additive attention bias, built once.
        fn attention_bias(&mut self) -> String {
            if let Some(bias) = &self.bias {
                return bias.clone();
            }
            let p = "rw/bias";
            let n = |s: &str| format!("{p}/{s}");
            let i0 = self.i64s(&n("i0"), &[0], true);
            let i1 = self.i64s(&n("i1"), &[1], true);
            let axis0 = || vec![attribute("axis", Attr::Int(0))];
            let sh_ids = self.node(
                "Shape",
                &n("sh_ids"),
                &["input_ids"],
                &[&n("sh_ids")],
                vec![],
            );
            let sh_m = self.node(
                "Shape",
                &n("sh_m"),
                &["attention_mask"],
                &[&n("sh_m")],
                vec![],
            );
            let seq = self.node("Gather", &n("S"), &[&sh_ids, &i1], &[&n("S")], axis0());
            let total = self.node("Gather", &n("T"), &[&sh_m, &i1], &[&n("T")], axis0());
            let past = self.node("Sub", &n("P"), &[&total, &seq], &[&n("P")], vec![]);
            let rows = self.node(
                "Range",
                &n("rows"),
                &[&past, &total, &i1],
                &[&n("rows")],
                vec![],
            );
            let cols = self.node(
                "Range",
                &n("cols"),
                &[&i0, &total, &i1],
                &[&n("cols")],
                vec![],
            );
            let ax1 = self.i64s(&n("ax1"), &[1], false);
            let ax0 = self.i64s(&n("ax0"), &[0], false);
            let ax12 = self.i64s(&n("ax12"), &[1, 2], false);
            let rows2 = self.node(
                "Unsqueeze",
                &n("rows2"),
                &[&rows, &ax1],
                &[&n("rows2")],
                vec![],
            );
            let cols2 = self.node(
                "Unsqueeze",
                &n("cols2"),
                &[&cols, &ax0],
                &[&n("cols2")],
                vec![],
            );
            let causal = self.node(
                "LessOrEqual",
                &n("causal"),
                &[&cols2, &rows2],
                &[&n("causal")],
                vec![],
            );
            let keep = self.node(
                "Cast",
                &n("mb"),
                &["attention_mask"],
                &[&n("mb")],
                vec![attribute("to", Attr::Int(BOOL as i64))],
            );
            let keep4 = self.node(
                "Unsqueeze",
                &n("mb4"),
                &[&keep, &ax12],
                &[&n("mb4")],
                vec![],
            );
            let ok = self.node("And", &n("ok"), &[&causal, &keep4], &[&n("ok")], vec![]);
            let zero = self.f32_scalar(&n("zero"), 0.0);
            let neg = self.f32_scalar(&n("neg"), f32::MIN);
            let bias = self.node(
                "Where",
                &n("bias"),
                &[&ok, &zero, &neg],
                &[&n("bias")],
                vec![],
            );
            self.bias = Some(bias.clone());
            bias
        }
    }

    let graph = model.graph.mut_or_insert_default();
    let mut out = Out {
        nodes: Vec::new(),
        initializers: Vec::new(),
        eps: BTreeMap::new(),
        rope: None,
        bias: None,
    };
    out.i64s("rw/axes_last", &[-1], false);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for n in std::mem::take(&mut graph.node) {
        let p = format!("rw/{}", n.name);
        let op = n.op_type.as_str();
        if matches!(
            op,
            "SimplifiedLayerNormalization"
                | "SkipSimplifiedLayerNormalization"
                | "RotaryEmbedding"
                | "GroupQueryAttention"
        ) {
            *counts.entry(op.to_string()).or_default() += 1;
        }
        match op {
            "SimplifiedLayerNormalization" => {
                assert_eq!(read_int(&n, "axis").unwrap_or(-1), -1, "{}", n.name);
                assert!(n.output.iter().skip(1).all(String::is_empty), "{}", n.name);
                let eps = read_float(&n, "epsilon").expect("an epsilon");
                out.rms_norm(&n.input[0], &n.input[1], eps, &n.output[0], &p);
            }
            "SkipSimplifiedLayerNormalization" => {
                assert_eq!(n.input.len(), 3, "{}: no bias input", n.name);
                assert!(
                    n.output.iter().skip(1).take(2).all(String::is_empty),
                    "{}: no mean or inverse-variance output",
                    n.name
                );
                let sum = match n.output.get(3) {
                    Some(s) if !s.is_empty() => s.clone(),
                    _ => format!("{p}/sum"),
                };
                out.node(
                    "Add",
                    &format!("{p}/sum"),
                    &[&n.input[0], &n.input[1]],
                    &[&sum],
                    vec![],
                );
                let eps = read_float(&n, "epsilon").expect("an epsilon");
                out.rms_norm(&sum, &n.input[2], eps, &n.output[0], &p);
            }
            "RotaryEmbedding" => {
                assert_eq!(read_int(&n, "interleaved").unwrap_or(0), 0, "{}", n.name);
                assert_eq!(
                    read_int(&n, "rotary_embedding_dim").unwrap_or(0),
                    0,
                    "{}: a rotation over the whole head",
                    n.name
                );
                let [x, positions, cos_cache, sin_cache] = n.input.as_slice() else {
                    panic!("{}: four inputs", n.name);
                };
                let (cos, sin) = out.rope_tables(positions, cos_cache, sin_cache);
                let q = |s: &str| format!("{p}/{s}");
                let s4 = out.i64s(&q("s4"), &[0, 0, -1, HEAD_DIM], false);
                let s3 = out.i64s(&q("s3"), &[0, 0, -1], false);
                let halves = out.i64s(&q("halves"), &[HEAD_DIM / 2, HEAD_DIM / 2], false);
                let x4 = out.node("Reshape", &q("x4"), &[x, &s4], &[&q("x4")], vec![]);
                out.node(
                    "Split",
                    &q("split"),
                    &[&x4, &halves],
                    &[&q("x1"), &q("x2")],
                    vec![attribute("axis", Attr::Int(-1))],
                );
                let (x1, x2) = (q("x1"), q("x2"));
                let a = out.node("Mul", &q("a"), &[&x1, &cos], &[&q("a")], vec![]);
                let b = out.node("Mul", &q("b"), &[&x2, &sin], &[&q("b")], vec![]);
                let c = out.node("Mul", &q("c"), &[&x2, &cos], &[&q("c")], vec![]);
                let d = out.node("Mul", &q("d"), &[&x1, &sin], &[&q("d")], vec![]);
                let o1 = out.node("Sub", &q("o1"), &[&a, &b], &[&q("o1")], vec![]);
                let o2 = out.node("Add", &q("o2"), &[&c, &d], &[&q("o2")], vec![]);
                let o = out.node(
                    "Concat",
                    &q("o"),
                    &[&o1, &o2],
                    &[&q("o")],
                    vec![attribute("axis", Attr::Int(-1))],
                );
                out.node("Reshape", &q("out"), &[&o, &s3], &[&n.output[0]], vec![]);
            }
            "GroupQueryAttention" => {
                assert_eq!(read_int(&n, "do_rotary").unwrap_or(0), 0, "{}", n.name);
                assert_eq!(read_float(&n, "softcap").unwrap_or(0.0), 0.0, "{}", n.name);
                let bias = out.attention_bias();
                let [q, k, v, past_k, past_v, ..] = n.input.as_slice() else {
                    panic!("{}: q, k, v and the cache", n.name);
                };
                let outputs: Vec<&str> = n.output.iter().map(String::as_str).collect();
                out.node(
                    "Attention",
                    &p,
                    &[q, k, v, &bias, past_k, past_v],
                    &outputs,
                    vec![
                        attribute("q_num_heads", Attr::Int(read_int(&n, "num_heads").unwrap())),
                        attribute(
                            "kv_num_heads",
                            Attr::Int(read_int(&n, "kv_num_heads").unwrap()),
                        ),
                        attribute("scale", Attr::Float(read_float(&n, "scale").unwrap())),
                    ],
                );
            }
            _ => out.nodes.push(n),
        }
    }
    let expected: BTreeMap<String, usize> = [
        ("GroupQueryAttention", 28),
        ("RotaryEmbedding", 56),
        ("SimplifiedLayerNormalization", 57),
        ("SkipSimplifiedLayerNormalization", 56),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    assert_eq!(counts, expected, "is this the pinned graph?");

    // Drop what no output reads: the attention's own mask inputs.
    let mut live: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    let mut kept = Vec::new();
    for n in out.nodes.into_iter().rev() {
        if n.output.iter().any(|o| live.contains(o)) {
            live.extend(n.input.iter().filter(|i| !i.is_empty()).cloned());
            kept.push(n);
        }
    }
    kept.reverse();
    graph.node = kept;
    graph.initializer.extend(out.initializers);
    graph.value_info.clear();

    model
        .opset_import
        .retain(|o| o.domain.is_empty() || o.domain == "ai.onnx");
    assert_eq!(model.opset_import.len(), 1, "one default opset");
    model.opset_import[0].version = 23;
    model.ir_version = 10;
}
