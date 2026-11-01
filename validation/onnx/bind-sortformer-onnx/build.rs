//! Fetches the Sortformer references, each behind its own feature:
//! `onnx_gen` is `altunenes/parakeet-rs`'s ONNX export of NVIDIA's
//! `Nemotron-3-Diarization`, and `onnx_gen_v2_1` `Scrybl`'s of the streaming
//! Sortformer v2.1, which `sortformer-model-validation` compares bunsen
//! against. The checkpoints themselves are not fetched here: the validation
//! crate's tests load them through bunsen's own pinned rows, at run time,
//! into the weights cache.
//!
//! `burn-onnx` 0.22 cannot import the Nemotron export as published, so that
//! graph is rewritten first ([`rewrite_for_burn_onnx`]); the rewrite changes
//! no value the graph computes. The v2.1 export imports as published.
//!
//! Fetching, pinning and the overrides (`SORTFORMER_ONNX_NEMOTRON`,
//! `SORTFORMER_ONNX_V2_1`) follow `bunsen-fetch`'s rules.

// Each feature uses its own constants; the rest are dead in that build.
#![allow(unused)]

use bunsen_fetch::{
    Asset,
    out_dir,
};

/// The export: one streaming step, `fp32`, dynamic time axes, its weights
/// inside the file.
const NEMOTRON_ONNX: Asset = Asset {
    override_var: "SORTFORMER_ONNX_NEMOTRON",
    url: "https://huggingface.co/altunenes/parakeet-rs/resolve/4d2a8bc71f5c896ec40faa59732e6716295edaf2/nemotron-3-diarization/nemotron3_diar_v3.onnx",
    sha256: "915e4fa23b0192ed9fadeb1cdd26847df986d50c92012d177be28d0343bbe03a",
};

/// `Scrybl`'s export of NVIDIA's `diar_streaming_sortformer_4spk-v2.1`: one
/// `NeMo` streaming step, `fp32`, opset 17, dynamic time axes, its weights
/// inside the file.
const V2_1_ONNX: Asset = Asset {
    override_var: "SORTFORMER_ONNX_V2_1",
    url: "https://huggingface.co/Scrybl/diar_streaming_sortformer_4spk-v2.1/resolve/3be92087de17e4e9063c9cfabc104f9809b2d037/diar_streaming_sortformer_4spk-v2.1.onnx",
    sha256: "0182539aa7f8ecfc446ec56eb320b6f9131d686f98ca8f40644830af7d4d81c8",
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // `cfg` rather than a `CARGO_FEATURE_*` lookup: `burn_onnx` is an
    // optional build-dependency, so without the feature it is not linked.
    #[cfg(feature = "onnx_gen")]
    generate_reference();
    #[cfg(feature = "onnx_gen_v2_1")]
    generate_v2_1_reference();
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
    println!("cargo:rustc-env=SORTFORMER_ONNX_OUT_DIR={}", out.display());

    let fetched = NEMOTRON_ONNX.resolve(&out.join("nemotron3_diar_v3.onnx"));
    println!("cargo:rerun-if-changed={}", fetched.display());

    let bytes =
        std::fs::read(&fetched).unwrap_or_else(|e| panic!("read {}: {e}", fetched.display()));
    let mut model = ModelProto::parse_from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", fetched.display()));
    drop(bytes);
    rewrite_for_burn_onnx(&mut model);

    // `ModelGen` names the generated module after the file.
    let patched = out.join("nemotron_diarization.onnx");
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

/// Fetches the v2.1 export and generates the Rust model from it, as
/// published.
#[cfg(feature = "onnx_gen_v2_1")]
fn generate_v2_1_reference() {
    let out = out_dir();
    println!("cargo:rustc-env=SORTFORMER_ONNX_OUT_DIR={}", out.display());

    let fetched = V2_1_ONNX.resolve(&out.join("diar_streaming_sortformer_4spk-v2.1.onnx"));
    println!("cargo:rerun-if-changed={}", fetched.display());

    // `ModelGen` names the generated module after the file, which must
    // therefore be a Rust identifier: a link, not a second copy.
    let named = out.join("sortformer_v2_1.onnx");
    let _ = std::fs::remove_file(&named);
    if std::fs::hard_link(&fetched, &named).is_err() {
        std::fs::copy(&fetched, &named)
            .unwrap_or_else(|e| panic!("copy {} to {}: {e}", fetched.display(), named.display()));
    }

    burn_onnx::ModelGen::new()
        .input(named.to_str().expect("asset path is UTF-8"))
        .out_dir("./")
        .run_from_script();
}

/// Rewrites the two constructs `burn-onnx` 0.22 cannot import, as
/// `parakeet-rs`'s own burn export does (`sortformer_patches` in its
/// `generate_burn.py`):
///
/// 1. `ScatterElements(Shape(x), [k], u)`, a shape with its `k`-th dimension
///    replaced, which `onnx-ir` types as a rank-0 tensor, becomes
///    `Concat(Slice(shape, 0, k), u, Slice(shape, k + 1, end))`. The graph has
///    three.
/// 2. The default opset is declared 19 rather than 17, for `AveragePool`'s
///    `ceil_mode`. Nothing else in the graph reads differently at 19: every
///    `Split` names its sizes.
#[cfg(feature = "onnx_gen")]
fn rewrite_for_burn_onnx(model: &mut onnx_ir::ModelProto) {
    use std::collections::HashMap;

    use onnx_ir::{
        AttributeProto,
        AttributeType,
        NodeProto,
        TensorProto,
    };

    fn i64_constant(
        name: &str,
        value: i64,
    ) -> NodeProto {
        let mut tensor = TensorProto::new();
        tensor.dims = vec![1];
        tensor.data_type = 7; // INT64
        tensor.int64_data = vec![value];
        let mut attr = AttributeProto::new();
        attr.name = "value".into();
        attr.type_ = AttributeType::TENSOR.into();
        attr.t = Some(tensor).into();
        let mut node = NodeProto::new();
        node.op_type = "Constant".into();
        node.name = format!("{name}/const");
        node.output = vec![name.into()];
        node.attribute = vec![attr];
        node
    }

    fn node(
        op: &str,
        name: String,
        inputs: Vec<String>,
        output: String,
    ) -> NodeProto {
        let mut node = NodeProto::new();
        node.op_type = op.into();
        node.name = name;
        node.input = inputs;
        node.output = vec![output];
        node
    }

    /// An `INT64` constant's values.
    fn i64_values(tensor: &TensorProto) -> Option<Vec<i64>> {
        if tensor.data_type != 7 {
            return None;
        }
        if !tensor.int64_data.is_empty() {
            return Some(tensor.int64_data.clone());
        }
        Some(
            tensor
                .raw_data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect(),
        )
    }

    let graph = model.graph.mut_or_insert_default();

    let mut op_of: HashMap<String, String> = HashMap::new();
    let mut constants: HashMap<String, Vec<i64>> = HashMap::new();
    for n in &graph.node {
        for o in &n.output {
            op_of.insert(o.clone(), n.op_type.clone());
        }
        if n.op_type == "Constant"
            && let Some(values) = n
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .and_then(|a| a.t.as_ref())
                .and_then(i64_values)
        {
            constants.insert(n.output[0].clone(), values);
        }
    }

    // New constants go first, so every node still follows its inputs.
    let mut head = Vec::new();
    let mut body = Vec::new();
    let mut rewritten = 0;
    for n in std::mem::take(&mut graph.node) {
        let shape_scatter = n.op_type == "ScatterElements"
            && op_of.get(&n.input[0]).map(String::as_str) == Some("Shape")
            && constants.get(&n.input[1]).is_some_and(|v| v.len() == 1);
        if !shape_scatter {
            body.push(n);
            continue;
        }
        let k = constants[&n.input[1]][0];
        assert!(k >= 0, "a negative scatter index in {}", n.name);
        let out = n.output[0].clone();
        let name = |s: &str| format!("{out}/{s}");
        for (s, v) in [("k0", 0), ("k", k), ("k1", k + 1), ("end", 1i64 << 62)] {
            head.push(i64_constant(&name(s), v));
        }
        let mut parts = Vec::new();
        if k > 0 {
            body.push(node(
                "Slice",
                format!("{}/head", n.name),
                vec![n.input[0].clone(), name("k0"), name("k")],
                name("head"),
            ));
            parts.push(name("head"));
        }
        parts.push(n.input[2].clone());
        body.push(node(
            "Slice",
            format!("{}/tail", n.name),
            vec![n.input[0].clone(), name("k1"), name("end")],
            name("tail"),
        ));
        parts.push(name("tail"));
        let mut concat = node("Concat", n.name.clone(), parts, out);
        let mut axis = AttributeProto::new();
        axis.name = "axis".into();
        axis.type_ = AttributeType::INT.into();
        axis.i = 0;
        concat.attribute = vec![axis];
        body.push(concat);
        rewritten += 1;
    }
    assert_eq!(
        rewritten, 3,
        "the export has three shape scatters; is this the pinned graph?"
    );
    for n in body.iter().filter(|n| n.op_type == "Split") {
        assert!(
            n.input.len() > 1 && !n.input[1].is_empty(),
            "{}: a Split without sizes reads differently at opset 19",
            n.name
        );
    }
    head.extend(body);
    graph.node = head;

    for opset in model.opset_import.iter_mut() {
        if opset.domain.is_empty() || opset.domain == "ai.onnx" {
            assert_eq!(opset.version, 17, "the export declares opset 17");
            opset.version = 19;
        }
    }
}
