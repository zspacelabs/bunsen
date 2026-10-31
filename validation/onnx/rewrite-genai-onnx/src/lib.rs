//! Rewrites onnxruntime-genai ONNX exports to standard ONNX that `burn-onnx`
//! imports.
//!
//! An onnxruntime-genai export (`onnx-community`'s builds of `transformers`
//! models) computes its attention, rotary embedding and RMS norms with ops
//! that only onnxruntime runs. `burn-onnx` 0.22 imports none of them, so a
//! bundled-asset crate's build script calls [`rewrite_for_burn_onnx`] on the
//! parsed graph before handing it to `burn_onnx::ModelGen`.
//!
//! What each op becomes is listed on [`rewrite_for_burn_onnx`]. A rewrite is
//! only trusted once a validation crate has held the generated model to
//! onnxruntime's run of the published graph: `qwen3-model-validation` and
//! `voxtral-model-validation` do.

use std::collections::{
    BTreeMap,
    HashMap,
    HashSet,
};

/// The ONNX protos the rewrite reads and writes, as `burn-onnx` parses them.
pub use onnx_ir;
use onnx_ir::{
    AttributeProto,
    AttributeType,
    ModelProto,
    NodeProto,
    TensorProto,
};

const FLOAT: i32 = 1;
const INT64: i32 = 7;
const BOOL: i32 = 9;

/// The onnxruntime ops [`rewrite_for_burn_onnx`] replaces, as their
/// `op_type`.
pub const REWRITTEN_OPS: [&str; 4] = [
    "GroupQueryAttention",
    "RotaryEmbedding",
    "SimplifiedLayerNormalization",
    "SkipSimplifiedLayerNormalization",
];

/// Rewrites the export's onnxruntime ops ([`REWRITTEN_OPS`]) to standard ONNX
/// at opset 23:
///
/// 1. `SimplifiedLayerNormalization(x, w)`, an RMS norm, becomes
///    `x/Sqrt(ReduceMean(x*x,-1)+eps)*w`.
/// 2. `com.microsoft::SkipSimplifiedLayerNormalization(a, b, w)` becomes the
///    sum `s` of `a` and `b`; its first output is the RMS norm of `s`, its
///    fourth `s`.
/// 3. `com.microsoft::RotaryEmbedding(x, positions, cos, sin)`
///    (non-interleaved, over the whole head) becomes the half-split rotation
///    `(x1*cos-x2*sin, x2*cos+x1*sin)`, the tables gathered at the positions
///    once and shared by every node that reads the same ones. The head width is
///    twice the tables' last axis.
/// 4. `com.microsoft::GroupQueryAttention(q, k, v, past_k, past_v, ...)`
///    becomes opset 23's `Attention` with its query and key-value head counts
///    and scale, over the cache, under an additive bias built once per window.
///    Key `j` is visible to query `i`, at position `p=past+i`, when `j<=p`,
///    when `attention_mask` keeps it, and, under a `local_window_size` of
///    `w>0`, when `j>p-w`: a window of `w` keys, the query's own included. With
///    `do_rotary` set, the queries and keys are first rotated as in 3, at
///    positions `past..past+seq`, by the cosine and sine caches it reads
///    (inputs 7 and 8). The nodes that fed its own mask inputs are then dead,
///    and dropped.
///
/// And one rewrite that is not of an op: every initializer and node output
/// whose name holds a character other than ASCII letters, digits, `_`, `.`
/// and `/` is renamed `rw/v{n}`, everywhere it is read. `onnx-ir` 0.22 turns
/// a value's name into an identifier by dropping such characters, so two
/// names that differ only in them become one value: the exports name their
/// constants by value (`/model/constants/INT64/[-2]`, `.../[2]`), and a
/// `Slice` reads `starts = [-2]` as `[2]`. The graph's own inputs and outputs
/// keep their names.
///
/// The default opset is declared 23 (`Attention`'s), and `com.microsoft` is
/// no longer imported; no other op the pinned exports use reads differently
/// at 23.
///
/// # Arguments
/// * `model` - the parsed export, rewritten in place.
/// * `expected` - how many of each of [`REWRITTEN_OPS`] the pinned graph holds,
///   by `op_type`; ops it does not hold are left out.
///
/// # Panics
/// If the graph holds a different count of any rewritten op than `expected`
/// says (it is not the pinned graph), or an op uses a form not covered above
/// (interleaved or partial rotation, a softcap, a norm's statistics outputs).
pub fn rewrite_for_burn_onnx(
    model: &mut ModelProto,
    expected: &[(&str, usize)],
) {
    let graph = model.graph.mut_or_insert_default();
    let dims: HashMap<String, Vec<i64>> = graph
        .initializer
        .iter()
        .map(|t| (t.name.clone(), t.dims.clone()))
        .collect();
    let mut out = Out {
        nodes: Vec::new(),
        initializers: Vec::new(),
        eps: BTreeMap::new(),
        tables: BTreeMap::new(),
        steps: None,
        biases: BTreeMap::new(),
        dims,
    };
    out.i64s("rw/axes_last", &[-1], false);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for n in std::mem::take(&mut graph.node) {
        let p = format!("rw/{}", n.name);
        let op = n.op_type.as_str();
        if REWRITTEN_OPS.contains(&op) {
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
                let (cos, sin, head_dim) = out.rope_tables(positions, false, cos_cache, sin_cache);
                out.rotate(x, &cos, &sin, head_dim, &n.output[0], &p);
            }
            "GroupQueryAttention" => {
                assert_eq!(read_float(&n, "softcap").unwrap_or(0.0), 0.0, "{}", n.name);
                assert_eq!(
                    read_int(&n, "rotary_interleaved").unwrap_or(0),
                    0,
                    "{}",
                    n.name
                );
                let [q, k, v, past_k, past_v, ..] = n.input.as_slice() else {
                    panic!("{}: q, k, v and the cache", n.name);
                };
                let window = read_int(&n, "local_window_size").unwrap_or(-1);
                let bias = out.attention_bias(q, window);
                let (q, k) = if read_int(&n, "do_rotary").unwrap_or(0) != 0 {
                    let (Some(cos_cache), Some(sin_cache)) = (n.input.get(7), n.input.get(8))
                    else {
                        panic!("{}: do_rotary reads the caches", n.name);
                    };
                    let rows = out.steps.clone().expect("the bias built the positions").0;
                    let (cos, sin, head_dim) = out.rope_tables(&rows, true, cos_cache, sin_cache);
                    let rq = format!("{p}/q_rot");
                    let rk = format!("{p}/k_rot");
                    out.rotate(q, &cos, &sin, head_dim, &rq, &format!("{p}/rq"));
                    out.rotate(k, &cos, &sin, head_dim, &rk, &format!("{p}/rk"));
                    (rq, rk)
                } else {
                    (q.clone(), k.clone())
                };
                let outputs: Vec<&str> = n.output.iter().map(String::as_str).collect();
                out.node(
                    "Attention",
                    &p,
                    &[&q, &k, v, &bias, past_k, past_v],
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
    let expected: BTreeMap<String, usize> = expected
        .iter()
        .filter(|(_, count)| *count > 0)
        .map(|(op, count)| (op.to_string(), *count))
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
    rename_unsafe_values(graph);

    model
        .opset_import
        .retain(|o| o.domain.is_empty() || o.domain == "ai.onnx");
    assert_eq!(model.opset_import.len(), 1, "one default opset");
    model.opset_import[0].version = 23;
    model.ir_version = 10;
}

/// Whether `onnx-ir` keeps every character of `name` when it makes an
/// identifier of it.
fn is_safe_name(name: &str) -> bool {
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/'))
}

/// Renames every initializer and node output with an unsafe name
/// ([`is_safe_name`]) `rw/v{n}`, in every node that reads it; the graph's
/// inputs and outputs keep their names.
fn rename_unsafe_values(graph: &mut onnx_ir::GraphProto) {
    let io: HashSet<String> = graph
        .input
        .iter()
        .chain(graph.output.iter())
        .map(|v| v.name.clone())
        .collect();
    let mut renames: HashMap<String, String> = HashMap::new();
    let names: Vec<String> = graph
        .initializer
        .iter()
        .map(|t| t.name.clone())
        .chain(graph.node.iter().flat_map(|n| n.output.iter().cloned()))
        .collect();
    for name in names {
        if !name.is_empty() && !is_safe_name(&name) && !io.contains(&name) {
            let next = format!("rw/v{}", renames.len());
            renames.entry(name).or_insert(next);
        }
    }
    let rename = |name: &mut String| {
        if let Some(new) = renames.get(name.as_str()) {
            *name = new.clone();
        }
    };
    for t in &mut graph.initializer {
        rename(&mut t.name);
    }
    for n in &mut graph.node {
        n.input.iter_mut().for_each(rename);
        n.output.iter_mut().for_each(rename);
    }
}

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
    /// RMS epsilons, by bits, to their scalar initializer.
    eps: BTreeMap<u32, String>,
    /// Rotary tables, by (positions, cos cache), to (cos, sin, head width).
    tables: BTreeMap<(String, String), (String, String, i64)>,
    /// The query rows' positions `past..total` and the keys' `0..total`,
    /// once built.
    steps: Option<(String, String)>,
    /// Attention biases, by window.
    biases: BTreeMap<i64, String>,
    /// The input graph's initializer shapes, by name.
    dims: HashMap<String, Vec<i64>>,
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

    /// `x` RMS-normalized over its last axis, then scaled by `w`, into `out`.
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

    /// The cosine and sine tables at the positions, broadcastable against a
    /// `[batch, seq, heads, head_dim/2]` half, gathered once, and the head
    /// width.
    ///
    /// `positions` is `[batch, seq]`, or `[seq]` when `shared` (every row at
    /// the same positions).
    fn rope_tables(
        &mut self,
        positions: &str,
        shared: bool,
        cos_cache: &str,
        sin_cache: &str,
    ) -> (String, String, i64) {
        let key = (positions.to_string(), cos_cache.to_string());
        if let Some(t) = self.tables.get(&key) {
            return t.clone();
        }
        let half = match self.dims.get(cos_cache).map(Vec::as_slice) {
            Some([_, half]) => *half,
            other => panic!("{cos_cache}: a [positions, half] initializer, not {other:?}"),
        };
        let p = format!("rw/rope{}", self.tables.len());
        let axis0 = vec![attribute("axis", Attr::Int(0))];
        let c = self.node(
            "Gather",
            &format!("{p}/cos_at"),
            &[cos_cache, positions],
            &[&format!("{p}/cos_at")],
            axis0.clone(),
        );
        let s = self.node(
            "Gather",
            &format!("{p}/sin_at"),
            &[sin_cache, positions],
            &[&format!("{p}/sin_at")],
            axis0,
        );
        let axes = if shared { &[0, 2][..] } else { &[2][..] };
        let ax = self.i64s(&format!("{p}/axes"), axes, false);
        let cos = self.node(
            "Unsqueeze",
            &format!("{p}/cos"),
            &[&c, &ax],
            &[&format!("{p}/cos")],
            vec![],
        );
        let sin = self.node(
            "Unsqueeze",
            &format!("{p}/sin"),
            &[&s, &ax],
            &[&format!("{p}/sin")],
            vec![],
        );
        let t = (cos, sin, 2 * half);
        self.tables.insert(key, t.clone());
        t
    }

    /// `x`, `[batch, seq, heads*head_dim]`, rotated half-split by the tables,
    /// into `out`.
    fn rotate(
        &mut self,
        x: &str,
        cos: &str,
        sin: &str,
        head_dim: i64,
        out: &str,
        p: &str,
    ) {
        let q = |s: &str| format!("{p}/{s}");
        let s4 = self.i64s(&q("s4"), &[0, 0, -1, head_dim], false);
        let s3 = self.i64s(&q("s3"), &[0, 0, -1], false);
        let halves = self.i64s(&q("halves"), &[head_dim / 2, head_dim / 2], false);
        let x4 = self.node("Reshape", &q("x4"), &[x, &s4], &[&q("x4")], vec![]);
        self.node(
            "Split",
            &q("split"),
            &[&x4, &halves],
            &[&q("x1"), &q("x2")],
            vec![attribute("axis", Attr::Int(-1))],
        );
        let (x1, x2) = (q("x1"), q("x2"));
        let a = self.node("Mul", &q("a"), &[&x1, cos], &[&q("a")], vec![]);
        let b = self.node("Mul", &q("b"), &[&x2, sin], &[&q("b")], vec![]);
        let c = self.node("Mul", &q("c"), &[&x2, cos], &[&q("c")], vec![]);
        let d = self.node("Mul", &q("d"), &[&x1, sin], &[&q("d")], vec![]);
        let o1 = self.node("Sub", &q("o1"), &[&a, &b], &[&q("o1")], vec![]);
        let o2 = self.node("Add", &q("o2"), &[&c, &d], &[&q("o2")], vec![]);
        let o = self.node(
            "Concat",
            &q("o"),
            &[&o1, &o2],
            &[&q("o")],
            vec![attribute("axis", Attr::Int(-1))],
        );
        self.node("Reshape", &q("out"), &[&o, &s3], &[out], vec![]);
    }

    /// The query rows' positions `past..total`, `[seq]`, and the keys'
    /// `0..total`, `[total]`, built once from `q`'s sequence axis and
    /// `attention_mask`'s length.
    fn positions(
        &mut self,
        q: &str,
    ) -> (String, String) {
        if let Some((rows, cols)) = &self.steps {
            return (rows.clone(), cols.clone());
        }
        let p = "rw/pos";
        let n = |s: &str| format!("{p}/{s}");
        let i0 = self.i64s(&n("i0"), &[0], true);
        let i1 = self.i64s(&n("i1"), &[1], true);
        let axis0 = || vec![attribute("axis", Attr::Int(0))];
        let sh_q = self.node("Shape", &n("sh_q"), &[q], &[&n("sh_q")], vec![]);
        let sh_m = self.node(
            "Shape",
            &n("sh_m"),
            &["attention_mask"],
            &[&n("sh_m")],
            vec![],
        );
        let seq = self.node("Gather", &n("S"), &[&sh_q, &i1], &[&n("S")], axis0());
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
        self.steps = Some((rows.clone(), cols.clone()));
        (rows, cols)
    }

    /// The `[batch, 1, seq, total]` additive attention bias under `window`
    /// (`-1` or `0` for none), built once per window.
    fn attention_bias(
        &mut self,
        q: &str,
        window: i64,
    ) -> String {
        if let Some(bias) = self.biases.get(&window) {
            return bias.clone();
        }
        let (rows, cols) = self.positions(q);
        let p = format!("rw/bias{}", self.biases.len());
        let n = |s: &str| format!("{p}/{s}");
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
        let mut visible = self.node(
            "LessOrEqual",
            &n("causal"),
            &[&cols2, &rows2],
            &[&n("causal")],
            vec![],
        );
        if window > 0 {
            let w = self.i64s(&n("w"), &[window], true);
            let lo = self.node("Sub", &n("lo"), &[&rows2, &w], &[&n("lo")], vec![]);
            let recent = self.node(
                "Greater",
                &n("recent"),
                &[&cols2, &lo],
                &[&n("recent")],
                vec![],
            );
            visible = self.node(
                "And",
                &n("windowed"),
                &[&visible, &recent],
                &[&n("windowed")],
                vec![],
            );
        }
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
        let ok = self.node("And", &n("ok"), &[&visible, &keep4], &[&n("ok")], vec![]);
        let zero = self.f32_scalar(&n("zero"), 0.0);
        let neg = self.f32_scalar(&n("neg"), f32::MIN);
        let bias = self.node(
            "Where",
            &n("bias"),
            &[&ok, &zero, &neg],
            &[&n("bias")],
            vec![],
        );
        self.biases.insert(window, bias.clone());
        bias
    }
}

#[cfg(test)]
mod tests {
    use onnx_ir::ValueInfoProto;

    use super::*;

    fn op(
        op_type: &str,
        domain: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: Vec<AttributeProto>,
    ) -> NodeProto {
        let mut n = NodeProto::new();
        n.op_type = op_type.into();
        n.domain = domain.into();
        n.name = format!("/{op_type}");
        n.input = inputs.iter().map(|s| s.to_string()).collect();
        n.output = outputs.iter().map(|s| s.to_string()).collect();
        n.attribute = attrs;
        n
    }

    fn table(name: &str) -> TensorProto {
        let mut t = TensorProto::new();
        t.name = name.into();
        t.data_type = FLOAT;
        t.dims = vec![16, 4];
        t.float_data = vec![0.0; 64];
        t
    }

    /// A graph with one of each rewritten op, in the shapes the exports use:
    /// a norm, a standalone rotary embedding, a rotating windowed attention,
    /// and a skip norm over its output.
    fn genai_graph() -> ModelProto {
        let mut m = ModelProto::new();
        let g = m.graph.mut_or_insert_default();
        let eps = || attribute("epsilon", Attr::Float(1e-5));
        g.node = vec![
            op(
                "SimplifiedLayerNormalization",
                "",
                &["x", "w"],
                &["xn"],
                vec![eps()],
            ),
            op(
                "RotaryEmbedding",
                "com.microsoft",
                &["xn", "position_ids", "cos", "sin"],
                &["q"],
                vec![],
            ),
            op(
                "GroupQueryAttention",
                "com.microsoft",
                &[
                    "q", "xn", "xn", "past_k", "past_v", "seqlens", "total", "cos", "sin",
                ],
                &["y", "present_k", "present_v"],
                vec![
                    attribute("num_heads", Attr::Int(2)),
                    attribute("kv_num_heads", Attr::Int(1)),
                    attribute("scale", Attr::Float(0.35)),
                    attribute("do_rotary", Attr::Int(1)),
                    attribute("local_window_size", Attr::Int(4)),
                ],
            ),
            op(
                "SkipSimplifiedLayerNormalization",
                "com.microsoft",
                &["x", "y", "w"],
                &["out", "", "", "sum"],
                vec![eps()],
            ),
        ];
        g.initializer = vec![table("cos"), table("sin")];
        for name in ["out", "sum", "present_k", "present_v"] {
            let mut v = ValueInfoProto::new();
            v.name = name.into();
            g.output.push(v);
        }
        for (domain, version) in [("", 21), ("com.microsoft", 1)] {
            // `onnx-ir` does not name the opset proto; reach it through the
            // field.
            m.opset_import.push(Default::default());
            let o = m.opset_import.last_mut().unwrap();
            o.domain = domain.into();
            o.version = version;
        }
        m
    }

    const COUNTS: [(&str, usize); 4] = [
        ("GroupQueryAttention", 1),
        ("RotaryEmbedding", 1),
        ("SimplifiedLayerNormalization", 1),
        ("SkipSimplifiedLayerNormalization", 1),
    ];

    #[test]
    fn test_rewrites_every_op_to_standard_onnx() {
        let mut m = genai_graph();
        rewrite_for_burn_onnx(&mut m, &COUNTS);
        let g = m.graph.as_ref().unwrap();
        let ops: Vec<&str> = g.node.iter().map(|n| n.op_type.as_str()).collect();
        for gone in REWRITTEN_OPS {
            assert!(!ops.contains(&gone), "{gone} left in {ops:?}");
        }
        assert!(g.node.iter().all(|n| n.domain.is_empty()));
        assert_eq!(ops.iter().filter(|o| **o == "Attention").count(), 1);
        // The window is `j > p - w`.
        assert!(ops.contains(&"Greater"), "{ops:?}");
        // The attention reads the rotated queries and keys.
        let attention = g.node.iter().find(|n| n.op_type == "Attention").unwrap();
        assert!(
            attention.input[0].ends_with("q_rot"),
            "{:?}",
            attention.input
        );
        assert!(
            attention.input[1].ends_with("k_rot"),
            "{:?}",
            attention.input
        );
        // The standalone rotation and the attention's rotate at different
        // positions, so they gather separate tables.
        assert_eq!(ops.iter().filter(|o| **o == "Gather").count(), 2 * 2 + 2);
        assert_eq!(m.opset_import.len(), 1);
        assert_eq!(m.opset_import[0].version, 23);
    }

    /// Values named by their contents, which `onnx-ir` would merge, are
    /// renamed apart, everywhere they are read; the graph's inputs and
    /// outputs keep their names.
    #[test]
    fn test_renames_values_onnx_ir_would_merge() {
        let mut m = genai_graph();
        let g = m.graph.as_mut().unwrap();
        for (name, value) in [("/c/INT64/[-2]", -2), ("/c/INT64/[2]", 2)] {
            let mut t = TensorProto::new();
            t.name = name.into();
            t.data_type = INT64;
            t.int64_data = vec![value];
            g.initializer.push(t);
        }
        g.node.push(op(
            "Slice",
            "",
            &["sum", "/c/INT64/[-2]", "/c/INT64/[2]"],
            &["sliced [x]"],
            vec![],
        ));
        g.node
            .push(op("Identity", "", &["sliced [x]"], &["kept"], vec![]));
        let mut v = ValueInfoProto::new();
        v.name = "kept".into();
        g.output.push(v);

        rewrite_for_burn_onnx(&mut m, &COUNTS);
        let g = m.graph.as_ref().unwrap();
        let all_safe = g
            .node
            .iter()
            .flat_map(|n| n.input.iter().chain(&n.output))
            .chain(g.initializer.iter().map(|t| &t.name))
            .filter(|n| !n.is_empty())
            .all(|n| is_safe_name(n));
        assert!(all_safe);
        let slice = g.node.iter().find(|n| n.op_type == "Slice").unwrap();
        assert_ne!(slice.input[1], slice.input[2]);
        let value = |name: &str| {
            g.initializer
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.int64_data.clone())
        };
        assert_eq!(value(&slice.input[1]), Some(vec![-2]));
        assert_eq!(value(&slice.input[2]), Some(vec![2]));
        let identity = g.node.iter().find(|n| n.op_type == "Identity").unwrap();
        assert_eq!(identity.input[0], slice.output[0]);
        assert_eq!(identity.output[0], "kept");
    }

    #[test]
    #[should_panic(expected = "is this the pinned graph?")]
    fn test_refuses_a_graph_of_other_counts() {
        let mut m = genai_graph();
        rewrite_for_burn_onnx(&mut m, &[("GroupQueryAttention", 2)]);
    }
}
