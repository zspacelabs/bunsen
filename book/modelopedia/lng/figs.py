"""SVG figures for the language-model book. Each public function returns an <svg> string."""
import math

from .svgkit import add, arrow, box, esc, mul, rect, svg, text


def _tok(x, y, label, k="io", w=34, h=24):
    return rect(x, y, w, h, f"b {k}", 4) + text(x + w / 2, y + h / 2 + 1, label, "xs")


def _matrix(x, y, n, cell, fn, gap=1):
    """An n×n grid; fn(i, j) -> css class or None (blank)."""
    o = ""
    for i in range(n):
        for j in range(n):
            c = fn(i, j)
            o += rect(x + j * cell, y + i * cell, cell - gap, cell - gap, c or "cell")
    return o


# ================================================================== the model shape

def lm_macro():
    W, H = 960, 300
    o = ""
    words = ["The", "cat", "sat", "on", "the"]
    for i, w in enumerate(words):
        o += _tok(20, 46 + i * 34, w, "io", 46, 26)
    o += text(43, 30, "token ids", "xs")
    o += arrow([(70, 118), (96, 118)])
    o += box(98, 84, 104, 68, "embed", "io", "table [V, D]")
    o += arrow([(202, 118), (228, 118)])
    # the stream
    o += f'<path d="M230,118 L684,118" class="wire" style="stroke-width:5;opacity:.35"/>'
    o += text(456, 70, "the residual stream: one D-vector per position", "sm")
    for i, x in enumerate([252, 388, 552]):
        lab = "block 1" if i == 0 else ("block 2" if i == 1 else "block L")
        o += box(x, 92, 112, 52, lab, "mix", "mixer + MLP")
    o += text(526, 118, "…", "ttl")
    o += arrow([(684, 118), (706, 118)])
    o += box(708, 92, 78, 52, "norm", "norm", "final")
    o += arrow([(786, 118), (806, 118)])
    o += box(808, 92, 72, 52, "head", "io", "D → V")
    o += arrow([(880, 118), (900, 118)])
    for i in range(5):
        h = [10, 26, 6, 40, 14][i]
        o += rect(904 + i * 9, 140 - h, 7, h, "b act", 1)
    o += text(926, 156, "logits", "xs")
    # training targets
    tg = ["cat", "sat", "on", "the", "mat"]
    o += text(480, 196, "training: every position predicts the token after it, at once (the causal mask keeps it honest)", "sm")
    for i in range(5):
        o += _tok(270 + i * 86, 216, words[i], "io", 60, 24)
        o += arrow([(300 + i * 86, 241), (300 + i * 86, 257)])
        o += _tok(270 + i * 86, 260, tg[i], "act", 60, 24)
    o += text(240, 228, "input", "xs", "end") + text(240, 272, "target", "xs", "end")
    return svg(W, H, o, "The decoder-only language model")


def residual_stream():
    W, H = 960, 270
    o = ""
    y = 150
    o += f'<path d="M30,{y} L930,{y}" class="wire" style="stroke-width:6;opacity:.3"/>'
    o += arrow([(30, y), (930, y)])
    o += text(36, y + 22, "x₀ = embed(ids)", "mt", "start")
    o += text(926, y + 22, "→ final norm → head", "mt", "end")
    xs = [120, 300, 480, 660]
    labs = [("attention", "attn"), ("MLP", "act"), ("attention", "attn"), ("MLP", "act")]
    for i, (x, (lab, k)) in enumerate(zip(xs, labs)):
        o += box(x, 52, 120, 44, lab, k, "reads all positions" if k == "attn" else "reads one position")
        o += f'<path d="M{x+18},{y} L{x+18},{y-26}" class="ar"/>'
        o += box(x + 2, y - 44, 34, 18, "norm", "norm", None, 3, "xs")
        o += arrow([(x + 18, y - 44), (x + 18, 98)])
        o += arrow([(x + 102, 98), (x + 102, y - 12)], hi=True)
        o += add(x + 102, y)
        o += text(x + 60, 40, f"sublayer {i+1}", "caps")
    o += text(480, 210, "each sublayer reads a normalized copy of the stream and adds its output back:", "sm")
    o += text(480, 228, "x ← x + f(norm(x)).  Nothing overwrites the stream; every write is an addition.", "sm")
    o += text(840, 74, "a layer = 2 sublayers", "xs")
    return svg(W, H, o, "The residual stream")


def lowering():
    W, H = 960, 230
    o = ""
    o += box(20, 70, 200, 84, "family policy", "reg", "GemmaConfig · Qwen3Config · …")
    o += text(120, 172, "the paper's vocabulary:", "xs") + text(120, 186, "post_norms, layer_kinds, softcaps", "xs")
    o += arrow([(220, 112), (262, 112)], hi=True)
    o += text(241, 98, "lower", "xs")
    o += box(264, 52, 208, 120, "", "mix")
    o += text(368, 66, "stack policy", "ttl")
    o += box(280, 84, 176, 32, "Uniform: one block × L", "mix", None, 4, "xs")
    o += box(280, 124, 176, 32, "Patterned: kinds + layer→kind", "mix", None, 4, "xs")
    o += arrow([(472, 112), (514, 112)], hi=True)
    o += text(493, 98, "unroll", "xs")
    o += box(516, 52, 230, 120, "", "conv")
    o += text(631, 66, "structure: a config per layer", "ttl")
    for i in range(12):
        k = "attn" if (i + 1) % 6 == 0 else "norm"
        o += rect(532 + i * 17, 90, 14, 30, f"b {k}", 2)
    o += text(631, 140, "+ final norm · rotary tables · which", "xs") + text(631, 154, "layer uses which table", "xs")
    o += arrow([(746, 112), (788, 112)], hi=True)
    o += text(767, 98, "init", "xs")
    o += box(790, 70, 150, 84, "LmDecoderModel", "io", "embed · stack · head")
    o += text(480, 210, "every language kit ends in the same three-part module; what differs is what it filled in on the way", "sm")
    return svg(W, H, o, "From a family's policy to the generic model")


# ================================================================== embedding & head

def embed_head():
    W, H = 960, 300
    o = ""
    o += text(120, 24, "the token table E  [V × D]", "ttl")
    for r in range(9):
        cls = "b on" if r == 3 else "cell"
        o += rect(40, 40 + r * 18, 160, 16, cls)
        o += text(32, 48 + r * 18, str(r) if r < 8 else "V−1", "xs", "end")
    o += text(120, 216, "id 3 → row 3", "xs")
    o += arrow([(200, 103), (246, 103)], hi=True)
    o += box(248, 76, 100, 54, "row", "io", "1 × D")
    o += arrow([(348, 103), (372, 103)])
    o += box(374, 58, 212, 90, "", "norm")
    o += text(480, 74, "then, by family:", "xs")
    o += text(480, 94, "a parameter-free RMS (nanochat)", "xs")
    o += text(480, 110, "a LayerNorm, no bias (ModernBERT)", "xs")
    o += text(480, 126, "× √D (Gemma)", "xs")
    o += arrow([(586, 103), (608, 103)])
    o += text(624, 103, "x₀", "mt")
    o += text(810, 24, "the head: logits = h · Eᵀ  (tied)", "ttl")
    o += box(672, 50, 70, 40, "h", "io", "final state")
    o += arrow([(742, 70), (766, 70)])
    o += box(768, 50, 170, 40, "Eᵀ  or  W_out", "conv", "D → V")
    o += arrow([(853, 90), (853, 110)])
    o += box(768, 112, 170, 40, "slice · cap · f32", "act", "as configured")
    o += arrow([(853, 152), (853, 172)])
    o += text(853, 186, "logits over V", "xs")
    o += f'<path d="M120,228 C 120,256 853,256 853,198" class="ar dash"/>'
    o += text(486, 282, "tied: the same matrix reads tokens in and scores them out; the score of token v is h · e_v", "sm")
    return svg(W, H, o, "Token embedding and head")


def softcap():
    W, H = 600, 300
    o = ""
    x0, y0, sx, sy = 280, 150, 3.2, 2.0
    o += f'<path d="M20,{y0} L540,{y0} M{x0},16 L{x0},284" class="grid"/>'
    for c, cls in [(15, "curve2"), (30, "curve"), (50, "curve")]:
        pts = []
        for i in range(-80, 81, 2):
            v = c * math.tanh(i / c)
            pts.append(f"{x0 + i * sx:.1f},{y0 - v * sy:.1f}")
        st = ' style="opacity:.55"' if c == 50 else ""
        o += f'<path d="M{" L".join(pts)}" class="{cls}"{st}/>'
        o += text(x0 + 80 * sx + 4, y0 - c * math.tanh(80 / c) * sy, f"c = {c}", "xs", "start")
    o += f'<path d="M{x0-60*sx},{y0+60*sy} L{x0+60*sx},{y0-60*sy}" class="ar dash"/>'
    o += text(x0 + 60 * sx - 12, y0 - 60 * sy - 6, "no cap", "xs")
    o += text(x0 + 6, 24, "c · tanh(s / c)", "mt", "start")
    o += text(520, y0 + 14, "s", "mt")
    o += text(280, 292, "linear near 0, bounded by ±c: large scores keep their order but cannot run away", "xs")
    return svg(W, H, o, "Soft-capping")


# ================================================================== attention

def causal_mask():
    W, H = 900, 280
    o = ""
    n, cell = 8, 24
    toks = ["<s>", "The", "cat", "sat", "on", "the", "mat", "."]
    x, y = 120, 44
    import random
    rnd = random.Random(3)
    for i in range(n):
        o += text(x - 8, y + i * cell + cell / 2, toks[i], "xs", "end")
        o += text(x + i * cell + cell / 2, y - 10, toks[i], "xs")
        ws = [rnd.random() ** 2 for _ in range(i + 1)]
        s = sum(ws)
        for j in range(n):
            if j <= i:
                a = ws[j] / s
                o += rect(x + j * cell, y + i * cell, cell - 1, cell - 1, "on", 0, f'style="fill-opacity:{0.12 + 0.85 * a:.2f}"')
            else:
                o += rect(x + j * cell, y + i * cell, cell - 1, cell - 1, "off")
    o += text(x + n * cell / 2, y + n * cell + 18, "keys →", "xs") + text(x - 60, y + n * cell / 2, "queries", "xs")
    o += text(560, 52, "A = softmax(Q Kᵀ · s + M)", "mt")
    o += text(560, 80, "M = 0 at and before the query, −∞ after it", "sm")
    o += text(560, 112, "each row is a distribution over the past; the", "sm")
    o += text(560, 128, "output is that mixture of the value vectors", "sm")
    o += text(560, 164, "one pass trains all T next-token predictions:", "sm")
    o += text(560, 180, "no position can read the token it must predict", "sm")
    o += text(560, 216, "s = 1/√d by default; Gemma's query_pre_attn_scalar", "xs")
    o += text(560, 230, "and nanochat 2026's QK gain replace it", "xs")
    return svg(W, H, o, "The causal attention map")


def gqa():
    W, H = 960, 270
    o = ""
    panels = [("MHA", 8, 8, 60), ("GQA", 8, 2, 360), ("MQA", 8, 1, 660)]
    for name, hq, hkv, x0 in panels:
        o += text(x0 + 130, 24, f"{name}: {hq} query heads, {hkv} K/V head{'s' if hkv > 1 else ''}", "ttl")
        g = hq // hkv
        for i in range(hq):
            o += rect(x0 + i * 32, 46, 26, 26, "b attn", 3) + text(x0 + i * 32 + 13, 60, f"q{i}", "xs")
        kw = (hq * 32 - 6) / hkv
        for j in range(hkv):
            kx = x0 + j * (hq * 32) / hkv
            o += rect(kx + 2, 128, kw - 4, 26, "b conv", 3) + text(kx + kw / 2, 142, f"k,v{j}" if hkv > 1 else "k,v", "xs")
            for i in range(j * g, (j + 1) * g):
                o += f'<path d="M{x0 + i * 32 + 13},{72} L{kx + kw / 2},{128}" class="line"/>'
        o += text(x0 + 130, 172, f"cache per token per layer: {2 * hkv}·d", "xs")
        o += rect(x0, 184, 256 * hkv / 8, 14, "b conv", 2)
        o += rect(x0, 184, 256, 14, "slot", 2)
    o += text(480, 230, "the queries keep their heads; only the keys and values are shared within a group", "sm")
    o += text(480, 248, "Qwen3: 16 q / 8 kv · Qwen3-4B, -8B: 32 / 8 · Gemma 3 1B: 4 / 1 · Gemma 2B (v1): 8 / 1", "xs")
    return svg(W, H, o, "Grouped-query attention")


def attn_internals():
    W, H = 960, 250
    o = ""
    o += box(20, 96, 60, 44, "x", "io", "norm'd")
    o += arrow([(80, 118), (104, 118)])
    o += box(106, 40, 80, 32, "W_Q", "conv", None, 4)
    o += box(106, 102, 80, 32, "W_K", "conv", None, 4)
    o += box(106, 164, 80, 32, "W_V", "conv", None, 4)
    for yy in (56, 118, 180):
        o += f'<path d="M104,118 L104,{yy} L106,{yy}" class="line"/>'
    o += arrow([(186, 56), (212, 56)]) + arrow([(186, 118), (212, 118)])
    o += box(214, 40, 104, 32, "QK-norm", "norm", None, 4)
    o += box(214, 102, 104, 32, "QK-norm", "norm", None, 4)
    o += arrow([(318, 56), (344, 56)]) + arrow([(318, 118), (344, 118)])
    o += box(346, 40, 90, 32, "RoPE", "pool", None, 4)
    o += box(346, 102, 90, 32, "RoPE", "pool", None, 4)
    o += text(330, 26, "order: before (Qwen, Gemma 3) or after (nanochat) RoPE", "xs")
    o += arrow([(436, 56), (472, 90)]) + arrow([(436, 118), (472, 104)])
    o += box(474, 72, 132, 48, "scores", "attn", "q·k · scale → cap")
    o += arrow([(606, 96), (632, 96)])
    o += box(634, 72, 100, 48, "mask", "io", "causal · window")
    o += arrow([(734, 96), (756, 96)])
    o += box(758, 72, 80, 48, "softmax", "attn", "in f32")
    o += arrow([(798, 120), (798, 170)])
    o += f'<path d="M186,180 L780,180" class="ar"/>'
    o += box(758, 160, 80, 40, "A · V", "attn", None, 4)
    o += text(500, 196, "+ value residual (nanochat 2026)", "xs")
    o += arrow([(838, 180), (856, 180)])
    o += box(858, 140, 84, 32, "× σ(gate)", "act", None, 4)
    o += box(858, 186, 84, 32, "W_O", "conv", None, 4)
    o += arrow([(900, 172), (900, 186)])
    o += text(900, 128, "Qwen3.5", "xs")
    o += text(480, 236, "every box but the projections and the softmax is optional: each family is a choice of which ones are present", "sm")
    return svg(W, H, o, "Inside causal self-attention")


def output_gate():
    W, H = 900, 240
    o = ""
    o += box(30, 90, 110, 50, "heads' output", "attn", "Σ a · v  [H·d]")
    o += arrow([(140, 115), (200, 115)])
    o += mul(212, 115)
    o += box(150, 24, 124, 40, "σ(W_g x)", "act", "from the input")
    o += arrow([(212, 64), (212, 104)], hi=True)
    o += arrow([(223, 115), (270, 115)])
    o += box(272, 90, 80, 50, "W_O", "conv")
    o += arrow([(352, 115), (384, 115)])
    o += text(400, 115, "y", "mt")
    o += text(200, 176, "Elementwise (Qwen3.5): one gate per channel;", "xs")
    o += text(200, 190, "W_Q is twice as wide, queries then gates per head", "xs")
    o += text(200, 210, "Headwise: one gate per head", "xs")
    # attention-sink cartoon
    x0 = 520
    o += text(x0 + 170, 22, "why: softmax must put its mass somewhere", "ttl")
    for k, (lab, vals) in enumerate([("ungated", [0.62, .06, .08, .05, .07, .12]), ("gated", [.14, .18, .2, .1, .22, .16])]):
        yy = 48 + k * 92
        o += text(x0 - 6, yy + 30, lab, "xs", "end")
        for i, v in enumerate(vals):
            o += rect(x0 + i * 52, yy + 60 - v * 90, 40, v * 90, "b attn" if not (k == 0 and i == 0) else "b on", 2)
        o += f'<path d="M{x0-4},{yy+60} L{x0+310},{yy+60}" class="line"/>'
    o += text(x0 + 170, 228, "without a gate, a head that wants to say nothing dumps mass on the first token (a sink)", "xs")
    return svg(W, H, o, "The attention output gate")


# ================================================================== rotary positions

def rope_rotate():
    W, H = 920, 290
    o = ""
    cx, cy, r = 150, 150, 100
    o += f'<circle cx="{cx}" cy="{cy}" r="{r}" class="grid"/>'
    o += f'<path d="M{cx-r-14},{cy} L{cx+r+14},{cy} M{cx},{cy-r-14} L{cx},{cy+r+14}" class="grid"/>'
    for ang, lab, cls in [(20, "q at m", "ar hi"), (75, "k at n", "ar")]:
        a = math.radians(ang)
        x, y = cx + r * 0.9 * math.cos(a), cy - r * 0.9 * math.sin(a)
        mk = 'marker-end="url(#ahh)"' if "hi" in cls else 'marker-end="url(#ah)"'
        o += f'<path d="M{cx},{cy} L{x:.1f},{y:.1f}" class="{cls}" {mk}/>'
        o += text(x + (14 if ang < 45 else -4), y - 10, lab, "xs")
    a1, a2 = math.radians(20), math.radians(75)
    o += f'<path d="M{cx+40*math.cos(a1):.1f},{cy-40*math.sin(a1):.1f} A40,40 0 0,0 {cx+40*math.cos(a2):.1f},{cy-40*math.sin(a2):.1f}" class="curve2"/>'
    o += text(cx + 58, cy - 46, "(n−m)ωᵢ", "xs")
    o += text(cx, 276, "one pair (x₂ᵢ, x₂ᵢ₊₁) of a head, as a plane", "xs")
    o += text(330, 50, "RoPE rotates each pair of a query or key by its position × that pair's frequency:", "sm", "start")
    o += text(330, 82, "R(mωᵢ) q · R(nωᵢ) k  =  q · R((n−m)ωᵢ) k", "mt", "start")
    o += text(330, 114, "so a score depends only on the offset n − m, never on m or n alone.", "sm", "start")
    o += text(330, 146, "ωᵢ = θ^(−2i/d):  pair 0 spins once a step, the last pair barely turns.", "sm", "start")
    o += text(330, 178, "fast pairs resolve nearby order; slow pairs carry long-range position", "sm", "start")
    o += text(330, 194, "(and, nearly constant, act as position-free channels).", "sm", "start")
    o += text(330, 230, "no parameters, nothing added to the stream: positions live only inside", "sm", "start")
    o += text(330, 246, "attention's scores, which is why the stream itself is position-free.", "sm", "start")
    return svg(W, H, o, "Rotary position embedding")


def rope_spectrum():
    W, H = 900, 300
    o = ""
    x0, x1, y0, y1 = 90, 860, 30, 240
    # y: log10 wavelength 0..9
    def Y(v):
        return y1 - (v / 9) * (y1 - y0)
    o += f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1}" class="line"/>'
    for v in range(0, 10, 1):
        o += text(x0 - 8, Y(v), f"10{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in str(v))}", "xs", "end")
        o += f'<path d="M{x0},{Y(v)} L{x1},{Y(v)}" class="grid"/>'
    d = 128
    for theta, cls, lab in [(1e4, "curve", "θ = 10⁴ (nanochat 2025, Gemma 1/2, local tables)"),
                            (1e6, "curve2", "θ = 10⁶ (Qwen3, Gemma 3 global)"),
                            (1e7, "curve", "θ = 10⁷ (Qwen3.5)")]:
        pts = []
        for i in range(d // 2):
            lam = 2 * math.pi * theta ** (2 * i / d)
            pts.append(f"{x0 + (x1 - x0) * i / (d / 2 - 1):.1f},{Y(math.log10(lam)):.1f}")
        st = ' style="opacity:.5"' if theta == 1e7 else ""
        o += f'<path d="M{" L".join(pts)}" class="{cls}"{st}/>'
        dy = 14 if theta == 1e4 else -10
        o += text(x1 - 4, Y(math.log10(2 * math.pi * theta)) + dy, lab, "xs", "end")
    for ctx, lab in [(4096, "4K"), (32768, "32K"), (262144, "256K")]:
        o += f'<path d="M{x0},{Y(math.log10(ctx))} L{x1},{Y(math.log10(ctx))}" class="ar dash"/>'
        o += text(x0 + 6, Y(math.log10(ctx)) - 7, f"context {lab}", "xs", "start")
    o += text((x0 + x1) / 2, 262, "pair index i (0 … d/2 − 1), head width 128", "xs")
    o += text(28, (y0 + y1) / 2, "wavelength", "xs", "middle", 'transform="rotate(-90 28 135)"')
    o += text((x0 + x1) / 2, 284, "pairs whose wavelength exceeds the context never complete a turn; a larger θ gives more of them, for long contexts", "xs")
    return svg(W, H, o, "Rotary frequency spectrum")


def partial_rope():
    W, H = 900, 170
    o = ""
    x0, w = 60, 780
    o += text(x0, 26, "one 256-wide head of Qwen3.5's attention layer (partial_rotary_factor 0.25)", "ttl", "start")
    o += rect(x0, 44, w * 0.25, 34, "b pool", 4) + text(x0 + w * 0.125, 61, "dims 0–63: rotated", "xs")
    o += rect(x0 + w * 0.25, 44, w * 0.75, 34, "b io", 4) + text(x0 + w * 0.625, 61, "dims 64–255: no position, pure content", "xs")
    o += text(x0, 106, "the frequencies θ^(−2i/64) are spread over the rotated span only, so it holds the full spectrum, fast to slow", "sm", "start")
    o += text(x0, 126, "the rest of q · k matches content irrespective of distance: an attention layer that is mostly position-free", "sm", "start")
    o += text(x0, 146, "(its DeltaNet neighbours already know order, through their short conv and their recurrence)", "sm", "start")
    return svg(W, H, o, "Partial rotary embedding")


def rope_scaling():
    W, H = 900, 200
    o = ""
    x0, x1 = 80, 860
    o += text(x0, 26, "linear scaling (position interpolation): divide every frequency by f, so position p is rotated as p / f", "ttl", "start")
    for row, (lab, f) in enumerate([("trained", 1), ("× 8", 8)]):
        y = 64 + row * 64
        o += text(x0 - 10, y + 10, lab, "xs", "end")
        o += f'<path d="M{x0},{y+10} L{x1},{y+10}" class="line"/>'
        for k in range(0, 33):
            p = k * 1024 if f == 8 else k * 128
            x = x0 + (x1 - x0) * k / 32
            o += f'<path d="M{x:.1f},{y+4} L{x:.1f},{y+16}" class="line"/>'
        o += text(x1, y - 4, "4096 angles" if f == 1 else "the same angles reach 32768 positions", "xs", "end")
    o += text(x0, 182, "Gemma 3 4B scales only its global table (×8, θ 10⁶); its sliding layers never see far enough to need it", "sm", "start")
    return svg(W, H, o, "Rotary scaling")


# ================================================================== masks & windows

def mask_kinds():
    W, H = 960, 250
    o = ""
    n, cell = 10, 15
    panels = [
        ("causal", lambda i, j: "on" if j <= i else "off", "every decoder layer"),
        ("sliding, w = 4", lambda i, j: "on" if i - 4 < j <= i else "off", "Gemma 2/3, nanochat S"),
        ("bidirectional", lambda i, j: "on", "ModernBERT global, Laya head"),
        ("two-sided, ±2", lambda i, j: "on" if abs(i - j) <= 2 else "off", "ModernBERT local (±64)"),
        ("padding", lambda i, j: "on" if j < 7 else "off", "encoder key mask"),
    ]
    for k, (name, fn, who) in enumerate(panels):
        x = 26 + k * 188
        o += text(x + n * cell / 2, 26, name, "ttl")
        o += _matrix(x, 44, n, cell, fn)
        o += text(x + n * cell / 2, 44 + n * cell + 16, who, "xs")
    o += text(480, 236, "rows are queries, columns keys; a mask is the attention's only notion of direction or distance besides RoPE", "sm")
    return svg(W, H, o, "Attention masks")


PATTERNS = [
    ("Qwen3, Gemma 1", "F" * 28, "every layer full"),
    ("Gemma 2", "WF" * 13, "window 4096"),
    ("Gemma 3", ("WWWWWF" * 5)[:26], "window 512 (1B)"),
    ("Qwen3.5", "DDDF" * 6, "D: gated DeltaNet"),
    ("ModernBERT", ("FWW" * 8)[:22], "local ±64, two-sided"),
    ("nanochat 2026", "WWWF" * 5, "pattern SSSL"),
]
PK = dict(F="attn", W="norm", D="act")


def layer_patterns():
    W, H = 960, 300
    o = ""
    for r, (name, pat, note) in enumerate(PATTERNS):
        y = 30 + r * 40
        o += text(150, y + 11, name, "lbl", "end")
        for i, c in enumerate(pat):
            o += rect(164 + i * 21, y, 17, 22, f"b {PK[c]}", 3)
            o += text(164 + i * 21 + 8.5, y + 11.5, c, "xs")
        o += text(164 + len(pat) * 21 + 10, y + 11, note, "xs", "start")
    o += rect(164, 272, 14, 14, "b attn", 3) + text(184, 279, "F: full, global or long attention", "xs", "start")
    o += rect(384, 272, 14, 14, "b norm", 3) + text(404, 279, "W: a sliding or local window", "xs", "start")
    o += rect(584, 272, 14, 14, "b act", 3) + text(604, 279, "D: linear attention (gated DeltaNet)", "xs", "start")
    return svg(W, H, o, "Layer patterns")


def receptive():
    W, H = 900, 250
    o = ""
    x0, n, cw = 60, 34, 22
    rows = 4
    w = 4
    q = n - 1
    for l in range(rows):
        y = 190 - l * 40
        reach = min(q, (l + 1) * (w - 1))
        for i in range(n):
            cls = "on" if q - reach <= i <= q else "cell"
            o += rect(x0 + i * cw, y, cw - 3, 22, cls, 3, f'style="fill-opacity:{0.85 - 0.15 * l:.2f}"' if cls == "on" else "")
        o += text(x0 - 8, y + 11, f"layer {l+1}", "xs", "end")
    o += text(x0 + q * cw + 10, 38, "query", "xs", "end")
    o += text(450, 236, "with a window of w per layer, information can travel (w − 1) positions per layer: L layers see L·(w − 1) back;", "xs")
    o += text(450, 22, "one global layer every few resets the reach to the whole context, so windows cost locality only between them", "xs")
    return svg(W, H, o, "Receptive field of stacked windows")


def encoder_vs_decoder():
    W, H = 900, 230
    o = ""
    n, cell = 7, 20
    o += text(140, 22, "decoder: predict the next token", "ttl")
    o += _matrix(70, 40, n, cell, lambda i, j: "on" if j <= i else "off")
    o += text(140, 196, "causal · cached · generative", "xs")
    o += text(460, 22, "encoder: read the whole input", "ttl")
    o += _matrix(390, 40, n, cell, lambda i, j: "on")
    o += text(460, 196, "bidirectional · one pass · a state per token", "xs")
    o += text(760, 50, "the same DecoderStack builds both:", "sm")
    o += text(760, 70, "the mask is a config field,", "sm")
    o += text(760, 90, "AttentionMask::Causal or ::Bidirectional", "xs")
    o += text(760, 126, "an encoder takes no KV cache;", "sm")
    o += text(760, 146, "its padding is a key mask instead", "sm")
    o += text(760, 182, "ModernBERT, the Laya head, and the", "xs")
    o += text(760, 196, "speech encoders are all such stacks", "xs")
    return svg(W, H, o, "Encoder and decoder masks")


# ================================================================== logit control

def qk_growth():
    W, H = 920, 270
    o = ""
    x0, x1, y0, y1 = 70, 420, 40, 220
    o += f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1}" class="line"/>'
    o += text((x0 + x1) / 2, 240, "training step →", "xs")
    o += text(x0 - 10, (y0 + y1) / 2, "max |score|", "xs", "end")
    pts1 = [f"{x0 + t * 3.5:.1f},{y1 - min(175, 4 * math.exp(t / 26)):.1f}" for t in range(0, 101)]
    pts2 = [f"{x0 + t * 3.5:.1f},{y1 - 52 - 6 * math.sin(t / 7):.1f}" for t in range(0, 101)]
    o += f'<path d="M{" L".join(pts1)}" class="curve2"/>' + f'<path d="M{" L".join(pts2)}" class="curve"/>'
    o += text(x1 - 10, y0 + 6, "no QK-norm: logits grow, softmax saturates", "xs", "end")
    o += text(x0 + 10, y1 - 82, "QK-norm: |q|, |k| fixed, scores bounded", "xs", "start")
    o += text(680, 34, "three ways to keep scores in range", "ttl")
    rows = [("QK-norm", "RMS-normalize each head's q and k:", "|q·k| ≤ g²·d, whatever the weights do", "norm"),
            ("score scale", "replace 1/√d by a chosen s:", "Gemma 2/3: ÷√256; nanochat: g²/√d", "io"),
            ("soft-cap", "c·tanh(s/c) on the scores:", "Gemma 2's c = 50 (and 30 on the logits)", "act")]
    for i, (a, b, c, k) in enumerate(rows):
        y = 56 + i * 64
        o += box(470, y, 110, 46, a, k)
        o += text(596, y + 15, b, "sm", "start") + text(596, y + 33, c, "xs", "start")
    o += text(680, 256, "and run the softmax in f32: a bf16 softmax over 10⁵ keys rounds its small weights to zero", "xs")
    return svg(W, H, o, "Controlling attention logits")


def qk_order():
    W, H = 900, 170
    o = ""
    for r, (name, seq, who) in enumerate([("BeforeRope", ["W_Q x", "RMS", "RoPE"], "Qwen3, Qwen3.5, Gemma 3"),
                                          ("AfterRope", ["W_Q x", "RoPE", "RMS"], "nanochat")]):
        y = 34 + r * 66
        o += text(30, y + 18, name, "ttl", "start")
        x = 170
        for i, s in enumerate(seq):
            k = "conv" if i == 0 else ("norm" if s == "RMS" else "pool")
            o += box(x, y, 104, 36, s, k)
            if i < 2:
                o += arrow([(x + 104, y + 18), (x + 130, y + 18)])
            x += 132
        o += text(x + 10, y + 18, who, "xs", "start")
    o += text(450, 160, "a rotation preserves length, so a parameter-free RMS commutes with RoPE; with a learned per-dim gain, the order matters and must match the checkpoint", "xs")
    return svg(W, H, o, "QK-norm placement")


# ================================================================== feed-forward

def ffn_kinds():
    W, H = 960, 280
    o = ""
    o += text(200, 24, "plain MLP", "ttl")
    o += box(150, 44, 100, 36, "W₁", "conv", "D → 4D")
    o += arrow([(200, 80), (200, 100)])
    o += box(150, 102, 100, 36, "φ", "act", "ReLU² · GELU")
    o += arrow([(200, 138), (200, 158)])
    o += box(150, 160, 100, 36, "W₂", "conv", "4D → D")
    o += text(200, 216, "nanochat: ReLU², no bias", "xs") + text(200, 230, "Laya head: ReLU, biased", "xs")
    o += text(200, 248, "params 8D²", "xs")
    o += text(620, 24, "gated MLP (GLU)", "ttl")
    o += box(470, 44, 120, 36, "W_gate", "conv", "D → h")
    o += box(650, 44, 120, 36, "W_up", "conv", "D → h")
    o += arrow([(530, 80), (530, 100)])
    o += box(480, 102, 100, 36, "φ", "act", "SiLU · GELU")
    o += arrow([(530, 138), (600, 160)]) + arrow([(710, 80), (620, 160)])
    o += mul(610, 170)
    o += arrow([(610, 181), (610, 200)])
    o += box(550, 202, 120, 36, "W_down", "conv", "h → D")
    o += text(796, 104, "SwiGLU: φ = SiLU", "sm", "start") + text(796, 120, "Qwen3, Qwen3.5", "xs", "start")
    o += text(796, 146, "GeGLU: φ = GELU", "sm", "start") + text(796, 162, "Gemma (tanh),", "xs", "start")
    o += text(796, 176, "ModernBERT (exact)", "xs", "start")
    o += text(796, 204, "params 3Dh: equal to the", "xs", "start") + text(796, 218, "plain MLP's at h = 8D/3", "xs", "start")
    o += text(620, 262, "one branch decides, per hidden unit, how much of the other passes", "xs")
    return svg(W, H, o, "Plain and gated feed-forwards")


def activations():
    W, H = 600, 280
    o = ""
    x0, y0, sx, sy = 300, 200, 60, 60
    o += f'<path d="M30,{y0} L570,{y0} M{x0},20 L{x0},266" class="grid"/>'
    def gelu(x):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    fns = [("ReLU²", lambda x: max(0, x) ** 2, "curve2", ""),
           ("GELU", gelu, "curve", ""),
           ("SiLU", lambda x: x / (1 + math.exp(-x)), "curve", ' style="opacity:.5;stroke-dasharray:5 3"')]
    for name, f, cls, st in fns:
        pts = []
        for i in range(-400, 401, 8):
            x = i / 100
            v = f(x)
            if v * sy > 190:
                break
            pts.append(f"{x0 + x * sx:.1f},{y0 - v * sy:.1f}")
        o += f'<path d="M{" L".join(pts)}" class="{cls}"{st}/>'
    o += text(x0 + 1.25 * sx, y0 - 2.6 * sy, "ReLU²", "xs", "end")
    o += text(x0 + 3.1 * sx, y0 - 2.75 * sy, "GELU", "xs", "start")
    o += text(x0 + 3.1 * sx, y0 - 2.2 * sy, "SiLU", "xs", "start")
    o += text(x0 - 2.5 * sx, y0 + 40, "GELU and SiLU dip below 0 near −1;", "xs")
    o += text(x0 - 2.5 * sx, y0 + 54, "ReLU² is exactly 0 there: sparse", "xs")
    o += text(x0 + 2 * sx, y0 + 30, "x", "mt")
    return svg(W, H, o, "Activations")


# ================================================================== sublayers & norms

def norm_geometry():
    W, H = 920, 280
    o = ""
    # LayerNorm: centre then scale
    o += text(220, 22, "LayerNorm: subtract the mean, divide by the std", "ttl")
    o += text(700, 22, "RMSNorm: divide by the root mean square", "ttl")
    for k, cx in enumerate([220, 700]):
        cy = 150
        o += f'<circle cx="{cx}" cy="{cy}" r="80" class="grid"/>'
        o += f'<path d="M{cx-120},{cy} L{cx+120},{cy} M{cx},{cy-110} L{cx},{cy+110}" class="grid"/>'
        pts = [(cx + 100, cy - 70), (cx - 30, cy - 105), (cx + 55, cy + 40)]
        for (x, y) in pts:
            dx, dy = x - cx, y - cy
            if k == 0:
                dx -= 18
                dy += 10
            n = math.hypot(dx, dy)
            tx, ty = cx + 80 * dx / n, cy + 80 * dy / n
            o += f'<circle cx="{x}" cy="{y}" r="3.5" class="dot"/>'
            o += arrow([(x, y), (tx + (2 if dx > 0 else -2), ty)], dash=True)
            o += f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4" class="on"/>'
    o += text(220, 254, "onto a sphere in the hyperplane ⟂ 1: removes one direction and a scale", "xs")
    o += text(700, 254, "onto the sphere of radius √D: removes only the scale; cheaper, no mean", "xs")
    o += text(460, 274, "both then multiply by a learned gain (Gemma, Qwen3.5 store it as w and apply 1 + w); LayerNorm may add a bias", "xs")
    return svg(W, H, o, "LayerNorm and RMSNorm")


def norm_placements():
    W, H = 960, 250
    o = ""
    def lane(x, title, items, who):
        s = text(x + 100, 22, title, "ttl")
        s += f'<path d="M{x+10},50 L{x+10},200" class="wire"/>'
        s += add(x + 10, 200 - 10, 9)
        s += f'<circle cx="{x+10}" cy="64" r="3.5" class="pin"/>'
        s += f'<path d="M{x+10},64 L{x+40},64" class="wire"/>'
        yy = 54
        for i, (lab, k) in enumerate(items):
            s += box(x + 42, yy, 120, 26, lab, k, None, 4, "xs")
            if i < len(items) - 1:
                s += arrow([(x + 102, yy + 26), (x + 102, yy + 34)])
            yy += 34
        s += arrow([(x + 102, yy - 8), (x + 102, 190), (x + 21, 190)])
        s += text(x + 100, 226, who, "xs")
        return s
    o += lane(20, "pre-norm", [("norm", "norm"), ("f", "attn")], "Qwen3, nanochat, Gemma 1, Laya head")
    o += lane(260, "sandwich", [("norm", "norm"), ("f", "attn"), ("norm", "norm")], "Gemma 2 and 3")
    o += lane(500, "no input norm", [("f", "attn")], "ModernBERT's first layer")
    o += text(800, 22, "why a sandwich", "ttl")
    o += text(800, 50, "pre-norm keeps the stream clean,", "sm") + text(800, 66, "but lets each write's size drift;", "sm")
    o += text(800, 92, "a norm on the branch's output", "sm") + text(800, 108, "fixes every write's scale before", "sm")
    o += text(800, 124, "the add, at the cost of two", "sm") + text(800, 140, "more norms per layer", "sm")
    o += text(800, 172, "ModernBERT's embedding is", "xs") + text(800, 186, "normalized already, so its", "xs")
    o += text(800, 200, "first layer skips the duplicate", "xs")
    return svg(W, H, o, "Norm placements in the language kits")


# ================================================================== linear attention

def kv_vs_state():
    W, H = 940, 270
    o = ""
    o += text(220, 22, "softmax attention: a growing memory", "ttl")
    for i in range(12):
        o += rect(40 + i * 30, 50, 24, 40, "b conv", 3) + text(52 + i * 30, 70, "k", "xs")
        o += rect(40 + i * 30, 96, 24, 40, "b act", 3) + text(52 + i * 30, 116, "v", "xs")
    o += arrow([(400, 93), (430, 93)])
    o += text(220, 160, "every past (k, v) is kept; a step reads all t of them:", "xs")
    o += text(220, 174, "memory O(t·d), time O(t) per token, exact recall", "xs")
    o += text(700, 22, "linear attention: a fixed-size memory", "ttl")
    o += _matrix(620, 46, 6, 18, lambda i, j: "on3")
    o += text(674, 166, "S  [dₖ × dᵥ]", "mt")
    o += text(800, 60, "write:  S ← S + k vᵀ", "sm", "start")
    o += text(800, 84, "read:   o = Sᵀ q", "sm", "start")
    o += text(800, 120, "memory O(d²), time O(1)", "xs", "start")
    o += text(800, 134, "per token, whatever t is", "xs", "start")
    o += text(470, 214, "drop the softmax and attention becomes an outer-product memory: Σ v (kᵀq) = (Σ v kᵀ) q.", "sm")
    o += text(470, 232, "the price is capacity: S holds about d_k independent associations, and writes interfere", "sm")
    return svg(W, H, o, "KV cache versus recurrent state")


def delta_rule():
    W, H = 940, 290
    o = ""
    steps = [("decay", "S ← exp(g)·S", "forget a share of\neverything (g ≤ 0)", "norm"),
             ("read", "v̂ = Sᵀ k", "what the memory\nnow says k maps to", "pool"),
             ("correct", "S ← S + β k (v − v̂)ᵀ", "move k's answer\ntoward v by β ∈ (0,1)", "act"),
             ("query", "o = Sᵀ q", "read the updated\nmemory", "attn")]
    for i, (a, b, c, k) in enumerate(steps):
        x = 30 + i * 228
        o += box(x, 40, 200, 56, a, k, b)
        for j, line in enumerate(c.split("\n")):
            o += text(x + 100, 116 + j * 14, line, "xs")
        if i < 3:
            o += arrow([(x + 200, 68), (x + 226, 68)])
    o += text(470, 172, "plain linear attention only adds (S ← S + k vᵀ): re-writing a key piles a new value on the old.", "sm")
    o += text(470, 192, "the delta rule writes the error instead, one step of online least squares on ‖Sᵀk − v‖²", "sm")
    o += text(470, 208, "(Widrow–Hoff, 1960): re-writing a key replaces its value. The gate g adds Mamba-style forgetting.", "sm")
    o += text(470, 240, "q and k are L2-normalized first, so β is a true step size (‖k‖ = 1 makes the update a projection);", "xs")
    o += text(470, 254, "a decode step runs these four lines; a prefill runs them 64 positions at a time in matrix form, with the same result", "xs")
    return svg(W, H, o, "The gated delta rule")


def gdn_block():
    W, H = 960, 270
    o = ""
    o += box(20, 108, 56, 44, "x", "io")
    o += arrow([(76, 130), (100, 130)])
    o += f'<path d="M100,40 L100,224" class="line"/>'
    rows = [(40, "in_proj_qkv", "conv"), (100, "in_proj_b", "conv"), (150, "in_proj_a", "conv"), (204, "in_proj_z", "conv")]
    for y, lab, k in rows:
        o += f'<path d="M100,{y+12} L110,{y+12}" class="line"/>'
        o += box(110, y, 110, 26, lab, k, None, 4, "xs")
    o += arrow([(220, 53), (244, 53)])
    o += box(246, 38, 120, 30, "short conv", "pool", "causal, 4 taps")
    o += arrow([(366, 53), (388, 53)])
    o += box(390, 38, 60, 30, "SiLU", "act", None, 4, "xs")
    o += arrow([(450, 53), (472, 53)])
    o += box(474, 30, 104, 46, "split q, k, v", "io", "L2-norm q, k")
    o += arrow([(220, 113), (474, 113)])
    o += text(400, 106, "β = σ(·): write strength", "xs")
    o += arrow([(220, 163), (474, 163)])
    o += text(380, 156, "g = −exp(A)·softplus(· + dt): decay", "xs")
    o += box(476, 96, 120, 84, "gated delta rule", "mix", "state S per value head")
    o += arrow([(526, 76), (526, 96)])
    o += arrow([(596, 138), (640, 138)])
    o += box(642, 112, 120, 52, "gated RMSNorm", "norm", "w · x̂ · SiLU(z)")
    o += f'<path d="M220,217 L702,217 L702,166" class="ar"/>'
    o += text(460, 210, "z: the output gate", "xs")
    o += arrow([(762, 138), (790, 138)])
    o += box(792, 116, 100, 44, "out_proj", "conv", "Hᵥ·dᵥ → D")
    o += text(840, 196, "Qwen3.5: 16 key heads,", "xs") + text(840, 210, "16–48 value heads, d = 128", "xs")
    o += text(480, 254, "positions come from the conv and the recurrence itself: a linear layer takes no rotary table", "xs")
    return svg(W, H, o, "The gated DeltaNet")


def hybrid_stack():
    W, H = 940, 230
    o = ""
    for i in range(24):
        k = "attn" if (i + 1) % 4 == 0 else "act"
        o += rect(40 + i * 36, 40, 30, 48, f"b {k}", 4)
        o += text(55 + i * 36, 64, "F" if k == "attn" else "Δ", "xs")
    o += text(470, 24, "Qwen3.5-0.8B: 24 layers, every fourth full attention", "ttl")
    o += text(470, 118, "18 DeltaNet layers carry a fixed state each; 6 attention layers carry a KV cache, ¼ of a dense model's", "sm")
    o += text(470, 140, "the linear layers do the bulk of the mixing cheaply; the full layers give exact retrieval over the whole context", "sm")
    o += text(470, 176, "each kind takes the next slot of its own cache: attention layers index a KV cache 0…5, DeltaNet layers a state cache 0…17", "xs")
    o += text(470, 194, "(and a DeltaNet state cannot rewind: it summarizes its prefix, so it can only be snapshotted or reset)", "xs")
    return svg(W, H, o, "A hybrid stack")


# ================================================================== stream extras (nanochat 2026)

def stream_extras():
    W, H = 960, 330
    o = ""
    y = 170
    o += f'<path d="M40,{y} L900,{y}" class="wire" style="stroke-width:5;opacity:.3"/>'
    o += arrow([(40, y), (920, y)])
    o += box(40, 60, 92, 40, "embed + norm", "io", None, 4, "xs")
    o += arrow([(86, 100), (86, y - 12)])
    o += box(140, 60, 104, 40, "smear", "act", "+ λσ(g)·x₍t−1₎")
    o += arrow([(192, 100), (192, y - 12)], hi=True)
    o += text(192, 48, "① previous token", "xs")
    o += text(40, y + 22, "x₀", "mt", "start")
    for i, x in enumerate([300, 520, 740]):
        o += box(x, y - 22, 120, 44, f"block {['i', 'L/2', 'L'][i]}", "mix")
        o += text(x + 60, y - 34, "② x ← λᵢ·x + μᵢ·x₀", "xs")
    o += f'<path d="M86,{y+40} L86,{y+60} L790,{y+60} L790,{y+22}" class="ar dash"/>'
    o += text(260, y + 74, "x₀ is fed to every block (μᵢ), and the stream rescaled (λᵢ)", "xs")
    o += box(560, 252, 150, 40, "value embeddings", "conv", "alternate layers + last")
    o += arrow([(600, 252), (600, y + 22)], hi=True)
    o += text(635, 306, "③ token-indexed values, gated into V", "xs")
    o += f'<path d="M660,{y} L660,{y-70} L880,{y-70} L880,{y-11}" class="ar hi" marker-end="url(#ahh)"/>'
    o += add(880, y, 10, "−")
    o += text(770, y - 80, "④ backout: subtract β·x at L/2 before the final norm", "xs")
    o += text(480, 22, "upstream nanochat's 2026 additions: four extra paths around a plain pre-norm stack", "ttl")
    return svg(W, H, o, "nanochat 2026 stream extras")


# ================================================================== decoding state

def cache_growth():
    W, H = 900, 290
    o = ""
    x0, x1, y0, y1 = 80, 600, 30, 240
    o += f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1}" class="line"/>'
    o += text((x0 + x1) / 2, 262, "context length →", "xs")
    o += text(x0 - 10, (y0 + y1) / 2, "state", "xs", "end")
    o += f'<path d="M{x0},{y1} L{x1},{y0+10}" class="curve2"/>'
    o += text(x1 - 6, y0 + 2, "full attention: KV cache grows", "xs", "end")
    o += f'<path d="M{x0},{y1} L{x0+90},{y1-40} L{x1},{y1-40}" class="curve"/>'
    o += text(x1 - 6, y1 - 50, "sliding window: capped at w", "xs", "end")
    o += f'<path d="M{x0},{y1-14} L{x1},{y1-14}" class="curve" style="stroke-dasharray:6 4"/>'
    o += text(x1 - 6, y1 - 22, "DeltaNet: constant, dₖ·dᵥ per head", "xs", "end")
    o += text(760, 40, "what a decode keeps, per layer", "ttl")
    rows = [("attention", "every past k, v", "KVCache layer"), ("sliding", "the last w k, v", "ring of w"),
            ("DeltaNet", "conv tail + S", "DeltaNetState"), ("encoder", "nothing", "one pass")]
    for i, (a, b, c) in enumerate(rows):
        yy = 64 + i * 46
        o += text(640, yy, a, "lbl", "start") + text(640, yy + 15, f"{b} · {c}", "xs", "start")
    o += text(745, 252, "the model owns no state: the caller", "xs") + text(745, 266, "makes it and passes it to each step", "xs")
    return svg(W, H, o, "Decode state")


def prefill_decode():
    W, H = 940, 220
    o = ""
    o += text(240, 22, "prefill: the prompt in one call", "ttl")
    for i in range(8):
        o += _tok(40 + i * 46, 44, ["<s>", "Q", ":", "2", "+", "2", "=", "?"][i], "io", 40, 24)
    o += arrow([(220, 72), (220, 96)])
    o += box(120, 98, 200, 36, "stack (all positions)", "mix")
    o += arrow([(220, 134), (220, 154)])
    o += text(220, 166, "cache ← 8 positions · logits of the last", "xs")
    o += text(680, 22, "decode: one position per call", "ttl")
    for i in range(4):
        x = 520 + i * 100
        o += _tok(x, 44, ["4", ".", "<e>", ""][i] or "…", "act", 40, 24)
        o += box(x - 10, 98, 60, 36, "step", "mix", None, 4, "xs")
        o += arrow([(x + 20, 68), (x + 20, 98)])
        if i < 3:
            o += arrow([(x + 50, 116), (x + 90, 116)], dash=True)
    o += text(680, 166, "each step appends its k, v (or advances S), then the search picks a token", "xs")
    o += text(470, 204, "the search (greedy, sampling with top-k / top-p / min-p, beams) sits outside the model, over the last logits", "xs")
    return svg(W, H, o, "Prefill and decode")


# ================================================================== Laya

def laya():
    W, H = 960, 330
    o = ""
    toks = [("[CLS]", "io"), ("Q:", "io"), ("which", "io"), ("…", "io"), ("[MASK]", "on"), ("opt A", "io"),
            ("[MASK]", "on"), ("opt B", "io"), ("[MASK]", "on"), ("opt C", "io"), ("[SEP]", "io"), ("state …", "io")]
    for i, (t, k) in enumerate(toks):
        o += rect(30 + i * 74, 30, 68, 26, "b on" if k == "on" else "b io", 4) + text(64 + i * 74, 43, t, "xs")
    o += text(480, 18, "one prompt: a question, each option behind a [MASK] marker, then the state", "xs")
    o += arrow([(480, 56), (480, 76)])
    o += box(300, 78, 360, 36, "ModernBERT encoder", "mix", "bidirectional, local/global")
    o += arrow([(480, 114), (480, 134)])
    o += box(300, 136, 360, 30, "+ kind embedding (choice · score · yes/no), every position", "io", None, 4, "xs")
    o += arrow([(480, 166), (480, 184)])
    o += box(300, 186, 360, 34, "2 head layers", "mix", "pre-norm · MHA, no RoPE · ReLU 4D · biased")
    for i, x in enumerate([4, 6, 8]):
        cx = 64 + x * 74
        o += f'<path d="M{480 - 120 + i * 120},220 L{cx},246" class="ar"/>'
        o += box(cx - 40, 248, 80, 26, "scorer", "act", None, 4, "xs")
    o += text(390, 290, "LayerNorm → MLP d → d → 1 at each marker: one logit per option", "xs")
    o += box(740, 238, 200, 40, "softmax(logits / T)", "pool", "T by kind × option count")
    o += arrow([(714, 261), (738, 258)])
    o += text(850, 296, "calibrated probabilities", "xs")
    o += rect(740, 126, 200, 62, "b conv", 6) + text(840, 142, "act head", "lbl")
    o += text(840, 160, "first token's state + top prob,", "xs") + text(840, 174, "margin, entropy, count → acts", "xs")
    o += arrow([(660, 203), (738, 172)], dash=True)
    return svg(W, H, o, "The Laya decision model")
