"""SVG figures for the transformer chapters; same vocabulary as figs.py."""
import math
import random

from .figs import add, arrow, box, esc, mul, rect, svg, text


def _tokens(x, y, kinds, s=22, gap=4, labels=None):
    o = ""
    for i, k in enumerate(kinds):
        o += rect(x + i * (s + gap), y, s, s, f"b {k}", 3)
        if labels and labels[i]:
            o += text(x + i * (s + gap) + s / 2, y + s / 2 + 1, labels[i], "xs")
    return o


def _grid(x, y, n, cell, cls_fn, gap=0):
    o = ""
    for r in range(n):
        for c in range(n):
            o += rect(x + c * cell, y + r * cell, cell - gap, cell - gap, cls_fn(r, c))
    return o


# ---------------------------------------------------------------- the vision encoder

def vit_macro():
    W, H = 960, 250
    o = ""
    o += rect(20, 56, 128, 128, "cell")
    for i in range(1, 4):
        o += f'<path d="M{20+32*i},56 L{20+32*i},184 M20,{56+32*i} L148,{56+32*i}" class="line"/>'
    o += text(84, 200, "image  H × W × 3", "xs") + text(84, 214, "cut in p × p patches", "xs")
    o += arrow([(152, 120), (176, 120)])
    o += box(178, 92, 104, 56, "patch embed", "conv", "p×p conv, stride p")
    o += arrow([(282, 120), (304, 120)])
    kinds = ["io"] + ["conv"] * 8
    o += _tokens(308, 98, kinds, labels=["c"] + [""] * 8)
    o += _tokens(308, 126, ["norm"] * 9, s=22)
    o += text(308 + 9 * 26 / 2 - 2, 82, "[class] + N patch tokens, width D", "xs")
    o += text(308 + 9 * 26 / 2 - 2, 164, "+ learnt position embedding", "xs")
    o += arrow([(544, 120), (566, 120)])
    o += box(568, 84, 150, 72, "encoder block", "mix", "× L  (attention, MLP)")
    o += arrow([(718, 120), (740, 120)])
    o += box(742, 96, 84, 48, "pool", "pool", "class token")
    o += arrow([(826, 120), (848, 120)])
    o += box(850, 96, 90, 48, "head", "io", "norm → fc")
    o += text(480, 234, "one resolution throughout: every layer sees all N = HW/p² tokens at width D (isotropic)", "sm")
    return svg(W, H, o, "The ViT pipeline")


def patchify():
    W, H = 920, 250
    o = ""
    o += rect(30, 40, 160, 160, "cell")
    for r in range(4):
        for c in range(4):
            cls = "on" if (r, c) == (1, 2) else "cell"
            o += rect(30 + c * 40, 40 + r * 40, 40, 40, cls)
            o += text(50 + c * 40, 60 + r * 40, str(r * 4 + c + 1), "xs")
    o += text(110, 218, "p × p × C pixels per patch", "xs")
    o += arrow([(190, 100), (238, 100)], hi=True)
    o += box(240, 64, 74, 72, "flatten", "io", "p²C")
    o += arrow([(314, 100), (340, 100)])
    o += box(342, 64, 96, 72, "× W_E", "conv", "p²C → D")
    o += arrow([(438, 100), (464, 100)])
    o += box(466, 76, 62, 48, "token", "conv", "D")
    o += text(384, 168, "= a conv with kernel p and stride p:", "sm")
    o += text(384, 184, "one shared linear map per patch, no overlap", "sm")
    o += text(740, 30, "the grid is read row by row", "ttl")
    for i in range(16):
        x = 572 + i * 21
        o += rect(x, 88, 18, 24, "b on" if i == 6 else "b conv", 3) + text(x + 9, 101, str(i + 1), "xs")
    o += text(740, 132, "a set of N tokens; their positions are now only", "xs")
    o += text(740, 146, "what the position embedding tells the model", "xs")
    return svg(W, H, o, "Patch embedding")


def prefix_tokens():
    W, H = 940, 250
    o = ""

    def row(y, title, embed_prefix):
        s = text(20, y - 16, title, "ttl", "start")
        kinds = ["io", "attn", "reg", "reg"] + ["conv"] * 9
        labels = ["cls", "dist", "r₁", "r₂"] + [""] * 9
        x0 = 20
        s += _tokens(x0, y, kinds, s=26, gap=6, labels=labels)
        start = 0 if embed_prefix else 4
        for i in range(start, 13):
            s += rect(x0 + i * 32, y + 32, 26, 10, "b norm", 2)
        if not embed_prefix:
            s += text(x0 + 2 * 32 - 3, y + 38, "join after", "xs")
        s += text(x0 + 13 * 32 + 12, y + 13, "tokens", "xs", "start")
        s += text(x0 + 13 * 32 + 12, y + 38, "+ position embedding", "xs", "start")
        return s

    o += row(48, "default: the position embedding covers the prefix tokens too", True)
    o += row(150, "no_embed_class: positions over the patches alone; the prefix joins after", False)
    o += text(640, 70, "class: the token the head reads", "xs", "start")
    o += text(640, 86, "dist: DeiT's distillation token", "xs", "start")
    o += text(640, 102, "r₁…r_k: registers, read by", "xs", "start")
    o += text(640, 116, "nothing (scratch space)", "xs", "start")
    o += text(470, 236, "the prefix tokens are learnt vectors shared by every image; only the patch tokens carry the image", "sm")
    return svg(W, H, o, "Prefix tokens and the position embedding")


def hybrid():
    W, H = 960, 220
    o = ""
    o += box(16, 80, 76, 52, "image", "io", "224²×3")
    o += arrow([(92, 106), (112, 106)])
    o += f'<rect x="114" y="54" width="330" height="104" rx="10" class="slot"/>'
    o += text(279, 44, "ResNetV2 trunk: post-activation, GN + WS, SAME pads, no head", "xs")
    o += box(126, 80, 86, 52, "stem", "conv", "7×7 s2 + pool")
    o += arrow([(212, 106), (228, 106)])
    o += box(230, 80, 96, 52, "stages 1–3", "conv", "3, 4, 9 units")
    o += arrow([(326, 106), (342, 106)])
    o += box(344, 80, 88, 52, "map", "conv", "14² × 1024")
    o += arrow([(444, 106), (470, 106)], hi=True)
    o += box(472, 80, 112, 52, "patch embed", "conv", "1×1 patches")
    o += arrow([(584, 106), (604, 106)])
    o += box(606, 80, 110, 52, "tokens", "io", "196 + class")
    o += arrow([(716, 106), (736, 106)])
    o += box(738, 80, 110, 52, "ViT encoder", "mix", "× 12")
    o += arrow([(848, 106), (868, 106)])
    o += box(870, 80, 74, 52, "head", "io")
    o += text(480, 186, "the conv trunk does the early, local work; the patch grid is its stride-16 map, so each token is a", "sm")
    o += text(480, 202, "receptive field the convs already built (R50+ViT-B/16; the R26 and Ti hybrids cut other maps)", "sm")
    return svg(W, H, o, "Hybrid ViT")


# ---------------------------------------------------------------- attention

def attention_map():
    W, H = 960, 310
    o = ""
    o += text(110, 22, "project", "ttl") + text(470, 22, "score and normalize", "ttl") + text(820, 22, "mix", "ttl")
    o += box(30, 120, 60, 60, "X", "io", "N × D")
    for i, (nm, y) in enumerate([("Q", 56), ("K", 126), ("V", 196)]):
        o += arrow([(90, 150), (126, y + 22)])
        o += box(128, y, 82, 44, f"{nm} = X W_{nm.lower()}", "conv", "N × d_h")
    rnd = random.Random(11)
    n, cell, x0, y0 = 10, 17, 380, 56
    for r in range(n):
        ws = [rnd.random() ** 3 for _ in range(n)]
        ws[r] += 0.6
        tot = sum(ws)
        for c in range(n):
            a = ws[c] / tot
            cls = "on" if r == 3 else "on2"
            o += f'<rect x="{x0+c*cell}" y="{y0+r*cell}" width="{cell-1}" height="{cell-1}" class="{cls}" style="fill-opacity:{min(0.95, 0.08 + a*2.2):.2f};stroke:none"/>'
    o += rect(x0 - 1, y0 - 1, n * cell + 1, n * cell + 1, "grid")
    o += text(x0 + n * cell / 2, y0 + n * cell + 16, "A = softmax(Q Kᵀ / √d_h)", "mt")
    o += text(x0 + n * cell / 2, y0 + n * cell + 34, "N × N; each row sums to 1", "xs")
    o += arrow([(210, 78), (300, 78), (370, 90)])
    o += arrow([(210, 148), (300, 148), (370, 140)])
    o += text(x0 - 8, y0 + 3 * cell + 8, "query i →", "xs", "end")
    o += text(x0 + n * cell / 2, y0 - 10, "keys →", "xs")
    o += arrow([(210, 218), (660, 218), (660, 160), (690, 160)], dash=True)
    o += arrow([(x0 + n * cell + 4, 140), (690, 140)])
    o += box(692, 118, 110, 54, "Y = A V", "attn", "N × d_h per head")
    o += arrow([(802, 145), (824, 145)])
    o += box(826, 112, 112, 66, "concat · W_O", "conv", "h heads → D")
    o += text(480, 290, "h heads run in parallel, each on a D/h slice; a token's output is a convex mix of every token's value", "sm")
    return svg(W, H, o, "Multi-head self-attention")


def conv_vs_attn():
    W, H = 900, 270
    o = ""
    n, cell = 9, 20

    def grid(x0, title, fn, sub):
        s = text(x0 + n * cell / 2, 22, title, "ttl")
        for r in range(n):
            for c in range(n):
                s += fn(x0, r, c)
        s += text(x0 + n * cell / 2, 50 + n * cell + 16, sub[0], "xs")
        s += text(x0 + n * cell / 2, 50 + n * cell + 30, sub[1], "xs")
        return s

    def conv(x0, r, c):
        cls = "on" if (r, c) == (4, 4) else "on2" if abs(r - 4) <= 1 and abs(c - 4) <= 1 else "cell"
        return rect(x0 + c * cell, 40 + r * cell, cell - 1, cell - 1, cls)

    rnd = random.Random(5)
    ws = {(r, c): rnd.random() ** 4 for r in range(n) for c in range(n)}
    for k in [(0, 7), (1, 1), (6, 2), (8, 8), (3, 6)]:
        ws[k] = 0.9

    def attn(x0, r, c):
        if (r, c) == (4, 4):
            return rect(x0 + c * cell, 40 + r * cell, cell - 1, cell - 1, "on")
        a = ws[(r, c)]
        return (f'<rect x="{x0+c*cell}" y="{40+r*cell}" width="{cell-1}" height="{cell-1}" class="on3" '
                f'style="fill-opacity:{0.05 + a*0.9:.2f}"/>')

    o += grid(60, "3×3 conv", conv, ("local, and the same weights at every position:", "a fixed, translation-equivariant mixer"))
    o += grid(520, "self-attention", attn, ("global, and weights computed from the content:", "a data-dependent mixer over a set"))
    o += text(450, 120, "both mix tokens;", "sm") + text(450, 136, "they differ in where", "sm") + text(450, 152, "the weights come from", "sm")
    return svg(W, H, o, "Convolution versus self-attention")


def unit_vs_block():
    W, H = 960, 330
    o = ""
    o += text(230, 20, "ResNetV2 pre-activation bottleneck", "ttl")
    o += text(720, 20, "pre-norm transformer block", "ttl")
    # left: unit
    lx = 120
    items = [("BN · ReLU → 1×1", "conv", "channel mix, reduce"), ("BN · ReLU → 3×3", "mix", "spatial mix"),
             ("BN · ReLU → 1×1", "conv", "channel mix, expand")]
    y = 52
    o += arrow([(lx - 50, 40), (lx - 50, 290)], hi=True)
    o += text(lx - 58, 44, "x", "mt", "end")
    o += text(500 + 62, 44, "x", "mt", "end")
    for i, (l, k, s) in enumerate(items):
        o += box(lx, y, 190, 50, l, k, s)
        if i < 2:
            o += arrow([(lx + 95, y + 50), (lx + 95, y + 66)])
        y += 68
    o += arrow([(lx - 50, 46), (lx + 95, 46), (lx + 95, 50)])
    o += arrow([(lx + 95, 238), (lx + 95, 262), (lx - 39, 262)])
    o += add(lx - 50, 262)
    o += text(lx + 230, 140, "one residual:", "xs", "start") + text(lx + 230, 154, "both kinds of", "xs", "start")
    o += text(lx + 230, 168, "mixing inside", "xs", "start")
    # right: block
    rx = 620
    o += arrow([(rx - 50, 40), (rx - 50, 296)], hi=True)
    o += arrow([(rx - 50, 46), (rx + 95, 46), (rx + 95, 54)])
    o += box(rx, 56, 190, 44, "LN → self-attention", "attn", "token mix (all positions)")
    o += arrow([(rx + 95, 100), (rx + 95, 124), (rx - 39, 124)])
    o += add(rx - 50, 124)
    o += arrow([(rx - 50, 150), (rx + 95, 150), (rx + 95, 160)])
    o += box(rx, 162, 190, 58, "LN → MLP", "act", "D → 4D → D: two 1×1s")
    o += arrow([(rx + 95, 220), (rx + 95, 250), (rx - 39, 250)])
    o += add(rx - 50, 250)
    o += text(rx + 210, 80, "sublayer 1", "xs", "start") + text(rx + 210, 190, "sublayer 2", "xs", "start")
    o += text(480, 312, "the same pre-activation law (a clean identity stream) with the mixers factored apart: tokens, then channels", "sm")
    return svg(W, H, o, "A ResNetV2 unit beside a transformer block")


# ---------------------------------------------------------------- MLP

def mlp_kinds():
    W, H = 960, 280
    o = ""

    def col(x, title, sub):
        return text(x + 140, 22, title, "ttl") + text(x + 140, 266, sub, "xs")

    # plain
    x = 20
    o += col(x, "Mlp", "fc₂(act(fc₁ x)) · GELU, QuickGELU, tanh-GELU")
    o += box(x + 90, 40, 100, 34, "x", "io", None)
    o += arrow([(x + 140, 74), (x + 140, 88)])
    o += box(x + 70, 90, 140, 38, "fc₁", "conv", "D → rD")
    o += arrow([(x + 140, 128), (x + 140, 142)])
    o += box(x + 90, 144, 100, 32, "act", "act")
    o += arrow([(x + 140, 176), (x + 140, 190)])
    o += box(x + 70, 192, 140, 38, "fc₂", "conv", "rD → D")
    o += text(x + 248, 172, "(BEiT3: LN", "xs") + text(x + 248, 184, "on rD here)", "xs")
    # SwiGLU
    x = 330
    o += col(x, "SwiGLU", "fc₂(SiLU(x W_g) ⊙ x W_x) · AIMv2")
    o += box(x + 90, 40, 100, 34, "x", "io")
    o += arrow([(x + 120, 74), (x + 70, 92)]) + arrow([(x + 160, 74), (x + 210, 92)])
    o += box(x + 10, 94, 120, 38, "fc_g", "conv", "D → h")
    o += box(x + 150, 94, 120, 38, "fc_x", "conv", "D → h")
    o += arrow([(x + 70, 132), (x + 70, 142)])
    o += box(x + 30, 144, 80, 28, "SiLU", "act")
    o += arrow([(x + 70, 172), (x + 128, 186)]) + arrow([(x + 210, 132), (x + 152, 186)])
    o += mul(x + 140, 194)
    o += arrow([(x + 140, 205), (x + 140, 214)])
    o += box(x + 70, 216, 140, 34, "fc₂", "conv", "h → D")
    # packed
    x = 650
    o += col(x, "SwiGLU, packed", "one fc₁ of 2h rows, split in halves · DINOv2-g")
    o += box(x + 90, 40, 100, 34, "x", "io")
    o += arrow([(x + 140, 74), (x + 140, 88)])
    o += box(x + 60, 90, 160, 38, "fc₁", "conv", "D → 2h, then split")
    o += arrow([(x + 100, 128), (x + 80, 142)]) + arrow([(x + 180, 128), (x + 152, 186)])
    o += box(x + 40, 144, 80, 28, "SiLU(g)", "act")
    o += arrow([(x + 80, 172), (x + 128, 186)])
    o += text(x + 180, 158, "v", "mt", "start")
    o += mul(x + 140, 194)
    o += arrow([(x + 140, 205), (x + 140, 214)])
    o += box(x + 70, 216, 140, 34, "fc₂", "conv", "h → D")
    return svg(W, H, o, "MLP kinds")


# ---------------------------------------------------------------- the residual sublayer

def sublayer_slots():
    W, H = 960, 210
    o = ""
    y = 96
    o += f'<path d="M14,{y} L944,{y}" class="wire"/>'
    o += text(16, y - 12, "stream", "xs", "start")
    o += rect(70, y - 22, 80, 44, "pill", 6) + rect(70, y - 22, 80, 44, "slot", 6) + text(110, y, "input", "xs")
    o += f'<circle cx="172" cy="{y}" r="3.5" class="pin"/>'
    o += f'<path d="M172,{y} L172,{y+62} L800,{y+62} L800,{y+11}" class="wire"/>'
    o += text(480, y + 76, "skip: the identity", "xs")
    o += f'<rect x="186" y="{y-30}" width="560" height="60" rx="8" class="bg2"/>'
    items = [(196, 76, "pre", "slot"), (284, 120, "f: attention | MLP", "b mix"), (416, 76, "post", "slot"),
             (504, 96, "γ layer scale", "slot"), (612, 120, "drop path (train)", "slot")]
    for x, w, l, cls in items:
        o += rect(x, y - 20, w, 40, cls, 6) + text(x + w / 2, y, l, "xs")
    o += add(800, y)
    o += rect(830, y - 22, 80, 44, "pill", 6) + rect(830, y - 22, 80, 44, "slot", 6) + text(870, y, "out", "xs")
    o += text(466, y - 44, "the branch: residual = γ ⊙ post(f(pre(x)))", "sm")
    o += text(480, 196, "pre-norm fills pre · post-norm fills out · res-post-norm fills post · sandwich fills pre and post", "sm")
    return svg(W, H, o, "The residual sublayer's slots")


def norm_placement():
    W, H = 960, 330
    o = ""

    def panel(x, title, sub, kind):
        s = text(x + 140, 22, title, "ttl") + text(x + 140, 300, sub[0], "xs") + text(x + 140, 314, sub[1], "xs")
        sx = x + 60
        hi_ok = kind != "post"
        s += arrow([(sx, 40), (sx, 270)], hi=hi_ok)
        s += text(sx - 8, 44, "x", "mt", "end")
        if kind == "pre":
            s += arrow([(sx, 70), (sx + 130, 70), (sx + 130, 86)])
            s += box(sx + 90, 88, 80, 32, "LN", "norm")
            s += arrow([(sx + 130, 120), (sx + 130, 134)])
            s += box(sx + 80, 136, 100, 40, "f", "mix")
            s += arrow([(sx + 130, 176), (sx + 130, 210), (sx + 11, 210)])
            s += add(sx, 210)
            s += text(sx + 4, 236, "x + f(LN x)", "mt", "start")
        elif kind == "post":
            s += arrow([(sx, 70), (sx + 130, 70), (sx + 130, 96)])
            s += box(sx + 80, 98, 100, 40, "f", "mix")
            s += arrow([(sx + 130, 138), (sx + 130, 160), (sx + 11, 160)])
            s += add(sx, 160)
            s += f'<rect x="{sx-30}" y="186" width="60" height="30" rx="6" class="b norm"/>' + text(sx, 201, "LN", "lbl")
            s += text(sx + 40, 236, "LN(x + f(x))", "mt", "start")
        else:
            s += arrow([(sx, 70), (sx + 130, 70), (sx + 130, 86)])
            s += box(sx + 80, 88, 100, 40, "f", "mix")
            s += arrow([(sx + 130, 128), (sx + 130, 142)])
            s += box(sx + 90, 144, 80, 32, "LN", "norm")
            s += arrow([(sx + 130, 176), (sx + 130, 210), (sx + 11, 210)])
            s += add(sx, 210)
            s += text(sx + 4, 236, "x + LN(f x)", "mt", "start")
        return s

    o += panel(10, "post-norm", ("the 2017 Transformer; ResNet v1's act-after-add", "the stream is renormalized at every add"), "post")
    o += panel(330, "pre-norm", ("ViT, DeiT, CLIP, Swin V1; ResNetV2's law", "clean identity: x_L = x_l + Σ fᵢ"), "pre")
    o += panel(650, "res-post-norm", ("Swin V2, timm's rpn ViT", "clean identity, and each branch's size is capped"), "res")
    return svg(W, H, o, "Norm placement")


def layer_scale():
    W, H = 900, 240
    o = ""
    o += text(220, 22, "x + diag(γ) · f(x)", "mt")
    rnd = random.Random(2)
    for i in range(24):
        h0 = 3
        x = 40 + i * 15
        o += rect(x, 140 - h0, 10, h0, "b norm")
    o += f'<path d="M34,140 L410,140" class="line"/>'
    o += text(220, 158, "at init: γ small (CaiT: 0.1 … 10⁻⁶ by depth)", "xs")
    o += text(220, 172, "each block starts near the identity", "xs")
    for i in range(24):
        h = 8 + 70 * rnd.random() ** 1.5
        x = 490 + i * 15
        o += rect(x, 140 - h, 10, h, "b attn")
    o += f'<path d="M484,140 L860,140" class="line"/>'
    o += text(670, 158, "trained: one learned gain per channel", "xs")
    o += text(670, 172, "on each sublayer's residual", "xs")
    o += arrow([(420, 100), (476, 100)])
    o += text(450, 214, "the transformer's version of a ResNet's zero-initialized last BN γ: depth stays trainable", "sm")
    o += text(450, 230, "because a new block contributes almost nothing until it has learned something", "sm")
    return svg(W, H, o, "Layer scale")


# ---------------------------------------------------------------- pools

def token_pools():
    W, H = 960, 270
    o = ""

    def panel(x, title, sub, mode):
        s = text(x + 115, 22, title, "ttl") + text(x + 115, 246, sub[0], "xs") + text(x + 115, 260, sub[1], "xs")
        kinds = (["io", "attn"] if mode == "dist" else ["io"]) + ["conv"] * 6
        if mode in ("avg", "map"):
            kinds = ["reg"] + ["conv"] * 7
        tx, ty = x + 10, 170
        s += _tokens(tx, ty, kinds, s=22, gap=5)
        cx = lambda i: tx + i * 27 + 11
        if mode == "token":
            s += arrow([(cx(0), ty), (cx(0), 104)], hi=True)
            s += box(cx(0) - 30, 70, 60, 32, "fc", "io")
        elif mode == "avg":
            s += rect(x + 70, 76, 90, 32, "b pool", 6) + text(x + 115, 92, "mean", "lbl")
            for i in range(1, 8):
                s += f'<path d="M{cx(i)},{ty} L{x+115},{110}" class="line"/>'
            s += text(cx(0), ty + 36, "prefix", "xs") + text(cx(0), ty + 48, "left out", "xs")
        elif mode == "map":
            s += rect(x + 80, 92, 70, 28, "b attn", 6) + text(x + 115, 106, "latent q", "xs")
            rnd = random.Random(4)
            for i in range(1, 8):
                s += f'<path d="M{cx(i)},{ty} L{x+115},{122}" class="ar" style="stroke-opacity:{0.2+0.8*rnd.random():.2f}"/>'
            s += arrow([(x + 115, 92), (x + 115, 72)])
            s += rect(x + 70, 44, 90, 26, "b act", 6) + text(x + 115, 57, "+ MLP(LN)", "xs")
        else:
            s += arrow([(cx(0), ty), (cx(0), 120)], hi=True)
            s += arrow([(cx(1), ty), (cx(1) + 40, 120)], hi=True)
            s += box(cx(0) - 26, 88, 52, 30, "fc", "io")
            s += box(cx(1) + 14, 88, 60, 30, "fc_dist", "io")
            s += text(x + 115, 60, "average the logits", "xs")
        return s

    o += panel(20, "token", ("read the class token", "ViT, DeiT, CLIP"), "token")
    o += panel(255, "avg", ("mean of the patch tokens", "Swin, gap ViTs, AIMv2"), "avg")
    o += panel(490, "map", ("a learnt query attends over them", "SigLIP (an attention pool)"), "map")
    o += panel(725, "distilled", ("class + distillation tokens,", "two classifiers (DeiT)"), "dist")
    return svg(W, H, o, "Token pools")


# ---------------------------------------------------------------- windows

def window_partition():
    W, H = 960, 330
    o = ""
    n, cell = 8, 24
    cols = ["on2", "on3", "attn", "pool"]

    # regular
    x0, y0 = 30, 50
    o += text(x0 + 96, 26, "block l: W-MSA", "ttl")
    for r in range(n):
        for c in range(n):
            k = (r // 4) * 2 + (c // 4)
            o += rect(x0 + c * cell, y0 + r * cell, cell - 1, cell - 1, f"b {cols[k]}")
    o += text(x0 + 96, y0 + n * cell + 18, "4 windows of 4×4; attention inside each", "xs")
    # shifted: regions 3x3
    x1 = 360
    o += text(x1 + 96, 26, "block l+1: SW-MSA (shift by 2)", "ttl")
    reg = lambda i: 0 if i < 2 else 1 if i < 6 else 2
    names = "ABCDEFGHI"
    for r in range(n):
        for c in range(n):
            k = reg(r) * 3 + reg(c)
            cls = ["on2", "on3", "attn", "pool", "norm", "act", "io", "conv", "mix"][k]
            o += rect(x1 + c * cell, y0 + r * cell, cell - 1, cell - 1, f"b {cls}")
    for rr, (ra, rb) in enumerate([(0, 2), (2, 6), (6, 8)]):
        for cc, (ca, cb) in enumerate([(0, 2), (2, 6), (6, 8)]):
            o += text(x1 + (ca + cb) / 2 * cell, y0 + (ra + rb) / 2 * cell, names[rr * 3 + cc], "ttl")
    o += f'<path d="M{x1+2*cell},{y0} L{x1+2*cell},{y0+n*cell} M{x1+6*cell},{y0} L{x1+6*cell},{y0+n*cell} M{x1},{y0+2*cell} L{x1+n*cell},{y0+2*cell} M{x1},{y0+6*cell} L{x1+n*cell},{y0+6*cell}" class="ar" style="stroke-width:2.2"/>'
    o += text(x1 + 96, y0 + n * cell + 18, "windows straddle the old borders: 9 pieces", "xs")
    # cyclic shift
    x2 = 690
    o += text(x2 + 96, 26, "cyclic shift + mask", "ttl")
    for r in range(n):
        for c in range(n):
            rs, cs = (r + 2) % n, (c + 2) % n
            k = reg(rs) * 3 + reg(cs)
            cls = ["on2", "on3", "attn", "pool", "norm", "act", "io", "conv", "mix"][k]
            o += rect(x2 + c * cell, y0 + r * cell, cell - 1, cell - 1, f"b {cls}")
    o += f'<path d="M{x2+4*cell},{y0} L{x2+4*cell},{y0+n*cell} M{x2},{y0+4*cell} L{x2+n*cell},{y0+4*cell}" class="ar" style="stroke-width:2.2"/>'
    o += text(x2 + 96, y0 + n * cell + 18, "roll by −2: the pieces pack into 4 windows;", "xs")
    o += text(x2 + 96, y0 + n * cell + 32, "a mask stops pieces that were not neighbours", "xs")
    o += text(x2 + 96, y0 + n * cell + 46, "from attending to each other", "xs")
    o += arrow([(232, 146), (352, 146)])
    o += arrow([(562, 146), (682, 146)])
    o += text(480, 316, "alternating the two lets information cross window borders while every attention stays M×M sized", "sm")
    return svg(W, H, o, "Shifted windows")


def attn_cost():
    W, H = 900, 290
    o = ""
    x0, y0, pw, ph = 80, 30, 560, 210
    o += f'<path d="M{x0},{y0} L{x0},{y0+ph} L{x0+pw},{y0+ph}" class="line"/>'
    sides = [224 + 16 * i for i in range(0, 51)]
    smin, smax = 224, 1024
    C, M = 96, 7

    def cost_g(s):
        N = (s / 4) ** 2
        return 2 * N * N * C

    def cost_w(s):
        N = (s / 4) ** 2
        return 2 * M * M * N * C

    base = cost_w(224)
    lo, hi = 0, 5
    X = lambda s: x0 + (s - smin) / (smax - smin) * pw
    Y = lambda v: y0 + ph - (math.log10(v / base) - lo) / (hi - lo) * ph
    for k in range(lo, hi + 1):
        o += f'<path d="M{x0},{Y(base*10**k)} L{x0+pw},{Y(base*10**k)}" class="grid"/>'
        o += text(x0 - 8, Y(base * 10 ** k), f"{10**k:,}×", "xs", "end")
    for s in [224, 384, 512, 768, 1024]:
        o += text(X(s), y0 + ph + 14, str(s), "xs")
    o += text(x0 + pw / 2, y0 + ph + 32, "image side (pixels), stage-1 tokens N = (side/4)²", "xs")
    pg = " L".join(f"{X(s):.1f},{max(y0, Y(cost_g(s))):.1f}" for s in sides if Y(cost_g(s)) >= y0 - 1)
    pw_ = " L".join(f"{X(s):.1f},{Y(cost_w(s)):.1f}" for s in sides)
    o += f'<path d="M{pg}" class="curve2"/>'
    o += f'<path d="M{pw_}" class="curve"/>'
    o += text(x0 + pw + 12, 60, "global attention:", "sm", "start") + text(x0 + pw + 12, 76, "2N²C, grows as side⁴", "xs", "start")
    o += text(x0 + pw + 12, 190, "7×7 windows:", "sm", "start") + text(x0 + pw + 12, 206, "2M²NC, grows as side²", "xs", "start")
    o += text(x0 + pw + 12, 120, "(attention score cost", "xs", "start") + text(x0 + pw + 12, 134, "only, C = 96, relative", "xs", "start")
    o += text(x0 + pw + 12, 148, "to windows at 224)", "xs", "start")
    return svg(W, H, o, "Attention cost: global versus windowed")


def rel_pos_bias():
    W, H = 960, 300
    o = ""
    o += text(200, 22, "V1: a learnt table indexed by offset", "ttl")
    M, cell = 3, 30
    x0, y0 = 30, 60
    for r in range(M):
        for c in range(M):
            cls = "on" if (r, c) == (0, 0) else "on3" if (r, c) == (2, 1) else "cell"
            o += rect(x0 + c * cell, y0 + r * cell, cell - 1, cell - 1, cls)
    o += text(x0 + 15, y0 + 15, "q", "xs") + text(x0 + 45, y0 + 75, "k", "xs")
    o += text(x0 + 45, y0 + M * cell + 16, "window M = 3", "xs")
    o += text(x0 + 45, y0 + M * cell + 30, "Δ = k − q = (2, 1)", "xs")
    tx, ty, tc = 190, 40, 26
    n = 2 * M - 1
    for r in range(n):
        for c in range(n):
            cls = "on" if (r, c) == (2 + 2, 2 + 1) else "cell"
            o += rect(tx + c * tc, ty + r * tc, tc - 1, tc - 1, cls)
    o += text(tx + n * tc / 2, ty + n * tc + 16, "(2M−1)² entries per head", "xs")
    o += text(tx + n * tc / 2, ty + n * tc + 30, "score += B[Δ]", "mt")
    o += arrow([(124, 120), (184, 150)], hi=True)
    # V2
    o += text(700, 22, "V2: a continuous bias from an MLP", "ttl")
    lx, ly, lw = 520, 70, 380
    o += f'<path d="M{lx},{ly} L{lx+lw},{ly}" class="line"/>'
    o += f'<path d="M{lx},{ly+50} L{lx+lw},{ly+50}" class="line"/>'
    o += text(lx - 6, ly, "Δ", "mt", "end") + text(lx - 6, ly + 50, "Δ̂", "mt", "end")
    R = 11
    for d in range(-R, R + 1):
        xl = lx + lw / 2 + d * (lw / 2) / R
        dh = math.copysign(math.log2(1 + abs(d)), d) / math.log2(1 + R)
        xh = lx + lw / 2 + dh * lw / 2
        o += f'<path d="M{xl:.1f},{ly-4} L{xl:.1f},{ly+4}" class="line"/>'
        o += f'<path d="M{xh:.1f},{ly+46} L{xh:.1f},{ly+54}" class="line"/>'
        if d % 4 == 0 or abs(d) == R:
            o += f'<path d="M{xl:.1f},{ly+5} L{xh:.1f},{ly+45}" class="ar dash" style="stroke-opacity:.5"/>'
    o += text(lx + lw / 2, ly + 70, "Δ̂ = sign(Δ) · log(1 + |Δ|), scaled by the pretraining window", "xs")
    o += box(560, 168, 130, 44, "MLP 2 → 512 → h", "conv", "ReLU, no out bias")
    o += arrow([(690, 190), (716, 190)])
    o += box(718, 168, 90, 44, "16 · σ", "act", "bias ∈ (0, 16)")
    o += arrow([(808, 190), (834, 190)])
    o += box(836, 168, 100, 44, "B[Δ]", "attn", "per head")
    o += arrow([(710, ly + 78), (620, 166)])
    o += text(730, 248, "the bias is a smooth function of offset, so a window larger than the one trained", "xs")
    o += text(730, 262, "extrapolates gently instead of indexing table rows that never trained", "xs")
    return svg(W, H, o, "Relative position bias")


def cosine_attn():
    W, H = 900, 230
    o = ""
    cx, cy, r = 150, 120, 80
    o += f'<circle cx="{cx}" cy="{cy}" r="{r}" class="grid"/>'
    a1, a2 = math.radians(20), math.radians(70)
    o += arrow([(cx, cy), (cx + 1.5 * r * math.cos(a1), cy - 1.5 * r * math.sin(a1))], hi=True)
    o += arrow([(cx, cy), (cx + 0.7 * r * math.cos(a2), cy - 0.7 * r * math.sin(a2))])
    o += text(cx + 1.5 * r * math.cos(a1) + 10, cy - 1.5 * r * math.sin(a1), "q", "mt", "start")
    o += text(cx + 0.7 * r * math.cos(a2) + 4, cy - 0.7 * r * math.sin(a2) - 10, "k", "mt", "start")
    o += f'<path d="M{cx+30*math.cos(a1)},{cy-30*math.sin(a1)} A30,30 0 0 0 {cx+30*math.cos(a2)},{cy-30*math.sin(a2)}" class="line"/>'
    o += text(cx + 44, cy - 40, "θ", "mt")
    o += text(470, 50, "V1, scaled dot product:  q·k / √d = |q||k| cos θ / √d", "mt")
    o += text(470, 72, "unbounded: a few large-norm tokens can saturate the softmax", "xs")
    o += text(470, 120, "V2, scaled cosine:  cos θ · s,   s = exp(min(ℓ, ln 100))", "mt")
    o += text(470, 142, "logits bounded by ±100 per head, whatever the activations' size;", "xs")
    o += text(470, 156, "the learned scale s takes the role of 1/τ", "xs")
    o += text(470, 200, "with res-post-norm, the fix for large Swin models' runaway activations", "sm")
    return svg(W, H, o, "Cosine attention")


def patch_merge():
    W, H = 960, 270
    o = ""
    x0, y0, cell = 30, 50, 34
    cols = ["on2", "on3", "attn", "pool"]
    for r in range(4):
        for c in range(4):
            k = (r % 2) * 2 + (c % 2)
            o += rect(x0 + c * cell, y0 + r * cell, cell - 1, cell - 1, f"b {cols[k]}")
            o += text(x0 + c * cell + cell / 2, y0 + r * cell + cell / 2, str(k), "xs")
    o += f'<path d="M{x0+2*cell},{y0} L{x0+2*cell},{y0+4*cell} M{x0},{y0+2*cell} L{x0+4*cell},{y0+2*cell}" class="ar" style="stroke-width:2"/>'
    o += text(x0 + 2 * cell, y0 + 4 * cell + 18, "grid h × w, width C", "xs")
    o += arrow([(178, 118), (222, 118)])
    for i in range(4):
        o += rect(226, 62 + i * 28, 46, 26, f"b {cols[i]}", 3) + text(249, 75 + i * 28, f"x{i}", "xs")
    o += text(249, 186, "concat: 4C", "xs")
    o += text(249, 200, "(space → depth)", "xs")
    o += arrow([(272, 110), (302, 72)]) + arrow([(272, 126), (302, 162)])
    o += text(304, 36, "V1: norm, then reduce", "ttl", "start")
    o += box(304, 50, 80, 40, "LN", "norm", "4C")
    o += arrow([(384, 70), (404, 70)])
    o += box(406, 50, 130, 40, "linear, no bias", "conv", "4C → 2C")
    o += text(304, 130, "V2: reduce, then norm", "ttl", "start")
    o += box(304, 144, 130, 40, "linear, no bias", "conv", "4C → 2C")
    o += arrow([(434, 164), (454, 164)])
    o += box(456, 144, 80, 40, "LN", "norm", "2C")
    o += arrow([(540, 70), (600, 110)]) + arrow([(540, 164), (600, 126)])
    for r in range(2):
        for c in range(2):
            o += rect(610 + c * 46, 72 + r * 46, 45, 45, "b mix")
    o += text(656, 178, "h/2 × w/2, width 2C", "xs")
    o += text(820, 80, "the transformer's", "sm") + text(820, 96, "strided conv:", "sm")
    o += text(820, 116, "a 2×2, stride-2 conv", "xs") + text(820, 130, "written as unshuffle", "xs")
    o += text(820, 144, "+ 1×1 (odd sides pad)", "xs")
    o += text(480, 250, "four tokens become one with their channels side by side: the side halves and the width doubles, as in a conv stage", "sm")
    return svg(W, H, o, "Patch merging")


def pyramid():
    W, H = 960, 270
    o = ""
    o += text(230, 22, "ViT: isotropic", "ttl") + text(700, 22, "Swin: a pyramid, like a ResNet", "ttl")
    for i in range(4):
        x = 40 + i * 104
        o += rect(x, 70, 80, 120, "b mix", 4)
        o += text(x + 40, 130, "H/16", "xs")
        o += text(x + 40, 214, "D" if i else "D, N tokens", "xs")
        if i < 3:
            o += arrow([(x + 80, 130), (x + 102, 130)])
    o += text(230, 234, "one grid and one width, every layer", "xs")
    sides = [120, 60, 30, 15]
    labels = [("H/4", "C"), ("H/8", "2C"), ("H/16", "4C"), ("H/32", "8C")]
    x = 520
    for i, s in enumerate(sides):
        w = 40 + 18 * i
        o += rect(x, 130 - s / 2 - 10, w, s + 20, "b mix", 4)
        o += text(x + w / 2, 130, labels[i][0], "xs")
        o += text(x + w / 2, 214, labels[i][1], "xs")
        if i < 3:
            o += arrow([(x + w, 130), (x + w + 30, 130)], hi=True)
            o += text(x + w + 15, 116, "merge", "xs")
        x += w + 34
    o += text(700, 234, "merges halve the grid and double the width: feature maps at strides 4–32", "xs")
    o += text(480, 256, "the pyramid is what dense tasks (detection, segmentation) read, so Swin drops into a ResNet's place", "sm")
    return svg(W, H, o, "Isotropic versus pyramid")


# ---------------------------------------------------------------- families

def distill():
    W, H = 960, 260
    o = ""
    o += _tokens(30, 150, ["io", "attn"] + ["conv"] * 8, s=24, gap=5, labels=["cls", "dst"] + [""] * 8)
    o += text(160, 194, "class, distillation, patch tokens", "xs")
    o += arrow([(320, 162), (350, 162)])
    o += box(352, 130, 150, 64, "ViT encoder", "mix", "they all attend to each other")
    o += arrow([(502, 148), (560, 90)], hi=True)
    o += arrow([(502, 176), (560, 222)], hi=True)
    o += box(562, 66, 110, 48, "fc (class)", "io", "← true label")
    o += box(562, 198, 110, 48, "fc_dist", "io", "← teacher's label")
    o += arrow([(672, 90), (742, 140)]) + arrow([(672, 222), (742, 160)])
    o += box(744, 124, 96, 48, "average", "pool", "at inference")
    o += arrow([(840, 148), (868, 148)])
    o += box(870, 128, 70, 40, "logits", "io")
    o += box(562, 128, 110, 44, "conv teacher", "conv", "RegNetY-16GF")
    o += arrow([(617, 172), (617, 196)], dash=True)
    o += text(480, 20, "DeiT's distillation token: a second class token, trained to agree with a conv teacher", "sm")
    return svg(W, H, o, "Distillation through attention")


def registers():
    W, H = 960, 280
    o = ""
    n, cell = 10, 18
    rnd = random.Random(8)
    base = {(r, c): 0.15 + 0.5 * math.exp(-((r - 4.5) ** 2 + (c - 5) ** 2) / 8) + 0.05 * rnd.random()
            for r in range(n) for c in range(n)}
    outl = {(0, 8), (1, 1), (8, 0), (9, 7)}

    def grid(x0, with_out, x_shift=0):
        s = ""
        for (r, c), v in base.items():
            a = 0.95 if with_out and (r, c) in outl else v
            cls = "on" if with_out and (r, c) in outl else "on2"
            s += (f'<rect x="{x0+c*cell}" y="{50+r*cell}" width="{cell-1}" height="{cell-1}" class="{cls}" '
                  f'style="fill-opacity:{a:.2f};stroke:none"/>')
        return s

    o += text(130, 26, "no registers", "ttl")
    o += grid(40, True)
    o += text(130, 250, "a few background patches turn into", "xs") + text(130, 264, "high-norm global scratch tokens", "xs")
    o += text(560, 26, "with registers", "ttl")
    for i, (k, l) in enumerate(zip(["io", "reg", "reg", "reg", "reg"], ["cls", "r₁", "r₂", "r₃", "r₄"])):
        o += rect(410, 52 + i * 30, 26, 26, f"b {k}", 3) + text(423, 66 + i * 30, l, "xs")
    o += text(423, 214, "learnt;", "xs") + text(423, 228, "dropped at", "xs") + text(423, 242, "the output", "xs")
    o += grid(470, False)
    o += text(560, 250, "the scratch work moves into the registers;", "xs") + text(560, 264, "the patch map stays local and clean", "xs")
    o += text(820, 110, "registers are prefix", "sm") + text(820, 126, "tokens no pool reads:", "sm")
    o += text(820, 146, "the reg₁ / reg₄ in a", "xs") + text(820, 160, "name is their count", "xs")
    return svg(W, H, o, "Register tokens")


# ---------------------------------------------------------------- the transformer board

TX_KIND = dict(backbone="conv", patch="conv", tokens="io", norm_pre="norm", stack="mix", final="norm", pool="pool",
               pool_norm="norm", head="io", a_pre="norm", a_mix="attn", a_post="norm", a_scale="norm", a_out="norm",
               m_pre="norm", m_mlp="act", m_post="norm", m_scale="norm", m_out="norm")
TX_LINK = dict(backbone="f-vit.html#hybrid", patch="t-patch.html", tokens="t-patch.html#tokens",
               norm_pre="f-clip.html#clip", stack="tx-algebra.html#board", final="t-pool.html#norms",
               pool="t-pool.html", pool_norm="t-pool.html#norms", head="c-head.html#fc",
               a_pre="t-sublayer.html#placement", a_mix="t-attention.html", a_post="t-sublayer.html#placement",
               a_scale="t-sublayer.html#scale", a_out="t-sublayer.html#placement",
               m_pre="t-sublayer.html#placement", m_mlp="t-mlp.html", m_post="t-sublayer.html#placement",
               m_scale="t-sublayer.html#scale", m_out="t-sublayer.html#placement")
TX_NAME = dict(backbone="BACKBONE", patch="PATCH EMBED", tokens="TOKENS", norm_pre="NORM PRE", stack="STACK",
               final="FINAL NORM", pool="POOL", pool_norm="POOL NORM", head="HEAD", a_pre="PRE", a_mix="TOKEN MIXER",
               a_post="POST", a_scale="LAYER SCALE", a_out="OUT", m_pre="PRE", m_mlp="CHANNEL MIXER (MLP)",
               m_post="POST", m_scale="LAYER SCALE", m_out="OUT")
TX_CAPTION = dict(TX_NAME, a_pre="attention pre-norm", a_mix="token mixer", a_post="attention post-norm",
                  a_scale="attention layer scale", a_out="attention out-norm", m_pre="MLP pre-norm", m_mlp="MLP",
                  m_post="MLP post-norm", m_scale="MLP layer scale", m_out="MLP out-norm")


def _chip(x, y, w, h, slot, val, changed):
    o = text(x + 2, y - 8, TX_NAME[slot], "caps", "start")
    if val is None:
        o += rect(x, y, w, h, "slot", 6)
        o += text(x + w / 2, y + h - 11, "empty", "xs")
    else:
        title, params = val[0], val[1]
        link = val[2] if len(val) > 2 and val[2] else TX_LINK[slot]
        kind = val[3] if len(val) > 3 else TX_KIND[slot]
        inner = rect(x, y, w, h, f"b {kind}", 6) + text(x + w / 2, y + 15, title, "ct")
        for i, p in enumerate(params):
            inner += text(x + w / 2, y + 30 + i * 12.5, p, "xs")
        o += f'<a href="{link}"><title>{esc(TX_CAPTION[slot])}: {esc(title)}</title>{inner}</a>'
    if changed:
        o += rect(x - 3, y - 3, w + 6, h + 6, "chg", 8)
        o += text(x + w + 1, y - 8, "Δ", "chgt", "end")
    return o


def circuit_tx(spec, changed):
    W, H = 1040, 430
    o = ""
    my, mh = 26, 62
    wy = my + mh / 2
    o += f'<path d="M6,{wy} L1034,{wy}" class="wire"/>'
    macro = [("backbone", 100), ("patch", 116), ("tokens", 126), ("norm_pre", 84), ("stack", 152), ("final", 88),
             ("pool", 104), ("pool_norm", 88), ("head", 92)]
    x, pos = 14, {}
    for slot, w in macro:
        pos[slot] = x
        o += _chip(x, my, w, mh, slot, spec.get(slot), slot in changed)
        x += w + 8
    sx = pos["stack"]
    o += f'<path d="M{sx},{my+mh} L20,130" class="ar dash"/><path d="M{sx+152},{my+mh} L1020,130" class="ar dash"/>'
    o += rect(20, 130, 1000, 288, "board", 10)
    o += text(36, 148, spec.get("block", "ENCODER BLOCK  ×  L   (two residual sublayers on one stream)"), "caps", "start")
    lay = spec.get("layers", "")
    lw = len(lay) * 6.2 + 70
    o += rect(1004 - lw, 138, lw, 20, "pill", 10) + text(1004 - lw / 2, 148, f"layers: {lay}", "xs")
    if "layers" in changed:
        o += rect(1001 - lw, 135, lw + 6, 26, "chg", 12)
    h = 60
    rows = [("a", 206, ["a_pre", "a_mix", "a_post", "a_scale"], "ATTENTION SUBLAYER"),
            ("m", 332, ["m_pre", "m_mlp", "m_post", "m_scale"], "MLP SUBLAYER")]
    geo = [(76, 120), (212, 220), (448, 110), (574, 110)]
    plus_x, out_x, out_w = 722, 752, 104
    for r, (pfx, sy, slots, label) in enumerate(rows):
        y = sy - h / 2
        start = 28 if r == 0 else 36
        o += f'<path d="M{start},{sy} L{plus_x-11},{sy}" class="wire"/>'
        if r == 0:
            o += text(30, sy - 10, "x", "mt", "start")
        o += f'<circle cx="54" cy="{sy}" r="3.5" class="pin"/>'
        o += f'<path d="M54,{sy} L54,{sy+46} L{plus_x},{sy+46} L{plus_x},{sy+11}" class="wire"/>'
        o += text(388, sy + 56, "identity", "xs")
        for slot, (gx, gw) in zip(slots, geo):
            o += _chip(gx, y, gw, h, slot, spec.get(slot), slot in changed)
        o += add(plus_x, sy)
        o += f'<path d="M{plus_x+11},{sy} L{out_x},{sy}" class="wire"/>'
        oslot = f"{pfx}_out"
        o += _chip(out_x, y, out_w, h, oslot, spec.get(oslot), oslot in changed)
        o += text(1000, sy + 20, label, "caps", "end")
        if r == 0:
            o += f'<path d="M{out_x+out_w},{sy} L1008,{sy} L1008,{sy+58} L36,{sy+58} L36,{rows[1][1]}" class="wire"/>'
        else:
            o += f'<path d="M{out_x+out_w},{sy} L1010,{sy}" class="wire"/>' + text(1008, sy - 10, "y", "mt", "end")
    note = spec.get("note", "")
    o += text(36, 406, "drop path (train) on each residual before its add" + (f"  ·  {note}" if note else ""), "xs", "start")
    return svg(W, H, o, f"Circuit of {spec.get('name', 'a variant')}")
