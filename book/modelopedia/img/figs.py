"""SVG figures for the book. Each public function returns an <svg> string.

Vocabulary (CSS classes in style.css): box kinds conv, norm, act, pool, attn,
mix, io, reg; `ar` arrows; `add` circles. Colours come from CSS variables, so
every figure follows the page's light/dark theme.
"""
import math

DEFS = (
    '<defs>'
    '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>'
    '<marker id="ahh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M0,0 L10,5 L0,10 z" class="ahh"/></marker>'
    '</defs>'
)


def svg(w, h, body, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">{DEFS}{body}</svg>'


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="lbl", anchor="middle", extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" dominant-baseline="middle" {extra}>{esc(s)}</text>'


def box(x, y, w, h, label, k="conv", sub=None, rx=6, cls_text="lbl"):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="b {k}"/>'
    cx = x + w / 2
    if sub:
        out += text(cx, y + h / 2 - 7, label, cls_text) + text(cx, y + h / 2 + 9, sub, "xs")
    else:
        out += text(cx, y + h / 2, label, cls_text)
    return out


def arrow(pts, hi=False, dash=False, head=True):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    cls = "ar" + (" hi" if hi else "") + (" dash" if dash else "")
    m = (' marker-end="url(#ahh)"' if hi else ' marker-end="url(#ah)"') if head else ""
    return f'<path d="{d}" class="{cls}"{m}/>'


def add(x, y, r=11, sym="+"):
    return f'<circle cx="{x}" cy="{y}" r="{r}" class="add"/>' + text(x, y + 1, sym, "ttl")


def mul(x, y, r=11):
    return add(x, y, r, "×")


def rect(x, y, w, h, cls, rx=0, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}" {extra}/>'


def vchain(x, y0, w, h, gap, items):
    """A vertical chain of boxes joined by arrows; returns (svg, y_end)."""
    out, y = "", y0
    for i, it in enumerate(items):
        label, k = it[0], it[1]
        sub = it[2] if len(it) > 2 else None
        out += box(x, y, w, h, label, k, sub)
        if i < len(items) - 1:
            out += arrow([(x + w / 2, y + h), (x + w / 2, y + h + gap - 2)])
        y += h + gap
    return out, y - gap


# ---------------------------------------------------------------- macro

def macro():
    """The conv net's macro shape: stem, four stages, head; slabs sized by geometry."""
    W, H = 960, 330
    o = ""
    cy = 150
    # (label, sub, side, channels, kind)
    slabs = [
        ("image", "3 × 224²", 224, 3, "io"),
        ("stem", "64 × 56²  (÷4)", 56 * 2.2, 64, "conv"),
        ("stage 1", "256 × 56²  (÷4)", 56 * 2.2, 256, "mix"),
        ("stage 2", "512 × 28²  (÷8)", 28 * 2.2, 512, "mix"),
        ("stage 3", "1024 × 14²  (÷16)", 14 * 2.6, 1024, "mix"),
        ("stage 4", "2048 × 7²  (÷32)", 7 * 3.2, 2048, "mix"),
    ]
    x = 20
    prev = None
    for i, (lab, sub, side, ch, k) in enumerate(slabs):
        h = max(side * 0.95, 18)
        th = 10 + math.log2(ch) * 6.5 if i else 14
        dx, dy = 14, -10
        # pseudo-3d slab
        o += f'<polygon points="{x},{cy-h/2} {x+dx},{cy-h/2+dy} {x+dx+th},{cy-h/2+dy} {x+th},{cy-h/2}" class="b {k}" opacity=".7"/>'
        o += f'<polygon points="{x+th},{cy-h/2} {x+dx+th},{cy-h/2+dy} {x+dx+th},{cy+h/2+dy} {x+th},{cy+h/2}" class="b {k}" opacity=".55"/>'
        o += rect(x, cy - h / 2, th, h, f"b {k}")
        o += text(x + th / 2 + 6, cy + max(h / 2, 30) + 22, lab, "ttl")
        o += text(x + th / 2 + 6, cy + max(h / 2, 30) + 40, sub, "xs")
        if prev is not None:
            o += arrow([(prev + 18, cy), (x - 6, cy)])
        prev = x + th
        x += th + 52 if i < 5 else th
    # head
    x += 46
    o += arrow([(prev + 18, cy), (x - 6, cy)])
    o += box(x, cy - 22, 70, 44, "final", "act", "optional")
    o += arrow([(x + 70, cy), (x + 92, cy)])
    o += box(x + 94, cy - 22, 62, 44, "pool", "pool", "2048")
    o += arrow([(x + 156, cy), (x + 178, cy)])
    o += box(x + 180, cy - 22, 62, 44, "head", "io", "1000")
    o += text(x + 120, cy + 62, "ends: stem & head", "sm")
    o += text(480, 22, "spatial side shrinks ÷2 per stage; channels grow ×2  —  ResNet-50 at 224 px", "sm")
    return svg(W, H, o, "Macro architecture of a residual conv net")


def stage_chain():
    W, H = 900, 170
    o = ""
    xs = [20, 230, 400, 570, 740]
    o += box(20, 55, 150, 60, "unit 0", "mix", "stride s, C_in → C, projection")
    for i, x in enumerate([230, 400, 570]):
        o += box(x, 55, 120, 60, f"unit {i+1}", "mix", "stride 1, identity")
    o += text(735, 85, "…", "ttl")
    o += box(770, 55, 110, 60, "unit d−1", "mix", "stride 1, identity")
    for a, b in [(170, 230), (350, 400), (520, 570), (690, 725)]:
        o += arrow([(a, 85), (b - 4, 85)])
    o += arrow([(745, 85), (766, 85)])
    o += text(450, 22, "a stage = unit₀ ∘ unit₁ ∘ … ∘ unit_{d−1}: the first unit changes shape, the rest preserve it", "sm")
    o += text(95, 140, "changes (C, H, W)", "xs") + text(560, 140, "endomorphisms of one feature space  ℝ^{C×H×W}", "xs")
    return svg(W, H, o, "A stage is a chain of residual units")


def lowering():
    W, H = 960, 300
    o = ""
    fams = ["ResNet", "ResNeXt / Wide", "ResNetV2 / BiT", "Res2Net", "ResNeSt", "SKNet", "RegNet"]
    for i, f in enumerate(fams):
        y = 18 + i * 39
        o += box(20, y, 170, 30, f, "io")
        o += arrow([(190, y + 15), (318, 150)])
    o += box(322, 105, 190, 90, "conv-net policy", "mix", "stem · stages · layer choices · head")
    o += arrow([(512, 150), (566, 150)])
    o += box(570, 105, 170, 90, "structure", "conv", "every unit resolved")
    o += arrow([(740, 150), (794, 150)])
    o += box(798, 105, 140, 90, "model", "norm", "one module tree")
    o += text(420, 230, "family policies speak their paper's vocabulary", "xs")
    o += text(420, 246, "(depths, cardinality, radix, w0/wa/wm …)", "xs")
    o += text(655, 230, "strides past output stride → dilation;", "xs")
    o += text(655, 246, "unit kinds → conv paths", "xs")
    return svg(W, H, o, "Every family lowers to one generic conv-net policy")


# ---------------------------------------------------------------- the unit

def unit_schema():
    """The generic residual unit with every slot."""
    W, H = 900, 560
    o = ""
    cx = 360
    o += box(cx - 80, 14, 160, 34, "x", "io")
    o += arrow([(cx, 48), (cx, 70)])
    o += box(cx - 110, 72, 220, 38, "pre: norm → act", "norm", "pre-activation units only")
    o += arrow([(cx, 110), (cx, 130)])
    # split point
    o += f'<circle cx="{cx}" cy="134" r="4" class="dot"/>'
    # path column
    px = cx - 120
    o += f'<rect x="{px-30}" y="150" width="300" height="300" rx="10" class="slot"/>'
    o += text(px - 22, 162, "path", "sm", "start")
    o += arrow([(cx, 138), (cx, 172)])
    o += box(cx - 110, 174, 220, 36, "conv₀  (1×1 reduce)", "conv", "conv → norm → act")
    o += arrow([(cx, 210), (cx, 226)])
    o += box(cx - 110, 228, 220, 44, "MIXER  (k×k, strides)", "mix", "conv · Res2 · SplitAttn · SK")
    o += box(cx + 128, 236, 86, 28, "drop block", "reg")
    o += arrow([(cx + 110, 250), (cx + 126, 250)], dash=True, head=False)
    o += arrow([(cx, 272), (cx, 286)])
    o += box(cx - 110, 288, 220, 30, "anti-alias pool", "pool", None)
    o += arrow([(cx, 318), (cx, 332)])
    o += box(cx - 110, 334, 220, 30, "attention  @ mixer", "attn")
    o += arrow([(cx, 364), (cx, 378)])
    o += box(cx - 110, 380, 220, 36, "conv₂  (1×1 expand)", "conv", "conv → norm   (no act)")
    o += arrow([(cx, 450), (cx, 466)])
    o += box(cx - 110, 420 + 48, 220, 30, "attention  @ out", "attn")
    o += arrow([(cx, 498), (cx, 506)])
    o += box(cx - 70, 508, 140, 26, "drop path", "reg")
    o += arrow([(cx, 534), (cx, 541)])
    o += add(cx, 552 - 0)
    # shortcut
    sx = 700
    o += arrow([(cx, 134), (sx, 134), (sx, 250)])
    o += box(sx - 90, 252, 180, 70, "shortcut", "pool", "identity · 1×1 conv · avg → 1×1 · none")
    o += arrow([(sx, 322), (sx, 552), (cx + 14, 552)])
    o += text(sx + 8, 196, "projection reads pre(x);", "xs", "start")
    o += text(sx + 8, 210, "identity reads x or pre(x)", "xs", "start")
    o += arrow([(cx - 11, 552), (cx - 150, 552)], head=True)
    o += box(cx - 290, 538, 136, 28, "out act", "act")
    o += text(cx - 222, 528, "post-act units only", "xs")
    # legend
    lx = 30
    for i, (k, l) in enumerate([("conv", "conv block"), ("mix", "spatial mixer"), ("norm", "norm"), ("act", "activation"),
                                ("pool", "pool / shortcut"), ("attn", "channel attention"), ("reg", "train-only regularizer")]):
        o += rect(lx, 70 + i * 24, 16, 14, f"b {k}", 3) + text(lx + 24, 77 + i * 24, l, "sm", "start")
    o += text(lx, 250, "dashed slots are optional;", "xs", "start")
    o += text(lx, 264, "a basic unit has no conv₀", "xs", "start")
    return svg(W, H + 20, o, "The generic residual unit and its slots")


def resid_geometry():
    """Residual iteration as small steps along a flow (Euler view)."""
    W, H = 900, 300
    o = ""
    # left: one step
    o += rect(10, 10, 380, 280, "bg2", 8)
    ox, oy = 70, 240
    xx, xy = 230, 150
    o += arrow([(ox, oy), (xx, xy)])
    o += text((ox + xx) / 2 - 18, (oy + xy) / 2 - 8, "x", "mt")
    fx, fy = xx + 40, xy - 85
    o += arrow([(xx, xy), (fx, fy)], hi=True)
    o += text(xx + 36, xy - 36, "F(x)", "mt", "start")
    o += arrow([(ox, oy), (fx, fy)], dash=True)
    o += text(fx - 10, fy - 12, "x + F(x)", "mt", "end")
    o += text(200, 270, "a unit perturbs the identity: y = x + F(x)", "sm")
    o += text(200, 30, "one unit", "ttl")
    # right: flow
    o += rect(410, 10, 480, 280, "bg2", 8)
    o += text(650, 30, "a stage: many small steps", "ttl")
    pts = []
    x, y = 450, 240
    for i in range(9):
        t = i / 8
        nx = 450 + 400 * t
        ny = 240 - 160 * math.sin(t * 2.4) + 30 * t
        pts.append((nx, ny))
    for i in range(len(pts) - 1):
        o += arrow([pts[i], pts[i + 1]], hi=(i % 2 == 0))
        o += f'<circle cx="{pts[i][0]}" cy="{pts[i][1]}" r="3" class="dot"/>'
    o += text(pts[0][0], pts[0][1] + 16, "x₀", "mt")
    o += text(pts[-1][0] - 10, pts[-1][1] + 18, "x_d", "mt")
    o += text(650, 270, "x_{k+1} = x_k + F_k(x_k)  ≈  Euler step of  dx/dt = F(x, t)", "sm")
    return svg(W, H, o, "Geometry of residual updates")


def unrolled():
    W, H = 900, 260
    o = ""
    o += text(150, 20, "three units", "ttl")
    xs = [40, 120, 200, 280]
    for i in range(3):
        x = 40 + i * 90
        o += box(x + 20, 50, 50, 36, f"F{i+1}", "mix")
        o += arrow([(x, 120), (x + 88, 120)])
        o += arrow([(x + 10, 120), (x + 10, 68), (x + 18, 68)])
        o += arrow([(x + 70, 68), (x + 80, 68), (x + 80, 108)])
        o += add(x + 80, 120, 8)
    o += text(150, 160, "y = (1+F₃)(1+F₂)(1+F₁) x", "mt")
    o += text(640, 20, "unrolled: 2³ = 8 paths of length 0…3", "ttl")
    paths = ["", "1", "2", "3", "12", "13", "23", "123"]
    for i, p in enumerate(paths):
        y = 44 + i * 25
        o += f'<line x1="420" y1="{y}" x2="860" y2="{y}" class="line"/>'
        for j in range(3):
            x = 480 + j * 130
            if str(j + 1) in p:
                o += box(x, y - 9, 44, 18, f"F{j+1}", "mix", rx=4, cls_text="xs")
        o += text(880, y, str(len(p)), "xs")
    o += text(880, 30, "len", "xs")
    return svg(W, H, o, "Unrolled view of residual networks as path ensembles")


def post_vs_pre():
    W, H = 900, 420
    o = ""

    def col(x0, title, items, preact):
        s = text(x0 + 120, 18, title, "ttl")
        cx = x0 + 120
        s += box(cx - 40, 32, 80, 28, "x", "io")
        s += arrow([(cx, 60), (cx, 76)])
        y = 78
        for lab, k in items:
            s += box(cx - 70, y, 140, 30, lab, k)
            s += arrow([(cx, y + 30), (cx, y + 42)])
            y += 44
        s += add(cx, y + 6)
        s += arrow([(cx, 50), (x0 + 220, 50), (x0 + 220, y + 6), (cx + 12, y + 6)])
        if not preact:
            s += arrow([(cx, y + 17), (cx, y + 34)])
            s += box(cx - 50, y + 36, 100, 28, "ReLU", "act")
            s += text(cx, y + 84, "the sum is rectified: the", "xs")
            s += text(cx, y + 97, "shortcut is not a pure identity", "xs")
        else:
            s += arrow([(cx, y + 17), (cx, y + 40)])
            s += text(cx, y + 58, "nothing after the add: x_L = x_l + Σ F_i", "xs")
            s += text(cx, y + 71, "a clean identity highway", "xs")
        return s

    o += col(40, "v1  post-activation", [("conv", "conv"), ("BN", "norm"), ("ReLU", "act"), ("conv", "conv"), ("BN", "norm")], False)
    o += col(500, "v2  pre-activation", [("BN", "norm"), ("ReLU", "act"), ("conv", "conv"), ("BN", "norm"), ("ReLU", "act"), ("conv", "conv")], True)
    return svg(W, H, o, "Post-activation versus pre-activation units")


def hourglass():
    """Unit kinds drawn with widths proportional to channels."""
    W, H = 960, 250
    o = ""

    def unit(x0, title, layers, note):
        s = text(x0 + 110, 18, title, "ttl")
        y = 40
        for (w, lab, k) in layers:
            ww = max(w * 0.55, len(lab) * 7.2 + 16)
            s += box(x0 + 110 - ww / 2, y, ww, 34, lab, k, rx=4, cls_text="sm")
            y += 50
        s += text(x0 + 110, y + 4, note, "xs")
        return s

    o += unit(0, "basic", [(64 * 2.4, "3×3, 64", "mix"), (64 * 2.4, "3×3, 64", "conv")], "two full-width convs")
    o += unit(240, "bottleneck", [(256, "1×1, 64", "conv"), (110, "3×3, 64 (mixer)", "mix"), (256, "1×1, 256", "conv")],
              "reduce → mix → expand (×4)")
    o += unit(480, "grouped bottleneck", [(256, "1×1, 128", "conv"), (160, "3×3, 128, g=32", "mix"), (256, "1×1, 256", "conv")],
              "ResNeXt / RegNet: grouped mixer")
    o += unit(720, "inverted (RegNetZ)", [(140, "1×1, 192", "conv"), (330, "3×3, 192, g", "mix"), (140, "1×1, 48", "conv")],
              "expand → mix → project, linear out")
    o += text(480, 236, "box width ∝ channels; the input/output of each unit is the residual stream width", "sm")
    return svg(W, H, o, "Basic, bottleneck, grouped and inverted units")


def shortcut_kinds():
    W, H = 960, 250
    o = ""
    kinds = [
        ("identity", [], "shape preserved"),
        ("projection (B)", [("1×1 conv, s=2", "conv"), ("norm", "norm")], "torchvision downsample"),
        ("avg-down (D)", [("2×2 avg, s=2", "pool"), ("1×1 conv, s=1", "conv"), ("norm", "norm")], "Bag of Tricks"),
        ("none", None, "RegNetZ: no residual"),
    ]
    for i, (t, items, note) in enumerate(kinds):
        x = 20 + i * 240
        o += text(x + 100, 18, t, "ttl")
        if items is None:
            o += box(x + 40, 40, 120, 30, "x", "io")
            o += text(x + 100, 110, "unit = path(x)", "mt")
            o += text(x + 100, 130, "where shape changes", "xs")
        elif not items:
            o += box(x + 40, 40, 120, 30, "x", "io")
            o += arrow([(x + 100, 70), (x + 100, 190)])
        else:
            o += box(x + 40, 40, 120, 30, "x", "io")
            o += arrow([(x + 100, 70), (x + 100, 82)])
            y = 84
            for lab, k in items:
                o += box(x + 30, y, 140, 28, lab, k, cls_text="sm")
                y += 38
                o += arrow([(x + 100, y - 10), (x + 100, y - 2)])
        o += text(x + 100, 220, note, "xs")
    return svg(W, H, o, "Shortcut kinds")


def strided_grid():
    W, H = 900, 250
    o = ""

    def grid(x0, y0, title, mode):
        s = text(x0 + 80, y0 - 14, title, "ttl")
        for r in range(4):
            for c in range(4):
                cls = "cell"
                if mode == "s2" and r % 2 == 0 and c % 2 == 0:
                    cls = "on"
                if mode == "avg":
                    cls = ["on2", "on3"][(r // 2 + c // 2) % 2]
                s += rect(x0 + c * 40, y0 + r * 40, 38, 38, cls, 3)
        return s

    o += grid(60, 40, "1×1 conv, stride 2", "s2")
    o += text(140, 215, "reads 4 of 16 positions: 75% of the input", "xs")
    o += text(140, 228, "never reaches the next unit's shortcut", "xs")
    o += grid(360, 40, "2×2 avg, then 1×1", "avg")
    o += text(440, 215, "every position contributes;", "xs")
    o += text(440, 228, "a box filter before the decimation", "xs")
    o += grid(660, 40, "3×3 conv, stride 2", "s2")
    for r in range(4):
        for c in range(4):
            pass
    o += f'<rect x="{660-2}" y="{40-2}" width="{40*2+38+4-40}" height="{40*2+38+4-40}" class="slot"/>'
    o += text(740, 215, "the 3×3 window overlaps its", "xs")
    o += text(740, 228, "neighbours (ResNet-B moves stride here)", "xs")
    return svg(W, H, o, "What a strided 1x1 conv sees")


# ---------------------------------------------------------------- conv algebra

def conv_block():
    W, H = 900, 130
    o = ""
    o += box(20, 45, 90, 40, "x", "io", "C_in × H × W")
    o += arrow([(110, 65), (148, 65)])
    o += box(150, 40, 150, 50, "conv k×k", "conv", "stride s, dilation d, groups g")
    o += arrow([(300, 65), (338, 65)])
    o += box(340, 40, 110, 50, "norm", "norm", "BN · GN · EvoNorm")
    o += arrow([(450, 65), (488, 65)])
    o += box(490, 40, 110, 50, "hook  f", "reg", "drop block lives here")
    o += arrow([(600, 65), (638, 65)])
    o += box(640, 40, 100, 50, "act", "act", "ReLU · SiLU")
    o += arrow([(740, 65), (778, 65)])
    o += box(780, 45, 100, 40, "y", "io", "C_out × ⌈H/s⌉²")
    o += text(450, 115, "the atom: conv → norm → (f) → act.  Optional: weight standardization of the conv, SAME padding", "xs")
    return svg(W, H, o, "The conv block")


def grouped():
    W, H = 900, 270
    o = ""
    n = 16

    def mat(x0, y0, g, title, sub):
        s = text(x0 + 80, y0 - 18, title, "ttl")
        sz = 10
        for r in range(n):
            for c in range(n):
                blk = n // g
                cls = "on2" if r // blk == c // blk else "cell"
                s += rect(x0 + c * sz, y0 + r * sz, sz - 1, sz - 1, cls)
        s += text(x0 + 80, y0 + n * sz + 16, sub, "xs")
        return s

    o += mat(40, 50, 1, "dense, g = 1", "C_out·C_in·k² weights")
    o += mat(270, 50, 4, "grouped, g = 4", "÷ g weights and FLOPs")
    o += mat(500, 50, 16, "depthwise, g = C", "spatial-only mixing")
    o += text(800, 60, "channel-mixing matrix", "sm")
    o += text(800, 78, "per spatial tap", "sm")
    o += text(800, 110, "rows: output channels", "xs")
    o += text(800, 124, "cols: input channels", "xs")
    o += text(800, 160, "a grouped conv is a", "xs")
    o += text(800, 174, "direct sum of g convs:", "xs")
    o += text(800, 196, "W = W₁ ⊕ W₂ ⊕ … ⊕ W_g", "mt")
    return svg(W, H, o, "Grouped convolution as block-diagonal channel mixing")


def dilation():
    W, H = 900, 240
    o = ""

    def g(x0, d, title):
        s = text(x0 + 70, 20, title, "ttl")
        for r in range(7):
            for c in range(7):
                on = (r - 3) % d == 0 and (c - 3) % d == 0 and abs(r - 3) <= d and abs(c - 3) <= d
                s += rect(x0 + c * 20, 36 + r * 20, 18, 18, "on" if on else "cell", 2)
        return s

    o += g(40, 1, "3×3, dilation 1")
    o += text(110, 196, "reach 3", "xs")
    o += g(260, 2, "3×3, dilation 2")
    o += text(330, 196, "reach 5, same 9 weights", "xs")
    o += g(480, 3, "3×3, dilation 3")
    o += text(550, 196, "reach 7", "xs")
    o += text(790, 70, "reach = d·(k−1) + 1", "mt")
    o += text(790, 110, "dilation buys receptive", "xs")
    o += text(790, 124, "field without stride:", "xs")
    o += text(790, 138, "the map keeps its size", "xs")
    return svg(W, H, o, "Dilated convolution taps")


def rf_stack():
    W, H = 900, 220
    o = ""
    o += text(220, 18, "three 3×3 convs ≡ one 7×7 reach", "ttl")
    xs = list(range(7))
    for layer in range(4):
        y = 190 - layer * 50
        lo = 3 - layer
        hi = 3 + layer
        for i in xs:
            cls = "dot" if lo <= i <= hi else "off"
            o += f'<circle cx="{60 + i * 50}" cy="{y}" r="7" class="{cls}"/>'
        if layer:
            for i in range(lo, hi + 1):
                for j in (i - 1, i, i + 1):
                    if lo - 1 <= j <= hi + 1 and 3 - (layer - 1) <= j <= 3 + (layer - 1):
                        o += f'<line x1="{60 + i * 50}" y1="{y + 7}" x2="{60 + j * 50}" y2="{y + 43}" class="line"/>'
    o += text(470, 190, "input", "xs", "start") + text(470, 40, "after 3 convs", "xs", "start")
    o += text(700, 70, "params: 3·(3²) = 27 C²", "mt")
    o += text(700, 100, "vs 7² = 49 C²", "mt")
    o += text(700, 140, "and three nonlinearities", "xs")
    o += text(700, 154, "instead of one (VGG; deep stems)", "xs")
    return svg(W, H, o, "Receptive field growth of stacked small convs")


def output_stride():
    W, H = 900, 230
    o = ""
    o += text(450, 18, "output stride: once the net has strided enough, later strides become dilation", "ttl")
    rows = [("OS 32", [1, 1, 1, 1], [2, 2, 2]), ("OS 16", [1, 1, 1, 2], [2, 2, 1]), ("OS 8", [1, 1, 2, 4], [2, 1, 1])]
    names = ["stem", "stage 1", "stage 2", "stage 3", "stage 4"]
    for j, n in enumerate(names):
        o += text(200 + j * 140, 46, n, "sm")
    for i, (lab, dil, strides) in enumerate(rows):
        y = 70 + i * 44
        o += text(70, y + 14, lab, "ttl")
        side = [56, 56, 28, 14, 7]
        cur = 56
        for j in range(5):
            if j == 0:
                s, d = "÷4", 1
                cur = 56
            elif j == 1:
                s, d = "s1", 1
            else:
                st = strides[j - 2]
                d = dil[j - 1]
                if st == 2:
                    cur //= 2
                s = f"s{st}" + (f" d{d}" if d > 1 else "")
            k = "conv" if j == 0 else ("mix" if "d" not in s[2:] else "attn")
            o += box(140 + j * 140, y, 120, 34, f"{s}", k, f"map {cur}²", cls_text="sm")
    o += text(450, 218, "the map stays at 28² or 14²; dilation keeps each unit's receptive field as if it had strided", "xs")
    return svg(W, H, o, "Output stride and dilation")


# ---------------------------------------------------------------- norms

def norm_cubes():
    W, H = 900, 240
    o = ""

    def cube(x0, title, kind):
        s = text(x0 + 80, 20, title, "ttl")
        # a front face grid N (rows) x C (cols), depth for HW
        nN, nC = 4, 8
        cw, ch = 16, 26
        dx, dy = 30, -22
        for r in range(nN):
            for c in range(nC):
                on = False
                if kind == "bn":
                    on = c == 2
                if kind == "gn":
                    on = r == 1 and 4 <= c < 8
                if kind == "ln":
                    on = r == 1
                cls = "on" if on else "cell"
                x, y = x0 + c * cw, 70 + r * ch
                s += rect(x, y, cw, ch, cls)
        # top and side faces
        tx, ty, w, h = x0, 70, nC * cw, nN * ch
        s += f'<polygon points="{tx},{ty} {tx+dx},{ty+dy} {tx+w+dx},{ty+dy} {tx+w},{ty}" class="grid"/>'
        s += f'<polygon points="{tx+w},{ty} {tx+w+dx},{ty+dy} {tx+w+dx},{ty+h+dy} {tx+w},{ty+h}" class="grid"/>'
        s += text(x0 - 10, 70 + h / 2, "N", "sm", "end")
        s += text(x0 + w / 2, 70 + h + 14, "C", "sm")
        s += text(x0 + w + dx / 2 + 12, 70 + h / 2 - 4, "H·W", "sm", "start")
        return s

    o += cube(40, "batch norm", "bn")
    o += text(105, 210, "stats over (N, H, W) per channel", "xs")
    o += cube(330, "group norm", "gn")
    o += text(395, 210, "over (C/G, H, W) per sample, per group", "xs")
    o += cube(620, "layer norm", "ln")
    o += text(685, 210, "over (C, H, W) per sample", "xs")
    return svg(W, H, o, "Normalization axes")


# ---------------------------------------------------------------- attention

def se():
    W, H = 960, 190
    o = ""
    o += box(10, 70, 90, 50, "x", "io", "C × H × W")
    o += f'<circle cx="112" cy="95" r="3" class="dot"/>'
    o += arrow([(100, 95), (140, 95)])
    o += box(142, 70, 110, 50, "squeeze", "pool", "spatial mean → C")
    o += arrow([(252, 95), (282, 95)])
    o += box(284, 70, 100, 50, "fc₁  1×1", "conv", "C → C/r")
    o += arrow([(384, 95), (408, 95)])
    o += box(410, 70, 70, 50, "act", "act")
    o += arrow([(480, 95), (504, 95)])
    o += box(506, 70, 100, 50, "fc₂  1×1", "conv", "C/r → C")
    o += arrow([(606, 95), (630, 95)])
    o += box(632, 70, 80, 50, "σ", "attn", "gate ∈ (0,1)ᶜ")
    o += arrow([(712, 95), (766, 95)])
    o += mul(780, 95)
    o += arrow([(112, 95), (112, 160), (780, 160), (780, 108)])
    o += arrow([(792, 95), (840, 95)])
    o += box(842, 70, 100, 50, "x ⊙ g", "io", "C × H × W")
    o += text(450, 30, "excite: a two-layer bottleneck MLP over the channel descriptor; the gate rescales channels", "sm")
    return svg(W, H, o, "Squeeze-and-excitation")


def se_vs_eca():
    W, H = 900, 260
    o = ""
    n = 16

    def m(x0, title, f, sub):
        s = text(x0 + 80, 22, title, "ttl")
        for r in range(n):
            for c in range(n):
                s += rect(x0 + c * 10, 40 + r * 10, 9, 9, f(r, c))
        s += text(x0 + 80, 220, sub, "xs")
        return s

    import random
    rnd = random.Random(3)
    o += m(60, "SE: channel mixing", lambda r, c: "on2" if rnd.random() < 0.85 else "on3", "dense, low-rank (C → C/r → C), nonlinear")
    o += m(330, "ECA: channel mixing", lambda r, c: "on2" if abs(r - c) <= 2 else "cell", "banded: a 1-d conv of k = 5 across channels")
    o += text(720, 60, "both:  g = σ(f(mean_{hw} x))", "mt")
    o += text(720, 90, "y = diag(g) · x", "mt")
    o += text(720, 130, "a per-sample diagonal", "xs")
    o += text(720, 144, "rescaling of channel space;", "xs")
    o += text(720, 158, "they differ only in f", "xs")
    o += text(720, 190, "ECA kernel  k = odd(|log₂C + 1| / 2)", "xs")
    return svg(W, H, o, "SE versus ECA")


def gate_geometry():
    W, H = 900, 240
    o = ""
    o += rect(10, 10, 430, 220, "bg2", 8)
    o += text(225, 30, "channel gating: diagonal, input-conditioned", "ttl")
    cx, cy = 225, 130
    o += f'<line x1="{cx-150}" y1="{cy}" x2="{cx+150}" y2="{cy}" class="line"/><line x1="{cx}" y1="{cy-85}" x2="{cx}" y2="{cy+85}" class="line"/>'
    o += f'<circle cx="{cx}" cy="{cy}" r="70" class="grid" stroke-dasharray="4 3"/>'
    o += f'<ellipse cx="{cx}" cy="{cy}" rx="110" ry="35" class="curve2"/>'
    o += text(cx + 140, cy - 10, "c₁", "mt") + text(cx + 10, cy - 80, "c₂", "mt", "start")
    o += text(cx, 215, "g = (1.0, 0.3): axes stretch, never rotate", "xs")
    o += rect(460, 10, 430, 220, "bg2", 8)
    o += text(675, 30, "split / selective attention: per channel, a point on a simplex", "ttl")
    ax, ay, bx, by, tx, ty = 520, 190, 830, 190, 675, 60
    o += f'<polygon points="{ax},{ay} {bx},{by} {tx},{ty}" class="grid"/>'
    for (u, v, w) in [(0.6, 0.3, 0.1), (0.2, 0.2, 0.6), (0.34, 0.33, 0.33), (0.1, 0.8, 0.1)]:
        px, py = u * ax + v * bx + w * tx, u * ay + v * by + w * ty
        o += f'<circle cx="{px}" cy="{py}" r="5" class="on"/>'
    o += text(ax - 6, ay + 14, "split 1", "xs") + text(bx + 6, by + 14, "split 2", "xs") + text(tx, ty - 10, "split 3", "xs")
    o += text(675, 218, "softmax weights a_c ∈ Δ^{r−1}: y_c = Σ_i a_{c,i} U_{i,c}", "xs")
    return svg(W, H, o, "Geometry of gating and branch selection")


def attn_place():
    W, H = 900, 330
    o = ""

    def col(x0, title, order, note):
        s = text(x0 + 130, 18, title, "ttl")
        y = 34
        for lab, k in order:
            s += box(x0 + 40, y, 180, 30, lab, k, cls_text="sm")
            s += arrow([(x0 + 130, y + 30), (x0 + 130, y + 40)])
            y += 42
        s += add(x0 + 130, y + 8)
        s += text(x0 + 130, y + 36, note, "xs")
        return s

    o += col(60, "@ out  (SE-ResNet, ECA, ResNet-RS)", [("1×1 reduce", "conv"), ("3×3 mixer", "mix"), ("1×1 expand", "conv"), ("SE / ECA", "attn")], "gates the full-width residual branch")
    o += col(500, "@ mixer  (RegNetY / V / Z)", [("1×1", "conv"), ("3×3 grouped mixer", "mix"), ("SE (width ∝ unit input)", "attn"), ("1×1", "conv")], "gates the bottleneck, cheaper; mixed by the last 1×1")
    return svg(W, H, o, "Where the channel attention sits")


# ---------------------------------------------------------------- mixers

def res2():
    W, H = 960, 380
    o = ""
    o += box(380, 10, 200, 34, "1×1 reduce → s·w channels", "conv", None, cls_text="sm")
    o += arrow([(480, 44), (480, 62)])
    o += box(200, 64, 560, 26, "split into s = 4 chunks of w", "io", None, cls_text="sm")
    xs = [260, 400, 540, 680]
    for i, x in enumerate(xs):
        o += arrow([(x, 90), (x, 112)])
        o += text(x + 14, 102, f"x{i+1}", "mt", "start")
    # chunk 1: identity  (timm: last chunk passes; paper: first). Draw paper order.
    o += box(xs[0] - 40, 114, 80, 30, "identity", "io", None, cls_text="sm")
    o += box(xs[1] - 40, 114, 80, 30, "K₂ 3×3", "mix", None, cls_text="sm")
    o += add(xs[2], 180)
    o += box(xs[2] - 40, 200, 80, 30, "K₃ 3×3", "mix", None, cls_text="sm")
    o += add(xs[3], 260)
    o += box(xs[3] - 40, 280, 80, 30, "K₄ 3×3", "mix", None, cls_text="sm")
    o += arrow([(xs[2], 112), (xs[2], 168)])
    o += arrow([(xs[3], 112), (xs[3], 248)])
    o += arrow([(xs[1] + 40, 129), (xs[2] - 11, 180)], hi=True)
    o += arrow([(xs[2], 192), (xs[2], 198)])
    o += arrow([(xs[2] + 40, 215), (xs[3] - 11, 260)], hi=True)
    o += arrow([(xs[3], 272), (xs[3], 278)])
    o += box(200, 336, 560, 26, "concat → s·w  → 1×1 expand", "conv", None, cls_text="sm")
    for x, y in [(xs[0], 144), (xs[1], 144), (xs[2], 230), (xs[3], 310)]:
        o += arrow([(x, y), (x, 334)])
    o += text(110, 120, "y₁ = x₁", "mt")
    o += text(110, 142, "y₂ = K₂(x₂)", "mt")
    o += text(110, 215, "y₃ = K₃(x₃ + y₂)", "mt")
    o += text(110, 295, "y₄ = K₄(x₄ + y₃)", "mt")
    o += text(860, 120, "y₁: reach 1", "xs") + text(860, 142, "y₂: reach 3", "xs") + text(860, 215, "y₃: reach 5", "xs") + text(860, 295, "y₄: reach 7", "xs")
    return svg(W, H, o, "Res2Net's hierarchical scale split")


def res2_rf():
    W, H = 900, 180
    o = ""
    cx = 160
    for i, r in enumerate([1, 3, 5, 7]):
        s = r * 18
        o += f'<rect x="{cx - s/2}" y="{90 - s/2}" width="{s}" height="{s}" class="b {["io","mix","mix","mix"][i]}" style="fill-opacity:{0.9 - i*0.2}"/>'
    o += text(cx, 170, "one unit, four receptive fields", "xs")
    o += text(560, 50, "the branches form a chain of length s−1 inside one block:", "sm")
    o += text(560, 74, "a Res2 mixer is a small residual-style network on the channel axis", "sm")
    o += text(560, 110, "reach of y_i = 2(i−1) + 1   for 3×3 branches", "mt")
    o += text(560, 140, "the concat holds every scale at once; the 1×1 after it mixes them", "xs")
    return svg(W, H, o, "Receptive fields inside a Res2 mixer")


def splat():
    W, H = 960, 330
    o = ""
    o += box(20, 140, 80, 44, "x", "io", "C_in")
    o += arrow([(100, 162), (128, 162)])
    o += box(130, 130, 130, 64, "grouped conv", "mix", "g·r groups, → r·C")
    o += text(195, 210, "+ norm + act", "xs")
    for i, y in enumerate([60, 162, 264]):
        o += arrow([(260, 162), (300, y)])
        o += box(302, y - 18, 70, 36, f"U{i+1}", "conv", "C")
    o += text(337, 300, "r = 3 splits", "xs")
    o += add(420, 162)
    for y in [60, 162, 264]:
        o += arrow([(372, y), (409, 162 + (y - 162) * 0.05)])
    o += arrow([(431, 162), (452, 162)])
    o += box(454, 140, 70, 44, "GAP", "pool", "C")
    o += arrow([(524, 162), (544, 162)])
    o += box(546, 136, 104, 52, "fc₁ + BN + act", "conv", "C → d")
    o += arrow([(650, 162), (670, 162)])
    o += box(672, 136, 84, 52, "fc₂", "conv", "d → r·C")
    o += arrow([(756, 162), (776, 162)])
    o += box(778, 136, 80, 52, "r-softmax", "attn", "per channel")
    o += arrow([(818, 136), (818, 100)])
    o += box(770, 50, 100, 48, "Σ aᵢ ⊙ Uᵢ", "io", "C out")
    o += arrow([(372, 52), (390, 52), (390, 30), (820, 30), (820, 48)], dash=True)
    o += text(560, 22, "U₁ … U_r, weighed", "xs")
    o += text(480, 316, "the fuse (sum) and select (softmax) are SK's; ResNeSt adds cardinal groups and makes radix the knob", "xs")
    return svg(W, H, o, "ResNeSt split attention")


def card_radix():
    W, H = 900, 200
    o = ""
    o += text(450, 18, "the grouped conv's g·r groups: cardinal groups × radix splits", "ttl")
    g, r = 2, 3
    for k in range(g):
        x0 = 80 + k * 400
        o += f'<rect x="{x0-10}" y="36" width="{r*110+10}" height="110" rx="8" class="slot"/>'
        o += text(x0 + r * 55 - 5, 160, f"cardinal group {k+1}: softmax across its {r} splits", "xs")
        for i in range(r):
            o += box(x0 + i * 110, 56, 90, 70, f"split {i+1}", "mix", f"C/g channels")
    o += text(450, 188, "radix 1 degenerates to a sigmoid gate: SE on the mixer's output", "xs")
    return svg(W, H, o, "Cardinality and radix")


def sk():
    W, H = 960, 300
    o = ""
    o += text(150, 18, "split", "ttl") + text(500, 18, "fuse", "ttl") + text(800, 18, "select", "ttl")
    o += box(20, 120, 70, 44, "x", "io")
    o += arrow([(90, 142), (120, 80)])
    o += arrow([(90, 142), (120, 210)])
    o += box(122, 56, 150, 48, "3×3, d = 1", "mix", "reach 3")
    o += box(122, 186, 150, 48, "3×3, d = 2", "mix", "reach 5  (≈ 5×5)")
    o += add(340, 142)
    o += arrow([(272, 80), (331, 134)])
    o += arrow([(272, 210), (331, 150)])
    o += arrow([(351, 142), (380, 142)])
    o += box(382, 120, 70, 44, "GAP", "pool")
    o += arrow([(452, 142), (476, 142)])
    o += box(478, 116, 120, 52, "reduce", "conv", "1×1 + BN + act → d")
    o += arrow([(598, 142), (622, 142)])
    o += box(624, 116, 100, 52, "select", "conv", "1×1 → 2·C")
    o += arrow([(724, 142), (746, 142)])
    o += box(748, 116, 90, 52, "softmax", "attn", "over paths")
    o += arrow([(838, 142), (862, 142)])
    o += box(864, 116, 86, 52, "a·U₁ + b·U₂", "io", "a + b = 1")
    o += arrow([(272, 70), (906, 70), (906, 114)], dash=True)
    o += arrow([(272, 222), (906, 222), (906, 170)], dash=True)
    o += text(480, 270, "per channel, the net interpolates between two receptive fields: an input-dependent kernel size", "xs")
    return svg(W, H, o, "Selective kernel")


def sk_rf():
    W, H = 900, 200
    o = ""
    for j, (d, x0) in enumerate([(1, 60), (2, 260)]):
        for r in range(5):
            for c in range(5):
                on = (r - 2) % d == 0 and (c - 2) % d == 0 and abs(r - 2) <= d and abs(c - 2) <= d
                o += rect(x0 + c * 22, 30 + r * 22, 20, 20, ("on2" if j == 0 else "on") if on else "cell", 2)
        o += text(x0 + 55, 160, f"path {j+1}: d = {d}", "xs")
    o += text(470, 60, "effective kernel per channel c:", "sm", "start")
    o += text(470, 90, "K_c = a_c K₁ + (1 − a_c) K₂", "mt", "start")
    o += text(470, 120, "a_c moves along the segment Δ¹;", "xs", "start")
    o += text(470, 136, "at a_c = 1 the unit is a plain 3×3 ResNet mixer", "xs", "start")
    return svg(W, H, o, "Selective kernel receptive fields")


# ---------------------------------------------------------------- stems & sampling

def stems():
    W, H = 960, 330
    o = ""
    cols = [
        ("single (v1)", [("7×7, 64, s2", "conv")], "max 3×3 s2"),
        ("deep (C / D / S)", [("3×3, 32, s2", "conv"), ("3×3, 32", "conv"), ("3×3, 64", "conv")], "max 3×3 s2"),
        ("tiered (T)", [("3×3, 24, s2", "conv"), ("3×3, 32", "conv"), ("3×3, 64", "conv")], "max 3×3 s2"),
        ("RegNet", [("3×3, 32, s2", "conv")], "no pool"),
        ("ResNet-RS", [("3×3, 32, s2", "conv"), ("3×3, 32", "conv"), ("3×3, 64", "conv")], "3×3 conv s2"),
    ]
    for i, (t, items, pool) in enumerate(cols):
        x = 14 + i * 190
        o += text(x + 85, 18, t, "ttl")
        y = 36
        for lab, k in items:
            o += box(x + 10, y, 150, 34, lab, k, cls_text="sm")
            o += arrow([(x + 85, y + 34), (x + 85, y + 46)])
            y += 48
        o += box(x + 10, 190, 150, 34, pool, "conv" if "conv" in pool else ("io" if pool == "no pool" else "pool"), cls_text="sm")
        o += text(x + 85, 244, "÷4" if pool != "no pool" else "÷2", "sm")
    o += text(480, 280, "pool variants: max · avg (AA) · max s1 → blur s2 (blur AA) · zero-pad → max (BiT) · SAME-pad max · conv", "xs")
    o += text(480, 300, "widths shown for a 64-wide stem output (deep stem width 32)", "xs")
    return svg(W, H, o, "Stem variants")


def alias():
    W, H = 900, 260
    o = ""
    o += rect(10, 10, 430, 240, "bg2", 8)
    o += rect(460, 10, 430, 240, "bg2", 8)
    o += text(225, 30, "subsample a high-frequency signal…", "ttl")
    o += text(675, 30, "…blur first, then subsample", "ttl")

    def sig(x0, f, blur):
        pts = []
        for i in range(161):
            t = i / 160
            v = math.sin(2 * math.pi * f * t)
            if blur:
                v *= 0.25
            pts.append((x0 + 20 + t * 380, 130 - 60 * v))
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        return f'<path d="{d}" class="curve"/>'

    o += sig(10, 7.5, False)
    o += sig(460, 7.5, True)
    # samples every ~ 1/8: alias to slow wave
    sp = []
    for k in range(9):
        t = k / 8
        v = math.sin(2 * math.pi * 7.5 * t)
        sp.append((30 + t * 380, 130 - 60 * v))
        o += f'<circle cx="{sp[-1][0]:.1f}" cy="{sp[-1][1]:.1f}" r="5" class="on"/>'
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in sp)
    o += f'<path d="{d}" class="curve2" stroke-dasharray="5 4"/>'
    o += text(225, 220, "samples trace a slow false wave: aliasing;", "xs")
    o += text(225, 234, "a one-pixel shift changes the output", "xs")
    for k in range(9):
        t = k / 8
        v = 0.25 * math.sin(2 * math.pi * 7.5 * t)
        o += f'<circle cx="{480 + t*380:.1f}" cy="{130 - 60*v:.1f}" r="5" class="on"/>'
    o += text(675, 220, "high frequencies are attenuated before", "xs")
    o += text(675, 234, "the stride: shift-equivariance mostly survives", "xs")
    return svg(W, H, o, "Aliasing under stride")


def binomial():
    W, H = 900, 200
    o = ""
    taps = [1, 2, 1]
    for r in range(3):
        for c in range(3):
            v = taps[r] * taps[c]
            o += rect(60 + c * 46, 40 + r * 46, 44, 44, "on2", 4, f'style="fill-opacity:{0.15 + v/4*0.7}"')
            o += text(82 + c * 46, 62 + r * 46, f"{v}", "lbl")
    o += text(129, 190, "÷16 : [1 2 1]ᵀ[1 2 1] / 16", "xs")
    o += text(470, 50, "BlurPool(stride s) = subsample_s ∘ (binomial blur)", "mt", "start")
    o += text(470, 82, "taps C(n−1, k) / 2ⁿ⁻¹ — a discrete Gaussian; they sum to 1", "xs", "start")
    o += text(470, 104, "no parameters; reflect padding", "xs", "start")
    o += text(470, 136, "placement in a unit:  conv(s=1) → norm → act → BlurPool(s)", "xs", "start")
    o += text(470, 152, "the stride moves off the conv onto the low-pass", "xs", "start")
    return svg(W, H, o, "Binomial blur kernel")


def aa_place():
    W, H = 900, 270
    o = ""
    cols = [
        ("plain", [("3×3 mixer, s2", "mix")]),
        ("anti-aliased", [("3×3 mixer, s1", "mix"), ("blur / avg, s2", "pool")]),
        ("ResNeSt avd", [("split-attn, s1", "mix"), ("3×3 avg, s2", "pool")]),
        ("ResNeSt avd-first", [("3×3 avg, s2", "pool"), ("split-attn, s1", "mix")]),
    ]
    for i, (t, items) in enumerate(cols):
        x = 20 + i * 220
        o += text(x + 90, 18, t, "ttl")
        o += box(x + 20, 34, 140, 28, "1×1 reduce", "conv", cls_text="sm")
        y = 74
        o += arrow([(x + 90, 62), (x + 90, 72)])
        for lab, k in items:
            o += box(x + 20, y, 140, 30, lab, k, cls_text="sm")
            o += arrow([(x + 90, y + 30), (x + 90, y + 40)])
            y += 42
        o += box(x + 20, y, 140, 28, "1×1 expand", "conv", cls_text="sm")
    o += text(450, 250, "same total stride, different order of low-pass and spatial mixing", "xs")
    return svg(W, H, o, "Placement of anti-aliased downsampling")


# ---------------------------------------------------------------- regularizers

def drop_path():
    W, H = 900, 220
    o = ""
    o += text(225, 18, "stochastic depth: per sample, per unit", "ttl")
    for i in range(4):
        y = 40 + i * 40
        o += text(40, y + 12, f"sample {i+1}", "xs")
        for j in range(5):
            x = 90 + j * 66
            keep = (i * 7 + j * 3) % 4 != 0
            o += rect(x, y, 56, 24, "on2" if keep else "off", 4)
    o += text(225, 210, "a dropped unit is the identity; survivors scale by 1/(1−p)", "xs")
    o += text(675, 18, "drop rate ramps linearly with depth", "ttl")
    o += f'<line x1="500" y1="180" x2="860" y2="180" class="line"/><line x1="500" y1="180" x2="500" y2="40" class="line"/>'
    o += f'<line x1="500" y1="180" x2="860" y2="60" class="curve2"/>'
    o += text(680, 200, "unit index (whole net)", "xs")
    o += text(490, 60, "p_max", "xs", "end")
    o += text(490, 180, "0", "xs", "end")
    return svg(W, H, o, "Stochastic depth")


def drop_block():
    W, H = 900, 250
    o = ""

    def g(x0, title, f):
        s = text(x0 + 100, 18, title, "ttl")
        for r in range(10):
            for c in range(10):
                s += rect(x0 + c * 18, 30 + r * 18, 17, 17, "off" if f(r, c) else "on2")
        return s

    import random
    rnd = random.Random(7)
    drops = {(r, c) for r in range(10) for c in range(10) if rnd.random() < 0.2}
    o += g(80, "dropout: independent pixels", lambda r, c: (r, c) in drops)
    blocks = [(1, 2), (5, 6), (6, 1)]
    o += g(520, "DropBlock: contiguous blocks", lambda r, c: any(br <= r < br + 3 and bc <= c < bc + 3 for br, bc in blocks))
    o += text(450, 232, "neighbouring activations are correlated: an independent hole is filled in from its neighbours, a block is not", "xs")
    return svg(W, H, o, "Dropout versus DropBlock")


# ---------------------------------------------------------------- head

def head():
    W, H = 960, 220
    o = ""
    o += box(10, 70, 100, 50, "stage 4", "mix", "C × 7 × 7")
    o += arrow([(110, 95), (138, 95)])
    o += box(140, 66, 140, 58, "final  (optional)", "act", "act · norm+act · 1×1 conv")
    o += arrow([(280, 95), (308, 95)])
    o += box(310, 66, 140, 58, "global pool", "pool", "avg · max · avgmax · cat")
    o += arrow([(450, 95), (478, 95)])
    o += box(480, 66, 110, 58, "dropout", "reg", "train only")
    o += arrow([(590, 95), (618, 95)])
    o += box(620, 66, 120, 58, "fc", "conv", "C (or 2C) → classes")
    o += arrow([(740, 95), (768, 95)])
    o += box(770, 70, 100, 50, "logits", "io")
    o += text(480, 170, "global pooling collapses H×W: the classifier is translation-invariant and resolution-agnostic", "xs")
    o += text(480, 186, "headless (0 classes): stop after the pool, the pooled features are the output", "xs")
    return svg(W, H, o, "The classifier head")


# ---------------------------------------------------------------- family specific

def resnext_equiv():
    W, H = 960, 300
    o = ""
    o += text(240, 18, "split-transform-merge: 32 paths", "ttl")
    for i in range(6):
        x = 30 + i * 74
        if i == 4:
            o += text(x + 30, 130, "…", "ttl")
            continue
        o += box(x, 40, 60, 30, "1×1, 4", "conv", cls_text="xs")
        o += box(x, 110, 60, 30, "3×3, 4", "mix", cls_text="xs")
        o += box(x, 180, 60, 30, "1×1, 256", "conv", cls_text="xs")
        o += arrow([(x + 30, 70), (x + 30, 108)])
        o += arrow([(x + 30, 140), (x + 30, 178)])
        o += arrow([(x + 30, 210), (240, 250)])
    o += add(240, 260)
    o += text(500, 150, "≡", "ttl")
    o += text(720, 18, "one grouped bottleneck", "ttl")
    o += box(620, 40, 200, 34, "1×1, 128", "conv", cls_text="sm")
    o += arrow([(720, 74), (720, 106)])
    o += box(620, 108, 200, 34, "3×3, 128, groups = 32", "mix", cls_text="sm")
    o += arrow([(720, 142), (720, 174)])
    o += box(620, 176, 200, 34, "1×1, 256", "conv", cls_text="sm")
    o += text(720, 240, "concatenating the 32 reduces = one 1×1 to 128;", "xs")
    o += text(720, 254, "the independent 3×3s = a grouped conv;", "xs")
    o += text(720, 268, "summing the 32 expands = one 1×1 from 128", "xs")
    return svg(W, H, o, "ResNeXt aggregated transforms equal a grouped bottleneck")


def regnet_widths(cfg):
    """RegNet's linear width rule quantized to a geometric staircase."""
    depth, w0, wa, wm = cfg
    W, H = 900, 320
    o = ""
    xs0, ys0, xw, yh = 70, 270, 780, 220
    us = [w0 + wa * j for j in range(depth)]
    ss = [round(math.log(u / w0) / math.log(wm)) for u in us]
    ws = [int(round(w0 * wm ** s / 8) * 8) for s in ss]
    top = max(max(us), max(ws)) * 1.08

    def px(j):
        return xs0 + j / (depth - 1) * xw

    def py(v):
        return ys0 - v / top * yh

    o += f'<line x1="{xs0}" y1="{ys0}" x2="{xs0+xw}" y2="{ys0}" class="line"/><line x1="{xs0}" y1="{ys0}" x2="{xs0}" y2="{ys0-yh}" class="line"/>'
    for v in range(0, int(top), 200 if top > 800 else 100):
        o += f'<line x1="{xs0}" y1="{py(v)}" x2="{xs0+xw}" y2="{py(v)}" class="grid"/>' + text(xs0 - 6, py(v), str(v), "xs", "end")
    o += f'<path d="M{px(0)},{py(us[0])} L{px(depth-1)},{py(us[-1])}" class="curve" stroke-dasharray="6 4"/>'
    d = ""
    for j, w in enumerate(ws):
        x1 = px(j) - xw / (depth - 1) / 2
        x2 = px(j) + xw / (depth - 1) / 2
        d += f"M{max(x1, xs0)},{py(w)} L{min(x2, xs0+xw)},{py(w)} "
    o += f'<path d="{d}" class="curve2"/>'
    for j, w in enumerate(ws):
        o += f'<circle cx="{px(j)}" cy="{py(w)}" r="3" class="dot"/>'
    # stage labels
    stages = []
    for j, w in enumerate(ws):
        if not stages or stages[-1][0] != w:
            stages.append([w, j, j])
        else:
            stages[-1][2] = j
    for i, (w, a, b) in enumerate(stages):
        o += text(max((px(a) + px(b)) / 2, xs0 + 44), py(w) - 14, f"stage {i+1}: {b-a+1}× {w}", "xs")
    o += text(xs0 + xw / 2, ys0 + 22, "unit index j", "xs")
    o += text(xs0 + 10, 34, f"u_j = {w0} + {wa}·j  (dashed)", "mt", "start")
    o += text(xs0 + 10, 56, f"w_j = {w0} · {wm}^round(log(u_j/{w0}) / log {wm})  (solid, then ÷8 quantized)", "mt", "start")
    return svg(W, H, o, "RegNet width rule"), stages


def regnet_units():
    W, H = 960, 330
    o = ""
    cols = [
        ("X", [("1×1", "conv"), ("3×3 grouped, s", "mix"), ("1×1", "conv")], "post-act; conv shortcut"),
        ("Y", [("1×1", "conv"), ("3×3 grouped, s", "mix"), ("SE (¼ of unit input)", "attn"), ("1×1", "conv")], "X + SE at the mixer"),
        ("V", [("norm → act", "norm"), ("1×1", "conv"), ("3×3 grouped, s", "mix"), ("SE", "attn"), ("1×1", "conv")], "pre-activated Y; SiLU"),
        ("Z", [("1×1 expand ×4", "conv"), ("3×3 grouped, s", "mix"), ("SE", "attn"), ("1×1 project", "conv")], "inverted, linear out, no shortcut conv"),
    ]
    for i, (t, items, note) in enumerate(cols):
        x = 20 + i * 235
        o += text(x + 100, 18, f"RegNet{t}", "ttl")
        y = 36
        for lab, k in items:
            o += box(x + 15, y, 170, 30, lab, k, cls_text="sm")
            o += arrow([(x + 100, y + 30), (x + 100, y + 40)])
            y += 42
        o += add(x + 100, y + 8)
        o += text(x + 100, y + 36, note, "xs")
    return svg(W, H, o, "RegNet unit variants")


def preact_shortcut():
    W, H = 900, 250
    o = ""

    def col(x0, title, identity_raw, note):
        s = text(x0 + 180, 18, title, "ttl")
        s += box(x0 + 40, 36, 90, 28, "x", "io")
        s += arrow([(x0 + 85, 64), (x0 + 85, 80)])
        s += box(x0 + 20, 82, 130, 30, "pre: norm → act", "norm", cls_text="sm")
        s += arrow([(x0 + 85, 112), (x0 + 85, 128)])
        s += box(x0 + 20, 130, 130, 30, "path …", "mix", cls_text="sm")
        s += arrow([(x0 + 85, 160), (x0 + 85, 186)])
        s += add(x0 + 85, 198)
        if identity_raw:
            s += arrow([(x0 + 130, 50), (x0 + 260, 50), (x0 + 260, 198), (x0 + 97, 198)], hi=True)
            s += text(x0 + 268, 120, "raw x", "xs", "start")
        else:
            s += arrow([(x0 + 150, 120), (x0 + 260, 120), (x0 + 260, 198), (x0 + 97, 198)], hi=True)
            s += text(x0 + 268, 150, "pre(x)", "xs", "start")
        s += text(x0 + 160, 234, note, "xs")
        return s

    o += col(20, "ResNetV2 / BiT identity", True, "identity shortcut reads the raw stream")
    o += col(470, "RegNetV identity", False, "identity shortcut reads the pre-activated input")
    return svg(W, H, o, "Pre-activation shortcut sources")


def ws_fig():
    W, H = 900, 200
    o = ""
    o += text(225, 18, "a filter's weights, before", "ttl") + text(675, 18, "after weight standardization", "ttl")
    import random
    rnd = random.Random(11)
    vals = [rnd.gauss(0.6, 1.7) for _ in range(40)]
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    for i, v in enumerate(vals):
        o += rect(30 + i * 9.5, 110 - max(v, 0) * 12, 7, abs(v) * 12, "on2" if v >= 0 else "on")
        z = (v - mu) / sd
        o += rect(480 + i * 9.5, 110 - max(z, 0) * 18, 7, abs(z) * 18, "on2" if z >= 0 else "on")
    o += f'<line x1="20" y1="110" x2="420" y2="110" class="line"/><line x1="470" y1="110" x2="870" y2="110" class="line"/>'
    o += text(450, 186, "Ŵ = (W − mean(W)) / std(W), per output filter: the conv's gain is fixed, so GN can replace BN at batch size 1", "xs")
    return svg(W, H, o, "Weight standardization")


def pools():
    W, H = 900, 160
    o = ""
    vals = [[1, 2], [3, 6]]
    kinds = [("avg", "3", "mean"), ("max", "6", "max"), ("avgmax", "4.5", "(mean + max)/2"), ("catavgmax", "[3, 6]", "concat: 2C")]
    for i, (k, out, d) in enumerate(kinds):
        x = 30 + i * 220
        o += text(x + 80, 16, k, "ttl")
        for r in range(2):
            for c in range(2):
                o += rect(x + c * 34, 32 + r * 34, 32, 32, "on2", 3, f'style="fill-opacity:{0.2 + vals[r][c]/8}"')
                o += text(x + 16 + c * 34, 48 + r * 34, vals[r][c], "sm")
        o += arrow([(x + 74, 66), (x + 104, 66)])
        o += box(x + 106, 48, 64, 36, out, "pool", cls_text="sm")
        o += text(x + 80, 128, d, "xs")
    return svg(W, H, o, "Global pool kinds")


# ---------------------------------------------------------------- family circuits
# A fixed "board": the macro chassis on top, the unit's sockets below. A variant
# is a spec mapping each socket to a component (title, [params], link?, kind?)
# or None (an empty socket: the wire passes through).

SLOT_KIND = dict(stem="conv", stages="mix", final="act", pool="pool", head="io", pre="norm", conv0="conv",
                 mixer="mix", aa="pool", attn_mix="attn", conv2="conv", attn_out="attn", shortcut="pool",
                 out_act="act")
SLOT_LINK = dict(stem="c-sampling.html#stems", stages="algebra.html#macro", final="c-head.html#final",
                 pool="c-head.html#pool", head="c-head.html#fc", pre="c-residual.html#preact", conv0="c-conv.html",
                 mixer="c-conv.html", aa="c-sampling.html#aa", attn_mix="c-attention.html#place", conv2="c-conv.html",
                 attn_out="c-attention.html", shortcut="c-shortcut.html", out_act="c-residual.html#preact")
SLOT_NAME = dict(stem="STEM", stages="STAGES", final="FINAL", pool="POOL", head="HEAD", pre="PRE", conv0="CONV₀",
                 mixer="MIXER", aa="ANTI-ALIAS", attn_mix="ATTN @ MIXER", conv2="CONV₂", attn_out="ATTN @ OUT",
                 shortcut="SHORTCUT", out_act="OUT ACT")
PATH_W = dict(pre=84, conv0=88, mixer=104, aa=84, attn_mix=88, conv2=88, attn_out=88)
PATH_ORDER = ["pre", "conv0", "mixer", "aa", "attn_mix", "conv2", "attn_out"]


def chip(x, y, w, h, slot, val, changed):
    o = text(x + 2, y - 8, SLOT_NAME[slot], "caps", "start")
    if val is None:
        o += rect(x, y, w, h, "slot", 6)
        o += text(x + w / 2, y + h - 11, "empty", "xs")
    else:
        title, params = val[0], val[1]
        link = val[2] if len(val) > 2 and val[2] else SLOT_LINK[slot]
        kind = val[3] if len(val) > 3 else SLOT_KIND[slot]
        inner = rect(x, y, w, h, f"b {kind}", 6) + text(x + w / 2, y + 16, title, "ct")
        for i, p in enumerate(params):
            inner += text(x + w / 2, y + 32 + i * 13, p, "xs")
        o += f'<a href="{link}"><title>{esc(SLOT_NAME[slot].title())}: {esc(title)}</title>{inner}</a>'
    if changed:
        o += rect(x - 3, y - 3, w + 6, h + 6, "chg", 8)
        o += text(x + w + 1, y - 8, "Δ", "chgt", "end")
    return o


def circuit(spec, changed):
    W, H = 960, 404
    o = ""
    # ---- macro row
    my, mh = 26, 64
    wy = my + mh / 2
    o += f'<path d="M14,{wy} L944,{wy}" class="wire"/>'
    o += text(14, wy - 10, "image", "xs", "start")
    macro = [("stem", 60, 190), ("stages", 276, 240), ("final", 540, 108), ("pool", 670, 96), ("head", 788, 104)]
    for slot, x, w in macro:
        o += chip(x, my, w, mh, slot, spec.get(slot), slot in changed)
    o += text(944, wy - 10, "logits", "xs", "end")
    # zoom from the stages chip down to the unit board
    o += f'<path d="M276,{my+mh} L20,128" class="ar dash"/><path d="M516,{my+mh} L940,128" class="ar dash"/>'
    # ---- unit board
    o += rect(20, 128, 920, 258, "board", 10)
    o += text(36, 146, "RESIDUAL UNIT  ×  d  per stage  (the stage's first unit strides and reshapes)", "caps", "start")
    lay = spec.get("layers", "")
    lw = len(lay) * 6.4 + 70
    o += rect(924 - lw, 136, lw, 20, "pill", 10) + text(924 - lw / 2, 146, f"layers: {lay}", "xs")
    if "layers" in changed:
        o += rect(921 - lw, 133, lw + 6, 26, "chg", 12)
    y, h = 192, 66
    sy = y + h / 2
    order = spec.get("order", PATH_ORDER)
    x = 58
    pos = {}
    for slot in order:
        pos[slot] = (x, PATH_W[slot])
        x += PATH_W[slot] + 13
    plus_x = x + 12
    tap = spec.get("tap", "pre")
    nx = 44 if tap == "raw" else pos["pre"][0] + pos["pre"][1] + 6
    o += f'<path d="M28,{sy} L{plus_x-11},{sy}" class="wire"/>'
    o += text(30, sy - 10, "x", "mt", "start")
    for slot in order:
        px, pw = pos[slot]
        o += chip(px, y, pw, h, slot, spec.get(slot), slot in changed)
    # shortcut
    sx, sw, shy, shh = 330, 176, 300, 62
    smy = shy + shh / 2
    residual = spec.get("residual", True)
    o += f'<circle cx="{nx}" cy="{sy}" r="3.5" class="pin"/>'
    o += f'<path d="M{nx},{sy} L{nx},{smy} L{sx},{smy}" class="wire"/>'
    o += f'<path d="M{sx+sw},{smy} L{plus_x},{smy} L{plus_x},{sy+12}" class="wire"/>'
    o += chip(sx, shy, sw, shh, "shortcut", spec.get("shortcut"), "shortcut" in changed or "tap" in changed)
    tap_note = "identity reads raw x" if tap == "raw" else "identity reads the unit's input" if spec.get("pre") is None else "identity reads pre(x)"
    o += text(sx + sw + 10, smy - 8, tap_note, "xs", "start")
    o += text(sx + sw + 10, smy + 6, "identity: later units", "xs", "start")
    o += add(plus_x, sy)
    # output activation
    ox = plus_x + 20
    o += f'<path d="M{plus_x+11},{sy} L{ox},{sy}" class="wire"/>'
    o += chip(ox, y, 80, h, "out_act", spec.get("out_act"), "out_act" in changed)
    o += f'<path d="M{ox},{sy} L{ox+80},{sy}" class="wire"/>' if spec.get("out_act") is None else ""
    o += f'<path d="M{ox+80},{sy} L934,{sy}" class="wire"/>' + text(932, sy - 10, "y", "mt", "end")
    o += text(36, 376, "drop path (train) sits before the add; drop block after the mixer's norm" +
              ("" if residual else "  ·  units that reshape have no residual: the add is skipped"), "xs", "start")
    return svg(W, H, o, f"Circuit of {spec.get('name', 'a variant')}")
