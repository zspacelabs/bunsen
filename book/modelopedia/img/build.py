"""Build the book: wrap each page fragment in the shared shell.

Fragments live in pages/<slug>.html. Placeholders:
  {{fig:name}}           an SVG from figs.py
  {{cite:key[,key]}}     a numbered citation (per page), linked to references.html
  {{tree}} {{matrix}} {{refs}}  generated blocks
  {{regnet:depth,w0,wa,wm}}     RegNet width-rule figure
Run:  python3 build.py   (writes ../*.html)
"""
import html
import re
from pathlib import Path

from . import circuits
from . import figs
from . import figs_tx
from .refs import REFS

HERE = Path(__file__).parent
OUT = HERE.parent

PARTS = [
    ("Orientation", [
        ("index", "The Residual Lineage", "Overview", "Residual image models, from ResNet to Swin, read as one modular algebra."),
        ("algebra", "The Shared Algebra", "Orientation", "One model shape, a handful of slots, and the combinators that fill them."),
        ("tx-algebra", "From Units to Blocks", "Orientation", "The vision encoder: the same residual algebra, with tokens instead of feature maps."),
        ("names", "Reading the Names", "Orientation", "Model names are compressed policy: how to decode them into slot choices."),
    ]),
    ("Conv combinators", [
        ("c-conv", "Convolution Blocks", "Combinator", "The atom of every family: conv, norm, activation; groups, dilation and stride."),
        ("c-norm", "Normalization", "Combinator", "Batch, group and EvoNorm; weight standardization."),
        ("c-residual", "The Residual Unit", "Combinator", "y = act(shortcut(x) + F(x)): why it trains, and what it computes."),
        ("c-units", "Unit Kinds", "Combinator", "Basic, bottleneck, grouped, pre-activated and inverted conv paths."),
        ("c-shortcut", "Shortcuts", "Combinator", "Identity, projection, average-down, and none."),
        ("c-attention", "Channel Attention", "Combinator", "Squeeze-and-excitation and ECA: input-conditioned channel gains."),
        ("c-res2", "Scale Split (Res2)", "Spatial mixer", "A hierarchy of small convs inside one mixer: many receptive fields at once."),
        ("c-splitattn", "Split Attention", "Spatial mixer", "Radix splits weighed per channel by a softmax over their sum."),
        ("c-sk", "Selective Kernel", "Spatial mixer", "Two receptive fields, selected per channel."),
        ("c-sampling", "Stems & Downsampling", "Combinator", "Stems, stem pools, and anti-aliased strides."),
        ("c-drop", "Structured Regularizers", "Combinator", "Stochastic depth and DropBlock: train-time edits of the algebra."),
        ("c-head", "The Head", "Combinator", "Final layer, global pool, classifier."),
    ]),
    ("Conv families", [
        ("f-resnet", "ResNet", "Family", "The root: residual stages of basic or bottleneck units, and its refinements."),
        ("f-resnext", "ResNeXt & Wide ResNet", "Family", "Widening the mixer: by cardinality, or by brute width."),
        ("f-resnetv2", "ResNetV2 & BiT", "Family", "Pre-activation, then group norm and weight standardization for transfer."),
        ("f-res2net", "Res2Net", "Family", "A ResNet whose mixer is a scale hierarchy."),
        ("f-sknet", "SKNet", "Family", "A ResNet whose mixer selects its kernel size."),
        ("f-resnest", "ResNeSt", "Family", "A ResNeXt whose mixer attends over radix splits."),
        ("f-regnet", "RegNet", "Family", "A design space, not a design: widths from a quantized linear rule."),
    ]),
    ("Transformer combinators", [
        ("t-patch", "Patches & Tokens", "Combinator", "Patch embedding, prefix tokens, and the learnt position embedding."),
        ("t-attention", "Self-Attention", "Token mixer", "Multi-head attention: a global, data-dependent mixer over a set of tokens."),
        ("t-mlp", "The MLP", "Channel mixer", "The per-token feed-forward: GELU MLPs and gated SwiGLUs."),
        ("t-sublayer", "The Residual Sublayer", "Combinator", "Norm slots, norm placement, layer scale: the transformer's residual unit."),
        ("t-window", "Window Attention", "Token mixer", "Shifted windows, relative position bias, cosine attention."),
        ("t-merge", "Patch Merging", "Combinator", "The transformer's strided conv: 2×2 tokens into one, the width doubled."),
        ("t-pool", "Token Pools & Head", "Combinator", "Class token, mean, attention pool, distilled heads; where the final norm goes."),
    ]),
    ("Transformer families", [
        ("f-vit", "ViT", "Family", "Patches as words: the isotropic vision transformer, and its hybrids."),
        ("f-deit", "DeiT & DeiT III", "Family", "Training ViTs on ImageNet alone: a distillation token, then layer scale."),
        ("f-clip", "CLIP, SigLIP, AIMv2 & BEiT3", "Family", "Image towers of multimodal models: what each changes in the encoder."),
        ("f-dino", "DINOv2 & Registers", "Family", "Self-supervised towers, layer scale, and register tokens."),
        ("f-swin", "Swin", "Family", "Windowed attention in a pyramid: V1, Swin-S3 and V2."),
    ]),
    ("Back matter", [
        ("references", "References", "Back matter", "The papers behind each family and combinator."),
    ]),
]
FLAT = [p for _, ps in PARTS for p in ps]
TITLE = {p[0]: p[1] for p in FLAT}


# --------------------------------------------------------------------- tree
# (name, year, delta, chips, href, children, flags)
def N(name, year, delta, href, chips=(), kids=(), flags=""):
    return dict(name=name, year=year, delta=delta, href=href, chips=list(chips), kids=list(kids), flags=flags)


CH = {
    "unit": ("conv", "unit kind", "c-units.html"),
    "mixer": ("mix", "mixer", "algebra.html#slots"),
    "res2": ("mix", "Res2 mixer", "c-res2.html"),
    "splat": ("mix", "split-attn mixer", "c-splitattn.html"),
    "sk": ("mix", "SK mixer", "c-sk.html"),
    "groups": ("conv", "groups", "c-conv.html#groups"),
    "attn": ("attn", "channel attention", "c-attention.html"),
    "short": ("pool", "shortcut", "c-shortcut.html"),
    "stem": ("conv", "stem", "c-sampling.html#stems"),
    "aa": ("pool", "anti-alias", "c-sampling.html#aa"),
    "norm": ("norm", "norm", "c-norm.html"),
    "ws": ("norm", "weight std", "c-norm.html#ws"),
    "preact": ("norm", "pre-activation", "c-residual.html#preact"),
    "res": ("io", "residual", "c-residual.html"),
    "width": ("io", "width rule", "f-regnet.html#widths"),
    "final": ("act", "final layer", "c-head.html"),
    "act": ("act", "activation", "c-conv.html#act"),
    "patch": ("conv", "patch embed", "t-patch.html"),
    "tokens": ("io", "prefix tokens", "t-patch.html#tokens"),
    "pos": ("io", "position embedding", "t-patch.html#pos"),
    "mhsa": ("attn", "self-attention", "t-attention.html"),
    "window": ("attn", "window attention", "t-window.html"),
    "merge": ("pool", "patch merge", "t-merge.html"),
    "mlp": ("act", "MLP", "t-mlp.html"),
    "placement": ("norm", "norm placement", "t-sublayer.html#placement"),
    "lnorm": ("norm", "norm kind", "t-sublayer.html#kinds"),
    "scale": ("norm", "layer scale", "t-sublayer.html#scale"),
    "tpool": ("pool", "token pool", "t-pool.html"),
    "backbone": ("conv", "conv backbone", "f-vit.html#hybrid"),
}

CONV_TREE = N("Residual conv nets", "", "every family below is one generic conv net with its slots filled", "algebra.html",
         ["res"], [
    N("Plain deep nets", "2014", "VGG's stacks of 3×3 convs; Highway gates", "c-residual.html#roots", flags="boundary"),
    N("ResNet (v1)", "2015", "identity shortcuts; basic & bottleneck units", "f-resnet.html", ["res", "unit"], [
        N("ResNet-B/C/D (bag of tricks)", "2018", "stride on the 3×3; deep stem; avg-down shortcut", "f-resnet.html#tricks",
          ["stem", "short"], [
            N("S / T / E stems", "2018–", "deep stems of other widths; tiered stem", "f-resnet.html#tricks", ["stem"]),
        ]),
        N("Wide ResNet", "2016", "wider mixer, fewer layers", "f-resnext.html#wide", ["unit"]),
        N("ResNeXt", "2016", "cardinality: a grouped mixer", "f-resnext.html", ["groups"], [
            N("SKNet", "2019", "mixer selects between two receptive fields", "f-sknet.html", ["sk"], flags="kit"),
            N("ResNeSt", "2020", "radix splits × cardinal groups, attended", "f-resnest.html", ["splat", "aa"], flags="kit"),
            N("RegNet", "2020", "design space: generated widths, grouped bottleneck", "f-regnet.html", ["width", "groups"], [
                N("RegNetX", "2020", "the base design space", "f-regnet.html#x", ["groups"]),
                N("RegNetY", "2020", "+ SE at the mixer", "f-regnet.html#y", ["attn"]),
                N("RegNetV", "timm", "pre-activated RegNetY", "f-regnet.html#v", ["preact"]),
                N("RegNetZ", "2021", "inverted bottleneck, linear out, no shortcut convs", "f-regnet.html#z",
                  ["unit", "short", "final"]),
            ], flags="kit"),
        ]),
        N("Res2Net", "2019", "mixer is a hierarchical scale split", "f-res2net.html", ["res2"], [
            N("Res2NeXt", "2019", "scale split with grouped branches", "f-res2net.html#res2next", ["groups"]),
        ], flags="kit"),
        N("SE-ResNet / SE-ResNeXt", "2017", "squeeze-excite gates the residual branch", "f-resnet.html#attn", ["attn"], [
            N("ECA-ResNet", "2019", "SE's MLP replaced by a 1-d conv across channels", "f-resnet.html#attn", ["attn"]),
        ]),
        N("Anti-aliased ResNet", "2019", "strides move onto a blur / average pool", "f-resnet.html#aa", ["aa"]),
        N("ResNet-RS", "2021", "SE ¼ + deep stem + avg-down + conv stem pool, rescaled", "f-resnet.html#rs",
          ["attn", "stem", "short"]),
        N("ResNetV2 (pre-activation)", "2016", "norm → act before each conv; clean identity", "f-resnetv2.html", ["preact"], [
            N("BiT", "2019", "group norm + weight standardization, widened ×1–×4", "f-resnetv2.html#bit", ["norm", "ws"]),
            N("EvoNorm ResNetV2", "2020", "norm and activation fused into EvoNorm-S0", "f-resnetv2.html#evo", ["norm"]),
            N("→ hybrid ViT backbone", "2020", "a post-activation ResNetV2 trunk feeding a transformer: see Hybrid ViT",
              "f-vit.html#hybrid", ["backbone"], flags="boundary"),
        ], flags="kit"),
    ], flags="kit"),
 ])

TX_TREE = N("Vision transformers", "", "every family below is one generic vision encoder with its slots filled",
            "tx-algebra.html", ["mhsa", "mlp"], [
    N("Transformer", "2017", "attention + MLP residual sublayers, post-norm; text, not images", "t-sublayer.html#placement",
      ["placement"], flags="boundary"),
    N("ViT", "2020", "patches as tokens; class token; learnt positions; pre-norm encoder", "f-vit.html",
      ["patch", "tokens", "mhsa", "mlp"], [
        N("Hybrid ViT (R+ViT)", "2020", "a ResNetV2 trunk makes the map the patches are cut from", "f-vit.html#hybrid",
          ["backbone"]),
        N("ResPost ViT", "timm", "norm after each branch; no class token", "f-vit.html#rpn", ["placement", "tpool"]),
        N("GAP ViTs (sbb)", "2024", "no class token; mean pool; layer scale", "f-vit.html#gap", ["tpool", "scale"]),
        N("FlexiViT", "2022", "positions over the patches alone", "f-vit.html#flexi", ["pos"]),
        N("DeiT", "2020", "ImageNet-only training; a distillation token and second head", "f-deit.html",
          ["tokens", "tpool"], [
            N("DeiT III", "2022", "layer scale; positions over the patches alone", "f-deit.html#deit3", ["scale", "pos"]),
        ], flags="kit"),
        N("CLIP image tower", "2021", "a norm before the first block; QuickGELU; no patch bias", "f-clip.html#clip",
          ["lnorm", "patch"]),
        N("SigLIP image tower", "2023", "no class token; attention-pool (map) or mean", "f-clip.html#siglip", ["tpool"]),
        N("EVA", "2022", "a mean-pooled ViT-L", "f-clip.html#eva", ["tpool"]),
        N("BEiT3", "2022", "sub-LN: norms inside the attention and the MLP", "f-clip.html#beit3", ["lnorm", "mlp"]),
        N("AIMv2", "2024", "RMSNorm, SwiGLU, no biases, a normed patch embedding", "f-clip.html#aimv2",
          ["lnorm", "mlp", "tpool"]),
        N("DINOv2", "2023", "layer scale; a packed SwiGLU at giant size", "f-dino.html", ["scale", "mlp"], [
            N("Registers", "2023", "extra prefix tokens that no pool reads", "f-dino.html#registers", ["tokens"], [
                N("C-RADIO", "2024", "registers; no final norm", "f-dino.html#radio", ["tokens"]),
                N("sbb register ViTs", "2024", "reg1 / reg4 GAP baselines", "f-dino.html#sbb", ["tokens", "tpool"]),
            ]),
        ]),
        N("Swin", "2021", "attention in shifted windows; patch merges build a pyramid", "f-swin.html",
          ["window", "merge", "tpool"], [
            N("Swin-S3", "2021", "a searched window size per stage", "f-swin.html#s3", ["window"]),
            N("Swin V2", "2021", "res-post-norm; cosine attention; continuous position bias", "f-swin.html#v2",
              ["placement", "window", "merge"]),
        ], flags="kit"),
    ], flags="kit"),
])

TREE = N("Residual image models", "", "two generic models, one residual algebra", "algebra.html", ["res"],
         [CONV_TREE, TX_TREE])


def chips(keys):
    out = []
    for k in keys:
        kind, label, href = CH[k]
        out.append(f'<a class="chip {kind}" href="{href}">{label}</a>')
    return "".join(out)


def tree_node(n, depth=0):
    cls = "tn" + (" kit" if "kit" in n["flags"] else "") + (" boundary" if "boundary" in n["flags"] else "")
    search = html.escape(" ".join([n["name"], n["delta"], " ".join(CH[c][1] for c in n["chips"])]).lower())
    label = (f'<span class="{cls}"><a class="nm" href="{n["href"]}">{html.escape(n["name"])}</a>'
             f'<span class="yr">{n["year"]}</span><span class="dl">{html.escape(n["delta"])}</span>{chips(n["chips"])}</span>')
    if n["kids"]:
        kids = "".join(tree_node(k, depth + 1) for k in n["kids"])
        return f'<li data-s="{search}"><details open><summary>{label}</summary><ul>{kids}</ul></details></li>'
    return f'<li data-s="{search}">{label}</li>'


def tree():
    return (
        '<div class="tree-tools"><button data-act="open">expand all</button><button data-act="close">collapse all</button>'
        '<input type="search" placeholder="filter: e.g. attention, registers, 2019" aria-label="filter the tree"></div>'
        f'<ul class="tree" id="lineage">{tree_node(TREE)}</ul>'
    )


# --------------------------------------------------------------------- matrix
SLOTS = [("unit", "unit kind", "c-units.html"), ("mixer", "spatial mixer", "algebra.html#slots"),
         ("attn", "attention", "c-attention.html"), ("short", "shortcut", "c-shortcut.html"),
         ("stem", "stem", "c-sampling.html#stems"), ("aa", "anti-alias", "c-sampling.html#aa"),
         ("norm", "norm", "c-norm.html"), ("final", "final layer", "c-head.html")]
MATRIX = [
    ("ResNet v1", "f-resnet.html", ["basic / bottleneck", "3×3 conv", "—", "1×1 conv", "7×7 + max", "—", "BN", "—"]),
    ("ResNet-D", "f-resnet.html#tricks", ["bottleneck", "3×3 conv", "—", "avg → 1×1", "deep 3×3×3 + max", "—", "BN", "—"]),
    ("ResNeXt", "f-resnext.html", ["grouped bottleneck", "3×3, g = 32", "—", "1×1 conv", "7×7 + max", "—", "BN", "—"]),
    ("Wide ResNet", "f-resnext.html#wide", ["bottleneck ×2 wide", "3×3 conv", "—", "1×1 conv", "7×7 + max", "—", "BN", "—"]),
    ("SE / ECA-ResNet", "f-resnet.html#attn", ["bottleneck", "3×3 conv", "SE / ECA @ out", "either", "either", "—", "BN", "—"]),
    ("AA-ResNet", "f-resnet.html#aa", ["bottleneck", "3×3, s1", "—", "either", "+ avg / blur pool", "blur / avg", "BN", "—"]),
    ("ResNet-RS", "f-resnet.html#rs", ["bottleneck", "3×3 conv", "SE ¼ @ out", "avg → 1×1", "deep + conv pool", "—", "BN", "—"]),
    ("ResNetV2", "f-resnetv2.html", ["pre-act basic / bottleneck", "3×3 conv", "—", "1×1 (avg)", "7×7 or deep, bare", "—", "BN / GN / Evo", "norm + act"]),
    ("BiT", "f-resnetv2.html#bit", ["pre-act bottleneck", "3×3 conv, WS", "—", "1×1 WS", "7×7, zero-pad max", "—", "GN", "norm + act"]),
    ("Res2Net", "f-res2net.html", ["bottleneck", "Res2 scale split", "—", "either", "either", "—", "BN", "—"]),
    ("SKNet", "f-sknet.html", ["basic / bottleneck", "selective kernel", "—", "1×1 conv", "7×7 + max", "—", "BN", "—"]),
    ("ResNeSt", "f-resnest.html", ["bottleneck", "split attention", "(inside mixer)", "avg → 1×1", "deep + max", "avd (3×3 avg)", "BN", "—"]),
    ("RegNetX", "f-regnet.html#x", ["grouped bottleneck", "3×3, groups of k", "—", "1×1 conv", "3×3, no pool", "—", "BN", "—"]),
    ("RegNetY", "f-regnet.html#y", ["grouped bottleneck", "3×3, groups of k", "SE @ mixer", "1×1 conv", "3×3, no pool", "—", "BN", "—"]),
    ("RegNetV", "f-regnet.html#v", ["pre-act bottleneck", "3×3, groups of k", "SE @ mixer", "1×1 / avg", "3×3 bare", "—", "BN", "act"]),
    ("RegNetZ", "f-regnet.html#z", ["inverted, linear out", "3×3, groups of k", "SE @ mixer", "none", "3×3, no pool", "—", "BN", "act / 1×1 conv"]),
]


TX_SLOTS = [("tok", "tokenizer", "t-patch.html"), ("prefix", "prefix tokens", "t-patch.html#tokens"),
            ("pos", "positions", "t-patch.html#pos"), ("mixer", "token mixer", "t-attention.html"),
            ("mlp", "MLP", "t-mlp.html"), ("norm", "norm · placement", "t-sublayer.html#placement"),
            ("scale", "layer scale", "t-sublayer.html#scale"), ("pool", "pool · its norm", "t-pool.html")]
TX_MATRIX = [
    ("ViT", "f-vit.html", ["16×16 patches", "cls", "learnt, all", "global MHSA", "GELU 4×", "LN · pre", "—", "token · LN before"]),
    ("Hybrid ViT", "f-vit.html#hybrid", ["ResNetV2 map, 1×1", "cls", "learnt, all", "global MHSA", "GELU 4×", "LN · pre", "—", "token · LN before"]),
    ("ResPost ViT", "f-vit.html#rpn", ["16×16 patches", "—", "learnt", "global MHSA", "GELU 4×", "LN · res-post", "—", "avg · LN after"]),
    ("GAP ViT (sbb)", "f-vit.html#gap", ["16×16 patches", "—", "learnt", "global MHSA", "GELU 4×", "LN · pre", "γ", "avg · LN before"]),
    ("DeiT, distilled", "f-deit.html", ["16×16 patches", "cls + dist", "learnt, all", "global MHSA", "GELU 4×", "LN · pre", "—", "distilled · LN before"]),
    ("DeiT III", "f-deit.html#deit3", ["16×16 patches", "cls", "learnt, patches", "global MHSA", "GELU 4×", "LN · pre", "γ", "token · LN before"]),
    ("CLIP tower", "f-clip.html#clip", ["p×p, no bias", "cls", "learnt, all", "global MHSA", "(Quick)GELU 4×", "LN ε1e-5 · pre + ln_pre", "—", "token · LN before"]),
    ("SigLIP tower", "f-clip.html#siglip", ["p×p patches", "—", "learnt", "global MHSA", "(tanh) GELU", "LN · pre", "—", "map | avg · LN before"]),
    ("BEiT3", "f-clip.html#beit3", ["16×16 patches", "—", "learnt", "MHSA + sub-LN", "GELU 4× + sub-LN", "LN · pre", "—", "avg · LN after"]),
    ("AIMv2", "f-clip.html#aimv2", ["14×14, RMS-normed", "—", "learnt", "MHSA, no biases", "SwiGLU, no bias", "RMS · pre", "—", "avg · RMS before"]),
    ("DINOv2", "f-dino.html", ["14×14 patches", "cls", "learnt, all", "global MHSA", "GELU 4× | packed SwiGLU", "LN · pre", "γ", "token · LN before"]),
    ("DINOv2 + registers", "f-dino.html#registers", ["14×14 patches", "cls + 4 reg", "learnt, patches", "global MHSA", "GELU 4× | packed SwiGLU", "LN · pre", "γ", "token · LN before"]),
    ("C-RADIO", "f-dino.html#radio", ["16×16 patches", "cls + 7–9 reg", "learnt, patches", "global MHSA", "GELU", "LN · pre", "γ (B, L)", "token · none"]),
    ("Swin", "f-swin.html", ["4×4 + LN, then merges", "—", "— (relative bias)", "shifted windows", "GELU 4×", "LN · pre", "—", "avg · LN before"]),
    ("Swin-S3", "f-swin.html#s3", ["4×4 + LN, then merges", "—", "— (relative bias)", "windows per stage", "GELU 4×", "LN · pre", "—", "avg · LN before"]),
    ("Swin V2", "f-swin.html#v2", ["4×4 + LN, then merges", "—", "— (CPB-MLP)", "cosine, shifted windows", "GELU 4×", "LN · res-post", "—", "avg · LN before"]),
]


def matrix(which=""):
    slots, data = (TX_SLOTS, TX_MATRIX) if which == "tx" else (SLOTS, MATRIX)
    head = "".join(f'<th><a href="{h}">{l}</a></th>' for _, l, h in slots)
    rows = []
    for name, href, cells in data:
        tds = "".join(f'<td>{html.escape(c) if c != "—" else "<span style=opacity:.4>—</span>"}</td>' for c in cells)
        rows.append(f'<tr><th><a href="{href}">{name}</a></th>{tds}</tr>')
    return f'<div class="tbl-wrap"><table><thead><tr><th>design</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# --------------------------------------------------------------------- page shell
def nav(here):
    parts = []
    for part, pages in PARTS:
        items = "".join(
            f'<li><a href="{s}.html"{" class=here" if s == here else ""}>{t if s != "index" else "Overview"}</a></li>'
            for s, t, *_ in pages)
        parts.append(f"<h4>{part}</h4><ol>{items}</ol>")
    return "".join(parts)


def mobile_nav():
    items = "".join(f'<li><a href="{s}.html">{t if s != "index" else "Overview"}</a></li>' for s, t, *_ in FLAT)
    return f'<details class="mobile-nav"><summary>The Residual Lineage — contents</summary><ol>{items}</ol></details>'


SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{lede}">
<link rel="stylesheet" href="style.css">
<script>try{{var t=localStorage.getItem('rl-theme');if(t)document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
</head>
<body>
{mobile}
<div class="layout">
<nav class="side" aria-label="Book contents">
<a class="brand" href="index.html">The Residual Lineage</a>
<div class="tag">Residual image models, conv and transformer, as modular algebra</div>
{nav}
<div class="theme"><button type="button" id="theme">toggle theme</button></div>
</nav>
<main>
<header class="page"><div class="kicker">{kicker}</div><h1>{h1}</h1><p class="lede">{lede}</p></header>
{body}
{pagerefs}
{pager}
</main>
</div>
<script src="book.js"></script>
</body>
</html>
"""


def venue(k):
    _, y, _, v, _ = REFS[k]
    return v if str(y) in v else f"{v}, {y}"


def render_refs_page():
    items = []
    for k, (a, y, t, v, u) in sorted(REFS.items(), key=lambda kv: (kv[1][1], kv[1][0])):
        items.append(f'<li id="{k}"><b>{html.escape(t)}</b>. {html.escape(a.rstrip("."))}. <i>{html.escape(venue(k))}</i>. '
                     f'<a href="{u}">{html.escape(u)}</a> <span class="k">[{k}]</span></li>')
    return f'<ol class="refs">{"".join(items)}</ol>'


def build_page(i, slug, title, kicker, lede):
    src = (HERE / "pages" / f"{slug}.html").read_text()
    cited = []

    def cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        links = []
        for k in keys:
            if k not in REFS:
                raise KeyError(f"{slug}: unknown citation {k}")
            if k not in cited:
                cited.append(k)
            n = cited.index(k) + 1
            links.append(f'<a href="#ref-{n}" title="{html.escape(REFS[k][2])} ({REFS[k][1]})">{n}</a>')
        return f'<sup class="cite">[{",".join(links)}]</sup>'

    def fig(m):
        mod = figs_tx if hasattr(figs_tx, m.group(1)) else figs
        return getattr(mod, m.group(1))()

    def regnet(m):
        d, w0, wa, wm = m.group(1).split(",")
        s, _ = figs.regnet_widths((int(d), int(w0), float(wa), float(wm)))
        return s

    def circuit(m):
        key, _, note = m.group(1).partition("|")
        spec, parent = circuits.C[key]
        ch = circuits.changed(key)
        tx = spec.get("board") == "tx"
        extra = {"layers": "Layer choices", "order": "Socket order", "tap": "Shortcut tap"}

        def cname(c):
            if c in extra:
                return extra[c]
            if tx:
                n = figs_tx.TX_CAPTION[c]
                return n[0].upper() + n[1:] if not n.isupper() else n.title()
            return figs.SLOT_NAME[c].title()
        names = [cname(c) for c in circuits.SLOTS if c in ch]
        cap = f"<b>{html.escape(spec['name'])}.</b> "
        if parent:
            cap += f"Δ vs {html.escape(circuits.C[parent][0]['name'])}: {', '.join(names) or 'none'}. "
        cap += note.strip()
        board = figs_tx.circuit_tx(spec, ch) if tx else figs.circuit(spec, ch)
        return f'<figure class="fig circuit" id="circuit-{key}">{board}<figcaption>{cap}</figcaption></figure>'

    body = src
    body = re.sub(r"\{\{circuit:([a-z0-9-]+(?:\|[^}]*)?)\}\}", circuit, body)
    body = re.sub(r"\{\{fig:([a-z0-9_]+)\}\}", fig, body)
    body = re.sub(r"\{\{regnet:([0-9.,]+)\}\}", regnet, body)
    body = body.replace("{{tree}}", tree()).replace("{{matrix}}", matrix()).replace("{{matrix:tx}}", matrix("tx"))
    body = body.replace("{{refs}}", render_refs_page())
    body = re.sub(r"\{\{cite:([a-z0-9_, ]+)\}\}", cite, body)
    pagerefs = ""
    if cited:
        lis = "".join(
            f'<li id="ref-{n+1}">{html.escape(REFS[k][0].rstrip("."))}. <a href="references.html#{k}">{html.escape(REFS[k][2])}</a>. '
            f'{html.escape(venue(k))}. <a href="{REFS[k][4]}">↗</a></li>'
            for n, k in enumerate(cited))
        pagerefs = f'<section><h2>Cited on this page</h2><ol class="refs-page">{lis}</ol></section>'
    prev = FLAT[i - 1] if i > 0 else None
    nxt = FLAT[i + 1] if i + 1 < len(FLAT) else None
    pager = '<div class="pager">'
    pager += f'<a href="{prev[0]}.html"><span class="d">← previous</span>{prev[1]}</a>' if prev else "<span></span>"
    pager += f'<a href="{nxt[0]}.html" style="text-align:right"><span class="d">next →</span>{nxt[1]}</a>' if nxt else ""
    pager += "</div>"
    page_title = title if slug == "index" else f"{title} · Residual Lineage"
    out = SHELL.format(title=html.escape(page_title), lede=html.escape(lede), mobile=mobile_nav(), nav=nav(slug),
                       kicker=kicker, h1=html.escape(title), body=body, pagerefs=pagerefs, pager=pager)
    (OUT / f"{slug}.html").write_text(out)
    left = re.findall(r"\{\{[^}]*\}\}", out)
    if left:
        raise ValueError(f"{slug}: unresolved {left}")


if __name__ == "__main__":
    for i, (slug, title, kicker, lede) in enumerate(FLAT):
        build_page(i, slug, title, kicker, lede)
        print("built", slug)
