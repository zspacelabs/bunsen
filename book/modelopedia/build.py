"""Build The Modelopedia: one book over bunsen's image, language and speech models.

The two halves keep their own figure, circuit and table code as packages (img: conv nets and vision transformers;
lng: language and speech models). This script owns the page list, the merged bibliography, the unified tree and
overview, and each page's placeholders. Fragments live in pages/<slug>.html; assets/ holds the stylesheet and script.

Run:  python3 build.py [OUT]   (default OUT: ../src/modelopedia, where mdBook publishes it as static files)

The mdBook preprocessor (preprocess.py) runs this on every `mdbook build`. Files are written only when their
content changes, so `mdbook serve`, which watches src/, settles after one rebuild.
"""
import html
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of the source tree

from img import build as IB
from img import circuits as IC
from img import figs as IF
from img import figs_tx as IFT
from lng import build as LB
from lng import figs as LF
from lng import figs_sp as LFS
from lng import svgkit as K

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "modelopedia"
BOOK = "The Modelopedia"
TAG = "bunsen's image, language and speech models, as one modular algebra"

# ------------------------------------------------------------------ pages
IMETA = {s: (t, k, l) for _, ps in IB.PARTS for s, t, k, l in ps}
LMETA = {s: (t, k, l) for _, ps in LB.PARTS for s, t, k, l in ps}


def img(slug, title=None, lede=None):
    t, k, l = IMETA[slug]
    return (slug, title or t, k, lede or l, "img")


def lng(old, new=None, title=None, lede=None):
    t, k, l = LMETA[old]
    return (new or old, title or t, k, lede or l, "lng")


def part_pages(meta, prefix, exclude=(), mk=None):
    return [mk(s) for s in meta if s.startswith(prefix) and s not in exclude]


PARTS = [
    ("Orientation", [
        ("index", BOOK, "Overview", "bunsen's image, language and speech models, from ResNet to ZONOS3, read as one modular algebra.", "atlas"),
        ("names", "Reading the Names", "Orientation",
         "Model and checkpoint names are compressed policy: how to decode them into socket choices.", "atlas"),
    ]),
    ("I · Conv nets", [img("algebra", "The Conv-Net Algebra", "One conv-net shape, a handful of slots, and the combinators that fill them.")]
     + [img(s) for s in IMETA if s.startswith("c-")]),
    ("I · Conv families", [img(s) for s in IMETA if s.startswith("f-") and IMETA[s][1] == "Family"
                            and s in ("f-resnet", "f-resnext", "f-resnetv2", "f-res2net", "f-sknet", "f-resnest", "f-regnet")]),
    ("II · Vision transformers", [img("tx-algebra")] + [img(s) for s in IMETA if s.startswith("t-")]),
    ("II · Vision families", [img(s) for s in ("f-vit", "f-deit", "f-clip", "f-dino", "f-swin")]),
    ("III · Language models", [lng("algebra", "lm-algebra")] + [lng(s, "l-" + s[2:]) for s in LMETA if s.startswith("c-")]),
    ("III · Language families", [lng(s) for s in ("f-nanochat", "f-qwen3", "f-gemma", "f-qwen35", "f-modernbert", "f-laya")]),
    ("IV · Speech", [lng("sp-algebra")] + [lng(s) for s in LMETA if s.startswith("s-")]),
    ("IV · Speech families", [lng(s) for s in ("f-whisper", "f-qwen3asr", "f-voxtral", "f-parakeet", "f-qwen3tts", "f-zonos3")]),
    ("Back matter", [("references", "References", "Back matter", "The papers behind every family and combinator.", "atlas")]),
]
FLAT = [p for _, ps in PARTS for p in ps]

# ------------------------------------------------------------------ bibliography: one key space, duplicates by URL
REFS = dict(IB.REFS)
REFS.update(LB.REFS)
BY_URL = {}
for k, v in REFS.items():
    BY_URL.setdefault(v[4], []).append(k)

# ------------------------------------------------------------------ link rewriting for the language half
def lng_links(s):
    def fix(m):
        h = m.group(1)
        f, _, a = h.partition("#")
        if f == "algebra.html":
            f = "lm-algebra.html"
        elif re.fullmatch(r"c-[a-z0-9]+\.html", f):
            f = "l-" + f[2:]
        elif f == "names.html":
            a = {"examples": "lm-examples"}.get(a, "lm-grammar")
        elif f == "index.html":
            a = {"matrix": "lm-matrix", "spmatrix": "sp-matrix"}.get(a, a)
        return f'href="{f}{"#" + a if a else ""}"'
    return re.sub(r'href="([a-z0-9-]+\.html(?:#[^"]*)?)"', fix, s)


def img_links(s):
    s = s.replace('href="index.html#matrix"', 'href="index.html#conv-matrix"')
    s = s.replace('href="index.html#txmatrix"', 'href="index.html#tx-matrix"')
    return s.replace('href="names.html#examples"', 'href="names.html#img-examples"')


# ------------------------------------------------------------------ the unified tree and the map
def tree():
    root = ('<span class="tn"><a class="nm" href="index.html#map">Residual models</a><span class="yr"></span>'
            '<span class="dl">four generic models, one residual algebra</span></span>')
    kids = IB.tree_node(IB.CONV_TREE) + IB.tree_node(IB.TX_TREE) + lng_links(LB.tree_node(LB.TREE))
    return ('<div class="tree-tools"><button data-act="open">expand all</button><button data-act="close">collapse all</button>'
            '<input type="search" placeholder="filter: e.g. attention, codec, 2019" aria-label="filter the tree"></div>'
            f'<ul class="tree" id="lineage"><li data-s="residual models"><details open><summary>{root}</summary>'
            f'<ul>{kids}</ul></details></li></ul>')


def atlas_map():
    W, H = 960, 330
    o = ""
    cols = [
        ("conv net", "algebra.html", [("stem", "conv"), ("stages of units", "mix"), ("pool · head", "pool")], "ResNet … RegNet"),
        ("vision encoder", "tx-algebra.html", [("patches · tokens", "conv"), ("encoder blocks", "mix"), ("token pool · head", "pool")],
         "ViT · DeiT · DINOv2 · Swin"),
        ("language model", "lm-algebra.html", [("token embedding", "io"), ("decoder layers", "mix"), ("norm · head", "io")],
         "nanochat · Qwen · Gemma · ModernBERT"),
        ("speech model", "sp-algebra.html", [("front end · encoder", "conv"), ("bridge · decoder", "mix"), ("search | codec", "reg")],
         "Whisper · Parakeet · Qwen3-TTS · ZONOS3"),
    ]
    for i, (name, href, parts, who) in enumerate(cols):
        x = 24 + i * 234
        o += f'<a href="{href}">' + K.text(x + 102, 22, name, "ttl") + "</a>"
        for j, (lab, k) in enumerate(parts):
            o += K.box(x, 40 + j * 52, 204, 38, lab, k)
            if j < 2:
                o += K.arrow([(x + 102, 78 + j * 52), (x + 102, 92 + j * 52)])
        o += K.text(x + 102, 206, who, "xs")
        o += f'<path d="M{x+102},214 L{x+102},244" class="ar dash"/>'
    o += K.rect(24, 246, 906, 44, "b norm", 8)
    o += K.text(477, 262, "the residual unit / sublayer:  x ← x + f(norm(x))", "lbl")
    o += K.text(477, 280, "pre-activation · pre-norm · layer scale · the sandwich — one family of placements across all four", "xs")
    o += K.text(477, 314, "every family here lowers to one of these four generic models, and every one is built from the same residual sublayer", "sm")
    return K.svg(W, H, o, "Four generic models")


# ------------------------------------------------------------------ cross-links between the halves
SEE_ALSO = {
    "t-attention": [("l-attention", "causal attention in language models: GQA, the output gate")],
    "l-attention": [("t-attention", "self-attention in vision encoders"), ("c-attention", "channel attention in conv nets")],
    "t-mlp": [("l-mlp", "feed-forwards in language models: ReLU², SwiGLU, GeGLU")],
    "l-mlp": [("t-mlp", "the MLP in vision encoders")],
    "t-sublayer": [("l-sublayer", "the same sublayer in language models: RMSNorm, 1 + w, the sandwich"),
                   ("c-residual", "the conv residual unit it descends from")],
    "l-sublayer": [("t-sublayer", "all four norm slots, layer scale and drop path"), ("c-residual", "the conv residual unit")],
    "c-residual": [("t-sublayer", "the transformer's residual sublayer"), ("lm-algebra", "the residual stream")],
    "t-window": [("l-mask", "sliding windows and layer patterns in language models")],
    "l-mask": [("t-window", "window attention in Swin")],
    "c-attention": [("t-attention", "self-attention"), ("l-attention", "causal self-attention")],
    "tx-algebra": [("lm-algebra", "the same algebra for language models"), ("sp-algebra", "and for speech")],
    "lm-algebra": [("tx-algebra", "the vision encoder's version"), ("sp-algebra", "speech models built on it")],
    "s-encoder": [("t-patch", "patch embedding, vision's stem")],
    "s-stem": [("t-patch", "patch embedding, vision's stem"), ("t-merge", "patch merging")],
    "s-codec": [("t-merge", "patch merging, the downward twin of a patch-up")],
}


def see_also(slug):
    if slug not in SEE_ALSO:
        return ""
    title = {p[0]: p[1] for p in FLAT}
    items = " · ".join(f'<a href="{s}.html">{html.escape(title[s])}</a> ({html.escape(why)})' for s, why in SEE_ALSO[slug])
    return f'<section><div class="callout defer"><span class="t">Elsewhere in the Modelopedia</span>{items}</div></section>'


# ------------------------------------------------------------------ placeholders, per half
def img_circuit(m):
    key, _, note = m.group(1).partition("|")
    spec, parent = IC.C[key]
    ch = IC.changed(key)
    tx = spec.get("board") == "tx"
    extra = {"layers": "Layer choices", "order": "Socket order", "tap": "Shortcut tap"}

    def cname(c):
        if c in extra:
            return extra[c]
        if tx:
            n = IFT.TX_CAPTION[c]
            return n[0].upper() + n[1:] if not n.isupper() else n.title()
        return IF.SLOT_NAME[c].title()
    names = [cname(c) for c in IC.SLOTS if c in ch]
    cap = f"<b>{html.escape(spec['name'])}.</b> "
    if parent:
        cap += f"Δ vs {html.escape(IC.C[parent][0]['name'])}: {', '.join(names) or 'none'}. "
    cap += note.strip()
    board = IFT.circuit_tx(spec, ch) if tx else IF.circuit(spec, ch)
    return f'<figure class="fig circuit" id="circuit-{key}">{board}<figcaption>{cap}</figcaption></figure>'


def render_img(body):
    body = re.sub(r"\{\{circuit:([a-z0-9-]+(?:\|[^}]*)?)\}\}", img_circuit, body)
    body = re.sub(r"\{\{fig:([a-z0-9_]+)\}\}", lambda m: getattr(IFT if hasattr(IFT, m.group(1)) else IF, m.group(1))(), body)
    body = re.sub(r"\{\{regnet:([0-9.,]+)\}\}",
                  lambda m: IF.regnet_widths(tuple(t(v) for t, v in zip((int, int, float, float), m.group(1).split(","))))[0],
                  body)
    return img_links(body)


def render_lng(body):
    body = re.sub(r"\{\{circuit:([a-z0-9-]+(?:\|[^}]*)?)\}\}", LB.circuit, body)
    body = re.sub(r"\{\{fig:([a-z0-9_]+)\}\}", lambda m: getattr(LFS if hasattr(LFS, m.group(1)) else LF, m.group(1))(), body)
    body = body.replace("{{matrix:sp}}", LB.matrix("sp"))
    return lng_links(body)


def render_atlas(body):
    body = body.replace("{{fig:atlas_map}}", atlas_map()).replace("{{tree}}", tree())
    body = body.replace("{{matrix:conv}}", IB.matrix()).replace("{{matrix:tx}}", IB.matrix("tx"))
    body = body.replace("{{matrix:lm}}", lng_links(LB.matrix())).replace("{{matrix:sp}}", lng_links(LB.matrix("sp")))
    return body.replace("{{refs}}", refs_page())


def venue(k):
    _, y, _, v, _ = REFS[k]
    return v if str(y) in v else f"{v}, {y}"


def refs_page():
    items = []
    for u, keys in sorted(BY_URL.items(), key=lambda kv: (REFS[kv[1][0]][1], REFS[kv[1][0]][0])):
        k = keys[0]
        a, y, t, v, _ = REFS[k]
        alias = "".join(f'<span id="{x}"></span>' for x in keys[1:])
        items.append(f'<li id="{k}">{alias}<b>{html.escape(t)}</b>. {html.escape(a.rstrip("."))}. <i>{html.escape(venue(k))}</i>. '
                     f'<a href="{u}">{html.escape(u)}</a></li>')
    return f'<ol class="refs">{"".join(items)}</ol>'


# ------------------------------------------------------------------ shell
def nav(here):
    out = []
    for part, pages in PARTS:
        items = "".join(f'<li><a href="{p[0]}.html"{" class=here" if p[0] == here else ""}>'
                        f'{"Overview" if p[0] == "index" else p[1]}</a></li>' for p in pages)
        out.append(f"<h4>{part}</h4><ol>{items}</ol>")
    return "".join(out)


def mobile_nav():
    items = "".join(f'<li><a href="{p[0]}.html">{"Overview" if p[0] == "index" else p[1]}</a></li>' for p in FLAT)
    return f'<details class="mobile-nav"><summary>{BOOK} — contents</summary><ol>{items}</ol></details>'


SHELL = LB.SHELL.replace("rs-theme", "mp-theme").replace(
    '<div class="tag">Language and speech models, as modular algebra</div>',
    f'<div class="tag">{TAG}</div><div class="tag"><a href="../index.html">← The Bunsen Book</a></div>')


def render_page(i, slug, title, kicker, lede, src):
    body = (HERE / "pages" / f"{slug}.html").read_text()
    body = {"img": render_img, "lng": render_lng, "atlas": render_atlas}[src](body)
    body += see_also(slug)
    cited = []

    def cite(m):
        links = []
        for k in [k.strip() for k in m.group(1).split(",")]:
            if k not in REFS:
                raise KeyError(f"{slug}: unknown citation {k}")
            if k not in cited:
                cited.append(k)
            n = cited.index(k) + 1
            links.append(f'<a href="#ref-{n}" title="{html.escape(REFS[k][2])} ({REFS[k][1]})">{n}</a>')
        return f'<sup class="cite">[{",".join(links)}]</sup>'
    body = re.sub(r"\{\{cite:([a-z0-9_, ]+)\}\}", cite, body)
    pagerefs = ""
    if cited:
        lis = "".join(
            f'<li id="ref-{n+1}">{html.escape(REFS[k][0].rstrip("."))}. <a href="references.html#{k}">{html.escape(REFS[k][2])}</a>. '
            f'{html.escape(venue(k))}. <a href="{REFS[k][4]}">↗</a></li>' for n, k in enumerate(cited))
        pagerefs = f'<section><h2>Cited on this page</h2><ol class="refs-page">{lis}</ol></section>'
    prev = FLAT[i - 1] if i > 0 else None
    nxt = FLAT[i + 1] if i + 1 < len(FLAT) else None
    pager = '<div class="pager">'
    pager += f'<a href="{prev[0]}.html"><span class="d">← previous</span>{prev[1]}</a>' if prev else "<span></span>"
    pager += f'<a href="{nxt[0]}.html" style="text-align:right"><span class="d">next →</span>{nxt[1]}</a>' if nxt else ""
    pager += "</div>"
    page_title = title if slug == "index" else f"{title} · {BOOK}"
    out = SHELL.format(title=html.escape(page_title), lede=html.escape(lede), mobile=mobile_nav(), nav=nav(slug), book=BOOK,
                       kicker=kicker, h1=html.escape(title), body=body, pagerefs=pagerefs, pager=pager)
    left = re.findall(r"\{\{[^}]*\}\}", out)
    if left:
        raise ValueError(f"{slug}: unresolved {left}")
    return out


def build(out=OUT):
    """Renders every page into `out`, copies the assets, and removes files no page or asset produces.

    Returns the number of files written (unchanged files are left alone)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {f"{p[0]}.html": render_page(i, *p).encode() for i, p in enumerate(FLAT)}
    for a in sorted((HERE / "assets").glob("*.[cj]s*")):
        files[a.name] = a.read_bytes()
    written = 0
    for name, data in files.items():
        path = out / name
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
            written += 1
    for path in out.iterdir():
        if path.name not in files:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    n = build(target)
    print(f"{len(FLAT)} pages in {target} ({n} files written)")
