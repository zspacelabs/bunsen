"""Build the book: wrap each page fragment in the shared shell.

Fragments live in pages/<slug>.html. Placeholders:
  {{fig:name}}             an SVG from figs.py
  {{circuit:key|note}}     a variant's circuit board (board.py, circuits.py)
  {{cite:key[,key]}}       a numbered citation (per page), linked to references.html
  {{tree}} {{matrix}} {{refs}}  generated blocks
Run:  python3 build.py   (writes ../*.html)
"""
import html
import re
from pathlib import Path

from . import board
from . import board_sp
from . import circuits
from . import figs
from . import figs_sp
from .refs import REFS

HERE = Path(__file__).parent
OUT = HERE.parent
BOOK = "The Residual Stream"

PARTS = [
    ("Orientation", [
        ("index", BOOK, "Overview", "bunsen's language and speech models, from nanochat to ZONOS3, read as one modular algebra."),
        ("algebra", "The Language-Model Algebra", "Orientation",
         "One generic stack, a board of sockets, and the lowering from a family's config to it."),
        ("names", "Reading the Names", "Orientation",
         "Checkpoint names are compressed policy: generation, size, tuning, and the rest."),
        ("sp-algebra", "From Sound to Tokens and Back", "Orientation",
         "Speech models as language models with a signal path: an encoder before, or a codec after."),
    ]),
    ("Combinators", [
        ("c-embed", "Embedding & Head", "Combinator", "The token table, its norm and scale; the head, tied or not, sliced and capped."),
        ("c-attention", "Causal Self-Attention", "Token mixer",
         "Scores, heads, grouped queries, and the gate on attention's output."),
        ("c-rope", "Rotary Positions", "Combinator", "Positions as rotations: bases, spectra, partial and scaled tables."),
        ("c-mask", "Masks, Windows & Patterns", "Combinator",
         "Causal and bidirectional masks, sliding windows, and the layer patterns that mix them."),
        ("c-logits", "Logit Control", "Combinator", "QK-norm, score scales and soft-caps: keeping attention's scores in range."),
        ("c-mlp", "The Feed-Forward", "Channel mixer", "Plain and gated MLPs: ReLU², GELU, SwiGLU and GeGLU."),
        ("c-sublayer", "Sublayers & Norms", "Combinator",
         "RMSNorm and LayerNorm, 1 + w gains, pre-norm and the sandwich."),
        ("c-deltanet", "The Gated DeltaNet", "Token mixer",
         "Linear attention as a fast-weight memory, corrected by the delta rule and gated."),
        ("c-stream", "Stream Extras", "Combinator",
         "nanochat 2026's extra paths: the smear, x₀ mixing, value embeddings, the backout."),
        ("c-cache", "Decoding State", "Combinator", "What each mixer must remember between steps, and who owns it."),
    ]),
    ("Families", [
        ("f-nanochat", "nanochat", "Family", "The GPT-2 line, modernized: upstream's 2025 model and its 2026 additions."),
        ("f-qwen3", "Qwen3", "Family", "The Llama template plus QK-norm: the plainest modern decoder in the kits."),
        ("f-gemma", "Gemma 1, 2 & 3", "Family", "GeGLU and √D, then the sandwich, windows and caps, then QK-norm and two tables."),
        ("f-qwen35", "Qwen3.5", "Family", "A hybrid: three gated DeltaNet layers to one gated attention layer."),
        ("f-modernbert", "ModernBERT", "Family", "The decoder's parts, run bidirectionally: a modern text encoder."),
        ("f-laya", "Laya", "Family", "An encoder with a decision head: typed questions in, calibrated probabilities out."),
    ]),
    ("Speech combinators", [
        ("s-frontend", "Log-Mel Front Ends", "Combinator", "Waveforms to log-mel frames, and the level each is measured against."),
        ("s-stem", "Stems & Frame Rates", "Combinator", "Strided convolutions and frame stacking: from 100 frames a second to 12."),
        ("s-encoder", "Audio Encoders", "Combinator", "Positions and masks: one window, blocks, or causal and streaming."),
        ("s-conformer", "The Conformer", "Combinator", "The macaron block: attention with relative positions, and a convolution."),
        ("s-bridge", "Bridges", "Combinator", "Four ways audio meets text: cross-attention, splice, addition with a delay, a joint."),
        ("s-transducer", "Transducers", "Combinator", "Frame-synchronous decoding: blank, the joint, and token-and-duration steps."),
        ("s-codec", "Neural Codecs", "Combinator", "Residual quantization, and two decoders from codes to a waveform."),
        ("s-codes", "Language Models over Codes", "Combinator",
         "Several codebooks a frame: summed rows, the delay pattern, and the depth decoder."),
        ("s-routes", "Routes & Experts", "Combinator",
         "ZONOS3's stack: attention residuals in place of the stream, and a latent mixture of experts."),
        ("s-stream", "Streams & Windows", "Combinator", "Windows, commits, voice activity, and what a stream must carry."),
    ]),
    ("Speech families", [
        ("f-whisper", "Whisper", "Family", "The encoder–decoder recognizer: a 30 s window, cross-attention, timestamp tokens."),
        ("f-qwen3asr", "Qwen3-ASR", "Family", "A Whisper-style encoder spliced into a Qwen3 chat prompt."),
        ("f-voxtral", "Voxtral Realtime", "Family", "Causal throughout: audio added to each step, the text a fixed delay behind."),
        ("f-parakeet", "Parakeet TDT", "Family", "A FastConformer and a token-and-duration transducer."),
        ("f-qwen3tts", "Qwen3-TTS", "Family", "A Qwen3 talker, a depth decoder for the other codebooks, and a DAC-style codec."),
        ("f-zonos3", "ZONOS3-Mini", "Family",
         "Delayed codebooks, a hybrid stack without a residual stream, and an ISTFT codec."),
    ]),
    ("Back matter", [
        ("references", "References", "Back matter", "The papers behind each family and combinator."),
    ]),
]
FLAT = [p for _, ps in PARTS for p in ps]


# --------------------------------------------------------------------- tree
def N(name, year, delta, href, chips=(), kids=(), flags=""):
    return dict(name=name, year=year, delta=delta, href=href, chips=list(chips), kids=list(kids), flags=flags)


CH = {
    "embed": ("io", "embedding", "c-embed.html"),
    "head": ("io", "head", "c-embed.html#head"),
    "tied": ("io", "tied head", "c-embed.html#tied"),
    "attn": ("attn", "attention", "c-attention.html"),
    "gqa": ("attn", "grouped queries", "c-attention.html#gqa"),
    "gate": ("act", "output gate", "c-attention.html#gate"),
    "rope": ("pool", "rotary", "c-rope.html"),
    "mask": ("io", "mask", "c-mask.html"),
    "window": ("norm", "windows", "c-mask.html#windows"),
    "qknorm": ("norm", "QK-norm", "c-logits.html#qknorm"),
    "cap": ("act", "soft-cap", "c-logits.html#softcap"),
    "mlp": ("act", "feed-forward", "c-mlp.html"),
    "glu": ("act", "gated MLP", "c-mlp.html#glu"),
    "norm": ("norm", "norm kind", "c-sublayer.html#kinds"),
    "placement": ("norm", "norm placement", "c-sublayer.html#placement"),
    "deltanet": ("mix", "gated DeltaNet", "c-deltanet.html"),
    "stream": ("act", "stream extras", "c-stream.html"),
    "res": ("io", "residual stream", "algebra.html#stream"),
    "cache": ("conv", "decode state", "c-cache.html"),
    "dhead": ("pool", "decision head", "f-laya.html#head"),
    "fe": ("conv", "front end", "s-frontend.html"),
    "stem": ("pool", "stem", "s-stem.html"),
    "aenc": ("mix", "audio encoder", "s-encoder.html"),
    "conformer": ("mix", "Conformer", "s-conformer.html"),
    "bridge": ("attn", "bridge", "s-bridge.html"),
    "transducer": ("act", "transducer", "s-transducer.html"),
    "codec": ("reg", "codec", "s-codec.html"),
    "codes": ("act", "code layout", "s-codes.html"),
    "routes": ("mix", "attention residuals", "s-routes.html"),
    "moe": ("act", "mixture of experts", "s-routes.html#moe"),
    "stream_sp": ("io", "streaming", "s-stream.html"),
}

TREE = N("Residual language and speech models", "", "every kit below is one generic stack with its sockets filled", "algebra.html",
         ["res"], [
    N("Transformer", "2017", "encoder–decoder; post-norm; sinusoidal positions; ReLU MLP", "c-sublayer.html#placement",
      ["attn", "placement"], [
        N("Whisper", "2022", "encoder–decoder ASR: a log-mel encoder and a cross-attending text decoder", "f-whisper.html",
          ["fe", "aenc", "bridge"], [
            N("Whisper large-v3 / turbo", "2023–24", "128 mels; turbo prunes the decoder to 4 layers", "f-whisper.html#sizes",
              ["fe"]),
        ], flags="kit"),
        N("Conformer → FastConformer", "2020–23", "attention plus a depthwise conv in a macaron block; 8× subsampling",
          "s-conformer.html", ["conformer", "stem"], [
            N("Parakeet TDT", "2025", "a FastConformer read by a token-and-duration transducer", "f-parakeet.html",
              ["transducer"], flags="kit"),
        ], flags="boundary"),
        N("BERT", "2018", "a bidirectional encoder, trained to fill in masked tokens", "f-modernbert.html#bert",
          ["mask"], [
            N("ModernBERT", "2024", "RoPE, GeGLU, bias-free LayerNorm; local windows with a global layer every third",
              "f-modernbert.html", ["rope", "glu", "norm", "window"], [
                N("mmBERT", "2025", "the same encoder, multilingual, on Gemma's 256k vocabulary", "f-modernbert.html#mmbert",
                  ["embed"]),
                N("Laya", "2025", "a decision head over ModernBERT or mmBERT: options scored at [MASK] markers",
                  "f-laya.html", ["dhead"], flags="kit"),
            ], flags="kit"),
        ], flags="boundary"),
        N("GPT, GPT-2", "2018–19", "decoder-only; causal mask; GPT-2 moves the norms before each sublayer",
          "f-nanochat.html#gpt2", ["mask", "placement"], [
            N("nanochat (2025)", "2025", "RoPE, QK-norm after it, ReLU², parameter-free RMS, a soft-capped untied head",
              "f-nanochat.html", ["rope", "qknorm", "mlp", "norm", "cap"], [
                N("nanochat (2026)", "2026", "smear, x₀ mixing, value embeddings, backout, QK gain, windows",
                  "f-nanochat.html#v2026", ["stream", "window"]),
            ], flags="kit"),
            N("The PaLM / LLaMA template", "2022–23", "RMSNorm pre-norm, SwiGLU, RoPE, no biases; GQA from Llama 2",
              "algebra.html#template", ["norm", "glu", "rope", "gqa"], [
                N("Qwen3", "2025", "QK-norm before RoPE; heads of 128 decoupled from the width; tied to 4B",
                  "f-qwen3.html", ["qknorm", "gqa", "tied"], [
                    N("Qwen3-ASR", "2025", "a Whisper-style encoder's frames spliced into a Qwen3 chat prompt",
                      "f-qwen3asr.html", ["aenc", "bridge"], flags="kit"),
                    N("Qwen3-TTS", "2025", "a Qwen3 talker over codec codes, a depth decoder, a DAC-style codec",
                      "f-qwen3tts.html", ["codes", "codec"], flags="kit"),
                    N("Qwen3.5", "2026", "3 gated DeltaNet layers : 1 gated attention layer; partial RoPE; 1 + w norms",
                      "f-qwen35.html", ["deltanet", "gate", "rope", "window"], [
                        N("ZONOS3-Mini", "2026", "GDN hybrid over delayed codebooks; attention residuals; latent MoE",
                          "f-zonos3.html", ["routes", "moe", "codes", "codec"], flags="kit"),
                    ], flags="kit"),
                ], flags="kit"),
                N("Voxtral Realtime", "2026", "a causal audio encoder added step by step to a Mistral decoder, a delay behind",
                  "f-voxtral.html", ["aenc", "bridge", "stream_sp"], flags="kit"),
                N("Gemma", "2024", "GeGLU; embedding × √D; tied head; 1 + w RMS; MQA at 2B", "f-gemma.html#gemma1",
                  ["glu", "embed", "tied", "norm"], [
                    N("Gemma 2", "2024", "sandwich norms; sliding windows on alternate layers; soft-caps",
                      "f-gemma.html#gemma2", ["placement", "window", "cap"], [
                        N("Gemma 3", "2025", "QK-norm replaces the caps; 5 sliding : 1 full; two rotary tables",
                          "f-gemma.html#gemma3", ["qknorm", "window", "rope"]),
                    ]),
                ], flags="kit"),
            ], flags="boundary"),
        ], flags="boundary"),
    ], flags="boundary"),
    N("CTC → RNN-T → TDT", "2006–23", "frame-synchronous decoding: a blank symbol, then a joint, then durations",
      "s-transducer.html", ["transducer"], flags="boundary"),
    N("SoundStream → EnCodec → DAC → Mimi", "2021–24", "neural codecs: residual vector quantization of a learned latent",
      "s-codec.html", ["codec"], flags="boundary"),
    N("Linear attention → DeltaNet → Gated DeltaNet", "2020–24",
      "attention as a recurrent outer-product memory; then error-correcting writes; then a decay gate",
      "c-deltanet.html", ["deltanet", "cache"], flags="boundary"),
])


def chips(keys):
    return "".join(f'<a class="chip {CH[k][0]}" href="{CH[k][2]}">{CH[k][1]}</a>' for k in keys)


def tree_node(n):
    cls = "tn" + (" kit" if "kit" in n["flags"] else "") + (" boundary" if "boundary" in n["flags"] else "")
    search = html.escape(" ".join([n["name"], n["year"], n["delta"], " ".join(CH[c][1] for c in n["chips"])]).lower())
    label = (f'<span class="{cls}"><a class="nm" href="{n["href"]}">{html.escape(n["name"])}</a>'
             f'<span class="yr">{n["year"]}</span><span class="dl">{html.escape(n["delta"])}</span>{chips(n["chips"])}</span>')
    if n["kids"]:
        kids = "".join(tree_node(k) for k in n["kids"])
        return f'<li data-s="{search}"><details open><summary>{label}</summary><ul>{kids}</ul></details></li>'
    return f'<li data-s="{search}">{label}</li>'


def tree():
    return (
        '<div class="tree-tools"><button data-act="open">expand all</button><button data-act="close">collapse all</button>'
        '<input type="search" placeholder="filter: e.g. window, QK-norm, 2025" aria-label="filter the tree"></div>'
        f'<ul class="tree" id="lineage">{tree_node(TREE)}</ul>'
    )


# --------------------------------------------------------------------- matrix
SLOTS = [("embed · head", "c-embed.html"), ("token mixer", "c-attention.html"), ("positions", "c-rope.html"),
         ("QK · scores", "c-logits.html"), ("mask · windows", "c-mask.html"), ("feed-forward", "c-mlp.html"),
         ("norm · placement", "c-sublayer.html"), ("extras", "c-stream.html")]
MATRIX = [
    ("Llama template", "algebra.html#template",
     ["plain · own", "causal, GQA", "RoPE 10⁴", "q·k/√d", "causal, full", "SwiGLU 8D/3", "RMS · pre", "—"]),
    ("nanochat 2025", "f-nanochat.html",
     ["RMS'd · own, cap 15", "causal, MHA", "RoPE 10⁴, bf16", "RMS after RoPE", "causal, full", "ReLU² 4D",
      "RMS (no gain) · pre", "—"]),
    ("nanochat 2026", "f-nanochat.html#v2026",
     ["RMS'd, padded · own, cap 15", "causal, MHA + value emb.", "RoPE 10⁵", "RMS after · gain 1.2", "S / L pattern",
      "ReLU² 4D", "RMS (no gain) · pre", "smear · λ, μ · backout"]),
    ("Qwen3", "f-qwen3.html",
     ["plain · tied (≤ 4B)", "causal, GQA, head 128", "RoPE 10⁶", "RMS before RoPE", "causal, full", "SwiGLU",
      "RMS · pre", "—"]),
    ("Gemma", "f-gemma.html#gemma1",
     ["× √D · tied", "causal, MQA / MHA, head 256", "RoPE 10⁴", "q·k/√d", "causal, full", "GeGLU (tanh) 8D",
      "RMS 1+w · pre", "—"]),
    ("Gemma 2", "f-gemma.html#gemma2",
     ["× √D · tied, cap 30", "causal, GQA, head 256", "RoPE 10⁴", "÷√256 · cap 50", "4096-window, alternate",
      "GeGLU (tanh)", "RMS 1+w · sandwich", "—"]),
    ("Gemma 3", "f-gemma.html#gemma3",
     ["× √D · tied", "causal, GQA / MQA, head 256", "local 10⁴ · global 10⁶", "RMS before · ÷√256",
      "512–1024-window, 5:1", "GeGLU (tanh)", "RMS 1+w · sandwich", "—"]),
    ("Qwen3.5", "f-qwen35.html",
     ["plain · tied (≤ 4B)", "DeltaNet 3 : gated attn 1", "¼ RoPE 10⁷ (attn)", "RMS before (attn)", "causal, full",
      "SwiGLU", "RMS 1+w · pre", "—"]),
    ("ModernBERT", "f-modernbert.html",
     ["LN'd · none", "bidirectional MHA", "global 1.6·10⁵ · local 10⁴", "q·k/√d", "±64 local, global 1 : 2",
      "GeGLU (exact)", "LN, no bias · pre", "—"]),
    ("Laya head", "f-laya.html",
     ["encoder states + kind · scorer", "bidirectional MHA, biased", "none", "q·k/√d", "prompt tokens",
      "ReLU 4D, biased", "LN · pre", "—"]),
]


SP_SLOTS = [("front end", "s-frontend.html"), ("stem · rate", "s-stem.html"), ("encoder", "s-encoder.html"),
            ("bridge", "s-bridge.html"), ("language model", "algebra.html#board"), ("output", "s-codes.html"),
            ("codec", "s-codec.html")]
SP_MATRIX = [
    ("Whisper", "f-whisper.html", ["log₁₀ mel 80/128 · clip max", "conv1d ×2 · 50 Hz", "LN, learned pos, 30 s",
                                   "cross-attention", "LN decoder, learned pos", "tied · timestamps · beam", "—"]),
    ("Qwen3-ASR", "f-qwen3asr.html", ["log₁₀ mel 128 · window max", "conv2d ×3 per 1 s · 13 Hz", "LN, sinusoid per chunk, 8 s blocks",
                                      "splice into prompt", "Qwen3", "tied · greedy", "—"]),
    ("Voxtral Realtime", "f-voxtral.html", ["log₁₀ mel 128 · fixed level", "causal conv ×2 + stack 4 · 12.5 Hz",
                                            "RMS, RoPE, causal 15 s", "add per step + delay scale", "Ministral-style",
                                            "tied · greedy · PAD / [W]", "—"]),
    ("Parakeet TDT", "f-parakeet.html", ["ln mel 128 · per-mel norm", "dw-striding ×8 · 12.5 Hz", "Conformer, relative pos",
                                         "joint", "LSTM prediction net", "tokens + durations · greedy", "—"]),
    ("Qwen3-TTS", "f-qwen3tts.html", ["—", "—", "—", "—", "Qwen3 talker + depth decoder", "cb 0 + 15 depth heads · sample",
                                      "split RVQ 16 · DAC · 24 kHz"]),
    ("ZONOS3-Mini", "f-zonos3.html", ["fbank (speaker clip)", "—", "MECT-B2 speaker encoder", "speaker + condition rows",
                                      "AttnRes GDN hybrid + latent MoE", "10 heads, delayed · sample", "10 cb · ISTFT · 44.1 kHz"]),
]


def matrix(which=""):
    slots, data = (SP_SLOTS, SP_MATRIX) if which == "sp" else (SLOTS, MATRIX)
    head = "".join(f'<th><a href="{h}">{l}</a></th>' for l, h in slots)
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
    return f'<details class="mobile-nav"><summary>{BOOK} — contents</summary><ol>{items}</ol></details>'


SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{lede}">
<link rel="stylesheet" href="style.css">
<script>try{{var t=localStorage.getItem('rs-theme');if(t)document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
</head>
<body>
{mobile}
<div class="layout">
<nav class="side" aria-label="Book contents">
<a class="brand" href="index.html">{book}</a>
<div class="tag">Language and speech models, as modular algebra</div>
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


def circuit(m):
    key, _, note = m.group(1).partition("|")
    spec, parent = circuits.C[key]
    ch = circuits.changed(key)
    sp = spec.get("board") == "sp"

    def cname(c):
        if c in circuits.EXTRA:
            return circuits.EXTRA[c]
        n = (board_sp.CAPTION if sp else board.CAPTION)[c]
        return n[0].upper() + n[1:] if not n.isupper() else n.title()
    names = [cname(c) for c in circuits.SLOTS if c in ch]
    cap = f"<b>{html.escape(spec['name'])}.</b> "
    if parent:
        cap += f"Δ vs {html.escape(circuits.C[parent][0]['name'])}: {', '.join(names) or 'none'}. "
    cap += note.strip()
    return f'<figure class="fig circuit" id="circuit-{key}">{(board_sp.circuit_sp if sp else board.circuit_lm)(spec, ch)}<figcaption>{cap}</figcaption></figure>'


def build_page(i, slug, title, kicker, lede):
    src = (HERE / "pages" / f"{slug}.html").read_text()
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

    body = src
    body = re.sub(r"\{\{circuit:([a-z0-9-]+(?:\|[^}]*)?)\}\}", circuit, body)
    body = re.sub(r"\{\{fig:([a-z0-9_]+)\}\}", lambda m: getattr(figs_sp if hasattr(figs_sp, m.group(1)) else figs, m.group(1))(), body)
    body = body.replace("{{tree}}", tree()).replace("{{matrix}}", matrix()).replace("{{matrix:sp}}", matrix("sp")).replace("{{refs}}", render_refs_page())
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
    page_title = title if slug == "index" else f"{title} · {BOOK}"
    out = SHELL.format(title=html.escape(page_title), lede=html.escape(lede), mobile=mobile_nav(), nav=nav(slug),
                       book=BOOK, kicker=kicker, h1=html.escape(title), body=body, pagerefs=pagerefs, pager=pager)
    (OUT / f"{slug}.html").write_text(out)
    left = re.findall(r"\{\{[^}]*\}\}", out)
    if left:
        raise ValueError(f"{slug}: unresolved {left}")


if __name__ == "__main__":
    for i, (slug, title, kicker, lede) in enumerate(FLAT):
        build_page(i, slug, title, kicker, lede)
        print("built", slug)
