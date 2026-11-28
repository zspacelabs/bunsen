"""SVG figures for the speech chapters; same vocabulary as figs.py."""
import math

from .svgkit import add, arrow, box, esc, mul, rect, svg, text


def _wave(x, y, w, h, n=180, seed=1.0, cls="curve"):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        env = 0.25 + 0.75 * abs(math.sin(t * 3.1 + seed)) * (0.6 + 0.4 * math.sin(t * 11 + seed))
        v = env * math.sin(t * 95 + seed * 3) * 0.9
        pts.append(f"{x + t * w:.1f},{y + h / 2 - v * h / 2:.1f}")
    return f'<path d="M{" L".join(pts)}" class="{cls}" style="stroke-width:1.2"/>'


def _spec(x, y, w, h, cols=24, rows=10, seed=0):
    o = ""
    cw, rh = w / cols, h / rows
    for c in range(cols):
        for r in range(rows):
            v = 0.5 + 0.5 * math.sin(c * 0.7 + seed) * math.cos(r * 0.9 - c * 0.2)
            v *= (1 - r / rows * 0.6)
            o += rect(x + c * cw, y + (rows - 1 - r) * rh, cw - 0.5, rh - 0.5, "on3", 0,
                      f'style="fill-opacity:{0.1 + 0.8 * max(0, v):.2f}"')
    return o


# ================================================================== orientation

def speech_macro():
    W, H = 960, 330
    o = ""
    o += text(480, 20, "recognition: audio in, tokens out", "ttl")
    o += _wave(20, 44, 90, 44)
    o += arrow([(114, 66), (134, 66)])
    o += box(136, 42, 92, 48, "front end", "conv", "log-mel")
    o += arrow([(228, 66), (248, 66)])
    o += box(250, 42, 86, 48, "stem", "pool", "subsample")
    o += arrow([(336, 66), (356, 66)])
    o += box(358, 42, 120, 48, "audio encoder", "mix", "frames × D")
    o += arrow([(478, 66), (498, 66)])
    o += box(500, 42, 108, 48, "bridge", "attn", "4 ways in")
    o += arrow([(608, 66), (628, 66)])
    o += box(630, 42, 150, 48, "language model", "mix", "or prediction net")
    o += arrow([(780, 66), (800, 66)])
    o += box(802, 42, 140, 48, "search", "io", "text + times")
    o += text(480, 168, "synthesis: tokens in, audio out", "ttl")
    o += box(20, 190, 120, 48, "prompt rows", "io", "text · voice · conditions")
    o += arrow([(140, 214), (160, 214)])
    o += box(162, 190, 170, 48, "language model", "mix", "over codec codes")
    o += arrow([(332, 214), (352, 214)])
    o += box(354, 190, 140, 48, "code heads", "act", "K codebooks a frame")
    o += arrow([(494, 214), (514, 214)])
    o += box(516, 190, 130, 48, "dequantize", "reg", "codes → latent")
    o += arrow([(646, 214), (666, 214)])
    o += box(668, 190, 150, 48, "codec decoder", "conv", "latent → waveform")
    o += arrow([(818, 214), (838, 214)])
    o += _wave(840, 192, 100, 44, seed=2.2)
    o += f'<path d="M705,90 L705,140 L247,140 L247,190" class="ar dash"/>'
    o += text(476, 132, "the same decoder stack (often the same Qwen3 weights) sits in the middle of both", "xs")
    o += text(480, 280, "a speech model is a language model with a signal path bolted to one end: an encoder before it (ASR),", "sm")
    o += text(480, 298, "or a codec after it (TTS). The chapters split it there.", "sm")
    return svg(W, H, o, "Speech recognition and synthesis pipelines")


RATES = [
    ("waveform", 16000, "16 kHz (24, 44.1 kHz out)", "io"),
    ("log-mel frames", 100, "100 Hz: 10 ms hop", "conv"),
    ("Whisper, Voxtral encoder", 50, "50 Hz: 20 ms", "mix"),
    ("Qwen3-ASR encoder", 13, "13 Hz: 13 per 1 s chunk", "mix"),
    ("Parakeet, Voxtral decoder", 12.5, "12.5 Hz: 80 ms", "attn"),
    ("Qwen3-TTS codes", 12.5, "12.5 Hz: 1920 samples", "act"),
    ("ZONOS3 codes", 15, "15 Hz: 2940 samples", "act"),
]


def frame_rates():
    W, H = 960, 300
    o = ""
    x0, x1 = 230, 940
    span = 0.4
    o += text(x0, 20, "0.4 s of audio on each grid", "ttl", "start")
    for r, (name, hz, note, k) in enumerate(RATES):
        y = 40 + r * 34
        o += text(x0 - 10, y + 6, name, "lbl", "end")
        o += text(x0 - 10, y + 20, note, "xs", "end")
        if hz > 1000:
            o += _wave(x0, y - 2, x1 - x0, 26, n=400)
            continue
        n = int(span * hz)
        step = (x1 - x0) / (span * hz)
        for i in range(n):
            o += rect(x0 + i * step + 1, y, max(step - 2, 1), 22, f"b {k}", 2)
    o += text(480, 290, "every model steps down from samples to a frame rate a transformer can afford: 10 to 50 frames a second", "sm")
    return svg(W, H, o, "Frame rates")


# ================================================================== front ends

def log_mel():
    W, H = 960, 260
    o = ""
    o += _wave(16, 70, 140, 60)
    o += text(86, 150, "16 kHz samples", "xs")
    for i in range(5):
        o += rect(20 + i * 22, 56 + (i % 2) * 6, 50, 6, "on", 2, 'style="fill-opacity:.35"')
    o += text(86, 44, "25 ms windows, 10 ms hop", "xs")
    o += arrow([(160, 100), (186, 100)])
    o += box(188, 72, 92, 56, "|STFT|²", "conv", "201 bins")
    o += arrow([(280, 100), (306, 100)])
    # mel filters
    o += rect(308, 60, 150, 80, "bg2", 4)
    for i in range(9):
        c = 316 + 136 * ((i + 1) / 10) ** 1.6
        wdt = 6 + 22 * ((i + 1) / 10) ** 1.6
        o += f'<path d="M{c-wdt:.1f},132 L{c:.1f},72 L{c+wdt:.1f},132" class="line"/>'
    o += text(383, 152, "mel filterbank (80 or 128)", "xs")
    o += arrow([(458, 100), (484, 100)])
    o += box(486, 72, 70, 56, "log", "act", "log₁₀ / ln")
    o += arrow([(556, 100), (582, 100)])
    o += box(584, 72, 130, 56, "level", "norm", "floor · scale")
    o += arrow([(714, 100), (740, 100)])
    o += _spec(742, 60, 190, 80)
    o += text(837, 152, "frames × mels", "xs")
    o += text(480, 196, "the mel scale spaces filters by perceived pitch: dense at low frequencies, sparse at high", "sm")
    o += text(480, 216, "the log turns loudness into an additive offset; the level step decides what that offset is measured against", "sm")
    return svg(W, H, o, "The log-mel front end")


def level_policies():
    W, H = 960, 250
    o = ""
    panels = [("Whisper", "max − 8, then (x+4)/4", "the clip's (or stream's) loudest", 0),
              ("Voxtral", "fixed level 1.5", "a constant: chunking can't change it", 1),
              ("Qwen3-ASR", "each window's max", "per 30 s window", 2),
              ("Parakeet", "per-mel mean, std", "each channel standardized", 3)]
    for k, (name, rule, note, kind) in enumerate(panels):
        x = 20 + k * 236
        o += text(x + 106, 22, name, "ttl")
        o += f'<path d="M{x},180 L{x+212},180" class="line"/>'
        for i in range(16):
            h = 30 + 80 * abs(math.sin(i * 0.9 + k)) * (0.4 + 0.6 * (i % 5) / 4)
            o += rect(x + 6 + i * 13, 180 - h, 10, h, "b conv", 1)
        if kind in (0, 2):
            o += f'<path d="M{x},62 L{x+212},62" class="ar dash"/>' + text(x + 210, 56, "max", "xs", "end")
            o += f'<path d="M{x},118 L{x+212},118" class="curve2"/>' + text(x + 210, 112, "floor", "xs", "end")
        elif kind == 1:
            o += f'<path d="M{x},80 L{x+212},80" class="curve2"/>' + text(x + 210, 74, "fixed", "xs", "end")
        else:
            o += f'<path d="M{x},130 L{x+212},130" class="curve2"/>' + text(x + 210, 124, "mean ± std", "xs", "end")
            o += f'<path d="M{x},100 L{x+212},100 M{x},160 L{x+212},160" class="ar dash"/>'
        o += text(x + 106, 200, rule, "sm") + text(x + 106, 218, note, "xs")
    o += text(480, 244, "a level that depends on the whole clip is exact offline and approximate on a stream; a fixed one streams exactly", "xs")
    return svg(W, H, o, "Level policies")


# ================================================================== stems

def stems():
    W, H = 960, 330
    o = ""
    rows = [("Whisper", [("conv1d k3", "pool"), ("conv1d k3 s2", "pool")], "100 → 50 Hz", "GELU after each"),
            ("Voxtral", [("causal k3", "pool"), ("causal k3 s2", "pool"), ("stack ×4", "reg")], "100 → 50 → 12.5 Hz",
             "left padding only: no look-ahead"),
            ("Qwen3-ASR", [("conv2d s2", "pool"), ("conv2d s2", "pool"), ("conv2d s2", "pool"), ("linear", "conv")],
             "100 → 13 Hz", "per 1 s chunk: 100 → 50 → 25 → 13"),
            ("Parakeet", [("conv2d s2", "pool"), ("dw s2 + pw", "pool"), ("dw s2 + pw", "pool"), ("linear", "conv")],
             "100 → 12.5 Hz", "FastConformer: depthwise-separable")]
    for r, (name, parts, rate, note) in enumerate(rows):
        y = 30 + r * 74
        o += text(110, y + 22, name, "ttl", "end")
        x = 130
        for i, (lab, k) in enumerate(parts):
            o += box(x, y, 116, 44, lab, k, None, 5, "xs")
            if i < len(parts) - 1:
                o += arrow([(x + 116, y + 22), (x + 136, y + 22)])
            x += 136
        o += text(x + 4, y + 16, rate, "lbl", "start") + text(x + 4, y + 32, note, "xs", "start")
    o += text(480, 322, "2-d stems treat the spectrogram as an image (time × frequency), then flatten the frequency axis into the width", "xs")
    return svg(W, H, o, "Convolutional stems")


def stacking():
    W, H = 900, 180
    o = ""
    o += text(450, 20, "feature stacking: four 20 ms encoder frames become one 80 ms decoder position", "ttl")
    for i in range(16):
        g = i // 4
        o += rect(60 + i * 30 + g * 14, 44, 26, 34, "b mix", 3)
    for g in range(4):
        x = 60 + g * (4 * 30 + 14)
        o += f'<path d="M{x+4},82 L{x+56},116 M{x+112},82 L{x+60},116" class="line"/>'
        o += rect(x + 30, 118, 56, 30, "b attn", 4) + text(x + 58, 133, "4D → D", "xs")
    o += text(700, 70, "concatenate, then an MLP:", "sm", "start")
    o += text(700, 88, "a strided conv with kernel = stride = 4", "xs", "start")
    o += text(700, 120, "Voxtral: 4 × 1280 → 3072 → 3072", "xs", "start")
    return svg(W, H, o, "Feature stacking")


# ================================================================== audio encoders

def encoder_masks():
    W, H = 960, 260
    o = ""
    n, cell = 12, 13
    panels = [("Whisper: one 30 s window", lambda i, j: "on", "1500 frames, every pair; zero-padded"),
              ("Qwen3-ASR: 8 s blocks", lambda i, j: "on" if i // 4 == j // 4 else "off", "104 frames; positions restart per 1 s"),
              ("Voxtral: causal, 15 s window", lambda i, j: "on" if i - 6 < j <= i else "off", "750 frames back; streams"),
              ("Parakeet: whole window", lambda i, j: "on", "relative positions, ≤ 60 s windows")]
    for k, (name, fn, note) in enumerate(panels):
        x = 30 + k * 236
        o += text(x + n * cell / 2, 24, name, "ttl")
        for i in range(n):
            for j in range(n):
                o += rect(x + j * cell, 44 + i * cell, cell - 1, cell - 1, fn(i, j))
        o += text(x + n * cell / 2, 44 + n * cell + 18, note, "xs")
    o += text(480, 244, "the mask decides whether an encoder can stream: only a causal one never waits for audio it hasn't heard", "sm")
    return svg(W, H, o, "Audio encoder masks")


def conformer_block():
    W, H = 960, 300
    o = ""
    y = 150
    o += f'<path d="M30,{y} L900,{y}" class="wire"/>'
    o += arrow([(900, y), (930, y)])
    o += text(36, y - 12, "x", "mt", "start")
    parts = [("½ FFN", "act", "SiLU, 4D"), ("MHSA", "attn", "rel. positions"), ("conv module", "pool", "GLU · dw k9 · BN"),
             ("½ FFN", "act", "SiLU, 4D")]
    for i, (lab, k, sub) in enumerate(parts):
        x = 70 + i * 190
        o += f'<path d="M{x},{y} L{x},{y-30}" class="ar"/>'
        o += box(x - 18, y - 92, 36, 18, "LN", "norm", None, 3, "xs")
        o += f'<path d="M{x},{y-30} L{x},{y-74}" class="ar"/>'
        o += box(x + 20, y - 120, 128, 52, lab, k, sub)
        o += arrow([(x + 148, y - 94), (x + 162, y - 94), (x + 162, y - 12)])
        o += add(x + 162, y)
    o += box(834, y - 22, 50, 44, "LN", "norm")
    o += text(470, 196, "the macaron: two half-step feed-forwards sandwich attention and a convolution, each a pre-norm residual", "sm")
    o += text(470, 214, "x += ½ ff₁(x) · x += attn(x) · x += conv(x) · x += ½ ff₂(x) · y = LN(x)", "mt")
    o += text(470, 248, "attention mixes globally by content; the depthwise conv mixes locally by offset; the block gets both", "sm")
    o += text(470, 268, "(Parakeet: 24 blocks, D = 1024, 8 heads, kernel 9; no biases)", "xs")
    return svg(W, H, o, "The Conformer block")


def conv_module():
    W, H = 900, 150
    o = ""
    parts = [("pointwise", "conv", "D → 2D"), ("GLU", "act", "2D → D"), ("depthwise", "pool", "k = 9, per channel"),
             ("BatchNorm", "norm", None), ("SiLU", "act", None), ("pointwise", "conv", "D → D")]
    x = 20
    for i, (lab, k, sub) in enumerate(parts):
        o += box(x, 40, 128, 50, lab, k, sub)
        if i < len(parts) - 1:
            o += arrow([(x + 128, 65), (x + 148, 65)])
        x += 148
    o += text(450, 120, "a gated, depthwise 1-d convolution over time: each channel gets its own 9-frame filter (720 ms at 80 ms frames)", "xs")
    return svg(W, H, o, "The Conformer convolution module")


def relpos():
    W, H = 920, 260
    o = ""
    n, cell = 7, 22
    x0, y0 = 60, 50
    o += text(x0 + n * cell / 2, 30, "offset i − j, read from one table", "ttl")
    for i in range(n):
        for j in range(n):
            d = i - j
            k = "on" if d == 0 else ("on2" if d > 0 else "on3")
            op = 0.2 + 0.6 * (1 - abs(d) / n)
            o += rect(x0 + j * cell, y0 + i * cell, cell - 1, cell - 1, k, 0, f'style="fill-opacity:{op:.2f}"')
            o += text(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2 + 1, str(d), "xs")
    o += text(x0 + n * cell / 2, y0 + n * cell + 18, "each diagonal shares one position vector", "xs")
    o += text(560, 50, "Transformer-XL scores, per head:", "sm")
    o += text(560, 80, "(qᵢ + u) · kⱼ  +  (qᵢ + v) · Wₚ p(i − j)", "mt")
    o += text(560, 112, "content term: does key j match what query i wants?", "xs")
    o += text(560, 128, "position term: does offset i − j matter to this head?", "xs")
    o += text(560, 144, "u, v: learned global biases (content and position)", "xs")
    o += text(560, 180, "p is a sinusoidal table over 2T − 1 offsets; one rel-shift", "xs")
    o += text(560, 196, "(pad, reshape, slice) turns the T × (2T − 1) product into T × T", "xs")
    o += text(560, 230, "unlike RoPE, the position enters as its own additive term", "sm")
    return svg(W, H, o, "Relative positional attention")


# ================================================================== bridges

def bridges():
    W, H = 960, 330
    o = ""
    def frames(x, y, n=5, k="mix", w=22):
        s = ""
        for i in range(n):
            s += rect(x + i * (w + 4), y, w, 18, f"b {k}", 3)
        return s
    def toks(x, y, n=5, k="io", w=22, lab=None):
        s = ""
        for i in range(n):
            s += rect(x + i * (w + 4), y, w, 18, f"b {k}", 3)
        return s
    # 1 cross-attention
    x = 20
    o += text(x + 100, 22, "① cross-attention", "ttl") + text(x + 100, 38, "Whisper", "xs")
    o += frames(x + 30, 60)
    o += toks(x + 30, 160)
    for i in range(5):
        for j in range(5):
            o += f'<path d="M{x+41+j*26},{78} L{x+41+i*26},{160}" class="line" style="opacity:.35"/>'
    o += text(x + 100, 200, "every decoder layer reads", "xs") + text(x + 100, 214, "all encoder frames", "xs")
    # 2 splice
    x = 260
    o += text(x + 100, 22, "② splice", "ttl") + text(x + 100, 38, "Qwen3-ASR", "xs")
    o += frames(x + 70, 60, 3)
    o += toks(x + 18, 160, 2) + rect(x + 70, 160, 74, 18, "b mix", 3) + toks(x + 148, 160, 2)
    o += text(x + 107, 169, "frames", "xs")
    o += arrow([(x + 107, 80), (x + 107, 158)], hi=True)
    o += text(x + 100, 200, "projected frames replace", "xs") + text(x + 100, 214, "<|audio_pad|> tokens", "xs")
    # 3 add
    x = 500
    o += text(x + 100, 22, "③ add, with a delay", "ttl") + text(x + 100, 38, "Voxtral Realtime", "xs")
    o += frames(x + 40, 60)
    for i in range(5):
        o += rect(x + 40 + i * 26, 160, 22, 18, "b io", 3)
        o += arrow([(x + 51 + i * 26, 80), (x + 51 + i * 26, 158)], hi=(i == 2))
    o += text(x + 100, 200, "position i's input is", "xs") + text(x + 100, 214, "embed(token) + audio(i)", "xs")
    # 4 joint
    x = 740
    o += text(x + 100, 22, "④ joint", "ttl") + text(x + 100, 38, "Parakeet TDT", "xs")
    o += frames(x + 40, 60)
    o += box(x + 40, 150, 126, 30, "prediction net", "conv", None, 4, "xs")
    o += add(x + 103, 118, 9)
    o += arrow([(x + 103, 80), (x + 103, 108)], hi=True) + arrow([(x + 103, 150), (x + 103, 128)])
    o += text(x + 100, 200, "one frame + the text so far,", "xs") + text(x + 100, 214, "added, scored", "xs")
    o += text(480, 260, "how much of the audio each output step sees: all of it (①, ②), the frames up to now (③), or one frame (④)", "sm")
    o += text(480, 280, "①: a separate decoder layer kind · ②: no new parts, only a prompt · ③: no new parts, plus a delay · ④: no attention at all", "xs")
    return svg(W, H, o, "Four bridges from audio to text")


def delay_stream():
    W, H = 960, 220
    o = ""
    x0, cw = 80, 40
    o += text(40, 52, "audio", "xs", "end") + text(40, 112, "output", "xs", "end")
    words = {8: "Hel", 9: "lo", 10: "[W]", 11: "world"}
    for i in range(20):
        x = x0 + i * cw
        o += rect(x, 40, cw - 4, 24, "b mix" if i < 14 else "cell", 3)
        if i < 6:
            lab, k = "PAD", "io"
        else:
            lab, k = ({6: "PAD", 7: "[W]", 8: "Hel", 9: "lo", 10: "[W]", 11: "world", 12: "PAD"}.get(i, "PAD"), "act")
        o += rect(x, 100, cw - 4, 24, f"b {k}", 3) + text(x + cw / 2 - 2, 112, lab, "xs")
        o += arrow([(x + cw / 2 - 2, 64), (x + cw / 2 - 2, 98)])
    o += f'<path d="M{x0+1*cw},34 L{x0+1*cw},28 L{x0+7*cw},28 L{x0+7*cw},34" class="line"/>'
    o += text(x0 + 4 * cw, 20, "delay: 6 positions = 480 ms", "xs")
    o += text(480, 160, "one 80 ms audio group per position, added to that position's token; the text runs a fixed delay behind the speech,", "sm")
    o += text(480, 178, "so the model has heard the word before it must write it. [W] marks a word start; PAD means \"nothing new yet\"", "sm")
    o += text(480, 204, "each layer is also told the delay: a sinusoidal embedding of it scales the feed-forward's normalized input (1 + f(delay))", "xs")
    return svg(W, H, o, "Voxtral's delayed stream")


# ================================================================== transducers

def lattice():
    W, H = 960, 300
    o = ""
    T, U, c = 9, 5, 30
    for p, (title, jumps) in enumerate([("RNN-T: blank moves one frame", False), ("TDT: a duration moves several", True)]):
        x0, y0 = 60 + p * 470, 50
        o += text(x0 + T * c / 2, 24, title, "ttl")
        for t in range(T):
            for u in range(U):
                o += f'<circle cx="{x0+t*c}" cy="{y0+(U-1-u)*c}" r="3" class="dot" style="opacity:.3"/>'
        o += text(x0 + T * c / 2, y0 + U * c + 8, "encoder frames t →", "xs")
        o += text(x0 - 14, y0 + (U - 1) * c / 2, "tokens", "xs", "end")
        if not jumps:
            path = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (4, 2), (5, 2), (5, 3), (6, 3), (7, 3), (7, 4), (8, 4)]
        else:
            path = [(0, 0), (1, 0), (1, 1), (3, 1), (3, 2), (5, 2), (5, 3), (7, 3), (7, 4), (8, 4)]
        for (a, b), (cc, d) in zip(path, path[1:]):
            hi = d > b
            o += arrow([(x0 + a * c, y0 + (U - 1 - b) * c), (x0 + cc * c, y0 + (U - 1 - d) * c)], hi=hi)
        o += text(x0 + T * c + 10, y0 + 10, "↑ emit a token", "xs", "start")
        o += text(x0 + T * c + 10, y0 + 26, "→ " + ("blank: next frame" if not jumps else "skip d frames"), "xs", "start")
    o += text(480, 236, "a transducer walks a frames × tokens grid; each step scores (frame t, tokens so far) through the joint", "sm")
    o += text(480, 254, "TDT also predicts how many frames the current token lasts, so a greedy decode skips the frames in between", "sm")
    o += text(480, 274, "(Parakeet: durations {0, 1, 2, 3, 4}; a 0 lets several tokens share a frame, up to 10)", "xs")
    return svg(W, H, o, "Transducer lattices")


def joint():
    W, H = 900, 220
    o = ""
    o += box(20, 30, 150, 46, "encoder frame eₜ", "mix", "1024")
    o += arrow([(170, 53), (196, 53)])
    o += box(198, 30, 100, 46, "W_enc", "conv", "→ 640")
    o += box(20, 120, 150, 46, "token yᵤ₋₁", "io", "or blank at start")
    o += arrow([(170, 143), (196, 143)])
    o += box(198, 120, 100, 46, "embed + LSTM ×2", "conv", "640, own state", 6, "xs")
    o += arrow([(298, 143), (316, 143)])
    o += box(318, 120, 80, 46, "W_pred", "conv", "→ 640")
    o += arrow([(298, 53), (440, 53), (440, 88)]) + arrow([(398, 143), (440, 143), (440, 110)])
    o += add(440, 99, 10)
    o += arrow([(450, 99), (476, 99)])
    o += box(478, 78, 70, 42, "ReLU", "act")
    o += arrow([(548, 99), (572, 99)])
    o += box(574, 70, 140, 58, "linear", "conv", "→ 8193 + 5")
    o += arrow([(714, 86), (740, 70)]) + arrow([(714, 112), (740, 128)])
    o += box(742, 48, 140, 38, "token (or blank)", "act", None, 4, "xs")
    o += box(742, 112, 140, 38, "duration 0–4", "pool", None, 4, "xs")
    o += text(450, 196, "the prediction net is a small language model of the transcript so far; the joint adds its state to one audio frame's", "xs")
    return svg(W, H, o, "The TDT joint")


# ================================================================== codecs

def rvq():
    W, H = 920, 280
    o = ""
    cx, cy = 40, 190
    o += text(150, 22, "residual vector quantization", "ttl")
    steps = [(140, -100), (64, 52), (-34, -26), (16, 12)]
    x, y = cx, cy
    for i, (dx, dy) in enumerate(steps):
        mk = 'marker-end="url(#ahh)"' if i == 0 else 'marker-end="url(#ah)"'
        o += f'<path d="M{x},{y} L{x+dx},{y+dy}" class="{"ar hi" if i == 0 else "ar"}" {mk}/>'
        lx, ly = x + dx / 2, y + dy / 2
        if i < 2:
            o += text(lx + (-10 if i == 0 else 14), ly + (-6 if i == 0 else 4), f"c{'₀₁'[i]}", "xs")
        x, y = x + dx, y + dy
    o += f'<circle cx="{x}" cy="{y}" r="5" class="on"/>' + text(x + 10, y + 14, "latent z", "xs", "start")
    o += text(150, 236, "each codebook quantizes what the previous left over:", "xs")
    o += text(150, 250, "z ≈ c₀ + c₁ + c₂ + …, coarse to fine", "xs")
    o += text(600, 22, "decoding codes back to a latent", "ttl")
    o += text(330, 64, "Qwen3-TTS (split RVQ, 16 × 2048):", "sm", "start")
    o += rect(330, 76, 80, 26, "b attn", 3) + text(370, 89, "cb 0", "xs")
    o += rect(418, 76, 200, 26, "b reg", 3) + text(518, 89, "cb 1–15: Σ, one projection", "xs")
    o += text(640, 89, "→ added → 512-d latent", "xs", "start")
    o += text(330, 128, "first codebook: semantic (Mimi-style); the rest acoustic residuals", "xs", "start")
    o += text(330, 170, "ZONOS3 (10 × 4096, 8-d codes):", "sm", "start")
    o += rect(330, 182, 150, 26, "b attn", 3) + text(405, 195, "cb 0 → 4 ConvNeXt", "xs")
    o += rect(488, 182, 130, 26, "b reg", 3) + text(553, 195, "cb 1–9: 8 → 1024 each", "xs")
    o += text(640, 195, "→ Σ → 1024-d latent", "xs", "start")
    o += text(330, 234, "semantic codebook gets its own small network before the sum", "xs", "start")
    return svg(W, H, o, "Residual vector quantization")


def codec_decoders():
    W, H = 960, 300
    o = ""
    o += text(30, 26, "Qwen3-TTS: 12.5 Hz → 24 kHz (×1920)", "ttl", "start")
    parts = [("dequantize", "reg", "512"), ("causal conv", "pool", "→ 1024"), ("transformer", "mix", "8 layers, window 72"),
             ("↑2 ×2", "pool", "conv-T + ConvNeXt"), ("DAC decoder", "conv", "↑8·5·4·3, SnakeBeta")]
    x = 30
    for i, (a, k, b) in enumerate(parts):
        o += box(x, 40, 160, 50, a, k, b)
        if i < len(parts) - 1:
            o += arrow([(x + 160, 65), (x + 178, 65)])
        x += 180
    o += text(30, 136, "ZONOS3: 15 Hz → 44.1 kHz (×2940)", "ttl", "start")
    parts = [("dequantize", "reg", "1024"), ("5 attention stages", "mix", "patch-up ×2 between"), ("ConvNeXt ×8", "pool", "at 240 Hz"),
             ("2 ISTFT heads", "act", "n_fft 736 & 2208"), ("mix + resample", "io", "44160 → 44100")]
    x = 30
    for i, (a, k, b) in enumerate(parts):
        o += box(x, 150, 160, 50, a, k, b)
        if i < len(parts) - 1:
            o += arrow([(x + 160, 175), (x + 178, 175)])
        x += 180
    o += text(480, 238, "two ways to the waveform: upsample in the time domain with transposed convs and periodic activations (DAC),", "sm")
    o += text(480, 256, "or predict a spectrum per frame and invert it (ISTFT, as Vocos does), leaving the waveform's phase to the overlap-add", "sm")
    o += text(480, 280, "every layer up to the ISTFT is causal, so both stream with state; the ISTFT and resampler add a fixed lag", "xs")
    return svg(W, H, o, "Two codec decoders")


def snake():
    W, H = 560, 260
    o = ""
    x0, y0, sx, sy = 280, 130, 40, 18
    o += f'<path d="M20,{y0} L540,{y0} M{x0},16 L{x0},226" class="grid"/>'
    for a, cls in [(1.0, "curve2"), (3.0, "curve")]:
        pts = []
        for i in range(-240, 241, 2):
            xx = i / 40
            v = xx + (1 / a) * math.sin(a * xx) ** 2
            pts.append(f"{x0 + xx * sx:.1f},{y0 - v * sy:.1f}")
        o += f'<path d="M{" L".join(pts)}" class="{cls}"/>'
    o += text(x0 + 6, 22, "x + sin²(αx) / β", "mt", "start")
    o += f'<path d="M440,196 L462,196" class="curve2"/>' + text(468, 196, "α = 1", "xs", "start")
    o += f'<path d="M440,212 L462,212" class="curve"/>' + text(468, 212, "α = 3", "xs", "start")
    o += text(280, 248, "a monotone trend plus a learned ripple: a periodic inductive bias for waveforms", "xs")
    return svg(W, H, o, "The Snake activation")


def patch_up():
    W, H = 900, 180
    o = ""
    o += text(450, 20, "patch-up: [T, 2C] → [2T, C], a parameter-free 1-d pixel shuffle", "ttl")
    for t in range(3):
        for c in range(4):
            k = "on2" if c % 2 == 0 else "on"
            o += rect(80 + t * 70, 44 + c * 22, 60, 20, k, 2, 'style="fill-opacity:.55"')
            o += text(110 + t * 70, 54 + c * 22, f"t{t} c{c}", "xs")
    o += arrow([(300, 88), (360, 88)], hi=True)
    for t in range(6):
        for c in range(2):
            src_c = c * 2 + (t % 2)
            k = "on2" if src_c % 2 == 0 else "on"
            o += rect(380 + t * 70, 66 + c * 22, 60, 20, k, 2, 'style="fill-opacity:.55"')
            o += text(410 + t * 70, 76 + c * 22, f"t{t//2} c{src_c}", "xs")
    o += text(450, 150, "channel 2c + j at time t moves to channel c at time 2t + j: width traded for rate, ×16 over ZONOS3's four patch-ups", "xs")
    return svg(W, H, o, "Patch-up")


# ================================================================== language models over codes

def code_layouts():
    W, H = 960, 300
    o = ""
    K, F = 4, 5
    # flat
    o += text(160, 22, "flatten: K steps per frame", "ttl")
    for i in range(K * F):
        f, k = divmod(i, K)
        o += rect(20 + i * 14, 50, 12, 20, f"b {'attn' if k == 0 else 'act'}", 2)
    o += text(160, 92, "exact, but K× the sequence length", "xs")
    # delay pattern
    o += text(480, 22, "delay pattern: one row, all codebooks", "ttl")
    for r in range(F + K - 1):
        for k in range(K):
            f = r - k
            cls = "cell" if f < 0 or f >= F else f"b {'attn' if k == 0 else 'act'}"
            o += rect(400 + r * 22, 44 + k * 22, 20, 20, cls, 2)
            if 0 <= f < F:
                o += text(410 + r * 22, 54 + k * 22, str(f), "xs")
    o += text(480, 148, "codebook k of frame f sits in row f + k: when cb k", "xs")
    o += text(480, 162, "is sampled, cb 0…k−1 of its frame are already known", "xs")
    o += text(480, 178, "(ZONOS3, as MusicGen)", "xs")
    # depth
    o += text(810, 22, "depth decoder: a small LM per frame", "ttl")
    for f in range(F):
        o += rect(700 + f * 46, 120, 40, 22, "b attn", 3) + text(720 + f * 46, 131, f"f{f}", "xs")
    o += arrow([(812, 118), (812, 104)])
    for k in range(K):
        o += rect(780 + k * 18, 44 + (K - 1 - k) * 0, 16, 56, f"b {'attn' if k == 0 else 'act'}", 2)
    o += text(812, 160, "the main LM steps once per frame (cb 0);", "xs")
    o += text(812, 174, "a 5-layer decoder fills cb 1…15 within it", "xs")
    o += text(812, 190, "(Qwen3-TTS, as RQ-Transformer, Moshi)", "xs")
    o += text(480, 236, "the problem: a frame is K tokens that depend on one another; each layout factors p(frame) differently", "sm")
    o += text(480, 256, "flatten: fully autoregressive · delay: across codebooks by one row of lag · depth: exact within a frame, cheaply", "xs")
    return svg(W, H, o, "Laying out multi-codebook tokens")


def row_embed():
    W, H = 900, 170
    o = ""
    o += text(450, 20, "a row is a sum: one table per column", "ttl")
    cols = [("cb 0", "attn"), ("cb 1", "act"), ("…", "act"), ("cb K−1", "act"), ("text", "io")]
    for i, (lab, k) in enumerate(cols):
        x = 60 + i * 120
        o += rect(x, 40, 90, 24, f"b {k}", 3) + text(x + 45, 52, lab, "xs")
        o += arrow([(x + 45, 64), (x + 45, 84)])
        o += box(x, 86, 90, 26, "table", "conv", None, 3, "xs")
        o += f'<path d="M{x+45},112 L{x+45},130 L700,130" class="line"/>'
    o += add(712, 130, 10)
    o += arrow([(722, 130), (760, 130)])
    o += text(790, 130, "row", "lbl")
    o += text(450, 160, "speaker and condition rows replace the sum with a projected embedding or a bucket's learnt row", "xs")
    return svg(W, H, o, "Summed row embeddings")


# ================================================================== routes & experts

def attnres():
    W, H = 960, 300
    o = ""
    o += text(230, 22, "residual stream: each layer reads the sum", "ttl")
    y = 130
    o += f'<path d="M30,{y} L430,{y}" class="wire" style="stroke-width:5;opacity:.3"/>'
    for i in range(4):
        x = 70 + i * 92
        o += box(x, 60, 60, 30, f"f{i+1}", "mix", None, 4, "xs")
        o += arrow([(x + 10, y), (x + 10, 92)]) + arrow([(x + 50, 92), (x + 50, y - 10)], hi=True)
        o += add(x + 50, y, 8)
    o += text(230, 170, "xₗ = x₀ + f₁ + … + fₗ: fixed, equal weights", "xs")
    o += text(700, 22, "attention residuals: each sublayer picks its mix", "ttl")
    src = ["x₀", "a₁", "f₁", "a₂", "f₂"]
    for i, s in enumerate(src):
        o += rect(510 + i * 56, 50, 46, 26, "b io", 3) + text(533 + i * 56, 63, s, "xs")
    o += text(500, 63, "bank", "xs", "end")
    ws = [0.1, 0.05, 0.45, 0.1, 0.3]
    for i, w in enumerate(ws):
        o += f'<path d="M{533+i*56},76 L700,150" class="line" style="stroke-width:{0.6+5*w:.1f};stroke:var(--k-hi);opacity:.7"/>'
    o += box(640, 150, 120, 34, "softmax route", "attn", None, 4, "xs")
    o += text(776, 162, "a query per sublayer", "xs", "start")
    o += text(776, 176, "keys: normed sources", "xs", "start")
    o += arrow([(700, 184), (700, 204)])
    o += box(650, 206, 100, 30, "sublayer s", "mix", None, 4, "xs")
    o += f'<path d="M750,221 L880,221 L880,40 L800,40 L800,50" class="ar dash"/>'
    o += text(890, 120, "append", "xs", "start")
    o += text(480, 268, "ZONOS3: 23 learned queries (2 per layer + 1 for the output); nothing is ever added to a stream, every read is a convex mix", "sm")
    o += text(480, 286, "per position, so it stays causal and needs no cache of its own", "xs")
    return svg(W, H, o, "Attention residuals")


def latent_moe():
    W, H = 960, 250
    o = ""
    o += box(20, 96, 60, 40, "x", "io", "1024")
    o += arrow([(80, 116), (110, 116)])
    o += box(112, 40, 130, 40, "router", "attn", "RMS → softmax")
    o += f'<path d="M96,116 L96,60 L110,60" class="line"/>'
    o += box(112, 96, 130, 40, "latent_down", "conv", "1024 → 256")
    o += arrow([(242, 116), (276, 116)])
    for e in range(8):
        k = "act" if e in (2, 5) else "cell"
        o += rect(280 + e * 44, 80 + (e % 2) * 10, 38, 56, f"b {k}" if k == "act" else "cell", 3)
        o += text(299 + e * 44, 108 + (e % 2) * 10, f"E{e}" if e < 7 else "…", "xs")
    o += arrow([(242, 60), (380, 60), (380, 78)], hi=True) + arrow([(380, 60), (512, 60), (512, 88)], hi=True)
    o += text(450, 50, "top-2 of 64 (bias only for choosing)", "xs")
    o += arrow([(632, 116), (660, 116)])
    o += box(662, 96, 130, 40, "Σ weights · out", "mix", None, 4, "xs")
    o += arrow([(792, 116), (812, 116)])
    o += box(814, 96, 130, 40, "latent_up", "conv", "256 → 1024")
    o += text(480, 184, "each token runs 2 of 64 small SwiGLU experts, in a 256-wide latent instead of the 1024-wide stream:", "sm")
    o += text(480, 202, "the model holds 800M parameters but runs about 100M per token", "sm")
    o += text(480, 226, "balancing without an auxiliary loss: a per-expert bias steers selection, but not the mixture weights", "xs")
    return svg(W, H, o, "The latent mixture of experts")


# ================================================================== streaming

def windows_commit():
    W, H = 960, 220
    o = ""
    x0, sc = 40, 26
    o += _wave(x0, 30, 880, 40, n=600, seed=0.4)
    for w, (s, e, cut) in enumerate([(0, 30, 24), (24, 54, 49), (49, 79, 79)]):
        y = 90 + w * 32
        o += rect(x0 + s * 11, y, (e - s) * 11, 22, "b mix", 3, 'style="fill-opacity:.4"')
        o += rect(x0 + s * 11, y, (cut - s) * 11, 22, "b act", 3)
        o += text(x0 + s * 11 + 6, y + 11, f"window {w+1}: commit to the pause", "xs", "start")
        if cut < e:
            o += f'<path d="M{x0+cut*11},{y-4} L{x0+cut*11},{y+26}" class="curve2"/>'
    o += text(480, 204, "a full window commits the words before a pause in its last third, then decodes again from the cut with fresh audio", "xs")
    return svg(W, H, o, "Windowed streaming")
