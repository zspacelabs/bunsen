"""The speech circuit board: four rows of sockets from audio in to audio out.

Recognizers fill the top rows (audio in, bridge, tokens); synthesizers fill the bottom rows (tokens, audio out),
and a voice-cloning synthesizer also uses audio in for its reference clip.
"""
from .svgkit import arrow, esc, rect, svg, text

KIND = dict(fe="conv", stem="pool", enc="mix", enc_pos="pool", enc_mask="io",
            proj="conv", bridge="attn", cond="norm",
            rows="io", lm="mix", lm_ext="mix", heads="act", search="io",
            quant="reg", trunk="mix", upsample="pool", vocoder="conv")
LINK = dict(fe="s-frontend.html", stem="s-stem.html", enc="s-encoder.html", enc_pos="s-encoder.html#positions",
            enc_mask="s-encoder.html#masks", proj="s-bridge.html", bridge="s-bridge.html", cond="s-bridge.html#delay",
            rows="s-codes.html#rows", lm="algebra.html#board", lm_ext="s-codes.html#depth", heads="s-codes.html",
            search="c-cache.html#search", quant="s-codec.html#rvq", trunk="s-codec.html#decoders",
            upsample="s-codec.html#decoders", vocoder="s-codec.html#vocoders")
NAME = dict(fe="FRONT END", stem="STEM", enc="ENCODER LAYER × N", enc_pos="POSITIONS", enc_mask="MASK",
            proj="PROJECTOR", bridge="BRIDGE", cond="CONDITIONING",
            rows="INPUT ROWS", lm="LANGUAGE MODEL", lm_ext="SECOND DECODER", heads="HEADS", search="SEARCH",
            quant="DEQUANTIZE", trunk="TRUNK", upsample="UPSAMPLE", vocoder="TO WAVEFORM")
CAPTION = dict(fe="front end", stem="stem", enc="encoder layer", enc_pos="encoder positions", enc_mask="encoder mask",
               proj="projector", bridge="bridge", cond="conditioning", rows="input rows", lm="language model",
               lm_ext="second decoder", heads="heads", search="search", quant="dequantizer", trunk="codec trunk",
               upsample="upsampler", vocoder="waveform stage")
ROWS = [
    ("AUDIO IN", "signal → frames", [("fe", 168), ("stem", 160), ("enc", 222), ("enc_pos", 140), ("enc_mask", 150)]),
    ("BRIDGE", "frames meet tokens", [("proj", 220), ("bridge", 330), ("cond", 220)]),
    ("TOKENS", "the language model", [("rows", 160), ("lm", 250), ("lm_ext", 152), ("heads", 150), ("search", 128)]),
    ("AUDIO OUT", "codes → waveform", [("quant", 190), ("trunk", 230), ("upsample", 196), ("vocoder", 224)]),
]


def _chip(x, y, w, h, slot, val, changed):
    o = text(x + 2, y - 8, NAME[slot], "caps", "start")
    if val is None:
        o += rect(x, y, w, h, "slot", 6) + text(x + w / 2, y + h - 10, "empty", "xs")
    else:
        title, params = val[0], val[1]
        link = val[2] if len(val) > 2 and val[2] else LINK[slot]
        kind = val[3] if len(val) > 3 else KIND[slot]
        inner = rect(x, y, w, h, f"b {kind}", 6) + text(x + w / 2, y + 15, title, "ct")
        for i, p in enumerate(params):
            inner += text(x + w / 2, y + 29 + i * 12.5, p, "xs")
        o += f'<a href="{link}"><title>{esc(CAPTION[slot])}: {esc(title)}</title>{inner}</a>'
    if changed:
        o += rect(x - 3, y - 3, w + 6, h + 6, "chg", 8) + text(x + w + 1, y - 8, "Δ", "chgt", "end")
    return o


def circuit_sp(spec, changed):
    W, H = 1040, 486
    o = ""
    h, pitch, top = 58, 110, 34
    for r, (label, sub, slots) in enumerate(ROWS):
        y = top + r * pitch
        my = y + h / 2
        used = any(spec.get(s) is not None for s, _ in slots)
        o += text(16, my - 6, label, "caps", "start") + text(16, my + 8, sub, "xs", "start")
        o += f'<path d="M140,{my} L1028,{my}" class="wire"{"" if used else ' style="opacity:.25"'}/>'
        gap = (888 - sum(w for _, w in slots)) / (len(slots) - 1)
        x = 140
        for slot, w in slots:
            o += _chip(x, y, w, h, slot, spec.get(slot), slot in changed)
            x += w + gap
        if r < len(ROWS) - 1:
            gy = y + h + 22
            o += f'<path d="M1028,{my} L1028,{gy} L132,{gy} L132,{my + pitch} L140,{my + pitch}" class="ar dash"/>'
    note = spec.get("note", "")
    o += text(16, 476, note, "xs", "start")
    return svg(W, H, o, f"Circuit of {spec.get('name', 'a speech model')}")
