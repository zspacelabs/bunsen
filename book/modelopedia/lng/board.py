"""The language-model circuit board: the generic decoder with each family's components injected.

A spec maps socket -> (title, [params], link?, kind?) or None (an empty socket), plus:
  name     the variant's name
  layers   the pill: norm kind · activation · biases
  pattern  the stack's layer kinds, one letter per layer: F full, W windowed, D DeltaNet
  tied     True when the head is the embedding's table
  block    the board's title
  note     a footnote
"""
from .svgkit import add, esc, rect, svg, text

KIND = dict(embed="io", stream="act", stack="mix", final="norm", head="io",
            a_pre="norm", a_mix="attn", a_post="norm",
            qk="norm", pos="pool", score="attn", mask="io", vres="conv", gate="act",
            m_pre="norm", m_mlp="act", m_post="norm")
LINK = dict(embed="c-embed.html", stream="c-stream.html", stack="c-mask.html#patterns", final="c-sublayer.html#final",
            head="c-embed.html#head", a_pre="c-sublayer.html#placement", a_mix="c-attention.html",
            a_post="c-sublayer.html#sandwich", qk="c-logits.html#qknorm", pos="c-rope.html", score="c-logits.html",
            mask="c-mask.html", vres="c-stream.html#values", gate="c-attention.html#gate",
            m_pre="c-sublayer.html#placement", m_mlp="c-mlp.html", m_post="c-sublayer.html#sandwich")
NAME = dict(embed="EMBED", stream="STREAM EXTRAS", stack="STACK", final="FINAL NORM", head="HEAD",
            a_pre="PRE", a_mix="TOKEN MIXER", a_post="POST", qk="Q/K NORM", pos="POSITIONS", score="SCORES",
            mask="MASK · WINDOW", vres="VALUES", gate="OUTPUT GATE", m_pre="PRE", m_mlp="FEED-FORWARD", m_post="POST")
CAPTION = dict(NAME, a_pre="mixer pre-norm", a_mix="token mixer", a_post="mixer post-norm", qk="Q/K norm",
               pos="positions", score="scores", mask="mask and window", vres="values", gate="output gate",
               m_pre="feed-forward pre-norm", m_mlp="feed-forward", m_post="feed-forward post-norm",
               stream="stream extras", stack="stack", final="final norm", embed="embedding", head="head")
PAT_KIND = dict(F="attn", W="norm", D="act")
PAT_NAME = dict(F="full / global attention", W="windowed attention", D="gated DeltaNet")


def _chip(x, y, w, h, slot, val, changed):
    o = text(x + 2, y - 8, NAME[slot], "caps", "start")
    if val is None:
        o += rect(x, y, w, h, "slot", 6)
        o += text(x + w / 2, y + h - 10, "empty", "xs")
    else:
        title, params = val[0], val[1]
        link = val[2] if len(val) > 2 and val[2] else LINK[slot]
        kind = val[3] if len(val) > 3 else KIND[slot]
        step = 12.5 if h >= 56 else 10.5
        top = 15 if h >= 56 else 12
        inner = rect(x, y, w, h, f"b {kind}", 6) + text(x + w / 2, y + top, title, "ct")
        for i, p in enumerate(params):
            inner += text(x + w / 2, y + top + 14 + i * step, p, "xs")
        o += f'<a href="{link}"><title>{esc(CAPTION[slot])}: {esc(title)}</title>{inner}</a>'
    if changed:
        o += rect(x - 3, y - 3, w + 6, h + 6, "chg", 8)
        o += text(x + w + 1, y - 8, "Δ", "chgt", "end")
    return o


def _pattern(x, y, w, h, pat, changed):
    o = rect(x, y, w, h, "pill", 8)
    o += text(x + 10, y + 14, "LAYER PATTERN", "caps", "start")
    o += text(x + w - 10, y + 14, f"{len(pat)} layers", "xs", "end")
    per = 18
    cw = (w - 20) / per
    for i, c in enumerate(pat):
        r, k = divmod(i, per)
        cx, cy = x + 10 + k * cw, y + 26 + r * 19
        o += rect(cx, cy, cw - 3, 15, f"b {PAT_KIND[c]}", 2)
    kinds = [c for c in "FWD" if c in pat]
    for j, c in enumerate(kinds):
        ly = y + h - 14 - (len(kinds) - 1 - j) * 14
        o += rect(x + 10, ly - 5, 10, 10, f"b {PAT_KIND[c]}", 2) + text(x + 26, ly, PAT_NAME[c], "xs", "start")
    if changed:
        o += rect(x - 3, y - 3, w + 6, h + 6, "chg", 10)
        o += text(x + w + 1, y - 6, "Δ", "chgt", "end")
    return o


def circuit_lm(spec, changed):
    W, H = 1040, 584
    o = ""
    my, mh = 30, 62
    wy = my + mh / 2
    o += f'<path d="M6,{wy} L1034,{wy}" class="wire"/>'
    macro = [("embed", 190), ("stream", 160), ("stack", 250), ("final", 120), ("head", 232)]
    x, pos = 14, {}
    for slot, w in macro:
        pos[slot] = (x, w)
        o += _chip(x, my, w, mh, slot, spec.get(slot), slot in changed)
        x += w + 8
    if spec.get("tied"):
        ex, ew = pos["embed"]
        hx, hw = pos["head"]
        a, b = ex + ew * 0.7, hx + hw * 0.5
        o += f'<path d="M{a},{my-4} C{a},{my-26} {b},{my-26} {b},{my-4}" class="ar dash"/>'
        o += rect((a + b) / 2 - 70, my - 26, 140, 14, "pill", 7) + text((a + b) / 2, my - 19, "tied: head = embedᵀ", "xs")
        if "tied" in changed:
            o += rect((a + b) / 2 - 73, my - 29, 146, 20, "chg", 9)
    elif "tied" in changed:
        hx, hw = pos["head"]
        o += text(hx + hw / 2, my - 19, "untied: a head of its own", "xs")
    sx, sw = pos["stack"]
    o += f'<path d="M{sx},{my+mh} L20,134" class="ar dash"/><path d="M{sx+sw},{my+mh} L1020,134" class="ar dash"/>'
    o += rect(20, 134, 1000, 422, "board", 10)
    o += text(36, 152, spec.get("block", "DECODER LAYER  ×  L   (two residual sublayers on one stream)"), "caps", "start")
    lay = spec.get("layers", "")
    lw = len(lay) * 6.1 + 66
    o += rect(1006 - lw, 142, lw, 20, "pill", 10) + text(1006 - lw / 2, 152, f"layers: {lay}", "xs")
    if "layers" in changed:
        o += rect(1003 - lw, 139, lw + 6, 26, "chg", 12)
    h = 58
    geo = [(76, 130), (226, 250), (500, 130)]
    plus_x = 680
    rows = [("a", 212, ["a_pre", "a_mix", "a_post"], "TOKEN-MIXER SUBLAYER", 422),
            ("m", 488, ["m_pre", "m_mlp", "m_post"], "FEED-FORWARD SUBLAYER", 536)]
    for r, (pfx, sy, slots, label, byp) in enumerate(rows):
        y = sy - h / 2
        start = 28 if r == 0 else 36
        o += f'<path d="M{start},{sy} L{plus_x-11},{sy}" class="wire"/>'
        if r == 0:
            o += text(30, sy - 10, "x", "mt", "start")
        o += f'<circle cx="54" cy="{sy}" r="3.5" class="pin"/>'
        o += f'<path d="M54,{sy} L54,{byp} L{plus_x},{byp} L{plus_x},{sy+11}" class="wire"/>'
        o += text(600, byp + 9, "identity", "xs")
        for slot, (gx, gw) in zip(slots, geo):
            o += _chip(gx, y, gw, h, slot, spec.get(slot), slot in changed)
        o += add(plus_x, sy)
        o += text(1004, sy - 12, label, "caps", "end")
        if r == 0:
            o += f'<path d="M{plus_x+11},{sy} L1008,{sy} L1008,444 L36,444 L36,{rows[1][1]}" class="wire"/>'
        else:
            o += f'<path d="M{plus_x+11},{sy} L1010,{sy}" class="wire"/>' + text(1008, sy - 26, "y", "mt", "end")
    # inside the mixer
    o += f'<path d="M226,{212+h/2} L70,258 M476,{212+h/2} L664,258" class="ar dash"/>'
    o += rect(70, 258, 594, 156, "pill", 8)
    o += text(82, 270, "INSIDE THE TOKEN MIXER", "caps", "start")
    inner = [("qk", 82, 294), ("pos", 276, 294), ("score", 470, 294), ("mask", 82, 360), ("vres", 276, 360),
             ("gate", 470, 360)]
    for slot, ix, iy in inner:
        o += _chip(ix, iy, 182, 46, slot, spec.get(slot), slot in changed)
    o += _pattern(700, 258, 296, 156, spec.get("pattern", "F" * 12), "pattern" in changed)
    note = spec.get("note", "")
    o += text(36, 572, note, "xs", "start")
    return svg(W, H, o, f"Circuit of {spec.get('name', 'a variant')}")
