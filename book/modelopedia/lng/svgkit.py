"""SVG primitives for the book (shared with the Residual Lineage book).

Box kinds (CSS classes): conv, norm, act, pool, attn, mix, io, reg; `ar` arrows;
`add` circles. Colours are CSS variables, so figures follow the theme.
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
