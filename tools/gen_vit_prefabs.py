"""Generate the vit kit's prefabs from timm: every `vision_transformer.py` and `vision_transformer_hybrid.py` architecture with hub weights.

Run with a Python that has timm (and torch):

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_vit_prefabs.py
    cargo +nightly fmt --all

For each architecture `vision_transformer.py` registers that has a pretrained hub tag, timm builds it
on the meta device while the script records the constructor kwargs its entrypoint passes
(`build_model_with_cfg`), and the image size it ends up with (timm fills that in from the default
tag's input size). Each kwarg maps onto a `ViTContractConfig` knob; an architecture with a kwarg
the contract has no knob for is skipped, and the script names the kwarg. A few are left out on
purpose (`OUT`). A hybrid's backbone (its `HybridEmbed`) is read from the built model: timm's
non-preact `ResNetV2` of `_resnetv2`, checked knob by knob, becomes `with_resnetv2_backbone`.

The prefab functions and the prefab table are written between the `gen_vit_prefabs` markers in
`crates/public/bunsen/src/kits/images/classification/vit/pretrained/prefabs.rs`; everything else
in the file is hand-written. The output is rustfmt's to lay out, hence the `fmt` after it.
"""

import argparse
import functools
import pathlib
import re

import timm
import torch
from timm.models import _registry
from timm.layers import GroupNormAct, StdConv2dSame
from timm.layers.pool2d_same import MaxPool2dSame
from timm.models import vision_transformer, vision_transformer_hybrid
from timm.models.resnetv2 import Bottleneck, ResNetV2

PREFABS = pathlib.Path(__file__).resolve().parent.parent.joinpath(
    "crates/public/bunsen/src/kits/images/classification/vit/pretrained/prefabs.rs"
)

# Architectures left out on purpose, and why.
OUT = {
    "attn_layer": "differential attention: a new forward path for one trainer's runs ([par_bound])",
    "block_fn:ParallelScalingBlock": "parallel blocks: a new forward path for one trainer's runs ([par_bound])",
    "block_fn:DiffParallelScalingBlock": "parallel blocks: a new forward path for one trainer's runs ([par_bound])",
    "dynamic_img_size": "a position embedding resampled per input: waits for resampling",
    "test": "timm's test fixtures",
    "backbone:ConvStem": "MobileCLIP's ConvStem backbone: [hy_mci]",
}

MODULES = ("vision_transformer", "vision_transformer_hybrid")

# The families whose order the table keeps first; the rest follow by name.
FAMILY_ORDER = ["vit", "vit_miil", "vit_clip"]

PREFIX_LABELS = {"vit": "ViT", "flexivit": "FlexiViT", "eva": "EVA", "aimv2": "AIMv2", "beit3": "BEiT3"}

SIZE_LABELS = {
    "tiny": "Ti",
    "xsmall": "XS",
    "small": "S",
    "medium": "M",
    "base": "B",
    "large": "L",
    "huge": "H",
    "giant": "g",
    "gigantic": "G",
}


def norm(value):
    """A kwarg as plain data: a `partial` as its function's name and keywords, a type as its name."""
    if isinstance(value, functools.partial):
        return {"partial": getattr(value.func, "__name__", str(value.func)), **{k: norm(v) for k, v in value.keywords.items()}}
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return [norm(v) for v in value]
    return getattr(value, "__name__", type(value).__name__)


def record():
    """Each pretrained `vision_transformer.py` architecture: its kwargs, and its image size."""
    archs = sorted(
        {
            name.split(".")[0]
            for name in timm.list_pretrained()
            if _registry._model_to_module.get(name.split(".")[0]) in MODULES
            and getattr(timm.get_pretrained_cfg(name), "hf_hub_id", None)
        }
    )
    seen = {}
    build = vision_transformer.build_model_with_cfg

    def spy(cls, variant, pretrained, **kwargs):
        skip = ("pretrained_cfg", "pretrained_cfg_overlay", "pretrained_filter_fn", "feature_cfg", "pretrained_strict", "cache_dir")
        seen[variant] = {k: norm(v) for k, v in kwargs.items() if k not in skip}
        return build(cls, variant, pretrained, **kwargs)

    vision_transformer.build_model_with_cfg = spy
    vision_transformer_hybrid.build_model_with_cfg = spy
    out = {}
    for arch in archs:
        with torch.device("meta"):
            model = timm.create_model(arch, pretrained=False)
        size = tuple(model.patch_embed.img_size)
        assert size[0] == size[1], (arch, size)
        kwargs = seen[arch]
        if hasattr(model.patch_embed, "backbone"):
            kwargs = dict(kwargs, embed_layer=backbone_of(model.patch_embed.backbone))
        else:
            assert model.patch_embed.proj.in_channels == 3, arch
        out[arch] = (kwargs, size[0])
    vision_transformer.build_model_with_cfg = build
    vision_transformer_hybrid.build_model_with_cfg = build
    return out


def backbone_of(backbone):
    """A hybrid's backbone as `("ResNetV2", layers)`, timm's `_resnetv2(layers)`; or `(name, None)`, one
    the contract doesn't build, or a `ResNetV2` with a knob other than `_resnetv2`'s."""
    stem = backbone.stem if isinstance(backbone, ResNetV2) else backbone
    if not isinstance(backbone, (ResNetV2, torch.nn.Sequential)) or not hasattr(stem, "conv"):
        return (type(backbone).__name__, None)
    convs = [m for m in backbone.modules() if isinstance(m, torch.nn.Conv2d)]
    norms = [m for m in backbone.modules() if isinstance(m, torch.nn.GroupNorm)]
    ok = (
        all(isinstance(c, StdConv2dSame) and c.eps == 1e-8 for c in convs)
        and all(isinstance(n, GroupNormAct) and n.num_groups == 32 for n in norms)
        and isinstance(stem.pool, MaxPool2dSame)
        and stem.conv.out_channels == 64
        and stem.conv.kernel_size == (7, 7)
    )
    if not isinstance(backbone, ResNetV2):
        return ("ResNetV2", []) if ok else ("Sequential", None)
    layers = [len(stage.blocks) for stage in backbone.stages]
    widths = [stage.blocks[-1].conv3.out_channels for stage in backbone.stages]
    ok = (
        ok
        and isinstance(backbone.norm, torch.nn.Identity)
        and all(isinstance(b, Bottleneck) for stage in backbone.stages for b in stage.blocks)
        and widths == [256, 512, 1024, 2048][: len(layers)]
    )
    return ("ResNetV2", layers) if ok else ("ResNetV2 (not _resnetv2's)", None)


def out_reason(arch, kwargs):
    """Why `arch` is left out on purpose, if it is."""
    if arch.startswith("test_"):
        return OUT["test"]
    if "attn_layer" in kwargs:
        return OUT["attn_layer"]
    if "block_fn" in kwargs and f"block_fn:{kwargs['block_fn']}" in OUT:
        return OUT[f"block_fn:{kwargs['block_fn']}"]
    if kwargs.get("dynamic_img_size"):
        return OUT["dynamic_img_size"]
    backbone = kwargs.get("embed_layer")
    if isinstance(backbone, tuple) and f"backbone:{backbone[0]}" in OUT:
        return OUT[f"backbone:{backbone[0]}"]
    return None


def lower(arch, kwargs, image_size):
    """`(expression, notes)`: the Rust prefab of `arch`, and what sets it apart; or a `str`, the kwarg the contract lacks."""
    kw = dict(kwargs)
    kw.pop("img_size", None)
    patch = kw.pop("patch_size", 16)
    dim = kw.pop("embed_dim", 768)
    depth = kw.pop("depth", 12)
    heads = kw.pop("num_heads", 12)
    expr = f"Self::vit({image_size}, {patch}, {dim}, {depth}, {heads})"
    notes = []
    backbone = kw.pop("embed_layer", None)
    if backbone is not None:
        name, layers = backbone
        if layers is None:
            return f"embed_layer=HybridEmbed({name})"
        expr += f".with_resnetv2_backbone(&[{', '.join(map(str, layers))}])"
        notes.append(
            f"on a `ResNetV2` backbone of {tuple(layers)} units" if layers else "on a `ResNetV2` stem"
        )

    pre_norm = kw.pop("pre_norm", False)
    norm_layer = kw.pop("norm_layer", None)
    eps = None
    rms = False
    if norm_layer is not None:
        if norm_layer.get("partial") not in ("LayerNorm", "RmsNorm") or set(norm_layer) - {"partial", "eps"}:
            return f"norm_layer={norm_layer}"
        eps = norm_layer.get("eps")
        rms = norm_layer["partial"] == "RmsNorm"
    if rms:
        expr += ".with_norm_layer(ViTNormLayer::RmsNorm)"
        notes.append("RMS norms")
    embed_norm = kw.pop("embed_norm_layer", None)
    if embed_norm is not None:
        if embed_norm != norm_layer:
            return f"embed_norm_layer={embed_norm} (not norm_layer)"
        expr += ".with_embed_norm(true)"
        notes.append("a normed patch embedding")
    if pre_norm and eps == 1e-5 and not rms:
        expr += ".with_clip()"
        notes.append("CLIP's pre-norm image tower")
    else:
        if pre_norm:
            expr += ".with_pre_norm(true)"
            notes.append("pre-norm")
        if eps is not None and eps != 1e-6:
            expr += f".with_norm_epsilon({eps!r})"
            notes.append(f"norm epsilon {eps!r}")

    if not kw.pop("qkv_bias", True):
        expr += ".with_qkv_bias(false)"
        notes.append("MIIL's, with no `qkv` bias" if arch.endswith("_miil") else "no `qkv` bias")
    if not kw.pop("proj_bias", True):
        expr += ".with_proj_bias(false)"
        notes.append("no output or MLP biases")
    mlp = kw.pop("mlp_layer", None)
    if mlp == "SwiGLU":
        expr += ".with_mlp_layer(ViTMlpLayer::SwiGlu)"
        notes.append("a `SwiGLU` MLP")
    elif mlp == {"partial": "GluMlp", "act_layer": "SiLU", "gate_last": False}:
        expr += ".with_mlp_layer(ViTMlpLayer::SwiGluPacked)"
        notes.append("a packed `SwiGLU` MLP")
    elif mlp is not None:
        return f"mlp_layer={mlp}"
    ratio = kw.pop("mlp_ratio", 4.0)
    if float(ratio) != 4.0:
        expr += f".with_mlp_ratio({float(ratio)!r})"
        notes.append(f"an MLP ratio of {float(ratio)!r}")
    act = kw.pop("act_layer", None)
    if act is not None:
        acts = {
            "gelu_tanh": ("GeluApproximate", "tanh GELU"),
            "quick_gelu": ("QuickGelu", "`QuickGELU`"),
            "silu": ("Silu", "`SiLU`"),
            "SiLU": ("Silu", "`SiLU`"),
        }
        if act not in acts:
            return f"act_layer={act}"
        expr += f".with_activation(BunsenActivationConfig::{acts[act][0]})"
        notes.append(acts[act][1])

    if not kw.pop("class_token", True):
        expr += ".with_class_token(false)"
        notes.append("no class token")
    reg = kw.pop("reg_tokens", 0)
    if reg:
        expr += f".with_reg_tokens({reg})"
        notes.append(f"{reg} register token{'s' if reg != 1 else ''}")
    if kw.pop("no_embed_class", False):
        expr += ".with_no_embed_class(true)"
        notes.append("the position embedding over the patches alone")

    block = kw.pop("block_fn", "Block")
    init_values = kw.pop("init_values", None)
    if block == "ResPostBlock":
        expr += ".with_norm_placement(NormPlacement::ResPost)"
        notes.append("res-post-norm blocks")
    elif block != "Block":
        return f"block_fn={block}"
    elif init_values is not None:
        expr += ".with_layer_scale(true)"
        notes.append("layer scale")

    if kw.pop("scale_attn_norm", False):
        expr += ".with_scale_attn_norm(true)"
        notes.append("a norm before the attention's output projection")
    if kw.pop("scale_mlp_norm", False):
        expr += ".with_scale_mlp_norm(true)"
        notes.append("a norm on the MLP's hidden width")
    if not kw.pop("final_norm", True):
        expr += ".with_final_norm(false)"
        notes.append("no final norm")
    pool = kw.pop("global_pool", "token")
    if pool == "avg":
        expr += ".with_global_pool(TokenPoolConfig::Avg)"
        notes.append("average-pooled")
    elif pool == "map":
        expr += ".with_global_pool(TokenPoolConfig::Map)"
        notes.append("attention-pooled")
    elif pool != "token":
        return f"global_pool={pool!r}"
    fc_norm = kw.pop("fc_norm", None)
    if fc_norm is not None:
        expr += f".with_fc_norm(Some({str(fc_norm).lower()}))"
        notes.append("its norm after the pool" if fc_norm else "its norm before the pool")

    if image_size % patch:
        notes.append(f"the {image_size % patch} pixels past the last whole patch dropped")
    if kw:
        return ", ".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return expr, notes


def describe(arch, kwargs, image_size):
    """The family (for the table's order) and the label (`ViT-B/16`, `R50+ViT-B/16`) of `arch`."""
    hybrid = re.fullmatch(r"vit_([a-z]+)_(r\d*)_s(\d+)(?:_p\d+)?_\d+", arch)
    if hybrid:
        size, backbone, stride = hybrid.groups()
        return f"vit_hybrid_{backbone}", f"{backbone.upper()}+ViT-{SIZE_LABELS.get(size, size)}/{stride}"
    tokens = arch.split("_")
    prefix, size, rest = tokens[0], tokens[1], tokens[2:]
    if "plus" in rest:
        rest.remove("plus")
        size += "+"
    family = "_".join([prefix] + [t for t in rest if not re.fullmatch(r"patch\d+|\d+", t)])
    patch = kwargs.get("patch_size", 16)
    plus = size.endswith("+")
    word = size.rstrip("+")
    label = f"{PREFIX_LABELS.get(prefix, prefix)}-{SIZE_LABELS.get(word, word)}/{patch}{'+' if plus else ''}"
    return family, label


def render(prefabs):
    """The generated fns and the table rows."""
    fns, rows = [], []
    for arch, (expr, notes, label, dims) in prefabs:
        size, depth, dim, heads = dims
        doc = f"The `{arch}` prefab: `{label}` at {size} pixels: {depth} layers, {dim} wide, {heads} heads"
        if notes:
            doc += "; " + ", ".join(notes)
        fns.append(f"    /// {doc}.\n    pub fn {arch}() -> Self {{\n        {expr}\n    }}\n")
        rows.append(f'        ("{arch}", Self::{arch}),\n')
    return "\n".join(fns), "".join(rows)


def splice(text, marker, body):
    begin, end = f"// gen_vit_prefabs: begin {marker}\n", f"// gen_vit_prefabs: end {marker}"
    head, found, tail = text.partition(begin)
    assert found, f"no `{begin.strip()}` marker"
    indent = head[head.rfind("\n") + 1 :]
    _, found, tail = tail.partition(end)
    assert found, f"no `{end}` marker"
    return f"{head}{begin}{body}{indent}{end}{tail}"


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    built, out, skipped = [], [], []
    for arch, (kwargs, size) in record().items():
        reason = out_reason(arch, kwargs)
        if reason:
            out.append((arch, reason))
            continue
        lowered = lower(arch, kwargs, size)
        if isinstance(lowered, str):
            skipped.append((arch, lowered))
            continue
        expr, notes = lowered
        family, label = describe(arch, kwargs, size)
        dims = (size, kwargs.get("depth", 12), kwargs.get("embed_dim", 768), kwargs.get("num_heads", 12))
        rank = FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)
        key = (rank, family, dims[2], -kwargs.get("patch_size", 16), size, arch)
        built.append((key, arch, (expr, notes, label, dims)))
    built.sort()

    fns, rows = render([(arch, p) for _, arch, p in built])
    text = PREFABS.read_text()
    new = splice(splice(text, "fns", fns), "table", rows)
    for arch, reason in out:
        print(f"out      {arch}: {reason}")
    for arch, reason in skipped:
        print(f"skipped  {arch}: {reason}")
    print(f"{len(built)} prefabs; {len(skipped)} skipped; {len(out)} out")
    PREFABS.write_text(new)


if __name__ == "__main__":
    main()
