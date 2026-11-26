"""Generate the vit kit's toy fixtures: tiny seeded timm `VisionTransformer`s and their answers.

Fifteen toys, each timm's own `timm.models.vision_transformer.VisionTransformer` on 16x16
images in 4x4 patches (a 4x4 grid), 16 wide, two layers of two heads, 10 classes:

- `toy_token`: timm's defaults: a class token, its pool, layer norms of epsilon 1e-6, a
  biased `qkv`, `GELU`;
- `toy_avg`: the patch tokens' mean (`global_pool='avg'`), its norm after the pool
  (`fc_norm`) in place of the final norm;
- `toy_avgmax`: the same with `avgmax`;
- `toy_clip`: CLIP's pre-norm (`pre_norm=True`: a bias-free patch embedding and `norm_pre`),
  its layer norms of epsilon 1e-5;
- `toy_quickgelu`: `toy_clip` with `QuickGELU`, as the `_quickgelu` archs;
- `toy_reg`: two register tokens, the position embedding over the patches alone
  (`no_embed_class`), and layer scale (`init_values`), as DeiT III and the register ViTs;
- `toy_gap`: no class token, the mean pool (`class_token=False`, `global_pool='avg'`);
- `toy_miil`: no `qkv` bias (`qkv_bias=False`), as `vit_base_patch16_224_miil`;
- `toy_headless`: no classifier (`num_classes=0`): the output is the pooled features;
- `toy_openai`: `toy_clip` with an 8-wide projection head of no bias, its weights written in
  `OpenAI` CLIP's own layout (`visual.*`, with a stand-in for the text tower), as the `openai`
  repos' `pytorch_model.bin`s hold them. timm's `_convert_openai_clip` maps them back onto the
  model's state dict exactly, which the generator checks;
- `toy_rpn`: res-post-norm blocks (`block_fn=ResPostBlock`: each branch normed on its output),
  no class token, the mean pool, no `qkv` bias, as `vit_base_patch16_rpn_224`;
- `toy_bare`: no norm after the last layer (`final_norm=False`), with register tokens and layer
  scale, as the C-RADIO archs;
- `toy_remainder`: 18x18 images, the patch embedding dropping the 2 pixels past the last whole
  patch, as `vit_so400m_patch14_siglip_384`;
- `toy_bare_avg`: `toy_bare` pooled by the mean: no norm before the pool or after it;
- `toy_map`: the attention pool (`global_pool='map'`, `AttentionPoolLatent`), no class token,
  tanh `GELU`, as SigLIP's towers;
- `toy_map_cls`: the attention pool with a class token, which the pool leaves out.

The toy test builds each as bunsen's `ViTContractConfig`.

Each is built in float64 on the CPU with seeded weights: the norms' gains and shifts random, so
a misplaced norm shows. A two-image `[2, 3, 16, 16]` batch runs through
`forward_intermediates` (each layer's patch tokens as a map), the pooled features and the
logits (an 18x18 batch for `toy_remainder`). The fixture is the weights under timm's own state-dict names (what its hub safetensors
hold; `toy_openai`'s in `OpenAI`'s) and `io.*`: `images`, `layer_0`, `layer_1`, `features` (`forward_features`), `pooled`,
`logits`.

Run from the repository root:

    /media/Data/bunsen-tmp/timm/venv/bin/python -I tools/gen_vit_toy_fixtures.py

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0,
safetensors 0.8.0, numpy 2.5.5.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from functools import partial  # noqa: E402

import torch  # noqa: E402
from timm.layers import QuickGELU  # noqa: E402
from timm.models.vision_transformer import (  # noqa: E402
    ResPostBlock,
    VisionTransformer,
    _convert_openai_clip,
)

from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_weights,
    io_tensors,
    weights,
    write,
)

OUT = TESTDATA / "vit"

CLIP = dict(pre_norm=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-5))

# Each toy's seed is its position here: a new toy goes at the end.
TOYS = {
    "toy_token": dict(),
    "toy_avg": dict(global_pool="avg"),
    "toy_avgmax": dict(global_pool="avgmax"),
    "toy_clip": dict(CLIP),
    "toy_quickgelu": dict(CLIP, act_layer=QuickGELU),
    "toy_reg": dict(reg_tokens=2, no_embed_class=True, init_values=1e-5),
    "toy_gap": dict(class_token=False, global_pool="avg"),
    "toy_miil": dict(qkv_bias=False),
    "toy_headless": dict(num_classes=0),
    "toy_openai": dict(CLIP, num_classes=8),
    "toy_rpn": dict(block_fn=ResPostBlock, class_token=False, global_pool="avg", qkv_bias=False),
    "toy_bare": dict(final_norm=False, reg_tokens=2, no_embed_class=True, init_values=1e-5),
    "toy_remainder": dict(img_size=18),
    "toy_bare_avg": dict(final_norm=False, global_pool="avg"),
    "toy_map": dict(class_token=False, global_pool="map", act_layer="gelu_tanh"),
    "toy_map_cls": dict(global_pool="map"),
}

# The toys whose weights are written in `OpenAI` CLIP's layout.
OPENAI = {"toy_openai"}


def openai_clip(state: dict) -> dict:
    """A timm CLIP tower's state dict in `OpenAI`'s layout: what timm's `_convert_openai_clip`
    reads. The head is `visual.proj`, `[d, classes]`, its bias dropped (timm's is zero); the
    class token and the position embedding lose their unit axes; two text-tower tensors stand in
    for the rest of the model."""
    swaps = [
        ("patch_embed.proj", "conv1"),
        ("blocks.", "transformer.resblocks."),
        ("norm_pre", "ln_pre"),
        ("norm1", "ln_1"),
        ("norm2", "ln_2"),
        ("attn.qkv.", "attn.in_proj_"),
        ("attn.proj", "attn.out_proj"),
        ("mlp.fc1", "mlp.c_fc"),
        ("mlp.fc2", "mlp.c_proj"),
    ]
    out = {}
    for k, v in state.items():
        if k == "head.bias":
            assert not v.any(), "the head's bias is zero"
            continue
        if k == "head.weight":
            out["visual.proj"] = v.t().contiguous()
            continue
        if k == "cls_token":
            out["visual.class_embedding"] = v.reshape(-1)
            continue
        if k == "pos_embed":
            out["visual.positional_embedding"] = v[0]
            continue
        if k.startswith("norm."):
            k = "ln_post." + k[len("norm."):]
        for a, b in swaps:
            k = k.replace(a, b)
        out["visual." + k] = v
    out["text_projection"] = torch.zeros(4, 8, dtype=torch.float64)
    out["transformer.resblocks.0.ln_1.weight"] = torch.ones(4, dtype=torch.float64)
    return out


def main() -> None:
    for i, (name, args) in enumerate(TOYS.items()):
        args = dict(dict(num_classes=10, img_size=16), **args)
        size = args["img_size"]
        model = build(
            lambda: VisionTransformer(patch_size=4, embed_dim=16, depth=2, num_heads=2, **args),
            seed=4321 + i,
        )
        init_weights(model, norm_bias=0.1)
        if name in OPENAI:
            with torch.no_grad():
                model.head.bias.zero_()
        torch.manual_seed(77 + i)
        images = torch.randn(2, 3, size, size, dtype=torch.float64)
        with torch.no_grad():
            layers = model.forward_intermediates(images, intermediates_only=True)
            features = model.forward_features(images)
            pooled = model.forward_head(features, pre_logits=True)
            logits = model(images)
        assert len(layers) == 2, len(layers)
        io = {"images": images, "features": features, "pooled": pooled, "logits": logits}
        io.update({f"layer_{k}": x for k, x in enumerate(layers)})
        state = weights(model)
        if name in OPENAI:
            openai = openai_clip(model.state_dict())
            back = _convert_openai_clip(openai, model)
            assert back.keys() == model.state_dict().keys(), set(back) ^ set(model.state_dict())
            for k, v in model.state_dict().items():
                assert torch.equal(back[k].to(v.dtype), v), k
            state = {k: v.float().contiguous() for k, v in openai.items()}
        write(OUT / f"{name}.safetensors", state, io_tensors(io))
        print(name, {k: tuple(v.shape) for k, v in io.items()})


if __name__ == "__main__":
    main()
