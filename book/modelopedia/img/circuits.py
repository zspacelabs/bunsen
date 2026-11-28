"""Per-family variant circuits: each spec fills the board's sockets.

A variant is derived from its parent; the sockets it changes are computed and
marked Δ in its figure. Component = (title, [params], link?, kind?).
"""

SLOTS = ["stem", "stages", "final", "pool", "head", "layers", "pre", "conv0", "mixer", "aa", "attn_mix", "conv2",
         "attn_out", "shortcut", "out_act", "order", "tap",
         # the transformer board
         "backbone", "patch", "tokens", "norm_pre", "stack", "pool_norm", "a_pre", "a_mix", "a_post", "a_scale",
         "a_out", "m_pre", "m_mlp", "m_post", "m_scale", "m_out"]

C = {}  # key -> (spec, parent key or None)


def define(key, parent=None, **kw):
    spec = dict(C[parent][0]) if parent else {}
    spec.update(kw)
    C[key] = (spec, parent)


def changed(key):
    spec, parent = C[key]
    if not parent:
        return set()
    p = C[parent][0]
    return {s for s in SLOTS if spec.get(s) != p.get(s)}


# ------------------------------------------------------------------ generic board
define("generic", None, name="the generic conv net",
       stem=("stem", ["convs: 7×7 | 3×3 ×3 | 3×3", "pool: max · avg · blur · conv · none"]),
       stages=("stages", ["depths [d₁ … d₄]", "widths · strides · dilation"]),
       final=("final", ["— | act | norm+act", "| 1×1 conv"]),
       pool=("global pool", ["avg · max · avgmax", "· catavgmax"]),
       head=("classifier", ["dropout → fc", "(0 classes: none)"]),
       layers="norm · act · WS? · SAME?",
       pre=("norm → act", ["pre-act units"]),
       conv0=("conv block", ["1×1 reduce", "(bottleneck)"]),
       mixer=("mixer", ["conv · Res2", "SplitAttn · SK"], "algebra.html#slots"),
       aa=("pool", ["blur · avg · avd"]),
       attn_mix=("SE", ["RegNet place"]),
       conv2=("conv block", ["1×1 expand | 3×3", "no act"]),
       attn_out=("SE · ECA", ["ResNet place"]),
       shortcut=("shortcut", ["identity · 1×1 conv", "avg → 1×1 · none"]),
       out_act=("act", ["post-act only"]))

# ------------------------------------------------------------------ ResNet
define("resnet-basic", None, name="ResNet v1 (basic)",
       stem=("single stem", ["7×7 · 64 · s2", "max 3×3 · s2"]),
       stages=("4 stages", ["units ×[2, 2, 2, 2]", "widths 64→512 · s 1,2,2,2"]),
       final=None, pool=("global pool", ["avg"]), head=("classifier", ["fc → classes"]),
       layers="BN · ReLU",
       pre=None, conv0=None,
       mixer=("conv block", ["3×3 · stride s"]),
       aa=None, attn_mix=None,
       conv2=("conv block", ["3×3", "no act"]),
       attn_out=None,
       shortcut=("projection", ["1×1 · stride s · norm"]),
       out_act=("act", ["ReLU"]))
define("resnet", "resnet-basic", name="ResNet v1 (bottleneck)",
       stages=("4 stages", ["units ×[3, 4, 6, 3]", "widths 256→2048 · s 1,2,2,2"]),
       conv0=("conv block", ["1×1 · → p"]),
       mixer=("conv block", ["3×3 · p · stride s"]),
       conv2=("conv block", ["1×1 · → 4p", "no act"]))
define("resnet-d", "resnet", name="ResNet-D",
       stem=("deep stem", ["3×3 ×3 · 32, 32, 64", "max 3×3 · s2"]),
       shortcut=("avg-down", ["2×2 avg · s → 1×1 · norm"]))
define("resnet-se", "resnet", name="SE-ResNet / ECA-ResNet",
       attn_out=("SE | ECA", ["C → C/16 → C", "| 1-d conv, k(C)"]))
define("resnet-aa", "resnet", name="Anti-aliased ResNet",
       stem=("single stem", ["7×7 · 64 · s2", "max s1 → blur s2"]),
       mixer=("conv block", ["3×3 · p · stride 1"]),
       aa=("blur | avg", ["stride s", "after mixer act"]))
define("resnet-rs", "resnet", name="ResNet-RS",
       stem=("deep stem", ["3×3 ×3 · 32, 32, 64", "3×3 conv · s2 (no max)"]),
       attn_out=("SE", ["C → C/4 → C"]),
       shortcut=("avg-down", ["2×2 avg · s → 1×1 · norm"]))

# ------------------------------------------------------------------ ResNeXt / Wide
define("resnext", "resnet", name="ResNeXt",
       conv0=("conv block", ["1×1 · → g·w"]),
       mixer=("conv block", ["3×3 · g = 32", "g·w wide · stride s"]))
define("wide", "resnet", name="Wide ResNet",
       conv0=("conv block", ["1×1 · → 2p"]),
       mixer=("conv block", ["3×3 · 2p · stride s"]))

# ------------------------------------------------------------------ ResNetV2 / BiT
define("resnetv2", "resnet", name="ResNetV2 (pre-activation)",
       stem=("single stem", ["7×7 · 64 · s2 · bare", "max 3×3 · s2"]),
       final=("norm → act", ["BN · ReLU"]),
       pre=("norm → act", ["BN · ReLU", "on unit input"]),
       conv0=("conv", ["1×1 · → p"], "c-units.html#pre"),
       mixer=("pre-act conv", ["BN·ReLU → 3×3", "stride s"], "c-units.html#pre"),
       conv2=("pre-act conv", ["BN·ReLU → 1×1", "→ 4p"], "c-units.html#pre"),
       shortcut=("projection", ["1×1 · stride s", "reads pre(x) · no norm"]),
       out_act=None, tap="raw")
define("bit", "resnetv2", name="BiT",
       stem=("single stem", ["7×7 StdConv · s2", "zero-pad → max s2"]),
       stages=("4 stages", ["units ×[3, 4, 6, 3]", "widths ×k (k = 1…4)"]),
       final=("norm → act", ["GN · ReLU"]),
       layers="GN(32) · ReLU · weight std",
       pre=("norm → act", ["GN · ReLU", "on unit input"]),
       mixer=("pre-act conv", ["GN·ReLU → 3×3", "stride s"], "c-units.html#pre"),
       conv2=("pre-act conv", ["GN·ReLU → 1×1", "→ 4p"], "c-units.html#pre"))
define("evonorm", "resnetv2", name="ResNetV2-EvoNorm",
       stem=("deep stem", ["3×3 ×3 · bare", "max 3×3 · s2"]),
       final=("EvoNorm-S0", ["norm + act"], "c-norm.html#evo"),
       layers="EvoNorm-S0 (norm+act)",
       pre=("EvoNorm-S0", ["on unit input"], "c-norm.html#evo"),
       mixer=("pre-act conv", ["Evo → 3×3", "stride s"], "c-units.html#pre"),
       conv2=("pre-act conv", ["Evo → 1×1", "→ 4p"], "c-units.html#pre"),
       shortcut=("avg-down", ["2×2 avg → 1×1", "reads pre(x)"]))

# ------------------------------------------------------------------ Res2Net
define("res2net", "resnet", name="Res2Net",
       conv0=("conv block", ["1×1 · → s·w"]),
       mixer=("Res2 split", ["s chunks of w", "3×3 hierarchy"], "c-res2.html"))
define("res2net-d", "res2net", name="Res2Net-D",
       stem=("deep stem", ["3×3 ×3 · 32, 32, 64", "max 3×3 · s2"]),
       shortcut=("avg-down", ["2×2 avg · s → 1×1 · norm"]))

# ------------------------------------------------------------------ SKNet
define("skresnet", "resnet-basic", name="SKResNet",
       mixer=("selective kernel", ["3×3 d1 ‖ 3×3 d2", "split input"], "c-sk.html"))
define("skresnext", "resnext", name="SKResNeXt",
       mixer=("selective kernel", ["3×3 d1 ‖ 3×3 d2", "g = 32 · stride s"], "c-sk.html"))

# ------------------------------------------------------------------ ResNeSt
define("resnest", "resnet-d", name="ResNeSt",
       mixer=("split attention", ["radix r · card. g", "stride 1"], "c-splitattn.html"),
       aa=("avd", ["3×3 avg", "stride s"]))
define("resnest-first", "resnest", name="ResNeSt (avd first)",
       order=["pre", "conv0", "aa", "mixer", "attn_mix", "conv2", "attn_out"])

# ------------------------------------------------------------------ RegNet
define("regnetx", "resnext", name="RegNetX",
       stem=("RegNet stem", ["3×3 · 32 · s2", "no pool"]),
       stages=("4 stages, generated", ["widths from w₀, wₐ, wₘ", "depth d · group size k"], "f-regnet.html#widths"),
       conv0=("conv block", ["1×1 · b = 1"]),
       mixer=("conv block", ["3×3 · groups of k", "stride s"]),
       conv2=("conv block", ["1×1", "no act"]))
define("regnety", "regnetx", name="RegNetY",
       attn_mix=("SE", ["¼ of unit input", "after the mixer"]))
define("regnetv", "regnety", name="RegNetV",
       stem=("RegNet stem", ["3×3 · s2 · bare", "no pool"]),
       final=("act", ["SiLU"]),
       layers="BN · SiLU",
       pre=("norm → act", ["BN · SiLU", "on unit input"]),
       mixer=("pre-act conv", ["BN·SiLU → 3×3", "groups of k · s"], "c-units.html#pre"),
       conv2=("pre-act conv", ["BN·SiLU → 1×1"], "c-units.html#pre"),
       shortcut=("projection", ["1×1 | avg → 1×1", "reads pre(x)"]),
       out_act=None, tap="pre")
define("regnetz", "regnety", name="RegNetZ",
       final=("act | 1×1 conv", ["SiLU · (_h: → 1536)"]),
       layers="BN · SiLU",
       conv0=("conv block", ["1×1 · expand ×4"]),
       conv2=("conv block", ["1×1 · project", "no act"]),
       shortcut=("none", ["reshaping units:", "no residual"]),
       out_act=None, residual=False)


# ================================================================== transformers (board="tx")
LN = ("LayerNorm", ["ε 1e-6"])
define("tx-generic", None, board="tx", name="the generic vision encoder",
       backbone=("conv net", ["headless ResNetV2", "(hybrids only)"]),
       patch=("patch embed", ["p×p conv, stride p", "± bias · ± norm"]),
       tokens=("prefix + pos", ["cls · dist · regs", "learnt positions"]),
       norm_pre=("norm", ["CLIP's ln_pre"]),
       stack=("stack | stages", ["L blocks, or", "merges + d₁…d₄"]),
       final=("norm", ["after the stack"]),
       pool=("token pool", ["token · avg · max", "map · distilled"]),
       pool_norm=("norm", ["after the pool"]),
       head=("classifier", ["dropout → fc", "(+ fc_dist)"]),
       layers="norm kind · ε · activation · biases",
       a_pre=("norm", ["pre-norm"]), a_mix=("token mixer", ["global attention", "| window attention"]),
       a_post=("norm", ["res-post-norm"]), a_scale=("γ", ["per channel"]), a_out=("norm", ["post-norm"]),
       m_pre=("norm", ["pre-norm"]), m_mlp=("MLP", ["Mlp | SwiGLU", "| packed SwiGLU"]),
       m_post=("norm", ["res-post-norm"]), m_scale=("γ", ["per channel"]), m_out=("norm", ["post-norm"]))

# ------------------------------------------------------------------ ViT
define("vit", None, board="tx", name="ViT",
       backbone=None,
       patch=("patch embed", ["16×16 conv, s16", "→ D, bias"]),
       tokens=("cls + positions", ["1 class token", "learnt, over all"]),
       norm_pre=None,
       stack=("encoder", ["L uniform blocks", "D wide · h heads"]),
       final=("LayerNorm", ["ε 1e-6"]),
       pool=("token", ["the class token"]),
       pool_norm=None,
       head=("classifier", ["fc → classes"]),
       layers="LayerNorm ε 1e-6 · GELU · biases",
       a_pre=LN, a_mix=("self-attention", ["global, bidirectional", "h heads · qkv bias"]),
       a_post=None, a_scale=None, a_out=None,
       m_pre=LN, m_mlp=("MLP", ["D → 4D → D", "GELU"]),
       m_post=None, m_scale=None, m_out=None)
define("vit-hybrid", "vit", name="Hybrid ViT (R50+ViT-B/16)",
       backbone=("ResNetV2", ["stem + [3, 4, 9]", "post-act · GN · WS"]),
       patch=("patch embed", ["1×1 on the map", "1024 → D"]))
define("vit-rpn", "vit", name="ViT, res-post-norm (rpn)",
       tokens=("positions only", ["no class token"]),
       pool=("avg", ["mean of patches"]), pool_norm=("LayerNorm", ["after the pool"]), final=None,
       a_pre=None, a_post=LN, m_pre=None, m_post=LN,
       a_mix=("self-attention", ["global, bidirectional", "h heads · no qkv bias"]))
define("vit-gap", "vit", name="GAP ViT (timm sbb)",
       tokens=("positions only", ["no class token"]),
       pool=("avg", ["mean of patches"]),
       a_mix=("self-attention", ["global, bidirectional", "h heads · no qkv bias"]),
       a_scale=("γ", ["layer scale"]), m_scale=("γ", ["layer scale"]))
define("flexivit", "vit", name="FlexiViT",
       tokens=("cls + positions", ["1 class token", "learnt, patches only"]))

# ------------------------------------------------------------------ DeiT
define("deit-distilled", "vit", name="DeiT, distilled",
       tokens=("cls + dist + pos", ["class, distillation", "learnt, over all"]),
       pool=("distilled", ["cls → fc", "dist → fc_dist"]),
       head=("two classifiers", ["logits averaged"]))
define("deit3", "vit", name="DeiT III",
       tokens=("cls + positions", ["1 class token", "learnt, patches only"]),
       a_scale=("γ", ["layer scale"]), m_scale=("γ", ["layer scale"]))

# ------------------------------------------------------------------ language-supervised towers
define("clip", "vit", name="CLIP image tower",
       patch=("patch embed", ["p×p conv, s p", "→ D, no bias"]),
       norm_pre=("LayerNorm", ["ε 1e-5", "before block 1"]),
       final=("LayerNorm", ["ε 1e-5"]),
       layers="LayerNorm ε 1e-5 · GELU | QuickGELU",
       a_pre=("LayerNorm", ["ε 1e-5"]), m_pre=("LayerNorm", ["ε 1e-5"]),
       m_mlp=("MLP", ["D → 4D → D", "GELU | QuickGELU"]))
define("siglip", "vit", name="SigLIP image tower (map)",
       tokens=("positions only", ["no class token"]),
       pool=("map", ["latent query attends", "+ MLP · h heads"], "t-pool.html#map"),
       layers="LayerNorm ε 1e-6 · GELU (some tanh)",
       m_mlp=("MLP", ["D → rD → D", "GELU · tanh-GELU"]))
define("siglip-gap", "siglip", name="SigLIP image tower (gap)",
       pool=("avg", ["mean of patches"]))
define("aimv2", "vit", name="AIMv2",
       patch=("patch embed", ["14×14 conv, s14", "→ D · RMSNorm"]),
       tokens=("positions only", ["no class token"]),
       final=("RMSNorm", ["ε 1e-5"]),
       pool=("avg", ["mean of patches"]),
       layers="RMSNorm ε 1e-5 · SiLU · no biases",
       a_pre=("RMSNorm", ["ε 1e-5"]), m_pre=("RMSNorm", ["ε 1e-5"]),
       a_mix=("self-attention", ["global, bidirectional", "no qkv / proj bias"]),
       m_mlp=("SwiGLU", ["gate ⊙ value", "h = 2.75 D · no bias"]))
define("beit3", "vit", name="BEiT3",
       tokens=("positions only", ["no class token"]),
       pool=("avg", ["mean of patches"]), pool_norm=("LayerNorm", ["after the pool"]), final=None,
       layers="LayerNorm ε 1e-5 · GELU",
       a_mix=("self-attention", ["+ LN before W_O", "(sub-LN)"]),
       m_mlp=("MLP", ["D → 4D → D", "+ LN on 4D (sub-LN)"]))

# ------------------------------------------------------------------ DINOv2 and registers
define("dinov2", "vit", name="DINOv2",
       patch=("patch embed", ["14×14 conv, s14", "518 px: 37×37 grid"]),
       a_scale=("γ", ["layer scale"]), m_scale=("γ", ["layer scale"]))
define("dinov2-g", "dinov2", name="DINOv2 giant",
       layers="LayerNorm ε 1e-6 · SiLU · biases",
       m_mlp=("SwiGLU, packed", ["one fc₁ → 2h", "h = 4096 · SiLU"]))
define("dinov2-reg", "dinov2", name="DINOv2 with registers",
       tokens=("cls + 4 regs + pos", ["registers unread", "positions: patches only"]))
define("radio", "dinov2-reg", name="C-RADIO",
       tokens=("cls + 7–9 regs + pos", ["registers unread", "positions: patches only"]),
       patch=("patch embed", ["16×16 conv, s16", "512 px"]),
       final=None)

# ------------------------------------------------------------------ Swin
SLN = ("LayerNorm", ["ε 1e-5"])
define("swin", "vit", name="Swin Transformer (V1)",
       block="SWIN BLOCK  ×  dᵢ  per stage   (odd blocks shift their windows by half a window)",
       patch=("patch embed", ["4×4 conv, s4 → C", "+ LayerNorm"]),
       tokens=None,
       stack=("4 stages", ["[2, 2, 6, 2] · C→8C", "merge: LN → 4C→2C"], "t-merge.html"),
       final=("LayerNorm", ["ε 1e-5"]),
       pool=("avg", ["mean of tokens"]),
       layers="LayerNorm ε 1e-5 · GELU",
       a_pre=SLN, m_pre=SLN,
       a_mix=("window attention", ["M×M windows · shift", "rel-pos table"], "t-window.html"),
       note="no position embedding: positions enter as each window's relative bias")
define("swin-s3", "swin", name="Swin-S3",
       stack=("4 stages", ["[2, 2, 6, 2] · C→8C", "windows 7, 7, 14, 7"], "t-merge.html"),
       a_mix=("window attention", ["a window per stage", "rel-pos table"], "t-window.html"))
define("swinv2", "swin", name="Swin Transformer V2",
       stack=("4 stages", ["[2, 2, 6, 2] · C→8C", "merge: 4C→2C → LN"], "t-merge.html"),
       a_pre=None, a_post=SLN, m_pre=None, m_post=SLN,
       a_mix=("window attention", ["cosine · learned scale", "CPB-MLP · log offsets"], "t-window.html"))
