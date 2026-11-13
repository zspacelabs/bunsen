#!/usr/bin/env python3
"""Generate the ZONOS3-Mini toy-model fixtures: a tiny LM and a tiny codec decoder.

Run once; commit the outputs. Regenerating is not part of CI.

    git clone https://github.com/Zyphra/ZONOS3-Mini upstream
    git -C upstream checkout b171b19149d29b8bcc31e0342d5e3b0063dc479b
    # modeling_zonos3.py and configuration_zonos3.py of Zyphra/ZONOS3-Mini-hf @ 73500c5a:
    hf download Zyphra/ZONOS3-Mini-hf modeling_zonos3.py configuration_zonos3.py \\
        --revision 73500c5a1fc3e0c4da69908e3cffd9346cdd97c2 --local-dir hf
    python3.12 -m venv venv
    venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
    venv/bin/pip install transformers==4.57.3 safetensors numpy
    venv/bin/pip install --no-deps -e upstream/model
    venv/bin/python -I tools/gen_zonos3_toy_fixtures.py --upstream upstream --hf hf

Pinned versions used to produce the committed fixture:
    torch 2.9.1+cpu, transformers 4.57.3, safetensors 0.8.0, Python 3.12;
    upstream `Zyphra/ZONOS3-Mini` b171b191 (checked: `git rev-parse HEAD`);
    `Zyphra/ZONOS3-Mini-hf` 73500c5a's `modeling_zonos3.py` and `configuration_zonos3.py`
    (checked: sha256).

The fixture holds both halves of the system as `Zyphra/ZONOS3-Mini-hf` packages them: one
safetensors file with the LM under `lm.` and the codec decoder under `codec.`, and one
config with their `lm` and `codec` sections, in that repository's `config.json` form. The
oracles:

- The LM is upstream's `TINY` (`model/tests/test_reference_lm.py`: 2 layers, width 64,
  `gdn1_global1`, so layer 0 is a gated DeltaNet and layer 1 global attention; layer 0's
  feed-forward is dense, layer 1's a latent MoE of 4 experts, top 2, in a 32-wide latent),
  run by upstream's plain-PyTorch reference (`zonos3.reference.lm`, "the numerical
  oracle"), in float64. The reference keeps some steps in float32 whatever the model's
  dtype (the AttnRes scores, the rotary angles, the attention scores, the delta rule and
  its state, the router and the expert accumulation): its outputs hold a float32
  implementation to float32's precision. The reference fixes the GDN at the release's
  4 heads x 128 as module constants; the toy sets them to 2 x 16 (`GDN_TOY`), so the
  fixture stays small and a reader that assumed the release's geometry fails on it.
  Cross-checked: the same weights in the `-hf` port's `Zonos3MiniLM` (built from the
  written `lm` config section, which carries the GDN geometry as fields, so the section
  and the constants' override are checked too) give the same logits.
- The codec decoder is the `-hf` port's `Zonos3MiniCodec` (op for op with upstream's
  `ISTFTCodec`, weight norm fused) at toy sizes, in float32: the port casts its iSTFT
  head's linear outputs and the mixer's input to float32, so it does not run in float64
  (and upstream runs its latent, head and resampler in float32 by design). Two stages,
  so one patch-up between them; sliding windows shorter than the sequence; two iSTFT
  branches of `n_fft` 24 and 72 (not powers of two, the release's ratios to the hop);
  a 4 -> 3 resampler.

Weights are seeded random; the norm gains about one (`toy_fixture.init_weights`), so a
misplaced one is caught. The codec's layer scales are random too.

Outputs, under `crates/public/bunsen/testdata/zonos3/`:

`toy.safetensors`
    - `lm.*`, `codec.*`: the weights, under the `-hf` file's names.
    - `io.lm.*`: the LM's inputs and outputs:
        - `ids` `[72, 11]` (rows of 10 codebook ids and a text id; the text id of a
          non-text row is the padding row, a codebook id of a non-audio row the audio
          pad); row 0 is a speaker row, rows 1-3 condition rows, rows 4-47 text, rows
          48-71 audio;
        - `speaker_embedding` `[8]` (replacing row 0 through the speaker projection);
          `condition_buckets` `[3]` (rows 1-3; the second is the unknown bucket);
        - the reference's captures of the full prefill, under its own names
          (`tools/dump_golden.py lm`'s): `emb_sum`, `emb_cond`, `emb_norm`; per layer
          `l`, `attnres_attn_{l}`, `attn_in_{l}`, `attn_out_{l}`, `attnres_ffn_{l}`,
          `ffn_in_{l}`, `ffn_out_{l}`; layer 0's delta-rule internals `gdn_recurrent_0`
          `[72, 2, 16]`, `gdn_gated_0` `[72, 32]`, `gdn_state_0` `[2, 16, 16]`
          (`[heads, value, key]`), `gdn_conv_state_0` `[96, 3]`; layer 1's
          `router_logits_1` `[72, 4]`; `attnres_final`, `out_norm`, `logits`
          `[72, 10, 18]` (softcapped);
        - `chunked_logits` `[32, 10, 18]`: rows 40-71's logits from a prefill of rows
          0-39, then one of rows 40-71 on the same cache;
        - `decode_logits` `[8, 10, 18]`: rows 64-71's logits from a prefill of rows
          0-63, then eight one-row steps.
    - `io.codec.*`: the codec's inputs and outputs:
        - `codes` `[1, 3, 12]` (codebook 0, then the residual codebooks);
        - `latent` `[1, 24, 12]` (the summed codebook latents, the trunk's input);
          `stage_0` `[1, 32, 12]`, `patch_0` `[1, 16, 24]`, `stage_1` `[1, 32, 24]`
          (each trunk module's output; `stage_1` is the iSTFT head's input);
          `y0` `[1, 144]`, `y1` `[1, 144]` (the iSTFT branches, `n_fft` 24 and 72, at
          the internal rate); `mixer` `[1, 1, 144]`; `audio` `[1, 1, 108]`.
`toy.config.json`
    `{"lm": ..., "codec": ...}`, the `-hf` `config.json`'s sections for the toy.
"""

import argparse
import hashlib
import importlib.util
import pathlib
import subprocess
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from toy_fixture import (  # noqa: E402
    TESTDATA,
    Capture,
    build,
    init_weights,
    io_tensors,
    shapes,
    weights,
    write,
    write_json,
)

UPSTREAM_COMMIT = "b171b19149d29b8bcc31e0342d5e3b0063dc479b"
"""`Zyphra/ZONOS3-Mini` (GitHub): the reference LM."""

HF_FILES = {
    "modeling_zonos3.py": "975e2756f24d45aa95d49de45aae799e0009e0d3ff8747e6e9482609074c65e2",
    "configuration_zonos3.py": "16715663cdbb99dae70f1709853dfaf0243b1222fe5f19c5bcc613b86dd0018c",
}
"""`Zyphra/ZONOS3-Mini-hf` @ 73500c5a1fc3e0c4da69908e3cffd9346cdd97c2: the codec decoder
and the LM cross-check."""

OUT = TESTDATA / "zonos3"

N_CB, CODEBOOK, TEXT_VOCAB = 10, 16, 20

TINY = {
    "n_layers": 2,
    "dim": 64,
    "head_dim": 16,
    "n_heads": 4,
    "n_kv_heads": 2,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_seqlen": 256,
    "loss_softcap": 15.0,
    "ffn_dim_multiplier": 2.0,
    "multiple_of": 32,
    "moe_ffn_dim_multiplier": 1.0,
    "moe_multiple_of": 32,
    "moe_alpha": 0.5,
    "moe_n_experts": 4,
    "n_codebooks": N_CB,
    "codebook_size": CODEBOOK,
    "codebook_sizes": [CODEBOOK] * N_CB,
    "eoa_id": CODEBOOK,
    "audio_pad_id": CODEBOOK + 1,
    "text_vocab": TEXT_VOCAB,
    "speaker_embedding_dim": 8,
    "attention_pattern": "gdn1_global1",
    "residual_type": "full_attnres",
    "use_xsa": False,
    "speaker_enabled": True,
    "condition_tokens_enabled": True,
    "moe_router_type": "linear",
    "moe_router_use_eda": False,
    "moe_balancing_strategy": "quantile",
    "moe_router_topk": 2,
    "moe_start_from_layer": 1,
    "moe_end_from_layer": 0,
    "condition_count": 3,
    "condition_n_buckets": 4,
    "condition_names": ["loudness", "fmax", "speaking_rate"],
    "condition_edges": {
        "loudness": [-30.0, -20.0, -10.0],
        "fmax": [4e3, 8e3, 12e3],
        "speaking_rate": [10.0, 15.0, 20.0],
    },
}
"""Upstream's `TINY` (`model/tests/test_reference_lm.py` @ b171b191), as `params.json`'s
`model` section."""

GDN_TOY = (2, 16)
"""The toy's GDN heads and head width, in place of the reference's fixed 4 x 128."""

ROWS, TEXT_ROWS, AUDIO_ROWS = 72, slice(4, 48), slice(48, 72)
CHUNK_SPLIT, DECODE_FROM = 40, 64


def stage(input_dimension, output_dimension, context_frames):
    """A codec trunk stage's config: 2 layers, width 32, 4 heads."""
    return {
        "context_frames": context_frames,
        "d_model": 32,
        "dim_feedforward": 64,
        "input_dimension": input_dimension,
        "num_heads": 4,
        "num_layers": 2,
        "output_dimension": output_dimension,
    }


CODEC = {
    "codebook_dim": 4,
    "codebook_size": 16,
    "fir_kernel": 7,
    "head_depth": 2,
    "head_hidden": 48,
    "head_hop": 6,
    # per frame: head_hop x 2 (one patch-up) = 12 internal samples, x 3 / 4 resampled
    "hop": 9,
    "latent_dim": 24,
    "n_ffts": [24, 72],
    "resample_new": 3,
    "resample_orig": 4,
    "resample_width": 2,
    "residual_codebooks": 2,
    "semantic_depth": 2,
    "semantic_hidden": 32,
    "semantic_width": 20,
    "stages": [stage(24, 32, 5), stage(16, None, 7)],
}
"""The codec decoder's `config.json` section, at toy sizes."""

CODEC_FRAMES = 12


def check_upstream(upstream: pathlib.Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        sys.exit(f"{upstream} is at {head}, not the pinned {UPSTREAM_COMMIT}")


def load_hf(hf: pathlib.Path) -> types.ModuleType:
    """The `-hf` port's `modeling_zonos3`, its files checked against their pins, loaded as
    a package so its relative import resolves."""
    for name, digest in HF_FILES.items():
        got = hashlib.sha256((hf / name).read_bytes()).hexdigest()
        if got != digest:
            sys.exit(f"{hf / name}: sha256 {got}, not the pinned {digest}")
    package = types.ModuleType("zonos3_hf")
    package.__path__ = [str(hf)]
    sys.modules["zonos3_hf"] = package
    for name in ("configuration_zonos3", "modeling_zonos3"):
        spec = importlib.util.spec_from_file_location(f"zonos3_hf.{name}", hf / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"zonos3_hf.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["zonos3_hf.modeling_zonos3"]


def hf_lm_section(c) -> dict:
    """The `-hf` `config.json` `lm` section of a reference `Zonos3LMConfig`."""
    from zonos3.reference.lm import GDN_HEAD_DIM, GDN_HEADS, GDN_KERNEL

    return {
        "audio_pad_id": c.audio_pad_id,
        "codebook_size": c.codebook_size,
        "condition_bucket_counts": list(c.condition_bucket_counts),
        "condition_edges": {k: list(v) for k, v in c.condition_edges.items()},
        "condition_names": list(c.condition_names),
        "dim": c.dim,
        "eoa_id": c.eoa_id,
        "ffn_hidden": c.ffn_hidden,
        "gdn_head_dim": GDN_HEAD_DIM,
        "gdn_heads": GDN_HEADS,
        "gdn_kernel": GDN_KERNEL,
        "gdn_period": c.gdn_period,
        "head_dim": c.head_dim,
        "loss_softcap": c.loss_softcap,
        "max_seqlen": c.max_seqlen,
        "moe_hidden": c.moe_hidden,
        "moe_routed_dim": c.moe_routed_dim,
        "moe_topk": c.moe_topk,
        "n_codebooks": c.n_codebooks,
        "n_experts": c.n_experts,
        "n_heads": c.n_heads,
        "n_kv_heads": c.n_kv_heads,
        "n_layers": c.n_layers,
        "norm_eps": c.norm_eps,
        "reference_max_frames": round(c.speaker_reference_max_seconds * 15),
        "rope_theta": c.rope_theta,
        "speaker_embedding_dim": c.speaker_embedding_dim,
        "text_rows": c.text_rows,
        "text_vocab": c.text_vocab,
    }


def is_lm_norm(name: str) -> bool:
    """The LM's norm gains: the RMS norms' and the GDN output norm's (`out_weight`)."""
    return name.endswith(("norm.weight", "rmsnorm_eda.weight", "out_weight"))


def lm_inputs(c):
    g = torch.Generator().manual_seed(7)
    pad_text = c.text_vocab  # the padding row
    ids = torch.full((ROWS, N_CB + 1), c.audio_pad_id, dtype=torch.long)
    ids[:, N_CB] = pad_text
    n_text = TEXT_ROWS.stop - TEXT_ROWS.start
    ids[TEXT_ROWS, N_CB] = torch.randint(4, c.text_vocab, (n_text,), generator=g)
    n_audio = AUDIO_ROWS.stop - AUDIO_ROWS.start
    ids[AUDIO_ROWS, :N_CB] = torch.randint(0, c.codebook_size, (n_audio, N_CB), generator=g)
    speaker = torch.randn(c.speaker_embedding_dim, generator=g, dtype=torch.float64)
    buckets = [2, c.condition_bucket_counts[1], 0]  # the second is the unknown bucket
    return ids, speaker, buckets


def lm_fixture(hf_module):
    import zonos3.reference.lm as reference
    from zonos3.reference.lm import LMCache, Zonos3LM, Zonos3LMConfig

    reference.GDN_HEADS, reference.GDN_HEAD_DIM = GDN_TOY

    config = Zonos3LMConfig.from_params({"model": TINY})
    model = build(lambda: Zonos3LM(config), seed=1234)
    init_weights(model, is_norm=is_lm_norm)
    ids, speaker, buckets = lm_inputs(config)
    conditioning = {
        "speaker_rows": {0: speaker},
        "condition_rows": {1 + i: b for i, b in enumerate(buckets)},
    }

    with torch.inference_mode():
        capture: dict = {}
        logits = model(ids, LMCache(), capture=capture, **conditioning).logits

        cache = LMCache()
        model(ids[:CHUNK_SPLIT], cache, **conditioning)
        chunked = model(ids[CHUNK_SPLIT:], cache).logits

        cache = LMCache()
        model(ids[:DECODE_FROM], cache, **conditioning)
        decoded = torch.cat([model(ids[r : r + 1], cache).logits for r in range(DECODE_FROM, ROWS)])

    for name, other, rows in (
        ("chunked", chunked, slice(CHUNK_SPLIT, ROWS)),
        ("decode", decoded, slice(DECODE_FROM, ROWS)),
    ):
        gap = (other - logits[rows]).abs().max().item()
        # float32's scale, not float64's: the reference's float32 steps sum in another order
        # over one row than over many (upstream holds its own at 1e-5)
        assert gap < 1e-6, f"{name} logits differ from the prefill's by {gap}"

    lm_config = hf_lm_section(config)
    hf_model = hf_module.Zonos3MiniLM(lm_config).to(torch.float64).eval()
    hf_model.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        hf_logits = hf_model(ids, hf_module.Zonos3MiniCache(), **conditioning)
    hf_gap = (hf_logits - logits).abs().max().item()
    assert hf_gap < 1e-5, f"the -hf port's logits differ from the reference's by {hf_gap}"

    keep = {
        name: torch.from_numpy(value)
        for name, value in capture.items()
        if not name.startswith(("gdn_", "layer_x_"))
        or name in ("gdn_recurrent_0", "gdn_gated_0", "gdn_state_0", "gdn_conv_state_0")
    }
    io = {
        "ids": ids,
        "speaker_embedding": speaker,
        "condition_buckets": torch.tensor(buckets),
        **keep,
        "logits": logits,
        "chunked_logits": chunked,
        "decode_logits": decoded,
    }
    tensors = weights(model, rename=lambda k: f"lm.{k}")
    report = {
        "chunked vs prefill": (chunked - logits[CHUNK_SPLIT:]).abs().max().item(),
        "decode vs prefill": (decoded - logits[DECODE_FROM:]).abs().max().item(),
        "-hf port vs reference": hf_gap,
    }
    return tensors, io_tensors(io, prefix="io.lm."), lm_config, report


def codec_fixture(hf_module):
    model = build(lambda: hf_module.Zonos3MiniCodec(CODEC), seed=4321)
    init_weights(model, norm_bias=0.1)
    model = model.float()
    g = torch.Generator().manual_seed(11)
    codes = torch.randint(
        0, CODEC["codebook_size"], (1, 1 + CODEC["residual_codebooks"], CODEC_FRAMES), generator=g
    )

    capture = Capture()
    capture.input("latent", model.stages[0])
    for i, module in enumerate(model.stages):
        name = f"stage_{i // 2}" if i % 2 == 0 else f"patch_{i // 2}"
        capture.output(name, module)
    capture.input("branches", model.head.mixer)
    capture.output("mixer", model.head.mixer)
    with torch.inference_mode():
        audio = model(codes)
        again = model(codes)
    assert torch.equal(audio, again), "the codec toy is not deterministic"

    values = dict(capture.values)
    branches = values.pop("branches")
    io = {"codes": codes, **values, "y0": branches[:, 0], "y1": branches[:, 1], "audio": audio}
    return weights(model, rename=lambda k: f"codec.{k}"), io_tensors(io, prefix="io.codec.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--upstream", type=pathlib.Path, required=True, help="Zyphra/ZONOS3-Mini checkout"
    )
    parser.add_argument(
        "--hf", type=pathlib.Path, required=True, help="the -hf repo's two .py files"
    )
    args = parser.parse_args()
    check_upstream(args.upstream)
    hf_module = load_hf(args.hf)

    lm_weights, lm_io, lm_config, report = lm_fixture(hf_module)
    codec_weights, codec_io = codec_fixture(hf_module)

    write(OUT / "toy.safetensors", lm_weights, codec_weights, lm_io, codec_io)
    write_json(OUT / "toy.config.json", {"lm": lm_config, "codec": CODEC}, indent=2, sort_keys=True)

    for name, gap in report.items():
        print(f"{name}: max |diff| {gap:.3e}")
    for name, shape in shapes({**lm_io, **codec_io}).items():
        print(f"{name}: {list(shape)}")
    print(f"{len(lm_weights)} LM and {len(codec_weights)} codec tensors -> {OUT}")


if __name__ == "__main__":
    main()
