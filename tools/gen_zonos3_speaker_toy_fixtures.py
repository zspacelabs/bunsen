#!/usr/bin/env python3
"""Generate the ZONOS3-Mini speaker-encoder toy fixture: a tiny MECT-B2.

Run once; commit the outputs. Regenerating is not part of CI.

    # the venv and the -hf files of tools/gen_zonos3_toy_fixtures.py
    venv/bin/python -I tools/gen_zonos3_speaker_toy_fixtures.py --hf hf

Pinned versions used to produce the committed fixture:
    torch 2.9.1+cpu, transformers 4.57.3, safetensors 0.8.0, Python 3.12;
    `Zyphra/ZONOS3-Mini-hf` 73500c5a's `modeling_zonos3.py` and `configuration_zonos3.py`
    (checked: sha256, by `gen_zonos3_toy_fixtures.load_hf`).

The oracle is the `-hf` port's `Zonos3MiniSpeakerEncoder` (`zonos3.models.mect_speaker`'s
MECT-B2 and ASTP, op for op), at toy sizes, run in float64 on a fixed feature map: the
fbank front end is `ops::signal::KaldiFbank`'s, tested on its own. The toy keeps every path
of the release's network:

- three stages, so the stage mix weighs three;
- a stage whose first block halves the frequencies (`stride_f` 2, with the reduce
  convolution back to `cin * 2` channels), and blocks that halve the time (`stride_t` 2)
  and repeat it back, over an odd number of frames;
- expansions of 2 (a reduce convolution) and 1 (none), and a shortcut convolution where a
  block's first `ResBlock` changes the channels or strides;
- grouped convolutions with more than one input channel per group (`conv2`'s groups are
  the block's input channels);
- the token soft-MoE of 2 experts, 4 heads of width 2, the 113-tap position convolution
  over fewer frames than its taps.

Weights are seeded random (`toy_fixture.init_weights`); the batch norms' running variances,
which the port keeps as parameters, are then set to `[0.5, 1.5)` and their means to
`0.3 N(0, 1)`, so eval mode's normalization is not the identity and never divides by a
negative variance.

Outputs, under `crates/public/bunsen/testdata/zonos3/`:

`speaker_toy.config.json`
    `{"speaker_encoder": ...}`: the toy's section of the `-hf` `config.json`.
`speaker_toy.safetensors`
    - `speaker_encoder.*`: the weights, under the `-hf` file's names;
    - `io.speaker.features` `[1, 16, 23]`: the input, a mean-normalized fbank's layout
      (`[batch, mel bins, frames]`);
    - `io.speaker.head` `[1, 4, 16, 23]`: the head's output;
    - `io.speaker.block_{s}_{j}` `[1, c, f, 23]`: each MECT block's output;
    - `io.speaker.mix` `[1, 64, 23]`: the stage mix (`dense3`);
    - `io.speaker.pooled` `[1, 128]`: the ASTP pooling;
    - `io.speaker.embedding` `[1, 5]`: the embedding.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from gen_zonos3_toy_fixtures import load_hf  # noqa: E402
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

OUT = TESTDATA / "zonos3"

SPEAKER = {
    "bottleneck_dim": 6,
    "embed_dim": 5,
    "feat_dim": 16,
    "frame_length_ms": 25.0,
    "frame_shift_ms": 10.0,
    "hidden_dim": 8,
    "in_channels": 4,
    "max_seconds": 60.0,
    "num_experts": 2,
    "num_mel_bins": 16,
    "sample_rate": 16000,
    "stages": [
        [[2, 4, 1, 1, 2]],
        [[2, 8, 2, 1, 2], [1, 8, 1, 2, 2]],
        [[1, 16, 2, 1, 1], [1, 16, 1, 2, 1]],
    ],
    "waveform_scale": 32768.0,
}
"""The speaker encoder's `config.json` section, at toy sizes."""

FRAMES = 23


@torch.no_grad()
def set_running_stats(model: torch.nn.Module) -> None:
    """The batch norms' running statistics, which the port keeps as parameters: variances
    in `[0.5, 1.5)`, means `0.3 N(0, 1)`."""
    for name, p in model.named_parameters():
        if name.endswith("running_var"):
            p.copy_(0.5 + torch.rand_like(p))
        elif name.endswith("running_mean"):
            p.copy_(0.3 * torch.randn_like(p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hf", type=pathlib.Path, required=True, help="the -hf repo's two .py files"
    )
    args = parser.parse_args()
    hf = load_hf(args.hf)

    model = build(lambda: hf.Zonos3MiniSpeakerEncoder(SPEAKER), seed=2026)
    init_weights(model)
    set_running_stats(model)

    mect = model.feature_extractor
    capture = Capture()
    capture.output("head", mect.head)
    for s in range(mect.n_stages):
        for j, block in enumerate(getattr(mect, f"mect_block{s + 1}")):
            capture.output(f"block_{s}_{j}", block)
    capture.output("mix", mect.dense3)

    features = torch.randn(1, SPEAKER["feat_dim"], FRAMES, dtype=torch.float64)
    with torch.no_grad():
        pooled = model.encoding_layers(mect(features))
        embedding = model.linear(model.bn(pooled))
    io = {"features": features, **capture.values, "pooled": pooled, "embedding": embedding}
    io = io_tensors(io, prefix="io.speaker.")
    tensors = weights(model, rename=lambda name: f"speaker_encoder.{name}")

    write(OUT / "speaker_toy.safetensors", tensors, io)
    write_json(OUT / "speaker_toy.config.json", {"speaker_encoder": SPEAKER}, indent=2, sort_keys=True)
    for name, shape in shapes(io).items():
        print(f"{name}: {list(shape)}")
    print(f"{len(tensors)} speaker-encoder tensors -> {OUT}")


if __name__ == "__main__":
    main()
