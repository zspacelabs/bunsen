#!/usr/bin/env python3
"""Generate `zonos3-model-validation`'s speaker-encoder fixture: the `-hf` port's encoder.

Run once; commit the output. Regenerating is not part of CI.

    # the venv and the -hf files of validation/tools/gen_zonos3_fixtures.py, with
    venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torchaudio==2.9.1
    venv/bin/python -I validation/tools/gen_zonos3_speaker_fixtures.py --hf hf

Pinned versions used to produce the committed fixture:
    torch 2.9.1+cpu, torchaudio 2.9.1+cpu, transformers 4.57.3, Python 3.12; the `-hf` files
    checked against `gen_zonos3_fixtures.HF_FILES` (sha256), and the `-hf` `config.json`
    against `HF_CONFIG` (its `speaker_encoder` section is the encoder's geometry).

Everything runs on the CPU in float32, as upstream runs the encoder (autocast off). The
oracle is the `-hf` port's `Zonos3MiniSpeakerEncoder` (op for op with upstream's
`MECTFbankSpeakerEmbedding`: torchaudio's sinc resample and Kaldi fbank in plain torch, the
MECT-B2 network and ASTP pooling), its weights the `-hf` file's `speaker_encoder.*`, strictly.
Its clips are the `-hf` codec's renders (`Zonos3MiniCodec`, as `gen_zonos3_fixtures.py`
loads it) of `VOICES`' whole stock reference codes, at the codec's 44.1 kHz, so the encoder's
resample to 16 kHz is exercised. The fixture holds no audio: a test renders the same codes with
bunsen's codec, which `codec` checks against this one.

Output, `validation/models/zonos3-model-validation/testdata/speaker.safetensors` (metadata
`voices`, a JSON list, and the pins), for each voice under `<voice>.`:

- `audio.stats` `[4]` (f64): the render's sample count, sum, absolute sum and RMS;
- `prepared.stats` `[4]` (f64): the same of the encoder's input after its resample, 60 s cut
  and int16 scale;
- `features.stats` `[4]` (f64): the same of the mean-normalized fbank, `[80, T]`; and
  `features.sample` (every `SAMPLE_STRIDE`-th value);
- `pooled` `[5120]`: the ASTP pooling;
- `embedding` `[192]`: the encoder's raw embedding (checked equal to the encoder's own
  `forward(wav, 44100)`);
- `stock_cosine` `[1]` (f64): its cosine to the voice's shipped `.embedding`, which
  upstream computed from the voice's source audio, not from this render.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
import gen_zonos3_fixtures  # noqa: E402
from gen_zonos3_fixtures import HF_FILES, load_hf_codec, sha256, write  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "models/zonos3-model-validation/testdata"

HF_CONFIG = "1692a0e9b715d6c92cd9aa83e7caeab9ec4f5fb1185042cd969455b8b0fc3ecc"
"""`Zyphra/ZONOS3-Mini-hf` @ 73500c5a's `config.json`: bunsen's pinned row's."""

VOICES = ["Savannah", "Arjun", "Cillian"]
SAMPLE_STRIDE = 101


def stats(x: torch.Tensor) -> torch.Tensor:
    x = x.double().flatten()
    return torch.tensor(
        [x.numel(), x.sum().item(), x.abs().sum().item(), x.square().mean().sqrt().item()],
        dtype=torch.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf", type=pathlib.Path, required=True)
    args = parser.parse_args()
    for name, digest in {**HF_FILES, "config.json": HF_CONFIG}.items():
        got = sha256(args.hf / name)
        if got != digest:
            sys.exit(f"{args.hf / name}: sha256 {got}, not the pinned {digest}")
    from safetensors import safe_open

    codec = load_hf_codec(args.hf)
    modeling = sys.modules["zonos3_hf.modeling_zonos3"]
    section = json.loads((args.hf / "config.json").read_text())["speaker_encoder"]
    encoder = modeling.Zonos3MiniSpeakerEncoder(section)
    state, stock = {}, {}
    with safe_open(args.hf / "model.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith("speaker_encoder."):
                state[k[len("speaker_encoder.") :]] = f.get_tensor(k).float()
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.float().eval()

    tensors = {}
    with safe_open(args.hf / "voices.safetensors", "pt") as f, torch.inference_mode():
        for voice in VOICES:
            codes = f.get_tensor(f"{voice}.codes").long()
            shipped = f.get_tensor(f"{voice}.embedding").float()
            audio = codec(codes.T[None]).float().flatten()
            wav = encoder.prepare_input(audio, 44100)
            c = encoder.c
            feats = modeling.kaldi_fbank(
                wav[0] * float(c["waveform_scale"]),
                float(c["sample_rate"]),
                int(c["num_mel_bins"]),
                float(c["frame_length_ms"]),
                float(c["frame_shift_ms"]),
            )[None].transpose(1, 2)
            feats = feats - feats.mean(-1, keepdim=True)
            pooled = encoder.encoding_layers(encoder.feature_extractor(feats))
            embedding = encoder.linear(encoder.bn(pooled)).flatten()
            whole = encoder(audio, 44100).flatten()
            assert torch.equal(embedding, whole), "the staged pass is the encoder's"
            cosine = torch.nn.functional.cosine_similarity(
                embedding.double(), shipped.double(), dim=0
            ).item()
            at = lambda k: f"{voice}.{k}"  # noqa: E731
            tensors[at("audio.stats")] = stats(audio)
            tensors[at("prepared.stats")] = stats(wav * float(c["waveform_scale"]))
            tensors[at("features.stats")] = stats(feats)
            tensors[at("features.sample")] = feats.flatten()[::SAMPLE_STRIDE].float().contiguous()
            tensors[at("pooled")] = pooled.flatten().float().contiguous()
            tensors[at("embedding")] = embedding.float().contiguous()
            tensors[at("stock_cosine")] = torch.tensor([cosine], dtype=torch.float64)
            print(
                f"{voice}: {len(codes)} frames, {audio.numel()} samples, "
                f"{feats.shape[-1]} fbank frames, stock cosine {cosine:.4f}"
            )

    gen_zonos3_fixtures.OUT = OUT
    meta = {
        "hf_revision": "73500c5a1fc3e0c4da69908e3cffd9346cdd97c2",
        "torch": torch.__version__,
        "voices": json.dumps(VOICES),
        "sample_stride": str(SAMPLE_STRIDE),
    }
    write("speaker.safetensors", tensors, meta)


if __name__ == "__main__":
    main()
