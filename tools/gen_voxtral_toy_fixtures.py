#!/usr/bin/env python3
"""Generate the Voxtral Realtime toy-model fixture.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/vxvenv
    /tmp/vxvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/vxvenv/bin/pip install transformers safetensors
    /tmp/vxvenv/bin/python tools/gen_voxtral_toy_fixtures.py

Pinned versions used to produce the committed fixture:
    transformers 5.22.1, torch 2.15.0+cpu, safetensors 0.8.0

A tiny `VoxtralRealtimeForConditionalGeneration` -- `transformers`' own implementation -- with
seeded random weights, run on the CPU in float64 and stored as float32. The toy keeps the
shape rules that matter: a causal convolution stem that halves the frame rate, an audio
encoder whose sliding window is shorter than its input, four encoder frames stacked per decoder
position, a text model with grouped-query attention, a head width that is not
`hidden_size / num_attention_heads`, a sliding window shorter than the decode, the delay
conditioning, and a tied head.

Output, `crates/public/bunsen/testdata/voxtral/toy.safetensors`:

- the toy's weights under the names `mistralai/Voxtral-Mini-4B-Realtime-2602`'s
  `model.safetensors` gives them (`audio_tower.*`, `multi_modal_projector.*`,
  `language_model.model.*`; the head is tied and not stored);
- `io.*`: the inputs and the outputs bunsen is pinned against: `features` `[1, mels, 64]`;
  the audio tower's `audio_stem` (the embedder's output), `audio_layer0` and `audio` (its
  final output); `audio_embeds` (projected); then the whole model on `input_ids` at
  `num_delay_tokens` `DELAYS[0]`: `text_layer0`, `text_hidden` (after the final norm),
  `logits`; and `logits_delay_{d}` for each of `DELAYS`.
"""

import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import VoxtralRealtimeConfig, VoxtralRealtimeForConditionalGeneration

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/voxtral"

FEATURE_FRAMES = 64
INPUT_IDS = [1, 32, 32, 33, 40, 41, 32, 50]
DELAYS = [6, 2]

CONFIG = dict(
    audio_config=dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_mel_bins=16,
        rms_norm_eps=1e-5,
        rope_parameters=dict(rope_theta=1e6, rope_type="default"),
        sliding_window=5,
        max_position_embeddings=1500,
    ),
    text_config=dict(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=10,
        rms_norm_eps=1e-5,
        rope_parameters=dict(rope_theta=1e6, rope_type="default"),
        sliding_window=7,
        tie_word_embeddings=True,
    ),
    downsample_factor=4,
    default_num_delay_tokens=DELAYS[0],
)


def file_name(name: str) -> str:
    """The checkpoint file's name for a state-dict name."""
    for prefix, file_prefix in [("model.audio_tower.", "audio_tower."),
                                ("model.multi_modal_projector.", "multi_modal_projector."),
                                ("model.language_model.", "language_model.model.")]:
        if name.startswith(prefix):
            return file_prefix + name[len(prefix):]
    raise ValueError(name)


def main() -> None:
    torch.manual_seed(0)
    config = VoxtralRealtimeConfig(**CONFIG)
    model = VoxtralRealtimeForConditionalGeneration(config).to(torch.float64).eval()
    # Randomize what initializes to constants (norm weights), so a misplaced one is caught.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "layer_norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    OUT.mkdir(parents=True, exist_ok=True)

    n_mels = CONFIG["audio_config"]["num_mel_bins"]
    features = 0.5 * torch.randn(1, n_mels, FEATURE_FRAMES, dtype=torch.float64)
    input_ids = torch.tensor([INPUT_IDS])

    tower = model.model.audio_tower
    captured = {}

    def keep(name):
        def hook(_m, _i, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return hook

    tower.embedder.register_forward_hook(keep("audio_stem"))
    tower.layers[0].register_forward_hook(keep("audio_layer0"))
    model.model.language_model.layers[0].register_forward_hook(keep("text_layer0"))

    with torch.no_grad():
        audio = tower(input_features=features).last_hidden_state
        audio_embeds = model.model.get_audio_features(input_features=features).pooler_output
        out = model.model(input_ids=input_ids, input_features=features,
                          num_delay_tokens=DELAYS[0])
        logits = {d: model(input_ids=input_ids, input_features=features,
                           num_delay_tokens=d).logits for d in DELAYS}
    assert audio_embeds.shape[1] == len(INPUT_IDS)

    io = {
        "features": features,
        "input_ids": input_ids.to(torch.float64),
        "audio_stem": captured["audio_stem"],
        "audio_layer0": captured["audio_layer0"],
        "audio": audio,
        "audio_embeds": audio_embeds,
        "text_layer0": captured["text_layer0"],
        "text_hidden": out.last_hidden_state,
        "logits": logits[DELAYS[0]],
    }
    io.update({f"logits_delay_{d}": v for d, v in logits.items()})
    tensors = {file_name(k): v.float().contiguous()
               for k, v in model.state_dict().items() if k != "lm_head.weight"}
    tensors.update({f"io.{k}": v.float().contiguous() for k, v in io.items()})
    save_file(tensors, OUT / "toy.safetensors")
    (OUT / "toy.json").write_text(json.dumps({"config": CONFIG, "delays": DELAYS}, indent=1)
                                  + "\n")
    print("wrote", OUT, {k: tuple(v.shape) for k, v in io.items()})


if __name__ == "__main__":
    main()
