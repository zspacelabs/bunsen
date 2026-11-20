#!/usr/bin/env python3
"""Generate the Qwen3-TTS toy-model fixtures: a tiny model (talker and code predictor)
and a tiny codec decoder.

Run once; commit the outputs. Regenerating is not part of CI.

    python3.12 -m venv venv
    venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \\
        torch==2.9.1 torchaudio==2.9.1
    venv/bin/pip install transformers==4.57.3 accelerate==1.12.0 safetensors numpy \\
        einops librosa soundfile sox onnxruntime
    venv/bin/pip install --no-deps qwen-tts==0.1.1
    venv/bin/python -I tools/gen_qwen3_tts_toy_fixtures.py

`qwen_tts` imports `sox`, `onnxruntime`, `librosa` and `soundfile` at module level (its
25 Hz tokenizer and its audio I/O); the toy uses none of them.

Pinned versions used to produce the committed fixture:
    qwen-tts 0.1.1 (byte-identical to `QwenLM/Qwen3-TTS` @ 022e286b), transformers
    4.57.3, torch 2.9.1+cpu, safetensors 0.8.0, Python 3.12 (checked at start).

The oracles are `qwen_tts`'s own classes: `Qwen3TTSForConditionalGeneration` (the talker,
its text projection and codec head, and the code predictor) and
`Qwen3TTSTokenizerV2Decoder` (the 12 Hz codec's decoder), at toy sizes, seeded random
weights, on the CPU in float64 (the rotary tables are float32 by upstream's design), stored
as float32. The toy keeps the release's shape rules:

- Talker: a head width that is not `hidden_size / heads` (as 0.6B's), grouped-query
  attention, interleaved M-RoPE with `sum(mrope_section) == head_dim / 2` (every section
  reads the same positions, so it is 1-D RoPE), a text width that is not the talker's
  (so the resize MLP changes width), and the codec table laid out as the release's:
  the codes first, then 1024 control rows (`generate` suppresses the last 1024 rows but
  EOS), each control id the release's minus 2016 (`2048 - 32`).
- Code predictor: narrower than the talker, so `small_to_mtp_projection` is a linear
  (1.7B's; 0.6B's is the identity), with per-group tables at the talker's width.
- Codec decoder: the split RVQ (one semantic book, the rest summed), a transformer whose
  head width is not `hidden / heads`, a sliding window shorter than a chunk, two x2
  upsamples, two vocoder blocks; one codebook usage below the clamp, and in the codes;
  the last conv scaled down (`AUDIO_SCALE`), so the audio is mostly inside its clamp.

Weights are seeded random; the norm gains about one (`toy_fixture.init_weights`). The
codebooks' usages are drawn in `[0.5, 1.5)`, not by fan-in, since they divide.

Outputs, under `crates/public/bunsen/testdata/qwen3_tts/`:

`toy.safetensors`
    - `talker.*`: the model's weights, under the checkpoint's names.
    - `decoder.*`: the codec decoder's, under the codec checkpoint's names.
    - `io.<mode>.*`, one greedy take per prompt mode (`MODES`):
        - `input_ids` `[1, n]` (`<|im_start|>assistant\\n`, the text, then the five
          trailing ids, as `qwen_tts`' processor lays them out); `instruct_ids` `[1, m]`
          when the mode has an instruction;
        - `prompt_embeds` `[1, L, 32]`: the talker's prefill input, as `generate` builds
          it; `trailing_text` `[1, t, 32]`: what each step adds after it;
        - `codes` `[frames, 4]`: the take's codes (codebook 0 from the talker, 1-3 from
          the predictor), greedy under upstream's processors (the repetition penalty
          1.05 over codebook 0's history, the control rows but EOS suppressed, EOS
          blocked for 2 frames), the EOS frame not included;
        - `step_embeds` `[steps, 32]`: each decode step's talker input (the summed codec
          embeddings of the frame before, plus its trailing text or the pad);
        - `talker_logits` `[1 + steps, 1056]`: the talker's last-position logits at the
          prefill and at each step, before the logits processors (row `k` chooses frame
          `k`'s codebook 0; the last row's choice ends the take unrun);
        - `predictor_logits` `[frames, 3, 32]`: the predictor's logits for groups 1-3 of
          each frame, before the logits processors.
    - `io.prefill.*`: the first mode's prefill, through the stack: `layer_{i}` `[1, L,
      32]` each layer's output, `norm` the final norm's.
    - `io.codec.*`: the codec on fixed codes: `codes` `[1, 4, 20]`; `quantized`
      `[1, 16, 20]` (the RVQ sum); `pre_conv` `[1, 24, 20]`; `transformer` `[1, 20, 24]`
      (the pre-transformer's output, its projection included); `upsample_{i}` (after
      each x2 stage); `decoder_{i}` (after each vocoder module); `audio` `[1, 1, 480]`
      (the whole decode); `chunked_audio` `[1, 1, 480]` (`chunked_decode`, chunks of
      `CHUNK` frames with `CONTEXT` of left context).
`toy.config.json`
    `{"model": ..., "codec": {"decoder_config": ...}}`: the model's `config.json` and the
    codec's `decoder_config` section, as the checkpoints write them.
"""

import importlib.metadata
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from toy_fixture import (  # noqa: E402
    TESTDATA,
    build,
    init_weights,
    io_tensors,
    shapes,
    weights,
    write,
    write_json,
)

PINS = {"qwen-tts": "0.1.1", "transformers": "4.57.3", "torch": "2.9.1+cpu"}

OUT = TESTDATA / "qwen3_tts"

ARCHITECTURES = ["Qwen3TTSForConditionalGeneration"]
"""The model `config.json`'s, as the release's."""

CODES, GROUPS = 32, 4
"""Codebook bins and groups, the talker's and the codec's alike."""

CONTROL_SHIFT = 2048 - CODES
"""A release control id, minus this, is the toy's."""


def control(release_id: int) -> int:
    return release_id - CONTROL_SHIFT


TEXT_VOCAB = 48
IM_START, IM_END, ASSISTANT, NEWLINE, USER = 40, 41, 42, 43, 44
TTS_PAD, TTS_BOS, TTS_EOS = 45, 46, 47

TALKER = dict(
    vocab_size=CODES + 1024,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    hidden_act="silu",
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    rope_theta=1000000,
    rope_scaling={"interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
    attention_bias=False,
    num_code_groups=GROUPS,
    text_hidden_size=40,
    text_vocab_size=TEXT_VOCAB,
    codec_pad_id=control(2148),
    codec_bos_id=control(2149),
    codec_eos_token_id=control(2150),
    codec_think_id=control(2154),
    codec_nothink_id=control(2155),
    codec_think_bos_id=control(2156),
    codec_think_eos_id=control(2157),
    codec_language_id={
        "chinese": control(2055),
        "english": control(2050),
        "sichuan_dialect": control(2062),
    },
    spk_id={"ryan": control(3061), "eric": control(2875)},
    spk_is_dialect={"ryan": False, "eric": "sichuan_dialect"},
    code_predictor_config=dict(
        vocab_size=CODES,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        num_code_groups=GROUPS,
        layer_types=["full_attention"] * 2,
    ),
)

MODEL = dict(
    talker_config=TALKER,
    tokenizer_type="qwen3_tts_tokenizer_12hz",
    tts_model_size="toy",
    tts_model_type="custom_voice",
    im_start_token_id=IM_START,
    im_end_token_id=IM_END,
    assistant_token_id=ASSISTANT,
    tts_pad_token_id=TTS_PAD,
    tts_bos_token_id=TTS_BOS,
    tts_eos_token_id=TTS_EOS,
)

DECODER = dict(
    codebook_size=CODES,
    codebook_dim=16,
    hidden_size=16,
    latent_dim=24,
    max_position_embeddings=256,
    rope_theta=10000,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=12,
    attention_bias=False,
    sliding_window=4,
    intermediate_size=32,
    hidden_act="silu",
    layer_scale_initial_scale=0.01,
    rms_norm_eps=1e-5,
    num_hidden_layers=2,
    num_quantizers=GROUPS,
    num_semantic_quantizers=1,
    upsample_rates=[3, 2],
    upsampling_ratios=[2, 2],
    decoder_dim=16,
    attention_dropout=0.0,
)
"""The codec's `decoder_config`, at toy sizes: 3 x 2 x 2 x 2 = 24 samples a frame."""

CODEC_FRAMES, CHUNK, CONTEXT = 20, 8, 3
"""The codec case: 20 frames, decoded whole and in chunks of 8 with 3 of left context
(upstream's 300 and 25, scaled)."""

AUDIO_SCALE = 16.0
"""The vocoder's last conv is divided by this: at fan-in scale the toy's audio is nine
parts in ten on the clamp; divided, a few samples are."""

TAKE_FRAMES = 6
"""Each mode's greedy take is at most this many frames."""

MODES = {
    # name: (speaker, language, instruction, non_streaming)
    "english": ("ryan", "english", False, False),
    "english_upfront": ("ryan", "english", False, True),
    "auto": ("ryan", "auto", False, False),
    "dialect": ("eric", "auto", False, False),
    "instructed": ("ryan", "chinese", True, True),
    "described": (None, "english", True, True),
}
"""The prompt modes: a stock speaker with a language, text interleaved (upstream's
"streaming") or upfront ("non-streaming"); `auto` (no language row); a dialect speaker
under `auto` (upstream's override to its dialect); an instruction (1.7B CustomVoice's);
no speaker but a description (VoiceDesign's)."""

TEXT_LEN, INSTRUCT_LEN = 7, 4


def check_pins() -> None:
    for name, want in PINS.items():
        got = torch.__version__ if name == "torch" else importlib.metadata.version(name)
        if got != want:
            sys.exit(f"{name} is {got}, not the pinned {want}")


def text_ids(g: torch.Generator, n: int) -> torch.Tensor:
    return torch.randint(0, IM_START, (n,), generator=g)


def input_ids(g: torch.Generator) -> torch.Tensor:
    """`<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n`."""
    head = torch.tensor([IM_START, ASSISTANT, NEWLINE])
    tail = torch.tensor([IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE])
    return torch.cat([head, text_ids(g, TEXT_LEN), tail])[None]


def instruct_ids(g: torch.Generator) -> torch.Tensor:
    """`<|im_start|>user\\n{instruction}<|im_end|>\\n`."""
    head = torch.tensor([IM_START, USER, NEWLINE])
    tail = torch.tensor([IM_END, NEWLINE])
    return torch.cat([head, text_ids(g, INSTRUCT_LEN), tail])[None]


class Calls:
    """Each call of a module: its keyword arguments, its `inputs_embeds` when given, and
    its output's last-position `logits`."""

    def __init__(self, module: torch.nn.Module, *, logits: bool):
        self.embeds: list = []
        self.logits: list = []
        self.kwargs: list = []
        module.register_forward_pre_hook(self._pre, with_kwargs=True)
        if logits:
            module.register_forward_hook(self._post)

    def _pre(self, _module, _args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None:
            self.embeds.append(embeds.detach().clone())
        self.kwargs.append(kwargs)

    def _post(self, _module, _args, out):
        self.logits.append(out.logits[:, -1].detach().clone())

    def clear(self) -> None:
        self.embeds.clear()
        self.logits.clear()
        self.kwargs.clear()


def model_fixture():
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    config = Qwen3TTSConfig(**MODEL)
    model = build(lambda: Qwen3TTSForConditionalGeneration(config), seed=1234)
    init_weights(model)
    talker = model.talker

    talker_model = Calls(talker.model, logits=False)
    talker_out = Calls(talker, logits=True)
    predictor = Calls(talker.code_predictor, logits=True)
    layers: dict = {}

    def keep_first(name):
        """Keeps a module's output at its first call: the prefill's."""

        def hook(_m, _i, out):
            out = out[0] if isinstance(out, tuple) else out
            layers.setdefault(name, out.detach().clone())

        return hook

    for i, layer in enumerate(talker.model.layers):
        layer.register_forward_hook(keep_first(f"layer_{i}"))
    talker.model.norm.register_forward_hook(keep_first("norm"))

    g = torch.Generator().manual_seed(7)
    io: dict = {}
    report: dict = {}
    for mode, (speaker, language, instructed, non_streaming) in MODES.items():
        ids = input_ids(g)
        instruct = instruct_ids(g) if instructed else None
        for calls in (talker_model, talker_out, predictor):
            calls.clear()
        with torch.inference_mode():
            codes_list, _ = model.generate(
                input_ids=[ids],
                instruct_ids=[instruct],
                languages=[language],
                speakers=[speaker],
                non_streaming_mode=non_streaming,
                max_new_tokens=TAKE_FRAMES,
                do_sample=False,
                subtalker_dosample=False,
                repetition_penalty=1.05,
            )
        codes = codes_list[0]
        prefill = talker_out.kwargs[0]
        # the talker's inputs: the prefill, then one row a step
        prompt = talker_model.embeds[0]
        steps = torch.cat([e[0] for e in talker_model.embeds[1:]], dim=0)
        # the predictor runs once a frame per step after the prefill: a prefill of two
        # rows (the talker's hidden state, codebook 0's embedding), then one row a group
        per_frame = GROUPS - 1
        assert len(predictor.logits) % per_frame == 0
        predictor_logits = torch.stack(predictor.logits).view(-1, per_frame, CODES)
        frames = codes.shape[0]
        assert predictor_logits.shape[0] >= frames
        io[mode] = {
            "input_ids": ids,
            **({"instruct_ids": instruct} if instructed else {}),
            "prompt_embeds": prompt,
            "trailing_text": prefill["trailing_text_hidden"],
            "codes": codes,
            "step_embeds": steps,
            "talker_logits": torch.cat(talker_out.logits, dim=0),
            "predictor_logits": predictor_logits,
        }
        report[mode] = (tuple(prompt.shape), frames)

    tensors = weights(model)
    out = {}
    for mode, values in io.items():
        out.update(io_tensors(values, prefix=f"io.{mode}."))
    out.update(io_tensors(layers, prefix="io.prefill."))
    return tensors, out, model.config, report


@torch.no_grad()
def init_codebooks(model: torch.nn.Module, g: torch.Generator) -> None:
    """Each codebook's usages in `[0.5, 1.5)`, and the first book's code 5 below the
    clamp (its sum scaled down with it, so the code still reads as a unit vector)."""
    for name, p in model.named_parameters():
        if name.endswith("cluster_usage"):
            p.copy_(0.5 + torch.rand(p.shape, generator=g, dtype=p.dtype))
    first = model.quantizer.rvq_first.vq.layers[0]._codebook
    first.cluster_usage[5] = 3e-6
    first.embedding_sum[5] *= 1e-5


def codec_fixture():
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    config = Qwen3TTSTokenizerV2DecoderConfig(**DECODER)
    model = build(lambda: Qwen3TTSTokenizerV2Decoder(config), seed=4321)
    init_weights(model, norm_bias=0.1)
    g = torch.Generator().manual_seed(11)
    init_codebooks(model, g)
    with torch.no_grad():
        # the vocoder's last conv, scaled so the audio is mostly inside the clamp
        model.decoder[-1].conv.weight /= AUDIO_SCALE
        model.decoder[-1].conv.bias /= AUDIO_SCALE
    codes = torch.randint(0, CODES, (1, GROUPS, CODEC_FRAMES), generator=g)
    codes[0, 0, 3] = 5  # the clamped code

    values: dict = {}

    def keep(name):
        def hook(_m, _i, out):
            out = getattr(out, "last_hidden_state", out)
            values.setdefault(name, out.detach().clone())

        return hook

    model.pre_conv.register_forward_hook(keep("pre_conv"))
    model.pre_transformer.register_forward_hook(keep("transformer"))
    for i, stage in enumerate(model.upsample):
        stage[-1].register_forward_hook(keep(f"upsample_{i}"))
    for i, module in enumerate(model.decoder):
        module.register_forward_hook(keep(f"decoder_{i}"))

    with torch.inference_mode():
        quantized = model.quantizer.decode(codes)
        audio = model(codes)
        whole = dict(values)
        chunked = model.chunked_decode(codes, chunk_size=CHUNK, left_context_size=CONTEXT)
    assert chunked.shape == audio.shape
    gap = (chunked - audio).abs().max().item()
    io = {"codes": codes, "quantized": quantized, **whole, "audio": audio,
          "chunked_audio": chunked}
    tensors = weights(model, rename=lambda k: f"decoder.{k}")
    return tensors, io_tensors(io, prefix="io.codec."), config, gap


def main() -> None:
    check_pins()
    model_weights, model_io, model_config, report = model_fixture()
    codec_weights, codec_io, codec_config, chunk_gap = codec_fixture()

    write(OUT / "toy.safetensors", model_weights, codec_weights, model_io, codec_io)
    config = {
        "model": {**model_config.to_diff_dict(), "architectures": ARCHITECTURES},
        "codec": {"decoder_config": codec_config.to_diff_dict()},
    }
    for section in (config["model"], config["codec"]["decoder_config"]):
        for k in ("transformers_version", "dtype", "torch_dtype"):
            section.pop(k, None)
    write_json(OUT / "toy.config.json", config, indent=2, sort_keys=True)

    for mode, (prompt, frames) in report.items():
        print(f"{mode}: prompt {list(prompt)}, {frames} frames")
    print(f"chunked vs whole decode: max |diff| {chunk_gap:.3e}")
    for name, shape in shapes({**model_io, **codec_io}).items():
        print(f"{name}: {list(shape)}")
    print(f"{len(model_weights)} model and {len(codec_weights)} codec tensors -> {OUT}")


if __name__ == "__main__":
    main()
