"""Per-family variant circuits on the language-model board (board.py).

A variant derives from its parent; the sockets it changes are computed and marked Δ.
Component = (title, [params], link?, kind?); None is an empty socket.
"""

SLOTS = ["embed", "stream", "stack", "final", "head", "tied", "layers", "pattern",
         "a_pre", "a_mix", "a_post", "qk", "pos", "score", "mask", "vres", "gate", "m_pre", "m_mlp", "m_post",
         # the speech board
         "fe", "stem", "enc", "enc_pos", "enc_mask", "proj", "bridge", "cond", "rows", "lm", "lm_ext", "heads", "search",
         "quant", "trunk", "upsample", "vocoder"]
EXTRA = {"tied": "Head tying", "layers": "Layer choices", "pattern": "Layer pattern"}

C = {}  # key -> (spec, parent key or None)


def define(key, parent=None, **kw):
    spec = {k: v for k, v in C[parent][0].items() if k != "note"} if parent else {}
    spec.update(kw)
    C[key] = (spec, parent)


def changed(key):
    spec, parent = C[key]
    if not parent:
        return set()
    p = C[parent][0]
    return {s for s in SLOTS if spec.get(s) != p.get(s)}


# ------------------------------------------------------------------ the generic board
define("generic", None, name="the generic language model",
       embed=("token table", ["then: RMS · LN · × √D", "(any or none)"]),
       stream=("smear · x₀-mix · backout", ["(nanochat 2026)"]),
       stack=("Uniform | Patterned", ["kinds; layer → kind", "a rotary table per kind"]),
       final=("RMS | LN", ["over the stream"]),
       head=("tied | own projection", ["slice · soft-cap", "logits in f32"]),
       tied=True,
       layers="RMS | LN · activation · biases",
       a_pre=("RMS | LN", ["pre-norm"]),
       a_mix=("attention | DeltaNet", ["causal or bidirectional", "GQA · or a linear state"]),
       a_post=("RMS | LN", ["the sandwich"]),
       qk=("RMS per head", ["before | after RoPE"]),
       pos=("RoPE", ["θ · partial · scaled"]),
       score=("q·k · s", ["scale · soft-cap"]),
       mask=("causal | bidirectional", ["± a sliding window"]),
       vres=("value embedding", ["gated into V"]),
       gate=("σ(W_g x) ⊙ y", ["elementwise | headwise"]),
       m_pre=("RMS | LN", ["pre-norm"]),
       m_mlp=("MLP | GLU", ["ReLU² · GELU · SiLU"]),
       m_post=("RMS | LN", ["the sandwich"]),
       pattern="FFWWWFDDDFWWWWWFDDDF",
       note="each family below fills these sockets; an empty (dashed) socket passes the wire straight through")

# ------------------------------------------------------------------ the Llama template (a boundary, not a kit)
RMS = ("RMSNorm", ["pre-norm"])
define("llama", None, name="the Llama template",
       embed=("token table", ["plain lookup"]),
       stream=None,
       stack=("Uniform", ["one layer config × L"]),
       final=("RMSNorm", ["learned gain"]),
       head=("own projection", ["D → V, no bias"]),
       tied=False,
       layers="RMSNorm · SiLU · no biases",
       a_pre=RMS, a_post=None, m_pre=RMS, m_post=None,
       a_mix=("causal attention", ["GQA (Llama 2 70B, 3)", "head width D / H"]),
       qk=None,
       pos=("RoPE", ["θ 10⁴ · every dim", "rotate-half"]),
       score=("q·k / √d", ["no cap"]),
       mask=("causal", ["the whole context"]),
       vres=None, gate=None,
       m_mlp=("SwiGLU", ["h ≈ 8D/3", "no biases"]),
       pattern="F" * 32,
       note="not a bunsen kit: the 2023 consensus decoder the language kits are read against")

# ------------------------------------------------------------------ nanochat
NRMS = ("RMSNorm", ["no gain (ones)"])
define("nanochat", "llama", name="nanochat (2025)",
       embed=("token table + RMS", ["parameter-free norm", "x₀ = norm(E[id])"]),
       stack=("Uniform", ["d20: 20 × 1280", "heads of 128, H_kv = H"]),
       final=NRMS,
       head=("own projection", ["no bias", "soft-cap 15"]),
       layers="RMS, no gain · ReLU² · no biases",
       a_pre=NRMS, m_pre=NRMS,
       a_mix=("causal attention", ["MHA (H_kv = H)", "head 128"]),
       qk=("RMS per head", ["after RoPE", "no gain"]),
       pos=("RoPE", ["θ 10⁴ · bf16 tables", "reverse sign"]),
       m_mlp=("MLP", ["D → 4D → D", "ReLU²"]),
       pattern="F" * 20,
       note="upstream's norms have no parameters; bunsen's gains load as ones, which is the same function")
define("nanochat-2026", "nanochat", name="nanochat (2026)",
       embed=("token table + RMS", ["vocab padded to × 64", "parameter-free norm"]),
       stream=("smear · λ, μ · backout", ["24-channel smear gate", "backout at L/2"]),
       stack=("per-layer windows", ["d20, pattern SSSL", "last layer always long"]),
       pos=("RoPE", ["θ 10⁵ · exact tables", "reverse sign"]),
       score=("q·k · g² / √d", ["QK gain g = 1.2"]),
       mask=("causal", ["S: ¼ context (×128)", "L: the whole context"]),
       vres=("value embedding", ["alternate layers + last", "gate 12 ch → (0, 3)"]),
       pattern="WWWF" * 5,
       note="the d18 community checkpoint sets every layer long; the pattern is config, not architecture")

# ------------------------------------------------------------------ Qwen3
define("qwen3", "llama", name="Qwen3",
       stack=("Uniform", ["0.6B: 28 × 1024", "16 q / 8 kv heads of 128"]),
       head=("tied | own", ["tied to 4B, own at 8B"]),
       tied=True,
       layers="RMSNorm ε 1e-6 · SiLU · no biases",
       a_mix=("causal attention", ["GQA 2:1 to 4:1", "head 128 ≠ D / H"]),
       qk=("RMS per head", ["before RoPE", "learned gain"]),
       pos=("RoPE", ["θ 10⁶ (2507: 5·10⁶)", "rotate-half"]),
       pattern="F" * 28,
       note="also the text model of Qwen3-ASR and the talker stack of Qwen3-TTS")

# ------------------------------------------------------------------ Gemma
GRMS = ("RMSNorm, 1 + w", ["pre-norm"])
GPOST = ("RMSNorm, 1 + w", ["the sandwich"])
define("gemma1", "llama", name="Gemma (1)",
       embed=("token table × √D", ["scale rounded to dtype"]),
       head=("tied", ["logits in f32"]),
       tied=True,
       stack=("Uniform", ["2B: 18 × 2048", "7B: 28 × 3072"]),
       final=("RMSNorm, 1 + w", ["zero-centred gain"]),
       layers="RMS (1 + w) · GELU-tanh · no biases",
       a_pre=GRMS, m_pre=GRMS,
       a_mix=("causal attention", ["MQA (2B) · MHA (7B)", "head 256"]),
       m_mlp=("GeGLU", ["GELU (tanh)", "h = 8D"]),
       pattern="F" * 18)
define("gemma2", "gemma1", name="Gemma 2",
       stack=("Patterned", ["2B: 26 × 2304", "sliding, full alternate"]),
       head=("tied", ["soft-cap 30 · f32"]),
       a_post=GPOST, m_post=GPOST,
       a_mix=("causal attention", ["GQA 2:1 · head 256"]),
       score=("q·k / √256", ["query_pre_attn_scalar", "soft-cap 50"]),
       mask=("causal", ["window 4096 on", "alternate layers"]),
       pattern="WF" * 13)
define("gemma3", "gemma2", name="Gemma 3",
       stack=("Patterned", ["1B: 26 × 1152", "5 sliding : 1 full"]),
       head=("tied", ["no cap · f32"]),
       a_mix=("causal attention", ["1B: 4 q / 1 kv", "head 256"]),
       qk=("RMS per head, 1 + w", ["before RoPE"]),
       pos=("two RoPE tables", ["local θ 10⁴ · global 10⁶", "4B: global × 8, linear"]),
       score=("q·k / √256", ["query_pre_attn_scalar", "no cap"]),
       mask=("causal", ["window 512 (1B)", "1024 (4B)"]),
       pattern=("WWWWWF" * 5)[:26])

# ------------------------------------------------------------------ Qwen3.5
Q5RMS = ("RMSNorm, 1 + w", ["pre-norm"])
define("qwen35", "qwen3", name="Qwen3.5, full-attention layer",
       block="DECODER LAYER, FULL KIND  ×  L / 4   (every fourth layer)",
       stack=("Patterned (hybrid)", ["0.8B: 24 × 1024", "3 DeltaNet : 1 full"]),
       head=("tied | own", ["tied to 4B, own at 9B, 27B"]),
       final=("RMSNorm, 1 + w", ["zero-centred gain"]),
       layers="RMS (1 + w) · SiLU · no biases",
       a_pre=Q5RMS, m_pre=Q5RMS,
       a_mix=("causal attention", ["8 q / 2 kv heads", "head 256"]),
       qk=("RMS per head, 1 + w", ["before RoPE"]),
       pos=("partial RoPE", ["64 of 256 dims", "θ 10⁷"]),
       gate=("σ(W_g x) ⊙ y", ["elementwise", "W_Q twice as wide"]),
       pattern="DDDF" * 6,
       note="the vision tower and the multi-token-prediction head of the released checkpoints are not loaded")
define("qwen35-linear", "qwen35", name="Qwen3.5, DeltaNet layer",
       block="DECODER LAYER, LINEAR KIND  ×  3L / 4",
       a_mix=("gated DeltaNet", ["16 k heads, 16–48 v heads", "state 128 × 128 per v head"], "c-deltanet.html", "mix"),
       qk=("L2 per head", ["inside the rule"], "c-deltanet.html#rule"),
       pos=("short causal conv", ["4 taps, depthwise", "no RoPE"], "c-deltanet.html#block"),
       score=("the delta rule", ["decay g · write β", "no softmax"], "c-deltanet.html#rule"),
       mask=("causal", ["by recurrence"], "c-deltanet.html#state"),
       gate=("gated RMSNorm", ["w · x̂ · SiLU(z)"], "c-deltanet.html#block"),
       note="the feed-forward sublayer is the full layer's: only the token mixer changes kind")

# ------------------------------------------------------------------ ModernBERT and Laya
LN = ("LayerNorm", ["no bias · pre-norm"])
define("modernbert", "llama", name="ModernBERT",
       block="ENCODER LAYER  ×  L   (bidirectional; layer 0 has no mixer pre-norm)",
       embed=("token table + LN", ["LayerNorm, no bias"]),
       stack=("Patterned", ["base: 22 × 768", "large: 28 × 1024"]),
       final=("LayerNorm", ["no bias"]),
       head=("no head", ["hidden states out", "a user's head reads them"], "f-modernbert.html#head"),
       layers="LayerNorm (no bias) ε 1e-5 · GELU · no biases",
       a_pre=("LayerNorm", ["no bias", "(layer 0: none)"]), m_pre=LN,
       a_mix=("self-attention", ["MHA, head 64", "bidirectional"]),
       pos=("two RoPE tables", ["global θ 1.6·10⁵", "local θ 10⁴"]),
       mask=("bidirectional", ["local: ±64 window", "+ padding key mask"]),
       m_mlp=("GeGLU", ["GELU (exact)", "h = 1.5D (base)"]),
       pattern=("FWW" * 8)[:22],
       note="every third layer global, from the first; the others local")
define("laya-head", "modernbert", name="Laya's head layer",
       block="HEAD LAYER  ×  2   (PyTorch's TransformerEncoderLayer, pre-norm, over the encoder's states)",
       embed=("encoder states", ["+ question-kind embedding"], "f-laya.html#model"),
       stack=("Uniform", ["2 layers, width d"]),
       final=None,
       head=("option scorer", ["LN → MLP d → d → 1", "at each [MASK] · + act head"], "f-laya.html#head", "act"),
       layers="LayerNorm (biased) · ReLU · biases",
       a_pre=("LayerNorm", ["biased"]), m_pre=("LayerNorm", ["biased"]),
       a_mix=("self-attention", ["max(1, d/64) heads", "biased projections"]),
       pos=None,
       mask=("bidirectional", ["prompt tokens only"]),
       m_mlp=("MLP", ["D → 4D → D", "ReLU, biased"]),
       pattern="FF",
       note="positions reach the head only through the encoder's states: its own attention has no rotary table")


# ================================================================== the speech board (board="sp")
define("sp-generic", None, board="sp", name="the generic speech model",
       fe=("log-mel | fbank", ["log₁₀ | ln", "and a level rule"]),
       stem=("strided convs", ["1-d | 2-d", "× 2 to × 8"]),
       enc=("transformer | Conformer", ["pre-norm, D × N"]),
       enc_pos=("learned | sinusoidal", ["| RoPE | relative"]),
       enc_mask=("bidirectional | blocks", ["| causal window"]),
       proj=("linear | MLP", ["| frame stacking"]),
       bridge=("cross-attn | splice", ["| add | joint"]),
       cond=("delay | speaker", ["| condition rows"]),
       rows=("tokens | summed rows", ["text, codebooks"]),
       lm=("decoder stack", ["the LM board", "(or an LSTM)"]),
       lm_ext=("depth decoder", ["a frame's codebooks"]),
       heads=("vocab | codebooks", ["| + durations"]),
       search=("greedy | beam", ["| sample"]),
       quant=("RVQ", ["K codebooks → latent"]),
       trunk=("causal transformer", ["windowed"]),
       upsample=("conv-T | patch-up", ["+ ConvNeXt"]),
       vocoder=("DAC | ISTFT", ["waveform out"]),
       note="recognizers fill the top three rows; synthesizers the bottom two (and AUDIO IN, to clone a voice)")

define("whisper", None, board="sp", name="Whisper",
       fe=("log-mel, log₁₀", ["80 mels (v3: 128)", "max − 8, (x + 4) / 4"]),
       stem=("2 × conv1d", ["k3, the 2nd stride 2", "GELU · 100 → 50 Hz"]),
       enc=("pre-norm encoder", ["LN · MHA · GELU 4D", "biased, but keys not"]),
       enc_pos=("learned table", ["1500 slots", "(stored sinusoids)"]),
       enc_mask=("bidirectional", ["one 30 s window"]),
       proj=None,
       bridge=("cross-attention", ["a third sublayer", "in every decoder layer"]),
       cond=None,
       rows=("text tokens", ["+ learned positions", "448 slots"]),
       lm=("Transformer decoder", ["LN · self · cross · GELU", "4 (tiny) … 32 (large)"]),
       lm_ext=None,
       heads=("tied head", ["51864–51866 tokens", "incl. 1501 timestamps"]),
       search=("greedy | beam", ["timestamp rules", "temperature fallback"]),
       quant=None, trunk=None, upsample=None, vocoder=None,
       note=".en: GPT-2's vocabulary, no language or task token · large-v3: 128 mels · turbo: 4 decoder layers")
define("qwen3asr", "whisper", name="Qwen3-ASR",
       fe=("log-mel, log₁₀", ["128 mels", "each window's max − 8"]),
       stem=("3 × conv2d", ["s2 each, per 1 s chunk", "100 → 13 Hz · linear"]),
       enc=("pre-norm encoder", ["LN · MHA · GELU", "0.6B: 18 × 896, biased"]),
       enc_pos=("sinusoidal", ["restart every chunk"]),
       enc_mask=("blocks", ["8 s (104 frames)"]),
       proj=("MLP", ["896 → 896 → 1024", "GELU"]),
       bridge=("splice", ["frames replace the", "<|audio_pad|> tokens"]),
       rows=("chat prompt", ["system · user (audio)", "· assistant"]),
       lm=("Qwen3", ["0.6B: 28 × 1024", "its own circuit"], "f-qwen3.html"),
       heads=("tied head", ["151936 tokens"]),
       search=("greedy", ["\"language X<asr_text>…\"", "≤ 512 tokens"]),
       note="word times come from eight decoder heads' attention onto the audio rows, aligned by a monotonic DTW")
define("voxtral", "qwen3asr", name="Voxtral Realtime",
       fe=("log-mel, log₁₀", ["128 mels", "fixed level 1.5"]),
       stem=("2 × causal conv1d", ["left-padded, s2", "100 → 50 Hz"]),
       enc=("causal encoder", ["RMS · MHA · SwiGLU", "32 × 1280"]),
       enc_pos=("RoPE", ["θ 10⁶"]),
       enc_mask=("causal", ["15 s sliding window"]),
       proj=("stack 4 + MLP", ["4 × 1280 → 3072 → 3072", "80 ms per position"]),
       bridge=("add", ["embed(token) + audio(i)", "one group per position"]),
       cond=("delay", ["sinusoid → 1 + f(·)", "scales every FFN input"]),
       rows=("stream tokens", ["<s> + 38 PAD", "then one per 80 ms"]),
       lm=("Ministral-style", ["26 × 3072, GQA 32 / 8", "window 8192, tied"]),
       heads=("tied head", ["131072 tokens"]),
       search=("greedy", ["PAD · [W] · text", "words timed by position"]),
       note="every part is causal: one decoder step per 80 ms of audio, a fixed delay (480 ms by default) behind it")
define("parakeet", None, board="sp", name="Parakeet TDT",
       fe=("log-mel, ln", ["128 mels, preemphasis", "per-mel mean and std"]),
       stem=("dw-striding × 8", ["conv2d, 2 dw-separable", "100 → 12.5 Hz"]),
       enc=("Conformer", ["½FFN · MHSA · conv · ½FFN", "24 × 1024, no biases"], "s-conformer.html"),
       enc_pos=("relative", ["Transformer-XL u, v", "sinusoidal offsets"], "s-conformer.html#relpos"),
       enc_mask=("bidirectional", ["windows ≤ 60 s"]),
       proj=("encoder projector", ["1024 → 640"]),
       bridge=("joint", ["ReLU(enc + pred)", "one frame at a time"], "s-transducer.html#joint"),
       cond=None,
       rows=("last token", ["blank to start"], "s-transducer.html#joint"),
       lm=("prediction net", ["embed + LSTM × 2", "640 wide"], "s-transducer.html#joint", "conv"),
       lm_ext=None,
       heads=("token + duration", ["8192 + blank", "durations 0–4"], "s-transducer.html"),
       search=("greedy TDT", ["skip by duration", "≤ 10 tokens a frame"], "s-transducer.html#tdt"),
       quant=None, trunk=None, upsample=None, vocoder=None,
       note="Phonon-2 is the same circuit with re-quantized weights; Sortformer v2 reuses this encoder for diarization")
define("qwen3tts", None, board="sp", name="Qwen3-TTS",
       fe=None, stem=None, enc=None, enc_pos=None, enc_mask=None, proj=None, bridge=None, cond=None,
       rows=("text + codec rows", ["prompt: role · think · voice", "frame: Σ 16 codes + text"]),
       lm=("Qwen3 talker", ["0.6B: 28 × 1024", "1-d RoPE"], "f-qwen3.html"),
       lm_ext=("depth decoder", ["5-layer Qwen3", "cb 1–15 per frame"]),
       heads=("cb 0 head", ["+ 15 depth heads", "2048 codes each"]),
       search=("sample", ["T 0.9 · top-k 50", "rep. penalty 1.05"]),
       quant=("split RVQ", ["1 semantic + 15", "2048 codes, 12.5 Hz"]),
       trunk=("causal transformer", ["8 × 512, window 72", "layer scale"]),
       upsample=("↑ 2 × 2", ["conv-T + ConvNeXt"]),
       vocoder=("DAC decoder", ["↑ 8·5·4·3, SnakeBeta", "24 kHz"], "s-codec.html#vocoders"),
       note="no audio in: a voice is a stock speaker row (CustomVoice) or a description (VoiceDesign)")
define("zonos3", "qwen3tts", name="ZONOS3-Mini",
       fe=("Kaldi fbank", ["80 mels", "a reference clip"]),
       enc=("speaker encoder", ["MECT-B2: res convs", "+ soft-MoE layers"], "f-zonos3.html#speaker"),
       proj=("ASTP → 192", ["Linear 192 → 1024"]),
       bridge=("speaker row", ["replaces one row's", "summed embedding"]),
       cond=("condition rows", ["loudness · fmax ·", "speaking rate"]),
       rows=("11-id rows", ["10 codebooks + text", "delay pattern"]),
       lm=("AttnRes hybrid", ["11 × 1024: 9 GDN, 2 attn", "latent MoE, 64 / top-2"], "s-routes.html"),
       lm_ext=None,
       heads=("10 heads in one", ["4098 per codebook", "soft-cap 15"]),
       search=("sample", ["T 0.75 · k 80 · p 0.69", "rep. 2.4 on cb 0–5"]),
       quant=("10 codebooks", ["4096 × 8-d, 15 Hz", "cb 0 via 4 ConvNeXt"]),
       trunk=("5 attention stages", ["windows 10 s → 2 s", "patch-up × 2 between"]),
       upsample=("ConvNeXt × 8", ["at 240 Hz"]),
       vocoder=("2 ISTFT heads", ["n_fft 736 and 2208", "mixed → 44.1 kHz"], "s-codec.html#vocoders"),
       note="the stock voices speak from reference codes placed in the prompt; a speaker embedding is the alternative")
