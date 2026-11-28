# The Modelopedia

The kits' architectures have a book of their own:
**[The Modelopedia](../modelopedia/index.html)**. It reads every model family bunsen builds, from ResNet and ViT
to Gemma, Qwen3.5, Whisper and ZONOS3, as one modular algebra: four generic models, the sockets each family fills,
and the combinators that fill them, each with its theory and diagrams.

- **I · Conv nets**: the residual unit, conv blocks, norms, spatial mixers and channel attention; ResNet through
  RegNet.
- **II · Vision transformers**: patches, attention, the residual sublayer, windows and pools; ViT, DeiT, CLIP,
  SigLIP, DINOv2 and Swin.
- **III · Language models**: rotary positions, masks and windows, logit control, the gated DeltaNet; nanochat,
  Qwen3, Gemma, Qwen3.5, ModernBERT and Laya.
- **IV · Speech**: front ends, audio encoders, bridges, transducers and codecs; Whisper, Qwen3-ASR, Voxtral
  Realtime, Parakeet TDT, Qwen3-TTS and ZONOS3.

It documents architecture, not APIs: for those, see [Kits](kits.md) and the API docs. It is HTML rather than
chapters of this book, built from `book/modelopedia/` by an mdBook preprocessor on every book build.
