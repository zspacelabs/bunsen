# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.46.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.45.0...bunsen-catalog-v0.46.0) - 2026-11-27

### Added

- *(sknet)* the SKNet kit, timm's Selective Kernel Networks on the generic conv net ([#1053](https://github.com/zspacelabs/bunsen/pull/1053))
- *(resnest)* the ResNeSt kit, timm's ResNeSt on the generic conv net ([#1051](https://github.com/zspacelabs/bunsen/pull/1051))
- *(res2net)* the Res2Net kit, timm's Res2Net and Res2NeXt on the generic conv net ([#1049](https://github.com/zspacelabs/bunsen/pull/1049))
- *(vit)* the 11 hybrid ViT rows, three validated against timm ([#1046](https://github.com/zspacelabs/bunsen/pull/1046))
- *(resnetv2)* the ResNetV2 kit, the pre-activation ResNet and BiT on the generic conv net, with timm's checkpoints ([#1038](https://github.com/zspacelabs/bunsen/pull/1038))
- *(regnet)* RegNetV and RegNetZ, with pre-activation units, units with no residual and a final conv ([#1035](https://github.com/zspacelabs/bunsen/pull/1035))
- *(regnet)* the RegNet kit, RegNetX and RegNetY on the generic conv net, with timm's checkpoints ([#1032](https://github.com/zspacelabs/bunsen/pull/1032))
- *(vit)* BEiT3's sub-LN norms, and its rows ([#1029](https://github.com/zspacelabs/bunsen/pull/1029))
- *(vit)* gated MLPs, RMS norms and the bias-free projections; AIMv2 and the giant DINOv2 ([#1028](https://github.com/zspacelabs/bunsen/pull/1028))
- *(vit)* the attention pool (`map`), and the SigLIP and SigLIP 2 towers ([#1027](https://github.com/zspacelabs/bunsen/pull/1027))
- *(vit)* 143 timm rows on the generated prefabs, and eight validation rows ([#1026](https://github.com/zspacelabs/bunsen/pull/1026))
- *(swin)* the Swin V1 kit, and the Swin-S3 geometries, with timm's checkpoints ([#1016](https://github.com/zspacelabs/bunsen/pull/1016))
- *(swin)* [**breaking**] the Swin V2 kit on the generic staged vision encoder, with timm's checkpoints ([#1015](https://github.com/zspacelabs/bunsen/pull/1015))
- *(deit)* the `deit` kit: `DeiT` and `DeiT` III, distilled and plain ([#999](https://github.com/zspacelabs/bunsen/pull/999))
- *(vit)* `OpenAI` CLIP's own checkpoint layout, and its three timm rows ([#997](https://github.com/zspacelabs/bunsen/pull/997))
- *(vit)* 96 timm hub rows, and six of them validated against timm ([#996](https://github.com/zspacelabs/bunsen/pull/996))
- *(vit)* the `vit` kit: timm's `ViT`s as `VisionEncoderModel`s ([#995](https://github.com/zspacelabs/bunsen/pull/995))
- *(catalog)* ImageNet's 12k and 21k label sets; timm rows bind their head's labels ([#992](https://github.com/zspacelabs/bunsen/pull/992))
- *(resnet)* [**breaking**] the head's select-adaptive pools and dropout ([#990](https://github.com/zspacelabs/bunsen/pull/990))
- *(resnet)* [**breaking**] ResNet-RS, the strided conv stem pool ([#989](https://github.com/zspacelabs/bunsen/pull/989))
- *(resnet)* [**breaking**] anti-aliased downsampling, blur and average pools (the AA- and Blur-ResNets) ([#988](https://github.com/zspacelabs/bunsen/pull/988))
- *(resnet)* [**breaking**] channel attention, squeeze-excite and ECA (the SE- and ECA-ResNets) ([#987](https://github.com/zspacelabs/bunsen/pull/987))
- *(resnet)* [**breaking**] deep and tiered stems, and average-pool shortcuts (the C, D, S and T variants) ([#986](https://github.com/zspacelabs/bunsen/pull/986))
- *(resnet)* [**breaking**] grouped and widened bottlenecks: `ResNeXt` and wide `ResNet` ([#985](https://github.com/zspacelabs/bunsen/pull/985))
- *(resnet)* [**breaking**] timm's `in_chans` and `channels`, and the group norm (`resnet50_gn`) ([#984](https://github.com/zspacelabs/bunsen/pull/984))
- *(catalog-builder)* `cargo catalog timm-rows`: a kit's timm hub rows, from the Hub ([#983](https://github.com/zspacelabs/bunsen/pull/983))
- *(datasets)* webdataset shards, and ImageNet's validation set pulled from its gated mirror ([#975](https://github.com/zspacelabs/bunsen/pull/975))
- *(resnet)* [**breaking**] timm's ResNets from their hub repos: 42 rows, the old names aliases ([#971](https://github.com/zspacelabs/bunsen/pull/971))
- *(images)* timm's config.json, and a ResNet from a timm hub repo ([#970](https://github.com/zspacelabs/bunsen/pull/970))
- *(images)* [**breaking**] the image encoder and classifier seam; a ResNet load comes with its card ([#969](https://github.com/zspacelabs/bunsen/pull/969))
- *(images)* ImageNet's label sets, in the catalog and parsed ([#968](https://github.com/zspacelabs/bunsen/pull/968))

## [0.44.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.43.0...bunsen-catalog-v0.44.0) - 2026-11-23

### Added

- *(zonos3)* the construct loads the speaker encoder; validated on the release ([#891](https://github.com/zspacelabs/bunsen/pull/891))

### Other

- *(voxtral)* [**breaking**] the caches' pre-sizes are config fields ([#928](https://github.com/zspacelabs/bunsen/pull/928))

## [0.43.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.42.0...bunsen-catalog-v0.43.0) - 2026-11-22

### Added

- *(qwen3_asr)* [**breaking**] the audio encoder's stem, biases, norm and positions are config fields ([#848](https://github.com/zspacelabs/bunsen/pull/848))
- *(qwen3_asr)* [**breaking**] the base models read their geometry from config.json; no silent fallback ([#847](https://github.com/zspacelabs/bunsen/pull/847))
- *(qwen3_asr)* [**breaking**] the front end is read from the model's feature-extractor file ([#846](https://github.com/zspacelabs/bunsen/pull/846))
- *(qwen3)* [**breaking**] Qwen3's attention conventions and cache pre-size are config fields ([#842](https://github.com/zspacelabs/bunsen/pull/842))
- *(qwen3_tts)* the Qwen3-TTS catalog rows and their loader ([#769](https://github.com/zspacelabs/bunsen/pull/769))
- *(zonos3)* the `zyphra/zonos3/zonos3-mini` catalog row and its loader ([#709](https://github.com/zspacelabs/bunsen/pull/709))

### Other

- [**breaking**] the rotary base is an `f64` ([#792](https://github.com/zspacelabs/bunsen/pull/792))
- crutcher/tts voice info ([#729](https://github.com/zspacelabs/bunsen/pull/729))
- [**breaking**] the attention softmax widens through `Widening`, as `softmax_widening` ([#715](https://github.com/zspacelabs/bunsen/pull/715))

## [0.42.0](https://github.com/zspacelabs/bunsen/compare/bunsen-catalog-v0.41.0...bunsen-catalog-v0.42.0) - 2026-11-13

### Added

- *(resources)* provider descriptions, Hub mirrors, digest-filed downloads, and the cards rebuilt ([#686](https://github.com/zspacelabs/bunsen/pull/686))
- *(datasets)* [**breaking**] datasets are catalog builds ([#684](https://github.com/zspacelabs/bunsen/pull/684))
- *(pretrained)* [**breaking**] the kits load from the built-in catalog ([#678](https://github.com/zspacelabs/bunsen/pull/678))
- *(catalog)* the built-in catalog crate, and ryl YAML across the tree ([#677](https://github.com/zspacelabs/bunsen/pull/677))

### Other

- *(data)* [**breaking**] the catalog, resource and build names ([#681](https://github.com/zspacelabs/bunsen/pull/681))
- *(data)* [**breaking**] the `data` tree: resources, catalog, formats, builds, datasets ([#680](https://github.com/zspacelabs/bunsen/pull/680))
