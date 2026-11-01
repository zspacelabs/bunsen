//! # Qwen3 pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::Qwen3Construct;
use crate::{
    data::pretrained::{
        CONFIG,
        HfProvider,
        PretrainedCache,
        PretrainedFactory,
        PretrainedProvider,
        SAFETENSORS,
        SAFETENSORS_INDEX,
        StaticBase,
        StaticPretrained,
        StaticPretrainedGroup,
        StaticPretrainedTable,
        StaticResource,
        StaticResourceMap,
        WELL_KNOWN,
    },
    errors::BunsenResult,
    kits::gpts::qwen3::{
        GENERATION_CONFIG,
        Qwen3Bundle,
        TOKENIZER,
        TOKENIZER_CONFIG,
    },
};

/// The kit segment of a Qwen3 resource's path in the cache:
/// `<cache>/pretrained/qwen3/<namespace>/<sha256>/<file>`.
pub const QWEN3_KIT: &str = "qwen3";

/// The key of the checkpoint in a Qwen3 resource map: one
/// `model.safetensors`, or `checkpoint.index` and its shards
/// `checkpoint.<n>`.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` of a `config.json`.
pub const JSON: &str = "json";

/// The `kind` of a `transformers` `tokenizer.json`.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The `kind` of a `transformers` `tokenizer_config.json`.
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

/// The `kind` of a `transformers` `generation_config.json`.
pub const GENERATION_CONFIG_JSON: &str = "generation_config.json";

const QWEN_ORIGIN: &str = "https://huggingface.co/Qwen";

/// `Qwen/Qwen3-0.6B`'s files, at a fixed revision.
pub static QWEN3_0_6B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-0.6b",
    description: "Qwen3-0.6B: model.safetensors (bf16, 1.50 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-0.6B"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_0_6B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-0.6b",
    aliases: &[],
    description: "Qwen3-0.6B: thinking and non-thinking chat, bf16, 1.50 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-0.6B"),
    prefab: Some("qwen3-0.6b"),
    maps: &[&QWEN3_0_6B_FILES],
};

/// `Qwen/Qwen3-0.6B-Base`'s files, at a fixed revision.
pub static QWEN3_0_6B_BASE_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-0.6b-base",
    description: "Qwen3-0.6B-Base: model.safetensors (bf16, 1.19 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-0.6B-Base"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_0_6B_BASE: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-0.6b-base",
    aliases: &[],
    description: "Qwen3-0.6B-Base: the pretrained model, no chat tuning, bf16, 1.19 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-0.6B-Base"),
    prefab: Some("qwen3-0.6b"),
    maps: &[&QWEN3_0_6B_BASE_FILES],
};

/// `Qwen/Qwen3-1.7B`'s files, at a fixed revision.
pub static QWEN3_1_7B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-1.7b",
    description: "Qwen3-1.7B: 2 safetensors shards (bf16, 4.06 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-1.7B"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00002.safetensors",
            sha256: Some("169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00002.safetensors",
            sha256: Some("912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_1_7B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-1.7b",
    aliases: &[],
    description: "Qwen3-1.7B: thinking and non-thinking chat, bf16, 4.06 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-1.7B"),
    prefab: Some("qwen3-1.7b"),
    maps: &[&QWEN3_1_7B_FILES],
};

/// `Qwen/Qwen3-1.7B-Base`'s files, at a fixed revision.
pub static QWEN3_1_7B_BASE_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-1.7b-base",
    description: "Qwen3-1.7B-Base: model.safetensors (bf16, 3.44 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-1.7B-Base"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-1.7B-Base/resolve/ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("1bb33a92c3548fbc68b889b490e810440435253598835bd71dff0396060c12db"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_1_7B_BASE: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-1.7b-base",
    aliases: &[],
    description: "Qwen3-1.7B-Base: the pretrained model, no chat tuning, bf16, 3.44 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-1.7B-Base"),
    prefab: Some("qwen3-1.7b"),
    maps: &[&QWEN3_1_7B_BASE_FILES],
};

/// `Qwen/Qwen3-4B`'s files, at a fixed revision.
pub static QWEN3_4B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-4b",
    description: "Qwen3-4B: 3 safetensors shards (bf16, 8.04 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00003.safetensors",
            sha256: Some("328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00003.safetensors",
            sha256: Some("6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00003.safetensors",
            sha256: Some("e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_4B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-4b",
    aliases: &[],
    description: "Qwen3-4B: thinking and non-thinking chat, bf16, 8.04 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B"),
    prefab: Some("qwen3-4b"),
    maps: &[&QWEN3_4B_FILES],
};

/// `Qwen/Qwen3-4B-Base`'s files, at a fixed revision.
pub static QWEN3_4B_BASE_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-4b-base",
    description: "Qwen3-4B-Base: 3 safetensors shards (bf16, 8.04 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-Base"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-4B-Base/resolve/906bfd4b4dc7f14ee4320094d8b41684abff8539",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00003.safetensors",
            sha256: Some("4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00003.safetensors",
            sha256: Some("f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00003.safetensors",
            sha256: Some("c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_4B_BASE: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-4b-base",
    aliases: &[],
    description: "Qwen3-4B-Base: the pretrained model, no chat tuning, bf16, 8.04 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-Base"),
    prefab: Some("qwen3-4b"),
    maps: &[&QWEN3_4B_BASE_FILES],
};

/// `Qwen/Qwen3-4B-Instruct-2507`'s files, at a fixed revision.
pub static QWEN3_4B_INSTRUCT_2507_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-4b-instruct-2507",
    description: "Qwen3-4B-Instruct-2507: 3 safetensors shards (bf16, 8.04 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00003.safetensors",
            sha256: Some("75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00003.safetensors",
            sha256: Some("0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00003.safetensors",
            sha256: Some("7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_4B_INSTRUCT_2507: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-4b-instruct-2507",
    aliases: &[],
    description: "Qwen3-4B-Instruct-2507: non-thinking chat, 256K context, bf16, 8.04 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"),
    prefab: Some("qwen3-4b-2507"),
    maps: &[&QWEN3_4B_INSTRUCT_2507_FILES],
};

/// `Qwen/Qwen3-4B-Thinking-2507`'s files, at a fixed revision.
pub static QWEN3_4B_THINKING_2507_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-4b-thinking-2507",
    description: "Qwen3-4B-Thinking-2507: 3 safetensors shards (bf16, 8.04 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/resolve/768f209d9ea81521153ed38c47d515654e938aea",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00003.safetensors",
            sha256: Some("98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00003.safetensors",
            sha256: Some("57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00003.safetensors",
            sha256: Some("e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("60adcfc35f5f251e19a196ec977b4ca0fcfb534852911b3196d68d5e3b146aa0"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_4B_THINKING_2507: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-4b-thinking-2507",
    aliases: &[],
    description: "Qwen3-4B-Thinking-2507: thinking chat, 256K context, bf16, 8.04 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"),
    prefab: Some("qwen3-4b-2507"),
    maps: &[&QWEN3_4B_THINKING_2507_FILES],
};

/// `Qwen/Qwen3-8B`'s files, at a fixed revision.
pub static QWEN3_8B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-8b",
    description: "Qwen3-8B: 5 safetensors shards (bf16, 16.38 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-8B"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00005.safetensors",
            sha256: Some("31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00005.safetensors",
            sha256: Some("5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00005.safetensors",
            sha256: Some("c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00004",
            file: "model-00004-of-00005.safetensors",
            sha256: Some("b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00005",
            file: "model-00005-of-00005.safetensors",
            sha256: Some("20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_8B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-8b",
    aliases: &[],
    description: "Qwen3-8B: thinking and non-thinking chat, bf16, 16.38 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-8B"),
    prefab: Some("qwen3-8b"),
    maps: &[&QWEN3_8B_FILES],
};

/// `Qwen/Qwen3-8B-Base`'s files, at a fixed revision.
pub static QWEN3_8B_BASE_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-8b-base",
    description: "Qwen3-8B-Base: 5 safetensors shards (bf16, 16.38 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-8B-Base"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-8B-Base/resolve/49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("cd39d5e44c5aadfd1954b598883aea1c4a98822a2f7723d214ef2278b7fec2e4"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00005.safetensors",
            sha256: Some("9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00005.safetensors",
            sha256: Some("9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00005.safetensors",
            sha256: Some("c7dd5a191c8da555def0714550375b3c0117751add408fcdda0c2a79ac0186bc"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00004",
            file: "model-00004-of-00005.safetensors",
            sha256: Some("ad8b708792105133e03fe19e2d56c89c709f677f01285f8b3ede8947d59fd9fe"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00005",
            file: "model-00005-of-00005.safetensors",
            sha256: Some("fbf24915d47ea030bb68ab0b9488f4515a907185baa6dc26837c9c3f2326a550"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_8B_BASE: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-8b-base",
    aliases: &[],
    description: "Qwen3-8B-Base: the pretrained model, no chat tuning, bf16, 16.38 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-8B-Base"),
    prefab: Some("qwen3-8b"),
    maps: &[&QWEN3_8B_BASE_FILES],
};

/// `Qwen/Qwen3-4B-SafeRL`'s files, at a fixed revision.
pub static QWEN3_4B_SAFERL_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3-4b-saferl",
    description: "Qwen3-4B-SafeRL: 3 safetensors shards (bf16, 8.04 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-SafeRL"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3-4B-SafeRL/resolve/1b95ccb88cabaf284ffb6968719c40ae6cd4085b",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("466ba79f90c65a2b01d1e8a6e54a4247c8afda1550f1940ff29a711e28b6e4c1"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00003.safetensors",
            sha256: Some("c0190626eb435fa6dc3d433aed07f2841487df23c271a9592536098fb8f5f5dc"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00003.safetensors",
            sha256: Some("e4b710d16bf22ff30617b4f1147dbcd772d76ec15646aca7e679ae018d6756b8"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00003.safetensors",
            sha256: Some("3fcb4f4d1b33c73fc3b2e0d3fb3f512439d3a8352f62c914153378183257bdc9"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("d98f11fc551faeff1f0f91cf596f2c56b5e59791ecf0024dbf054ed4d5442f29"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3_4B_SAFERL: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3-4b-saferl",
    aliases: &[],
    description: "Qwen3-4B-SafeRL: Qwen3-4B tuned for safety by reinforcement learning, thinking and non-thinking chat, bf16, 8.04 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3-4B-SafeRL"),
    prefab: Some("qwen3-4b"),
    maps: &[&QWEN3_4B_SAFERL_FILES],
};

/// `Qwen/Qwen3Guard-Gen-0.6B`'s files, at a fixed revision.
pub static QWEN3GUARD_GEN_0_6B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3guard-gen-0.6b",
    description: "Qwen3Guard-Gen-0.6B: one safetensors file (bf16, 1.50 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B/resolve/fada3b2f655b89601929198343c94cd2f64d93cc",
    )],
    resources: &[
        StaticResource {
            key: CHECKPOINT,
            file: "model.safetensors",
            sha256: Some("4f3ce47ebd968cddb67de08d8764f8ede7c410a7d1fb9e08145a4c7a2f2e5c0f"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("4674816bdf440c113a9ca87aedfa069b67f7cabbd89e71da7d4e4a68adce4af4"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("934e187433b06c5d28519e0347dd293084cf28621f00a86ae7e24e42f8b53e81"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("5f0bc42aae9f779d06bf620ce685a02cb0cde802d0f27f05c7331aaf0e1fd2d7"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3GUARD_GEN_0_6B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3guard-gen-0.6b",
    aliases: &[],
    description: "Qwen3Guard-Gen-0.6B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories, bf16, 1.50 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B"),
    prefab: Some("qwen3-0.6b"),
    maps: &[&QWEN3GUARD_GEN_0_6B_FILES],
};

/// `Qwen/Qwen3Guard-Gen-4B`'s files, at a fixed revision.
pub static QWEN3GUARD_GEN_4B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3guard-gen-4b",
    description: "Qwen3Guard-Gen-4B: 3 safetensors shards (bf16, 8.82 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3Guard-Gen-4B"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3Guard-Gen-4B/resolve/6ec42827da0c1ff11e7a49dc269d2e810d27e108",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("d8971e86d2f9e8ba379e389e6ec0137c931952b13db1cdcc6e3ca75fbb7c0901"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00003.safetensors",
            sha256: Some("d73f9558059de0740f2cf77bd98705dab0073fc3b61f41460124252df60a717e"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00003.safetensors",
            sha256: Some("13f603d405366e68d19698e90a934eae216f9e9182e5b71c236b0043cb4f8b59"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00003.safetensors",
            sha256: Some("e12365059f1dade58ce0df2d22f9d8618eaf60e15b01b2fcceee55b6ac5a8832"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("49f228b735a13b8d723ac710c44875aa38e91dca7629dc79978dd004db72f3b4"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("934e187433b06c5d28519e0347dd293084cf28621f00a86ae7e24e42f8b53e81"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("5f0bc42aae9f779d06bf620ce685a02cb0cde802d0f27f05c7331aaf0e1fd2d7"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3GUARD_GEN_4B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3guard-gen-4b",
    aliases: &[],
    description: "Qwen3Guard-Gen-4B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories, bf16, 8.82 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3Guard-Gen-4B"),
    prefab: Some("qwen3-4b"),
    maps: &[&QWEN3GUARD_GEN_4B_FILES],
};

/// `Qwen/Qwen3Guard-Gen-8B`'s files, at a fixed revision.
pub static QWEN3GUARD_GEN_8B_FILES: StaticResourceMap<'static> = StaticResourceMap {
    name: "qwen/qwen3guard-gen-8b",
    description: "Qwen3Guard-Gen-8B: 5 safetensors shards (bf16, 16.38 GB), config, tokenizer, generation config",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3Guard-Gen-8B"),
    namespace: "qwen",
    bases: &[StaticBase::Url(
        "https://huggingface.co/Qwen/Qwen3Guard-Gen-8B/resolve/4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb",
    )],
    resources: &[
        StaticResource {
            key: "checkpoint.index",
            file: "model.safetensors.index.json",
            sha256: Some("cd39d5e44c5aadfd1954b598883aea1c4a98822a2f7723d214ef2278b7fec2e4"),
            kind: Some(SAFETENSORS_INDEX),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00001",
            file: "model-00001-of-00005.safetensors",
            sha256: Some("017fbd70389fd873bbd89f1b833fe867aa09fcd87438dd35d556a7a8b403b8ea"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00002",
            file: "model-00002-of-00005.safetensors",
            sha256: Some("497581027655f3188e40f221d4130a8659c1ebd2191f898500210d69594cc712"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00003",
            file: "model-00003-of-00005.safetensors",
            sha256: Some("c5b6112c09d0c89924d98be42a18887d516e1cce92ca537d20dc48ed9a0e29f2"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00004",
            file: "model-00004-of-00005.safetensors",
            sha256: Some("ea7cf87db6e61d88c0476dffac16a2c15b99015866ae0e5e733b28ccf756e5c3"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: "checkpoint.00005",
            file: "model-00005-of-00005.safetensors",
            sha256: Some("048a9615fa462914b5028b0e68256176feeee525235d194a6d9985c4973eb8c1"),
            kind: Some(SAFETENSORS),
            sources: &[],
        },
        StaticResource {
            key: CONFIG,
            file: "config.json",
            sha256: Some("984e02d7c38d5aae44f9d803287a5437faf7d5c39ad04409c9e4ed5810291898"),
            kind: Some(JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER,
            file: "tokenizer.json",
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            kind: Some(TOKENIZER_JSON),
            sources: &[],
        },
        StaticResource {
            key: TOKENIZER_CONFIG,
            file: "tokenizer_config.json",
            sha256: Some("934e187433b06c5d28519e0347dd293084cf28621f00a86ae7e24e42f8b53e81"),
            kind: Some(TOKENIZER_CONFIG_JSON),
            sources: &[],
        },
        StaticResource {
            key: GENERATION_CONFIG,
            file: "generation_config.json",
            sha256: Some("5f0bc42aae9f779d06bf620ce685a02cb0cde802d0f27f05c7331aaf0e1fd2d7"),
            kind: Some(GENERATION_CONFIG_JSON),
            sources: &[],
        },
    ],
};

static QWEN3GUARD_GEN_8B: StaticPretrained<'static> = StaticPretrained {
    name: "qwen3guard-gen-8b",
    aliases: &[],
    description: "Qwen3Guard-Gen-8B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories, bf16, 16.38 GB",
    license: Some("Apache-2.0"),
    origin: Some("https://huggingface.co/Qwen/Qwen3Guard-Gen-8B"),
    prefab: Some("qwen3-8b"),
    maps: &[&QWEN3GUARD_GEN_8B_FILES],
};

/// Qwen's dense Qwen3 language models up to 8B.
pub static QWEN: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
    name: "qwen",
    description: "Qwen's dense Qwen3 language models, as transformers ships them",
    license: Some("Apache-2.0"),
    origin: Some(QWEN_ORIGIN),
    items: &[
        &QWEN3_0_6B,
        &QWEN3_0_6B_BASE,
        &QWEN3_1_7B,
        &QWEN3_1_7B_BASE,
        &QWEN3_4B,
        &QWEN3_4B_BASE,
        &QWEN3_4B_INSTRUCT_2507,
        &QWEN3_4B_THINKING_2507,
        &QWEN3_8B,
        &QWEN3_8B_BASE,
        &QWEN3_4B_SAFERL,
        &QWEN3GUARD_GEN_0_6B,
        &QWEN3GUARD_GEN_4B,
        &QWEN3GUARD_GEN_8B,
    ],
};

/// The Qwen3 checkpoints bunsen knows by name, behind the [`WELL_KNOWN`]
/// provider: `well-known:qwen/qwen3-0.6b`, `qwen/qwen3-0.6b`, or
/// `qwen3-0.6b`.
pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
    name: WELL_KNOWN,
    description: "the Qwen3 checkpoints bunsen knows by name",
    groups: &[&QWEN],
};

/// The name [`default_qwen3_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_MODEL: &str = "qwen/qwen3-0.6b";

/// Qwen3's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`'
/// `Qwen3ForCausalLM` layout (its safetensors checkpoint, one file or
/// shards, and its `config.json`), with its [`TOKENIZER`],
/// [`TOKENIZER_CONFIG`] and [`GENERATION_CONFIG`] files when it has them.
pub fn default_qwen3_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(WELL_KNOWN_TABLE.to_table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(TOKENIZER, "tokenizer.json", TOKENIZER_JSON)
                .with_extra_file(
                    TOKENIZER_CONFIG,
                    "tokenizer_config.json",
                    TOKENIZER_CONFIG_JSON,
                )
                .with_extra_file(
                    GENERATION_CONFIG,
                    "generation_config.json",
                    GENERATION_CONFIG_JSON,
                ),
        ),
    ]
}

/// Qwen3's factory over [`default_qwen3_providers`], building through
/// [`Qwen3Construct`].
///
/// [`load_bundle`](PretrainedFactory::load_bundle) with `"[provider:]name"`
/// is the whole pathway from a name to a [`Qwen3Bundle`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_factory() -> BunsenResult<PretrainedFactory<Qwen3Construct>> {
    PretrainedFactory::new().with_providers(default_qwen3_providers())
}

impl PretrainedFactory<Qwen3Construct> {
    /// A name to a bundle: [`load`](Self::load), keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_bundle(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
    ) -> BunsenResult<Arc<Qwen3Bundle>> {
        Ok(self.load(spec, cache, device)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::gpts::qwen3::{
        pretrained::QWEN3_PREFABS,
        weights::qwen3_config_from_hf,
    };

    /// Every file of every row is pinned to one URL, and each row's
    /// `config.json` (committed under `testdata/qwen3/`, at the row's digest)
    /// is the geometry of the prefab the row names.
    #[test]
    fn test_rows_are_pinned_and_match_their_prefabs() {
        let factory = default_qwen3_factory().unwrap();
        assert_eq!(factory.kit(), QWEN3_KIT);
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["well-known", "hf"]
        );
        let names = [
            "qwen3-0.6b",
            "qwen3-0.6b-base",
            "qwen3-1.7b",
            "qwen3-1.7b-base",
            "qwen3-4b",
            "qwen3-4b-base",
            "qwen3-4b-instruct-2507",
            "qwen3-4b-thinking-2507",
            "qwen3-8b",
            "qwen3-8b-base",
            "qwen3-4b-saferl",
            "qwen3guard-gen-0.6b",
            "qwen3guard-gen-4b",
            "qwen3guard-gen-8b",
        ];
        assert_eq!(
            factory.ids(),
            names
                .iter()
                .map(|n| format!("well-known:qwen/{n}"))
                .collect::<Vec<_>>()
        );
        for name in names {
            let (_, row) = factory.lookup(name).unwrap();
            assert_eq!(row.name, format!("qwen/{name}"));
            let map = &row.resources;
            map.validate().unwrap();
            for resource in map.resources.values() {
                assert!(resource.is_pinned(), "{name}: {}", resource.key);
                assert_eq!(resource.urls().len(), 1, "{name}: {}", resource.key);
            }
            for key in [CONFIG, TOKENIZER, TOKENIZER_CONFIG, GENERATION_CONFIG] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }
            assert!(
                map.get(CHECKPOINT).is_some() || map.get("checkpoint.index").is_some(),
                "{name}"
            );

            let path = format!(
                "{}/testdata/qwen3/{name}.config.json",
                env!("CARGO_MANIFEST_DIR")
            );
            assert_eq!(
                Some(crate::data::cache::sha256_of(std::path::Path::new(&path)).unwrap()),
                map.get(CONFIG).unwrap().sha256,
                "{name}: the committed config.json is the pinned one"
            );
            let config = qwen3_config_from_hf(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let prefab = QWEN3_PREFABS
                .expect_lookup_prefab(row.prefab.as_deref().unwrap())
                .to_config();
            assert_eq!(format!("{config:?}"), format!("{prefab:?}"), "{name}");
        }
        assert_eq!(
            factory.lookup(DEFAULT_QWEN3_MODEL).unwrap().1.name,
            "qwen/qwen3-0.6b"
        );
    }
}
