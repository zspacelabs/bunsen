"""Generate the timm validation fixtures: timm's answers on ImageNet's validation images.

Reads the plan `timm-fixture-plan` writes (`timm-model-validation`'s docs), whose files are
bunsen's cache's own: each row's checkpoint, and the first shard of ImageNet's validation set
(`timm/imagenet-1k-wds`, gated, pulled with the user's Hub token). Its first `images` samples,
`.jpg` and `.cls`, are the images.

For each eval transform the rows use, timm's `transforms_imagenet_eval` (the plan's size, filter,
crop_pct, crop mode, mean and std) makes the preprocessed `[images, 3, h, w]` float32 batch,
written to the plan's `inputs` path, in the cache: local, never committed, as it is pixels.

For each row, timm's own model of the row's architecture and class count (its `config.json`'s;
1000 for a torchvision row), its checkpoint loaded strictly through the architecture's own
checkpoint filter (which converts `OpenAI` CLIP's layout), runs in
float64 on the CPU on that batch. The row's fixture (committed) is the logits and the pre-logits
features as float32, with, in the safetensors metadata, the row, the input key and the SHA-256 of
the input batch's float32 bytes, the sample keys, their classes, and the versions below.

Run from the repository root:

    cargo run -q --manifest-path validation/Cargo.toml -p timm-model-validation \\
        --features checkpoints --bin timm-fixture-plan > /media/Data/bunsen-tmp/timm/plan.json
    /media/Data/bunsen-tmp/timm/venv/bin/python -I validation/tools/gen_timm_fixtures.py \\
        /media/Data/bunsen-tmp/timm/plan.json

Pinned versions used to produce the committed fixtures: timm 1.0.31, torch 2.15.0, Pillow 13.0.0,
safetensors 0.8.0.
"""

import hashlib
import importlib
import io
import json
import pathlib
import sys
import tarfile

import PIL
import timm
import torch
from PIL import Image
from safetensors.torch import save_file
from timm.data.transforms_factory import transforms_imagenet_eval


def first_samples(shard: str, n: int) -> list:
    """The first `n` samples of a webdataset shard: `(key, jpeg bytes, class)`, in tar order."""
    samples: dict = {}
    order: list = []
    with tarfile.open(shard) as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, _, field = member.name.rpartition("/")[2].partition(".")
            if field not in ("jpg", "cls"):
                continue
            if key not in samples:
                if len(order) == n:
                    break
                order.append(key)
                samples[key] = {}
            samples[key][field] = tar.extractfile(member).read()
    return [(k, samples[k]["jpg"], int(samples[k]["cls"].decode().strip())) for k in order]


def eval_transform(t: dict):
    """timm's eval transform of a bunsen `ImageEvalTransformConfig`, as the plan serializes it."""
    return transforms_imagenet_eval(
        img_size=tuple(t["size"]),
        crop_pct=t["crop_pct"],
        crop_mode=t["crop_mode"].lower(),
        interpolation=t["interpolation"].lower(),
        mean=tuple(t["mean"]),
        std=tuple(t["std"]),
    )


def main(plan_path: str) -> None:
    plan = json.loads(pathlib.Path(plan_path).read_text())
    samples = first_samples(plan["shard"], plan["images"])
    keys = [k for k, _, _ in samples]
    labels = [str(c) for _, _, c in samples]
    versions = {"timm": timm.__version__, "torch": torch.__version__, "pillow": PIL.__version__}

    inputs: dict = {}
    for row in plan["rows"]:
        key = row["input_key"]
        if key in inputs:
            continue
        transform = eval_transform(row["transform"])
        batch = torch.stack(
            [transform(Image.open(io.BytesIO(jpg)).convert("RGB")) for _, jpg, _ in samples]
        ).to(torch.float32).contiguous()
        path = pathlib.Path(row["inputs"])
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"inputs": batch}, str(path), metadata={"input_key": key, **versions})
        sha = hashlib.sha256(batch.numpy().tobytes()).hexdigest()
        inputs[key] = (batch, sha)
        print("inputs", key, tuple(batch.shape), sha[:16], path)

    for row in plan["rows"]:
        batch, sha = inputs[row["input_key"]]
        model = timm.create_model(
            row["architecture"], pretrained=False, num_classes=row["classes"]
        )
        # The architecture's own checkpoint filter, as timm's pretrained load runs it: it
        # converts a checkpoint in another layout (OpenAI CLIP's), and leaves timm's own as it
        # is. A module with none (resnet.py) loads the state dict as it is.
        module = importlib.import_module(type(model).__module__)
        filter_fn = getattr(module, "checkpoint_filter_fn", None)
        timm.models.load_checkpoint(model, row["checkpoint"], strict=True, filter_fn=filter_fn)
        model = model.to(torch.float64).eval()
        with torch.no_grad():
            x = batch.to(torch.float64)
            pre_logits = model.forward_head(model.forward_features(x), pre_logits=True)
            logits = model.forward_head(model.forward_features(x))
        tensors = {
            "logits": logits.to(torch.float32).contiguous(),
            "pre_logits": pre_logits.to(torch.float32).contiguous(),
        }
        metadata = {
            "row": row["row"],
            "input_key": row["input_key"],
            "input_sha256": sha,
            "keys": json.dumps(keys),
            "labels": json.dumps(labels),
            **versions,
        }
        path = pathlib.Path(row["fixture"])
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata=metadata)
        top1 = logits.argmax(1).tolist()
        if row["classes"] == 1000:
            hits = sum(int(t == int(c)) for t, c in zip(top1, labels))
            print(row["row"], "top1", top1, f"{hits}/{len(labels)} correct", path.name)
        else:
            print(row["row"], "top1", top1, f"({row['classes']} classes)", path.name)


if __name__ == "__main__":
    main(sys.argv[1])
