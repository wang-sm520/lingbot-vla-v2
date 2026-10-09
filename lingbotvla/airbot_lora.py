from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.utils.data import Sampler
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_configuration(path: Path) -> tuple[dict, dict]:
    recipe = yaml.safe_load(path.read_text())
    configuration = yaml.safe_load((ROOT / recipe["base_config"]).read_text())
    configuration["train"].update(
        enable_fp32=False,
        enable_mixed_precision=False,
        freeze_vision_encoder=True,
        train_expert_only=False,
        use_compile=False,
        enable_resume=False,
        optimizer="adamw",
        max_steps=recipe["max_steps"],
        micro_batch_size=recipe["micro_batch_size"],
        global_batch_size=recipe["micro_batch_size"] * recipe["gradient_accumulation_steps"],
        gradient_accumulation_steps=recipe["gradient_accumulation_steps"],
        seed=recipe["seed"],
        enable_gradient_checkpointing=True,
        action_fp32=False,
    )
    for key in ("joints", "norm_type"):
        configuration["data"][key] = [str(value) for value in configuration["data"][key]]
    configuration["data"].update(image_augment=False, img_size=224, num_workers=recipe["num_workers"], chunk_size=50)
    for key in ("robot_config_root", "norm_stats_file"):
        configuration["data"][key] = str((ROOT / configuration["data"][key]).resolve())
    return recipe, configuration


def build_base(configuration: dict, device: str, *, empty: bool = False):
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

    apply_lingbot_qwen3_vl_patch()
    settings = {**configuration["model"], **configuration["train"]}
    if device == "cpu":
        settings["vit_attn_implementation"] = "eager"
    with init_empty_weights():
        model = LingbotVlaV2Policy(LingbotVLAV2Config(**settings)).to(dtype=torch.bfloat16)
    if not empty:
        load_model_weights(model, settings["model_path"], init_device=device, post_training=True, adanorm_time=True)
    return model


def add_adapters(model, recipe: dict):
    specification = recipe["lora"]
    configuration = LoraConfig(
        r=specification["rank"],
        lora_alpha=specification["alpha"],
        lora_dropout=specification["dropout"],
        target_modules=specification["target_modules"],
        modules_to_save=specification["modules_to_save"],
        bias="none",
    )
    adapted = get_peft_model(model, configuration)
    for parameter in adapted.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return adapted


def build_dataset(configuration: dict, model_config):
    from lingbotvla.data.vla_data.base_dataset import VLADataset

    data = configuration["data"]
    processor = AutoProcessor.from_pretrained(configuration["model"]["tokenizer_path"], local_files_only=True)
    return VLADataset(
        data["train_path"],
        data["data_name"],
        SimpleNamespace(**data),
        data["robot_config_root"],
        config=model_config,
        processor=processor,
        chunk_size=50,
        image_size=(data["img_size"], data["img_size"]),
        use_future_image=True,
        use_depth_align=True,
        image_augment=False,
    )


class ReplayableSampler(Sampler):
    def __init__(self, length: int, seed: int, consumed: int = 0):
        if length <= 0 or consumed < 0:
            raise ValueError("Expected positive dataset length and nonnegative consumed count.")
        self.length, self.seed, self.consumed = length, seed, consumed

    def __iter__(self):
        epoch, offset = divmod(self.consumed, self.length)
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.length, generator=generator).tolist()
            yield from order[offset:]
            epoch += 1
            offset = 0


def trainable_parameters(model) -> OrderedDict:
    return OrderedDict((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)


def build_optimizer(model, recipe: dict):
    settings = recipe["optimizer"]
    optimizer = torch.optim.AdamW(
        list(trainable_parameters(model).values()),
        lr=settings["learning_rate"],
        betas=tuple(settings["betas"]),
        weight_decay=settings["weight_decay"],
        foreach=False,
    )
    warmup = settings["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
    return optimizer, scheduler


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checkpoint_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "COMPLETED.json").read_text())
    if manifest["format"] != "airbot_lora_v1":
        raise ValueError(f"Unrecognized checkpoint: {directory}")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if path.resolve().parent != directory.resolve() or digest(path) != expected:
            raise ValueError(f"Checkpoint verification failed: {path}")
    return manifest


def save_checkpoint(
    root: Path, model, optimizer, scheduler, step: int, consumed: int, fingerprint: str, keep: int = 2
) -> Path:
    if keep < 1:
        raise ValueError("At least one complete checkpoint must be retained.")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"global_step_{step}"
    if destination.exists():
        manifest = checkpoint_manifest(destination)
        if manifest["fingerprint"] != fingerprint or manifest["consumed"] != consumed:
            raise ValueError(f"Checkpoint already exists with different training state: {destination}")
        return destination
    temporary = root / f".step_{step}_{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    tensors = {
        name: value.detach().cpu().contiguous().clone() for name, value in get_peft_model_state_dict(model).items()
    }
    save_file(tensors, temporary / "adapter.safetensors")
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng(),
        "parameter_names": list(trainable_parameters(model)),
        "step": step,
        "consumed": consumed,
    }
    torch.save(state, temporary / "training_state.pt")
    manifest = {
        "format": "airbot_lora_v1",
        "step": step,
        "consumed": consumed,
        "fingerprint": fingerprint,
        "files": {filename: digest(temporary / filename) for filename in ("adapter.safetensors", "training_state.pt")},
    }
    (temporary / "COMPLETED.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in temporary.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(temporary)
    temporary.rename(destination)
    sync_directory(root)
    checkpoints = []
    for path in root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if (
            match
            and (path / "COMPLETED.json").is_file()
            and not path.is_symlink()
            and not (path / "hf_ckpt/model.safetensors.index.json").is_file()
        ):
            checkpoints.append((int(match.group(1)), path))
    for _, path in sorted(checkpoints)[:-keep]:
        if path != destination:
            shutil.rmtree(path)
    return destination


def load_checkpoint(directory: Path, model, optimizer, scheduler, fingerprint: str) -> dict:
    manifest = checkpoint_manifest(directory)
    if manifest["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint configuration/data fingerprint does not match this run.")
    tensors = load_file(directory / "adapter.safetensors")
    expected = get_peft_model_state_dict(model)
    if tensors.keys() != expected.keys() or any(tensors[name].shape != expected[name].shape for name in expected):
        raise ValueError("Checkpoint adapter keys or shapes do not match the model.")
    set_peft_model_state_dict(model, tensors)
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    if state["parameter_names"] != list(trainable_parameters(model)):
        raise ValueError("Checkpoint optimizer parameter order differs from the model.")
    if state["step"] != manifest["step"] or state["consumed"] != manifest["consumed"]:
        raise ValueError("Checkpoint counters do not match the committed manifest.")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state


def export_weights(model, destination: Path, shard_limit: int = 2_000_000_000) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    native = model.merge_and_unload(safe_merge=True) if hasattr(model, "merge_and_unload") else model
    shard, shard_bytes, total_bytes, shard_number = {}, 0, 0, 0
    weight_map = {}

    def flush():
        nonlocal shard, shard_bytes, shard_number
        if not shard:
            return
        shard_number += 1
        filename = f"model-{shard_number:05d}.safetensors"
        save_file(shard, temporary / filename, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, filename))
        shard, shard_bytes = {}, 0

    for name, parameter in native.state_dict().items():
        tensor = parameter.detach().cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        if shard_bytes + size > shard_limit:
            flush()
        shard[name] = tensor.clone()
        shard_bytes += size
        total_bytes += size
    flush()
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (temporary / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    native.config.save_pretrained(temporary)
    for path in temporary.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(temporary)
    temporary.rename(destination)
    sync_directory(destination.parent)


def record_batch_change(output: Path, previous: dict, current: dict, checkpoint: Path) -> dict:
    return record_configuration_change(output, previous, current, checkpoint, kind="batch")


def record_step_extension(output: Path, previous: dict, current: dict, checkpoint: Path) -> dict:
    return record_configuration_change(output, previous, current, checkpoint, kind="step_extension")


def record_configuration_change(output: Path, previous: dict, current: dict, checkpoint: Path, *, kind: str) -> dict:
    before = {key: copy.deepcopy(previous[key]) for key in current}
    after = copy.deepcopy(current)
    if previous["fingerprint"] != hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest():
        raise ValueError("Previous configuration fingerprint is invalid.")
    if kind == "batch":
        for snapshot in (before, after):
            for key in ("micro_batch_size", "gradient_accumulation_steps"):
                snapshot["recipe"].pop(key)
                snapshot["configuration"]["train"].pop(key)
            snapshot["configuration"]["train"].pop("global_batch_size")
    elif kind == "step_extension":
        if current["recipe"]["max_steps"] <= previous["recipe"]["max_steps"]:
            raise ValueError("A step extension must increase the optimizer step target.")
        for snapshot in (before, after):
            if snapshot["recipe"]["max_steps"] != snapshot["configuration"]["train"]["max_steps"]:
                raise ValueError("Recipe and training step targets differ.")
            snapshot["recipe"].pop("max_steps")
            snapshot["configuration"]["train"].pop("max_steps")
    else:
        raise ValueError(f"Unsupported configuration migration: {kind}")
    if before != after:
        raise ValueError("Configuration migration cannot change settings outside its requested fields.")
    manifest = checkpoint_manifest(checkpoint)
    if manifest["fingerprint"] != previous["fingerprint"]:
        raise ValueError("Configuration migration checkpoint belongs to a different configuration.")
    if kind == "step_extension" and manifest["step"] >= current["recipe"]["max_steps"]:
        raise ValueError("Step extension target must be greater than the resume step.")
    history = output / "configuration_history" / previous["fingerprint"]
    history.mkdir(parents=True, exist_ok=True)
    archived_checkpoint = history / checkpoint.name
    if not archived_checkpoint.exists():
        temporary = history / f".{checkpoint.name}_{uuid.uuid4().hex}.tmp"
        shutil.copytree(checkpoint, temporary)
        if checkpoint_manifest(temporary) != manifest:
            raise ValueError("Archived configuration migration checkpoint differs from its source.")
        temporary.rename(archived_checkpoint)
    elif checkpoint_manifest(archived_checkpoint) != manifest:
        raise ValueError("Archived configuration migration checkpoint already exists with different contents.")
    (history / "run.json").write_text(json.dumps(previous, indent=2) + "\n")
    for filename in ("lingbotvla_cli.yaml", "export_verification.json", "status.json"):
        if (output / filename).is_file():
            shutil.copy2(output / filename, history / filename)
    if (output / "assets").is_dir():
        shutil.copytree(output / "assets", history / "assets", dirs_exist_ok=True)
    change = {
        "kind": kind,
        "source_checkpoint": str(checkpoint.resolve()),
        "archived_checkpoint": str(archived_checkpoint.resolve()),
        "source_manifest_sha256": digest(checkpoint / "COMPLETED.json"),
        "from_fingerprint": previous["fingerprint"],
        "to_fingerprint": hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
        "step": manifest["step"],
        "consumed": manifest["consumed"],
    }
    if kind == "batch":
        change["from_batch"] = [
            previous["recipe"]["micro_batch_size"],
            previous["recipe"]["gradient_accumulation_steps"],
        ]
        change["to_batch"] = [current["recipe"]["micro_batch_size"], current["recipe"]["gradient_accumulation_steps"]]
    else:
        change["from_max_steps"] = previous["recipe"]["max_steps"]
        change["to_max_steps"] = current["recipe"]["max_steps"]
    return change


def checkpoint_load_fingerprint(output: Path, checkpoint: Path, fingerprint: str) -> str:
    manifest = json.loads((checkpoint / "COMPLETED.json").read_text())
    if manifest["fingerprint"] == fingerprint:
        return fingerprint
    snapshot = json.loads((output / "run.json").read_text())
    for key in ("step_extension", "batch_change"):
        change = snapshot.get(key, {})
        if (
            snapshot["fingerprint"] == fingerprint
            and change.get("to_fingerprint") == fingerprint
            and change.get("from_fingerprint") == manifest["fingerprint"]
            and change.get("source_manifest_sha256") == digest(checkpoint / "COMPLETED.json")
            and change.get("step") == manifest["step"]
            and change.get("consumed") == manifest["consumed"]
        ):
            return manifest["fingerprint"]
    raise ValueError("Resume checkpoint has no matching recorded configuration migration.")


def snapshot_configuration(
    output: Path,
    recipe: dict,
    configuration: dict,
    *,
    batch_change_checkpoint: Path | None = None,
    step_extension_checkpoint: Path | None = None,
) -> str:
    if batch_change_checkpoint is not None and step_extension_checkpoint is not None:
        raise ValueError("Apply one configuration migration at a time.")
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    norm_path = Path(configuration["data"]["norm_stats_file"])
    source_audit = json.loads((ROOT / "logs/airbot_data_audit.json").read_text())
    dataset_root = Path(configuration["data"]["train_path"])
    for relative, expected in source_audit["source_sha256"].items():
        if digest(dataset_root / relative) != expected:
            raise ValueError(f"Dataset changed since the audited 600-episode snapshot: {relative}")
    if digest(norm_path) != source_audit["normalization_sha256"]:
        raise ValueError("Normalization differs from the audited dataset statistics.")
    pretrained_assets = {
        name: json.loads((ROOT / f"logs/asset_{name}.json").read_text()) for name in ("lingbot", "tokenizer", "depth")
    }
    fingerprint_data = {
        "recipe": recipe,
        "configuration": configuration,
        "source_sha256": source_audit["source_sha256"],
        "norm_sha256": digest(norm_path),
        "pretrained_assets": pretrained_assets,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    snapshot_path = output / "run.json"
    previous = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    batch_change = previous.get("batch_change") if previous else None
    step_extension = previous.get("step_extension") if previous else None
    if previous and previous["fingerprint"] != fingerprint:
        if batch_change_checkpoint is not None:
            batch_change = record_batch_change(output, previous, fingerprint_data, batch_change_checkpoint)
        elif step_extension_checkpoint is not None:
            step_extension = record_step_extension(output, previous, fingerprint_data, step_extension_checkpoint)
        else:
            raise ValueError("Output directory contains a different configuration or dataset fingerprint.")
    snapshot = copy.deepcopy(configuration)
    snapshot["data"]["norm_stats_file"] = "assets/norm_stats.json"
    shutil.copy2(norm_path, assets / "norm_stats.json")
    shutil.copy2(ROOT / "configs/robot_configs/airbot_play.yaml", assets / "airbot_play.yaml")
    (output / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))
    record = {"fingerprint": fingerprint, **fingerprint_data}
    if batch_change:
        record["batch_change"] = batch_change
    if step_extension:
        record["step_extension"] = step_extension
    temporary = snapshot_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(snapshot_path)
    return fingerprint
