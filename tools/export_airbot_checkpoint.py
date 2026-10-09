from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from lingbotvla.airbot_lora import add_adapters, build_base, checkpoint_manifest, digest, export_weights


def export(checkpoint: Path, output: Path) -> None:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    source = checkpoint.parent.parent
    manifest = checkpoint_manifest(checkpoint)
    run = json.loads((source / "run.json").read_text())
    if manifest["fingerprint"] != run["fingerprint"]:
        raise ValueError("Checkpoint does not match its current run configuration.")
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "checkpoints" / checkpoint.name
    shutil.copytree(checkpoint, frozen, ignore=shutil.ignore_patterns("hf_ckpt"))
    if checkpoint_manifest(frozen) != manifest:
        raise ValueError("Copied checkpoint differs from its source.")
    for name in ("run.json", "lingbotvla_cli.yaml"):
        shutil.copy2(source / name, output / name)
    shutil.copytree(source / "assets", output / "assets")
    print(json.dumps({"stage": "snapshot_saved", "step": manifest["step"], "output": str(output)}), flush=True)
    torch.set_num_threads(4)
    model = add_adapters(build_base(run["configuration"], "cpu"), run["recipe"])
    weights = load_file(frozen / "adapter.safetensors")
    expected = get_peft_model_state_dict(model)
    if weights.keys() != expected.keys() or any(weights[name].shape != expected[name].shape for name in expected):
        raise ValueError("Checkpoint adapter keys or shapes do not match the model.")
    set_peft_model_state_dict(model, weights)
    del weights, expected
    model = model.merge_and_unload(safe_merge=True).bfloat16()
    export_weights(model, frozen / "hf_ckpt")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(output)}\n" for path in files))
    print(json.dumps({"stage": "exported", "step": manifest["step"], "model": str(frozen / "hf_ckpt")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot and export an AIRBOT LoRA checkpoint on CPU.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.checkpoint, arguments.output)
