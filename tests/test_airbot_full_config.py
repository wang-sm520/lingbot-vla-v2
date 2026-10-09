import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from lingbotvla.optim.optimizer import AnyPrecisionAdamW, build_optimizer
from tools.configure_airbot_full import build_configuration, parser


@pytest.mark.parametrize("profile,world_size", [("bf16-kahan", 1), ("fp32-master", 2)])
def test_full_configuration_passes_the_actual_training_parser(tmp_path, monkeypatch, profile, world_size):
    from lingbotvla.utils.arguments import parse_args
    from tasks.vla.train_lingbotvla import Arguments

    arguments = parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dataset"),
            "--weights-root",
            str(tmp_path / "weights"),
            "--output-dir",
            str(tmp_path / "run"),
            "--profile",
            profile,
            "--world-size",
            str(world_size),
        ]
    )
    configuration = build_configuration(arguments)
    config_path = tmp_path / "full.yaml"
    config_path.write_text(yaml.safe_dump(configuration))
    for name, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": str(world_size)}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["train", str(config_path)])
    actual = parse_args(Arguments)
    assert actual.train.global_batch_size == 32
    assert actual.train.gradient_accumulation_steps == 32 // world_size
    assert not actual.train.freeze_vit
    assert not actual.train.freeze_vision_encoder
    assert not actual.train.train_expert_only
    assert actual.train.enable_mixed_precision == (profile == "fp32-master")
    assert actual.train.max_steps == 100000
    assert actual.train.chunk_size == 50
    assert actual.data.train_path == str(tmp_path / "dataset")
    assert actual.model.model_path == str(tmp_path / "weights/lingbot-vla-v2-6b")
    assert actual.train.align_params["depth"]["moge_path"] == str(tmp_path / "weights/moge-2-vitb-normal/model.pt")


def test_bf16_profile_optimizer_accumulates_small_updates():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = build_optimizer(model, lr=5e-5, optimizer_type="anyprecision_adamw", weight_decay=0.0)
    assert isinstance(optimizer, AnyPrecisionAdamW)
    for _ in range(100):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
    assert torch.all(model.weight < 1.0)
    assert torch.isfinite(model.weight).all()
    assert optimizer.state[model.weight]["compensation"].dtype == torch.bfloat16


def test_incompatible_batch_sizes_are_rejected(tmp_path):
    arguments = parser().parse_args(
        [
            "--dataset",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--world-size",
            "3",
            "--global-batch-size",
            "32",
        ]
    )
    with pytest.raises(ValueError, match="divisible"):
        build_configuration(arguments)


def test_train_launcher_preserves_arguments_and_training_failure(tmp_path):
    torchrun = tmp_path / "torchrun"
    torchrun.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(23)\n")
    torchrun.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "NPROC_PER_NODE": "2",
        "NNODES": "1",
        "NODE_RANK": "0",
        "LINGBOT_TRAIN_LOG": str(tmp_path / "train.log"),
    }
    result = subprocess.run(
        ["bash", str(Path(__file__).resolve().parents[1] / "train.sh"), "train file.py", "value with spaces"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23
    arguments = json.loads(result.stdout.splitlines()[-1])
    assert arguments[arguments.index("--nproc-per-node") + 1] == "2"
    assert arguments[-2:] == ["train file.py", "value with spaces"]
