from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def check(model: Path, fixtures: Path, output: Path) -> None:
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    torch.set_num_threads(4)
    started = time.monotonic()
    policy = LingbotVLAv2Server(str(model), use_length=-1, chunk_ret=True, use_bf16=True)
    policy.reset(robo_name="airbot_play")
    tasks = json.loads((fixtures / "tasks.json").read_text())
    results = []
    for task in tasks:
        with np.load(fixtures / task["file"], allow_pickle=False) as sample:
            observation = {
                "observation.state": sample["state"],
                "observation.images.head": sample["head"],
                "observation.images.wrist": sample["wrist"],
                "task": task["prompt"],
            }
        measured = []
        for _ in range(2):
            torch.cuda.synchronize()
            inference_started = time.monotonic()
            actions = np.asarray(policy.infer(observation)["action"])
            torch.cuda.synchronize()
            measured.append(time.monotonic() - inference_started)
            if actions.shape != (50, 7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions for {task['key']}: {actions.shape}")
        results.append({**task, "shape": list(actions.shape), "finite": True, "seconds": measured})
        print(json.dumps(results[-1]), flush=True)
    report = {
        "status": "passed",
        "model": str(model),
        "hardware_initialized": False,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "tasks": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check exported AIRBOT actions using recorded observations.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    check(arguments.model.resolve(), arguments.fixtures.resolve(), arguments.output.resolve())
