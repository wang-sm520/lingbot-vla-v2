from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def benchmark(arguments: argparse.Namespace) -> None:
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    if arguments.repeats < 2 or not arguments.steps or any(steps <= 0 for steps in arguments.steps):
        raise ValueError("Use at least two repetitions and positive denoising step counts.")
    torch.set_num_threads(4)
    started = time.monotonic()
    policy = LingbotVLAv2Server(
        str(arguments.model.resolve()),
        use_length=-1,
        chunk_ret=True,
        use_bf16=True,
        use_compile=arguments.use_compile,
        num_inference_steps=arguments.steps[0],
    )
    policy.reset(robo_name="airbot_play")
    load_seconds = time.monotonic() - started
    tasks = json.loads((arguments.fixtures / "tasks.json").read_text())
    references = {}
    baseline_seconds = {}
    results = []
    report = {
        "status": "running",
        "hardware_initialized": False,
        "model": str(arguments.model.resolve()),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "use_compile": arguments.use_compile,
        "seed": arguments.seed,
        "prediction_horizon": policy.config.chunk_size,
        "reference_steps": arguments.steps[0],
        "load_seconds": load_seconds,
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for steps in arguments.steps:
        policy.config.num_steps = steps
        for task in tasks:
            with np.load(arguments.fixtures / task["file"], allow_pickle=False) as sample:
                observation = {
                    "observation.state": sample["state"],
                    "observation.images.head": sample["head"],
                    "observation.images.wrist": sample["wrist"],
                    "task": task["prompt"],
                }
            measured = []
            predictions = []
            normalized_predictions = []
            with torch.inference_mode():
                for repeat in range(arguments.repeats + 1):
                    torch.manual_seed(arguments.seed + repeat)
                    torch.cuda.synchronize()
                    inference_started = time.monotonic()
                    prediction = policy.infer(observation, return_normalized=True)
                    torch.cuda.synchronize()
                    duration = time.monotonic() - inference_started
                    actions = np.asarray(prediction["action"])
                    if actions.shape != (50, 7) or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid actions for {task['key']}: {actions.shape}")
                    if repeat == 0:
                        warmup_seconds = duration
                    else:
                        measured.append(duration)
                        predictions.append(actions.copy())
                        normalized_predictions.append(prediction["_normalized_actions"].float().cpu().numpy().copy())
            actions = np.stack(predictions)
            normalized = np.stack(normalized_predictions)
            median_seconds = float(np.median(measured))
            if task["key"] not in references:
                references[task["key"]] = (actions, normalized)
                baseline_seconds[task["key"]] = median_seconds
            reference_actions, reference_normalized = references[task["key"]]
            difference = np.abs(actions - reference_actions)
            result = {
                "task": task["key"],
                "denoising_steps": steps,
                "warmup_seconds": warmup_seconds,
                "seconds": measured,
                "median_seconds": median_seconds,
                "speedup": baseline_seconds[task["key"]] / median_seconds,
                "shape": list(actions.shape[1:]),
                "finite": True,
                "joint_mean_abs_delta_rad": float(difference[..., :6].mean()),
                "joint_max_abs_delta_rad": float(difference[..., :6].max()),
                "gripper_mean_abs_delta_m": float(difference[..., 6].mean()),
                "gripper_max_abs_delta_m": float(difference[..., 6].max()),
                "normalized_mean_abs_delta": float(np.abs(normalized - reference_normalized).mean()),
            }
            results.append(result)
            report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            report["elapsed_seconds"] = time.monotonic() - started
            arguments.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    report["status"] = "completed"
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark recorded AIRBOT observations without initializing hardware."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 5, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=False)
    benchmark(parser.parse_args())
