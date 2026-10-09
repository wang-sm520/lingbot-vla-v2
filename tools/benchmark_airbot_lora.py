from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from torch.utils.data import DataLoader
from zoneinfo import ZoneInfo

from lingbotvla.airbot_lora import (
    ROOT,
    ReplayableSampler,
    add_adapters,
    build_base,
    build_dataset,
    build_optimizer,
    capture_rng,
    load_checkpoint,
    read_configuration,
    restore_rng,
    trainable_parameters,
)


def write_report(path: Path, report: dict) -> None:
    report["time"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(report), flush=True)


def benchmark(arguments: argparse.Namespace) -> int:
    from lingbotvla.data.multimodal.data_collator import VLADataCollatorWithPacking
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.vision_models.module_utils import build_depth_model, build_video_model
    from tasks.vla.train_airbot_lora import teacher_targets

    recipe, configuration = read_configuration(arguments.config)
    output = ROOT / recipe["output_dir"]
    original_run = json.loads((output / "run.json").read_text())
    if recipe != original_run["recipe"] or configuration != original_run["configuration"]:
        raise ValueError("Benchmark must use the active run's unchanged model and recipe.")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    if arguments.report.exists():
        raise FileExistsError(arguments.report)
    report = {
        "status": "preparing",
        "checkpoint": str(arguments.checkpoint),
        "micro_batch_size": arguments.micro_batch_size,
        "gradient_accumulation_steps": arguments.accumulation_steps,
        "effective_batch_size": arguments.micro_batch_size * arguments.accumulation_steps,
        "warmup_steps": arguments.warmup_steps,
        "measure_steps": arguments.measure_steps,
        "configuration_fingerprint": original_run["fingerprint"],
        "iterations": [],
    }
    write_report(arguments.report, report)
    try:
        random.seed(recipe["seed"])
        np.random.seed(recipe["seed"])
        torch.manual_seed(recipe["seed"])
        if arguments.prepare_only:
            model_config = LingbotVLAV2Config(**{**configuration["model"], **configuration["train"]})
            dataset = build_dataset(configuration, model_config)
            samples = [dataset[index] for index in range(arguments.micro_batch_size)]
            batch = VLADataCollatorWithPacking()(samples)
            report.update(
                status="prepared",
                shapes={key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
            )
            write_report(arguments.report, report)
            return 0
        if arguments.checkpoint is None:
            raise ValueError("A committed checkpoint is required for GPU benchmarks.")
        torch.cuda.manual_seed_all(recipe["seed"])
        report["phase"] = "loading_model"
        write_report(arguments.report, report)
        model = add_adapters(build_base(configuration, "cuda"), recipe)
        model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer, scheduler = build_optimizer(model, recipe)
        saved = load_checkpoint(arguments.checkpoint, model, optimizer, scheduler, original_run["fingerprint"])
        dataset = build_dataset(configuration, model.get_base_model().config)
        loader = DataLoader(
            dataset,
            batch_size=arguments.micro_batch_size,
            sampler=ReplayableSampler(len(dataset), recipe["seed"], saved["consumed"]),
            num_workers=recipe["num_workers"],
            collate_fn=VLADataCollatorWithPacking(),
            generator=torch.Generator().manual_seed(recipe["seed"]),
            pin_memory=True,
            multiprocessing_context="spawn" if recipe["num_workers"] else None,
        )
        restored_rng = capture_rng()
        depth_models = build_depth_model(configuration["train"]["align_params"])
        video_model = build_video_model(configuration["train"]["align_params"]["video"])
        restore_rng(restored_rng)
        parameters = list(trainable_parameters(model).values())
        batches = iter(loader)
        report.update(status="running", resumed_step=saved["step"])
        write_report(arguments.report, report)
        for iteration in range(arguments.warmup_steps + arguments.measure_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            losses = np.zeros(6, dtype=np.float64)
            for _ in range(arguments.accumulation_steps):
                report["phase"] = "data_loading"
                cpu_batch = next(batches)
                if cpu_batch["state"].shape[0] != arguments.micro_batch_size:
                    raise ValueError("Loader did not provide the requested physical batch size.")
                batch = {
                    key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in cpu_batch.items()
                }
                report["phase"] = "teacher_forward"
                targets = teacher_targets(batch, depth_models, video_model, configuration["train"]["align_params"])
                report["phase"] = "policy_forward"
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**batch, **targets)
                    loss = result[0] / arguments.accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError("Benchmark produced nonfinite loss.")
                report["phase"] = "backward"
                loss.backward()
                losses += (
                    np.asarray(
                        [float(value.detach()) if isinstance(value, torch.Tensor) else value for value in result[:6]]
                    )
                    / arguments.accumulation_steps
                )
                del result, loss, targets, batch
            report["phase"] = "optimizer_update"
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, recipe["optimizer"]["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            record = {
                "iteration": iteration + 1,
                "warmup": iteration < arguments.warmup_steps,
                "seconds_per_update": elapsed,
                "samples_per_second": report["effective_batch_size"] / elapsed,
                "loss": float(losses[0]),
                "action_loss": float(losses[1]),
                "grad_norm": float(grad_norm),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            report["iterations"].append(record)
            report["phase"] = "iteration_complete"
            write_report(arguments.report, report)
        measured = [record for record in report["iterations"] if not record["warmup"]]
        average = mean(record["seconds_per_update"] for record in measured)
        report.update(
            status="passed",
            mean_seconds_per_update=average,
            median_seconds_per_update=median(record["seconds_per_update"] for record in measured),
            samples_per_second=report["effective_batch_size"] / average,
            peak_allocated_gib=max(record["peak_allocated_gib"] for record in measured),
            peak_reserved_gib=max(record["peak_reserved_gib"] for record in measured),
        )
        write_report(arguments.report, report)
        return 0
    except Exception as error:
        out_of_memory = isinstance(error, torch.cuda.OutOfMemoryError)
        report.update(
            status="out_of_memory" if out_of_memory else "failed", error_type=type(error).__name__, error=str(error)
        )
        if torch.cuda.is_initialized():
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        write_report(arguments.report, report)
        if out_of_memory:
            return 2
        raise


def matrix(arguments: argparse.Namespace) -> None:
    results = []
    try:
        for batch_size in (32, 16, 8, 4, 2, 1):
            report_path = arguments.report.parent / f"batch_{batch_size}_accum_{32 // batch_size}.json"
            command = [
                sys.executable,
                "-m",
                "tools.benchmark_airbot_lora",
                "--checkpoint",
                str(arguments.checkpoint),
                "--config",
                str(arguments.config),
                "--micro-batch-size",
                str(batch_size),
                "--accumulation-steps",
                str(32 // batch_size),
                "--warmup-steps",
                str(arguments.warmup_steps),
                "--measure-steps",
                str(arguments.measure_steps),
                "--report",
                str(report_path),
            ]
            completed = subprocess.run(command, cwd=ROOT)
            result = json.loads(report_path.read_text())
            results.append(result)
            write_report(arguments.report, {"status": "running", "results": results})
            if completed.returncode == 0:
                break
            if result["status"] != "out_of_memory":
                raise RuntimeError(f"Benchmark batch {batch_size} failed: {result.get('error')}")
        if results[-1]["status"] != "passed":
            raise RuntimeError("No tested batch completed an optimizer update.")
        if results[-1]["micro_batch_size"] != 1:
            report_path = arguments.report.parent / "batch_1_accum_32.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.benchmark_airbot_lora",
                    "--checkpoint",
                    str(arguments.checkpoint),
                    "--config",
                    str(arguments.config),
                    "--micro-batch-size",
                    "1",
                    "--accumulation-steps",
                    "32",
                    "--warmup-steps",
                    str(arguments.warmup_steps),
                    "--measure-steps",
                    str(arguments.measure_steps),
                    "--report",
                    str(report_path),
                ],
                cwd=ROOT,
                check=True,
            )
            results.append(json.loads(report_path.read_text()))
        write_report(arguments.report, {"status": "completed", "results": results})
    finally:
        subprocess.run(["systemctl", "--user", "start", "lingbot-airbot-lora-after-pi05.service"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vla/airbot_play/airbot_play_lora.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    arguments = parser.parse_args()
    if (
        min(arguments.micro_batch_size, arguments.accumulation_steps, arguments.measure_steps) < 1
        or arguments.warmup_steps < 0
    ):
        parser.error("Batch size, accumulation and measured steps must be positive; warmup must be nonnegative.")
    if arguments.matrix:
        matrix(arguments)
    else:
        raise SystemExit(benchmark(arguments))
