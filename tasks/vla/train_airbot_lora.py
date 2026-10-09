from __future__ import annotations

import argparse
import json
import random
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from zoneinfo import ZoneInfo

from lingbotvla.airbot_lora import (
    ROOT,
    ReplayableSampler,
    add_adapters,
    build_base,
    build_dataset,
    build_optimizer,
    capture_rng,
    checkpoint_load_fingerprint,
    export_weights,
    load_checkpoint,
    read_configuration,
    restore_rng,
    save_checkpoint,
    snapshot_configuration,
    trainable_parameters,
)


def write_status(output: Path, stage: str, **values) -> None:
    record = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "stage": stage, **values}
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output / "status.json")
    print(json.dumps(record), flush=True)


def teacher_targets(batch: dict, depth_models, video_model, alignment: dict) -> dict:
    from lingbotvla.models.vla.vision_models.module_utils import get_depth_target, get_video_target

    images = batch.pop("pil_images")
    future_images = batch.pop("future_pil_images")
    effective_fps = batch.pop("future_video_effective_fps", None)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        current_depth, _ = get_depth_target("MoRGBD", depth_models, images)
        future_depth, _ = get_depth_target("MoRGBD", depth_models, future_images)
        video = get_video_target(video_model, images, future_images, alignment["video"], effective_fps=effective_fps)
    return {
        "depth_targets": current_depth,
        "future_depth_targets": future_depth,
        "future_video_targets": video["patch"],
        "future_video_cls_targets": video.get("cls"),
        "future_video_current_patch": video.get("current_patch"),
    }


def inference_probe(model, batch: dict, noise: torch.Tensor) -> torch.Tensor:
    keys = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "image_grid_thw")
    inputs = {key: batch[key] for key in keys if key in batch}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        native = model.get_base_model() if hasattr(model, "get_base_model") else model
        native.config.use_cache = True
        native.config.attention_implementation = "eager"
        backbone = native.model.qwenvl_with_expert
        if backbone.config.attention_implementation != "eager":
            backbone.config.attention_implementation = "eager"
            backbone.attention_interface = backbone.get_attention_interface()
        for name in ("pos_embeds", "position_embeddings", "cu_seqlens", "visual_split_sizes", "visual_max_seqlen"):
            setattr(backbone, name, None)
        actions = native.sample_actions(**inputs, noise=noise.clone())
    if actions.shape != (1, 50, 55) or not torch.isfinite(actions).all():
        raise ValueError("Offline policy inference returned invalid normalized actions.")
    return actions.detach().cpu()


def train(arguments: argparse.Namespace) -> None:
    from lingbotvla.data.multimodal.data_collator import VLADataCollatorWithPacking
    from lingbotvla.models.vla.vision_models.module_utils import build_depth_model, build_video_model

    recipe, configuration = read_configuration(arguments.config)
    output = (ROOT / (arguments.output or recipe["output_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.smoke:
        recipe["max_steps"] = 3
        configuration["train"]["max_steps"] = 3
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("The training/smoke entrypoint requires the available CUDA GPU.")
        if min(recipe["micro_batch_size"], recipe["gradient_accumulation_steps"]) < 1:
            raise ValueError("Micro-batch size and gradient accumulation must be positive.")
        if not arguments.resume and any((output / "checkpoints").glob("global_step_*")):
            raise FileExistsError("Output already contains checkpoints; select an explicit --resume checkpoint.")
        fingerprint = snapshot_configuration(output, recipe, configuration)
        random.seed(recipe["seed"])
        np.random.seed(recipe["seed"])
        torch.manual_seed(recipe["seed"])
        torch.cuda.manual_seed_all(recipe["seed"])
        write_status(output, "loading_model", total_steps=recipe["max_steps"])
        model = add_adapters(build_base(configuration, "cuda"), recipe)
        model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer, scheduler = build_optimizer(model, recipe)
        step, consumed = 0, 0
        save_after_configuration_change = False
        if arguments.resume:
            resume_checkpoint = arguments.resume.resolve()
            load_fingerprint = checkpoint_load_fingerprint(output, resume_checkpoint, fingerprint)
            state = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, load_fingerprint)
            step, consumed = state["step"], state["consumed"]
            save_after_configuration_change = load_fingerprint != fingerprint
        dataset = build_dataset(configuration, model.get_base_model().config)
        if len(dataset) != 105882 or dataset.dataset_meta.total_episodes != 600:
            raise ValueError("Expected the complete audited 600-episode dataset.")
        loader = DataLoader(
            dataset,
            batch_size=recipe["micro_batch_size"],
            sampler=ReplayableSampler(len(dataset), recipe["seed"], consumed),
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
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
            print(f"Signal {signum}: finish the current optimizer step and save.", flush=True)

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        parameters = list(trainable_parameters(model).values())
        writer = SummaryWriter(str(output / "tensorboard"))
        batches = iter(loader)
        fixed_batch = None
        torch.cuda.reset_peak_memory_stats()
        write_status(
            output,
            "training",
            step=step,
            consumed=consumed,
            total_steps=recipe["max_steps"],
            micro_batch_size=recipe["micro_batch_size"],
            gradient_accumulation_steps=recipe["gradient_accumulation_steps"],
            effective_batch_size=recipe["micro_batch_size"] * recipe["gradient_accumulation_steps"],
            trainable_parameters=sum(parameter.numel() for parameter in parameters),
        )
        checkpoint = None
        while step < recipe["max_steps"]:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = np.zeros(6, dtype=np.float64)
            started = time.monotonic()
            learning_rate = optimizer.param_groups[0]["lr"]
            for _ in range(recipe["gradient_accumulation_steps"]):
                cpu_batch = next(batches)
                if arguments.smoke and fixed_batch is None:
                    fixed_batch = {
                        key: value[:1].clone() for key, value in cpu_batch.items() if isinstance(value, torch.Tensor)
                    }
                batch = {
                    key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in cpu_batch.items()
                }
                targets = teacher_targets(batch, depth_models, video_model, configuration["train"]["align_params"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**batch, **targets)
                    loss = result[0] / recipe["gradient_accumulation_steps"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss before optimizer step {step + 1}")
                loss.backward()
                losses += (
                    np.asarray(
                        [float(value.detach()) if isinstance(value, torch.Tensor) else value for value in result[:6]]
                    )
                    / recipe["gradient_accumulation_steps"]
                )
                consumed += cpu_batch["state"].shape[0]
                del result, loss, targets, batch
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, recipe["optimizer"]["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = {
                "loss": float(losses[0]),
                "action_loss": float(losses[1]),
                "depth_loss": float(losses[2]),
                "future_depth_loss": float(losses[3]),
                "video_loss": float(losses[4]),
                "balance_loss": float(losses[5]),
                "grad_norm": float(grad_norm),
                "learning_rate": learning_rate,
                "seconds_per_step": time.monotonic() - started,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "micro_batch_size": recipe["micro_batch_size"],
                "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
                "effective_batch_size": recipe["micro_batch_size"] * recipe["gradient_accumulation_steps"],
            }
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "consumed": consumed, **metrics}) + "\n")
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)
            if step == 1 or step % 10 == 0 or arguments.smoke:
                write_status(
                    output, "training", step=step, total_steps=recipe["max_steps"], consumed=consumed, **metrics
                )
            if (
                step % recipe["save_steps"] == 0
                or step == recipe["max_steps"]
                or stop_requested
                or save_after_configuration_change
            ):
                checkpoint = save_checkpoint(
                    output / "checkpoints",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    consumed,
                    fingerprint,
                    recipe["keep_checkpoints"],
                )
                writer.flush()
                save_after_configuration_change = False
                write_status(output, "checkpoint_saved", step=step, checkpoint=str(checkpoint), **metrics)
            if stop_requested:
                write_status(output, "paused", step=step, checkpoint=str(checkpoint))
                writer.close()
                return
        if checkpoint is None:
            checkpoint = save_checkpoint(
                output / "checkpoints",
                model,
                optimizer,
                scheduler,
                step,
                consumed,
                fingerprint,
                recipe["keep_checkpoints"],
            )
        del depth_models, video_model, batches, loader
        torch.cuda.empty_cache()
        if arguments.smoke:
            if fixed_batch is None:
                fixed_batch = {
                    key: value.clone()
                    for key, value in next(
                        iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=VLADataCollatorWithPacking()))
                    ).items()
                    if isinstance(value, torch.Tensor)
                }
            probe_batch = {key: value.cuda() for key, value in fixed_batch.items()}
            noise = torch.randn((1, 50, 55), device="cuda", dtype=probe_batch["state"].dtype)
            reference = inference_probe(model, probe_batch, noise)
            for parameter in parameters:
                parameter.data.zero_()
            load_checkpoint(checkpoint, model, optimizer, scheduler, fingerprint)
            restored = inference_probe(model, probe_batch, noise)
            torch.testing.assert_close(restored, reference, atol=1e-5, rtol=1e-5)
            write_status(output, "smoke_reload_passed", step=step, checkpoint=str(checkpoint))
            model = model.merge_and_unload(safe_merge=True)
            model.bfloat16()
            model.config.attention_implementation = "eager"
            backbone = model.model.qwenvl_with_expert
            backbone.config.attention_implementation = "eager"
            backbone.attention_interface = backbone.get_attention_interface()
            merged_reference = inference_probe(model, probe_batch, noise)
            torch.save(
                {"batch": fixed_batch, "noise": noise.cpu(), "actions": merged_reference},
                output / "inference_reference.pt",
            )
            write_status(
                output,
                "smoke_merge_passed",
                step=step,
                merge_action_max_difference=float((merged_reference - reference).abs().max()),
            )
            del probe_batch, noise, reference, restored, merged_reference
        del optimizer, scheduler, parameters
        write_status(output, "exporting", step=step, checkpoint=str(checkpoint))
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload(safe_merge=True)
        model.bfloat16()
        model.cpu()
        torch.cuda.empty_cache()
        export_weights(model, checkpoint / "hf_ckpt")
        writer.close()
        write_status(
            output, "smoke_exported" if arguments.smoke else "completed", step=step, checkpoint=str(checkpoint)
        )
    except Exception as error:
        write_status(output, "failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vla/airbot_play/airbot_play_lora.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    train(parser.parse_args())
