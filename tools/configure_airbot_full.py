from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def build_configuration(arguments: argparse.Namespace) -> dict:
    if min(arguments.steps, arguments.micro_batch_size, arguments.global_batch_size, arguments.world_size) <= 0:
        raise ValueError("Steps and batch/world sizes must be positive.")
    per_iteration = arguments.micro_batch_size * arguments.world_size
    if arguments.global_batch_size % per_iteration:
        raise ValueError("Global batch size must be divisible by micro batch size times world size.")
    if not 0 <= arguments.warmup_steps <= arguments.steps or arguments.learning_rate <= 0:
        raise ValueError("Use a positive learning rate and a warmup within the training step count.")
    configuration = yaml.safe_load((ROOT / "configs/vla/airbot_play/airbot_play.yaml").read_text())
    weights = arguments.weights_root.expanduser().resolve()
    model = weights / "lingbot-vla-v2-6b"
    configuration["model"].update(
        model_path=str(model),
        tokenizer_path=str(weights / "Qwen3-VL-4B-Instruct"),
    )
    configuration["data"].update(
        train_path=str(arguments.dataset.expanduser().resolve()),
        robot_config_root=str(ROOT / "configs/robot_configs"),
        norm_stats_file=str(arguments.norm_stats.expanduser().resolve()),
        num_workers=2,
        img_size=224,
        image_augment=False,
    )
    train = configuration["train"]
    train.update(
        output_dir=str(arguments.output_dir.expanduser().resolve()),
        max_steps=arguments.steps,
        seed=42,
        micro_batch_size=arguments.micro_batch_size,
        global_batch_size=arguments.global_batch_size,
        gradient_accumulation_steps=arguments.global_batch_size // per_iteration,
        freeze_vit=False,
        freeze_vision_encoder=False,
        train_expert_only=False,
        train_state_proj=True,
        chunk_size=50,
        action_fp32=False,
        use_compile=False,
        enable_gradient_checkpointing=True,
        enable_fp32=False,
        enable_mixed_precision=arguments.profile == "fp32-master",
        optimizer="adamw" if arguments.profile == "fp32-master" else "anyprecision_adamw",
        data_parallel_mode="fsdp2",
        enable_full_shard=True,
        init_device="cuda",
        lr=arguments.learning_rate,
        lr_min=arguments.learning_rate,
        lr_decay_style="constant",
        lr_warmup_ratio=arguments.warmup_steps / arguments.steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        use_moe_expert_lr=False,
        save_steps=2500,
        save_epochs=0,
        enable_resume=False,
    )
    train.pop("num_train_epochs", None)
    train["align_params"]["depth"].update(
        moge_path=str(weights / "moge-2-vitb-normal/model.pt"),
        morgbd_path=str(model / "depth/model.pt"),
    )
    train["align_params"]["video"].update(
        ckpt_path=str(model / "dino_video/teacher_step_10000.pth"),
        config_path=str(model / "dino_video/config.yaml"),
    )
    return configuration


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate a portable AIRBOT full-policy fine-tuning configuration.")
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--weights-root", type=Path, default=ROOT / "weights")
    result.add_argument("--norm-stats", type=Path, default=ROOT / "assets/norm_stats/airbot_play_multitask.json")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, default=ROOT / "configs/local/airbot_full.yaml")
    result.add_argument("--profile", choices=("bf16-kahan", "fp32-master"), default="bf16-kahan")
    result.add_argument("--steps", type=int, default=100000)
    result.add_argument("--world-size", type=int, default=1)
    result.add_argument("--micro-batch-size", type=int, default=1)
    result.add_argument("--global-batch-size", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=5e-5)
    result.add_argument("--warmup-steps", type=int, default=1000)
    return result


def main() -> None:
    arguments = parser().parse_args()
    configuration = build_configuration(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as target:
        yaml.safe_dump(configuration, target, sort_keys=False)
    print(
        json.dumps(
            {
                "config": str(arguments.output.resolve()),
                "profile": arguments.profile,
                "optimizer": configuration["train"]["optimizer"],
                "world_size": arguments.world_size,
                "micro_batch_size": arguments.micro_batch_size,
                "gradient_accumulation_steps": configuration["train"]["gradient_accumulation_steps"],
                "global_batch_size": arguments.global_batch_size,
                "max_steps": arguments.steps,
                "lora": False,
                "freeze_vision_encoder": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
