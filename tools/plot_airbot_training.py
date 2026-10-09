from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter


def moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def plot_training(output: Path, window: int) -> None:
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    run = json.loads((output / "run.json").read_text())
    steps = np.array([row["step"] for row in rows])
    if len(steps) == 0 or not np.all(np.diff(steps) == 1):
        raise ValueError("Expected continuous optimizer steps without missing or duplicated records.")
    change = run.get("batch_change")
    switch_step = change["step"] if change else None
    extension = run.get("step_extension")
    resume_step = extension["step"] if extension else None
    destination = output / "plots" / f"step_{int(steps[-1])}"
    destination.mkdir(parents=True, exist_ok=True)
    metric_keys = (
        "loss",
        "action_loss",
        "depth_loss",
        "future_depth_loss",
        "video_loss",
        "balance_loss",
        "seconds_per_step",
    )
    metrics = {key: np.array([row[key] for row in rows], dtype=np.float64) for key in metric_keys}
    if any(not np.isfinite(values).all() for values in metrics.values()):
        raise ValueError("Training metrics contain nonfinite values.")
    segments = [np.ones(len(steps), dtype=bool)] if change is None else [steps <= switch_step, steps > switch_step]
    smoothed = {}
    for key, values in metrics.items():
        smoothed[key] = np.empty_like(values)
        for segment in segments:
            if segment.any():
                smoothed[key][segment] = moving_mean(values[segment], window)

    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#526078",
            "text.color": "#172b4d",
            "xtick.color": "#526078",
            "ytick.color": "#526078",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9.3))
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.16, top=0.79, hspace=0.47, wspace=0.19)
    figure.text(0.075, 0.95, "LingBot-VLA 2.0 · 训练曲线", fontsize=23, weight="bold")
    figure.text(
        0.075,
        0.906,
        f"{steps[-1]:,} 个优化器更新  |  600 条轨迹 · 易拉罐 / 红纸袋 / 西瓜皮  |  RTX 5090 · LoRA",
        fontsize=11,
        color="#526078",
    )
    figure.text(
        0.075,
        0.857,
        f"最后 500 步均值：总 loss {metrics['loss'][-500:].mean():.4f}    动作 loss {metrics['action_loss'][-500:].mean():.4f}",
        fontsize=13,
        color="#125b88",
    )
    if change:
        old_batch = "×".join(map(str, change["from_batch"]))
        new_batch = "×".join(map(str, change["to_batch"]))
        figure.text(
            0.98,
            0.857,
            f"虚线：step {switch_step:,} 后 batch {old_batch} → {new_batch}",
            ha="right",
            fontsize=10,
            color="#9b590c",
        )

    def trend(axis, key, color, label, raw=False):
        if raw:
            axis.plot(steps, metrics[key], color=color, alpha=0.12, linewidth=0.45, rasterized=True)
        for index, segment in enumerate(segments):
            axis.plot(
                steps[segment], smoothed[key][segment], color=color, linewidth=1.8, label=label if index == 0 else None
            )

    for axis, key, title, color in (
        (axes[0, 0], "loss", "总训练损失", "#2463a6"),
        (axes[0, 1], "action_loss", "动作预测损失", "#128375"),
    ):
        trend(axis, key, color, f"{window} 步移动均值", raw=True)
        axis.set_title(title, loc="left", pad=13)
        axis.set_yscale("log")
        axis.set_ylim(min(0.02, metrics[key].min() * 0.8), max(4, metrics[key].max() * 1.1))
        axis.set_yticks([0.02, 0.03, 0.1, 0.3, 1, 3])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_ylabel("loss（对数刻度）")
        axis.legend(loc="upper right", frameon=False)

    auxiliary = axes[1, 0]
    for key, color, label in (
        ("depth_loss", "#6a57a5", "当前深度"),
        ("future_depth_loss", "#c06124", "未来深度"),
        ("video_loss", "#187d9a", "视频"),
        ("balance_loss", "#727781", "专家负载均衡"),
    ):
        trend(auxiliary, key, color, label)
    auxiliary.set_title("辅助损失（日志中的加权值）", loc="left", pad=13)
    auxiliary.set_yscale("log")
    auxiliary.set_yticks([0.001, 0.003, 0.01, 0.02])
    auxiliary.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
    auxiliary.yaxis.set_minor_formatter(NullFormatter())
    auxiliary.set_ylabel("loss（对数刻度）")
    auxiliary.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)

    speed = axes[1, 1]
    trend(speed, "seconds_per_step", "#2463a6", f"{window} 步移动均值", raw=True)
    speed.set_title("每次优化器更新耗时", loc="left", pad=13)
    speed.set_ylabel("秒 / 更新")
    speed.set_ylim(0, max(10, metrics["seconds_per_step"].max() * 1.1))
    speed.legend(loc="upper right", frameon=False)

    for axis in axes.flat:
        axis.set_xlim(0, steps[-1])
        axis.set_xticks(np.arange(0, steps[-1] + 1, 5000))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value / 1000:g}k" if value else "0"))
        axis.set_xlabel("优化器更新步数")
        axis.grid(axis="y", which="major", color="#e5eaf0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#cad2dd")
        if change:
            axis.axvline(switch_step + 0.5, color="#b77c24", linestyle=(0, (4, 4)), linewidth=1.2)
        if extension and steps[-1] > resume_step:
            axis.axvspan(resume_step + 0.5, steps[-1], color="#7858a6", alpha=0.045, linewidth=0)
            axis.axvline(resume_step + 0.5, color="#7858a6", linestyle=(0, (4, 4)), linewidth=1.2)

    resume_note = f"紫色虚线：step {resume_step:,} 后断点续训。" if extension else ""
    figure.text(
        0.075,
        0.092,
        f"浅色为单步原始值，深色为 {window} 步向后移动均值，平滑在 batch 切换处分段。{resume_note}",
        fontsize=9,
        color="#526078",
    )
    figure.text(
        0.075,
        0.06,
        "全部曲线来自训练集；有效 batch 从 8 增至 32，切换前后的单步波动与样本数不同。曲线不代表真机抓取成功率。",
        fontsize=9,
        color="#526078",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"training_curves.{suffix}", dpi=160, facecolor="white")
    plt.close(figure)

    with (destination / "training_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "optimizer_steps": int(steps[-1]),
        "metric_records": len(rows),
        "smoothing_window": window,
        "batch_change_step": switch_step,
        "resume_step": resume_step,
        "first_500_mean_loss": float(metrics["loss"][:500].mean()),
        "last_500_mean_loss": float(metrics["loss"][-500:].mean()),
        "last_500_mean_action_loss": float(metrics["action_loss"][-500:].mean()),
        "artifacts": [
            str(destination / name) for name in ("training_curves.png", "training_curves.pdf", "training_metrics.csv")
        ],
    }
    if extension:
        before_resume = (steps > resume_step - 500) & (steps <= resume_step)
        if before_resume.any():
            previous_loss = float(metrics["loss"][before_resume].mean())
            previous_action_loss = float(metrics["action_loss"][before_resume].mean())
            summary.update(
                pre_resume_last_500_mean_loss=previous_loss,
                pre_resume_last_500_mean_action_loss=previous_action_loss,
                mean_loss_reduction_percent=100 * (1 - summary["last_500_mean_loss"] / previous_loss),
                mean_action_loss_reduction_percent=100
                * (1 - summary["last_500_mean_action_loss"] / previous_action_loss),
            )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/airbot_play_multitask_lora_30k"))
    parser.add_argument("--window", type=int, default=200)
    arguments = parser.parse_args()
    if arguments.window < 1:
        parser.error("Smoothing window must be positive.")
    plot_training(arguments.output.resolve(), arguments.window)
