from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoProcessor

from lingbotvla.data.vla_data.base_dataset import VLADataset
from tools.prepare_airbot_data import action_windows, read_parquet_tree


def check(config_path: Path, report_path: Path) -> dict:
    configuration = yaml.safe_load(config_path.read_text())
    data = configuration["data"]
    for key in ("joints", "norm_type"):
        data[key] = [str(value) if isinstance(value, dict) else value for value in data[key]]
    processor = AutoProcessor.from_pretrained(configuration["model"]["tokenizer_path"], local_files_only=True)
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config

    model_config = LingbotVLAV2Config(**configuration["model"], **configuration["train"])
    dataset = VLADataset(
        data["train_path"],
        "airbot_play",
        SimpleNamespace(**data),
        data["robot_config_root"],
        config=model_config,
        processor=processor,
        use_future_image=data["use_future_image"],
        use_depth_align=True,
    )
    source = Path(data["train_path"])
    episodes = read_parquet_tree(source / "meta/episodes").sort_by("episode_index").to_pylist()
    frames = read_parquet_tree(source / "data").sort_by("index")
    states = np.asarray(frames["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(frames["action"].to_pylist(), dtype=np.float32)
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pandas()
    chosen = []
    for prompt in tasks.index:
        matches = [episode for episode in episodes if episode["tasks"] == [prompt]]
        chosen.extend((matches[0], matches[-1]))
        tokens = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if len(tokens) > model_config.tokenizer_max_length:
            raise ValueError(f"Prompt would be truncated: {prompt}")
    results = []
    for episode in chosen:
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        windows = action_windows(actions[start:end], states[start:end], model_config.chunk_size)
        for index in (start, end - 1):
            training_item = dataset.getitem(index)
            np.testing.assert_allclose(training_item["future_video_effective_fps"], 25 / 49, atol=1e-7)
            raw = dataset.dataset[index]
            np.testing.assert_allclose(raw["observation.state"], states[index], atol=1e-7)
            expected = actions[np.minimum(np.arange(index, index + model_config.chunk_size), end - 1)]
            np.testing.assert_allclose(raw["action"], expected, atol=1e-7)
            item = dataset.feature_transform.apply(dataset.check_lerobot_item(raw))
            np.testing.assert_array_equal(item["action_joint_mask"].nonzero().flatten(), [0, 1, 2, 3, 4, 5, 28])
            if item["actions"].shape != (50, 55) or item["state"].shape != (55,):
                raise ValueError("Unexpected model action/state dimensions.")
            for key in ("images", "state", "actions"):
                if not torch.isfinite(item[key]).all():
                    raise ValueError(f"Nonfinite transformed {key}")
            action_stats = dataset.feature_transform.normalizer.norm_stats["action.arm.position"]
            expected_delta = (windows[index - start, :, :6] - action_stats["mean"]) / (action_stats["std"] + 1e-6)
            np.testing.assert_allclose(item["actions"][:, :6], expected_delta, atol=2e-6)
            restored = dataset.feature_transform.unapply(item)
            np.testing.assert_allclose(restored["action"], expected, atol=2e-6)
            np.testing.assert_allclose(restored["observation.state"], states[index], atol=2e-6)
            results.append(
                {
                    "index": index,
                    "episode": episode["episode_index"],
                    "prompt": raw["task"],
                    "images": list(item["images"].shape),
                    "padding_steps": int(item["action_is_pad"].sum()),
                    "max_action_roundtrip_error": float(np.max(np.abs(np.asarray(restored["action"]) - expected))),
                }
            )
            print(f"Verified frame {index}, task={raw['task']}", flush=True)
    report = {"status": "passed", "samples": results, "gpu_used": False, "hardware_initialized": False}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/vla/airbot_play/airbot_play.yaml"))
    parser.add_argument("--report", type=Path, default=Path("logs/airbot_data_preflight.json"))
    arguments = parser.parse_args()
    check(arguments.config, arguments.report)
