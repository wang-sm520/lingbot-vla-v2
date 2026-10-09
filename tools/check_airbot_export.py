from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from lingbotvla.airbot_lora import build_dataset


def check(checkpoint: Path, reference: Path | None = None) -> None:
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    output = checkpoint.parent.parent
    configuration = yaml.safe_load((output / "lingbotvla_cli.yaml").read_text())
    configuration["data"]["norm_stats_file"] = str(output / configuration["data"]["norm_stats_file"])
    server = LingbotVLAv2Server(
        str(checkpoint / "hf_ckpt"), use_length=-1, chunk_ret=True, use_bf16=True, use_fp32=False, use_compile=False
    )
    server.reset(robo_name="airbot_play")
    report = {"checkpoint": str(checkpoint), "hardware_initialized": False}
    if reference is not None:
        saved = torch.load(reference, map_location="cpu", weights_only=True)
        keys = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "image_grid_thw")
        batch = {key: saved["batch"][key].cuda() for key in keys if key in saved["batch"]}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = server.vla.sample_actions(**batch, noise=saved["noise"].cuda()).cpu()
        torch.testing.assert_close(predicted, saved["actions"], atol=1e-5, rtol=1e-5)
        report["merged_reload_max_error"] = float((predicted - saved["actions"]).abs().max())
    dataset = build_dataset(configuration, server.config)
    metadata = pq.read_table(Path(configuration["data"]["train_path"]) / "meta/episodes").to_pylist()
    selected = {}
    for episode in metadata:
        selected.setdefault(episode["tasks"][0], episode["dataset_from_index"])
    results = []
    for prompt, index in selected.items():
        sample = dataset.dataset[index]
        observation = {"observation.state": sample["observation.state"].numpy(), "task": prompt}
        for camera in ("observation.images.head", "observation.images.wrist"):
            frames = sample[camera]
            current = frames[0] if frames.ndim == 4 else frames
            observation[camera] = current.permute(1, 2, 0).numpy()
        actions = np.asarray(server.infer(observation)["action"])
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid exported policy output for {prompt}: {actions.shape}")
        results.append(
            {
                "prompt": prompt,
                "shape": list(actions.shape),
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
            }
        )
    report.update(status="passed", tasks=results)
    (output / "export_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    arguments = parser.parse_args()
    check(arguments.checkpoint.resolve(), arguments.reference)
