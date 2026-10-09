from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from deploy.airbot_play_safety import G2_LIMITS, JOINT_LIMITS
from lingbotvla.utils import normalize


def action_windows(actions: np.ndarray, states: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0 or len(actions) == 0 or actions.shape != states.shape:
        raise ValueError("Expected nonempty matching state/action arrays and a positive horizon.")
    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon), len(actions) - 1)
    windows = actions[indices].copy()
    windows[:, :, :6] -= states[:, None, :6]
    return windows


def read_parquet_tree(root: Path) -> pa.Table:
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {root}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(dataset_root: Path, stats_path: Path, report_path: Path, horizon: int = 50) -> dict:
    info = json.loads((dataset_root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 25:
        raise ValueError("Expected the AIRBOT LeRobot v3 dataset recorded at 25 FPS.")
    frames = read_parquet_tree(dataset_root / "data").sort_by("index")
    episodes = read_parquet_tree(dataset_root / "meta/episodes").sort_by("episode_index")
    tasks = pq.read_table(dataset_root / "meta/tasks.parquet").to_pandas()
    states = np.asarray(frames["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(frames["action"].to_pylist(), dtype=np.float32)
    for name, values in (("state", states), ("action", actions)):
        if values.shape != (info["total_frames"], 7) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} shape or nonfinite values: {values.shape}")
    if episodes.num_rows != info["total_episodes"] or episodes.num_rows != 600:
        raise ValueError("Expected all 600 episodes.")
    np.testing.assert_array_equal(frames["index"].to_numpy(), np.arange(len(states)))
    np.testing.assert_array_equal(episodes["episode_index"].to_numpy(), np.arange(600))
    counts = {}
    keys = (
        "observation.state.arm.position",
        "observation.state.effector.position",
        "action.arm.position",
        "action.effector.position",
    )
    statistics = {key: normalize.RunningStats() for key in keys}
    video_paths = set()
    previous_end = 0
    for episode in episodes.to_pylist():
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        length = episode["length"]
        if start != previous_end or end - start != length or length <= 0:
            raise ValueError(f"Noncontiguous episode boundaries: {episode['episode_index']}")
        previous_end = end
        episode_frames = frames.slice(start, length)
        np.testing.assert_array_equal(episode_frames["episode_index"].to_numpy(), episode["episode_index"])
        np.testing.assert_array_equal(episode_frames["frame_index"].to_numpy(), np.arange(length))
        np.testing.assert_allclose(episode_frames["timestamp"].to_numpy(), np.arange(length) / 25, atol=1e-5)
        task_indices = np.unique(episode_frames["task_index"].to_numpy())
        if len(task_indices) != 1:
            raise ValueError("Multiple task indices within an episode.")
        prompt = tasks.index[tasks["task_index"] == int(task_indices[0])].item()
        if episode["tasks"] != [prompt]:
            raise ValueError(f"Task metadata mismatch for episode {episode['episode_index']}")
        task_counts = counts.setdefault(prompt, {"episodes": 0, "frames": 0})
        task_counts["episodes"] += 1
        task_counts["frames"] += length
        for camera in ("observation.images.head", "observation.images.wrist"):
            prefix = f"videos/{camera}"
            relative = info["video_path"].format(
                video_key=camera,
                chunk_index=episode[f"{prefix}/chunk_index"],
                file_index=episode[f"{prefix}/file_index"],
            )
            if not (dataset_root / relative).is_file():
                raise FileNotFoundError(dataset_root / relative)
            duration = episode[f"{prefix}/to_timestamp"] - episode[f"{prefix}/from_timestamp"]
            if abs(duration - length / 25) > 1 / 25:
                raise ValueError(f"Video duration mismatch in episode {episode['episode_index']}")
            video_paths.add(relative)
        episode_states = states[start:end]
        windows = action_windows(actions[start:end], episode_states, horizon).reshape(-1, 7)
        for key, values in zip(
            keys, (episode_states[:, :6], episode_states[:, 6:], windows[:, :6], windows[:, 6:]), strict=True
        ):
            statistics[key].update(values.astype(np.float64))
    if previous_end != len(states) or sorted(value["episodes"] for value in counts.values()) != [100, 200, 300]:
        raise ValueError("Unexpected task/frame coverage.")
    limits = np.vstack((JOINT_LIMITS, G2_LIMITS))
    range_audit = {}
    for name, values in (("state", states), ("action", actions)):
        outside = (values < limits[:, 0]) | (values > limits[:, 1])
        range_audit[name] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "outside_per_dimension": outside.sum(axis=0).tolist(),
            "frames_outside_any_dimension": int(outside.any(axis=1).sum()),
        }
    norm_stats = {key: value.get_statistics() for key, value in statistics.items()}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = stats_path.with_suffix(".json.tmp")
    normalize.save(temporary, norm_stats, count=len(states))
    temporary.replace(stats_path)
    source_paths = sorted((dataset_root / "meta").rglob("*.parquet"))
    source_paths += sorted((dataset_root / "data").rglob("*.parquet"))
    source_paths.append(dataset_root / "meta/info.json")
    report = {
        "dataset": str(dataset_root.resolve()),
        "episodes": 600,
        "frames": len(states),
        "fps": 25,
        "prediction_horizon": horizon,
        "tasks": counts,
        "normalization": str(stats_path.resolve()),
        "normalization_sha256": file_digest(stats_path),
        "action_statistics_samples": len(states) * horizon,
        "boundary_padding": "repeat final action within each episode",
        "arm_action": "target joint radians minus current observation joint radians",
        "gripper_action": "absolute meters",
        "video_files": sorted(video_paths),
        "range_audit": range_audit,
        "range_policy": "report only; source data and hardware limits unchanged",
        "source_sha256": {str(path.relative_to(dataset_root)): file_digest(path) for path in source_paths},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=Path("assets/norm_stats/airbot_play_multitask.json"))
    parser.add_argument("--report", type=Path, default=Path("logs/airbot_data_audit.json"))
    arguments = parser.parse_args()
    result = prepare(arguments.dataset, arguments.stats, arguments.report)
    print(json.dumps({key: result[key] for key in ("episodes", "frames", "tasks", "range_audit")}, indent=2))
