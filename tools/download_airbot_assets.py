from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


ASSETS = {
    "lingbot": ("robbyant/lingbot-vla-v2-6b", "11c703bf6a5c1f45b3b69168482da11fdbba53d7", "lingbot-vla-v2-6b", None),
    "tokenizer": (
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "Qwen3-VL-4B-Instruct",
        ["*.json", "*.txt", "*.jinja"],
    ),
    "depth": ("Ruicheng/moge-2-vitb-normal", "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368", "moge-2-vitb-normal", None),
}


def download(name: str, weights_root: Path, reports_root: Path) -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    repo_id, revision, directory, patterns = ASSETS[name]
    destination = weights_root / directory
    print(f"Downloading {repo_id}@{revision}", flush=True)
    for attempt in range(1, 9):
        try:
            snapshot_download(
                repo_id, revision=revision, local_dir=destination, allow_patterns=patterns, max_workers=2
            )
            metadata = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
            files = []
            for remote in metadata.siblings:
                if patterns is not None and not any(fnmatch(remote.rfilename, pattern) for pattern in patterns):
                    continue
                path = destination / remote.rfilename
                if path.stat().st_size != remote.size:
                    raise ValueError(f"Size mismatch: {path}")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if remote.lfs and digest != remote.lfs.sha256:
                    raise ValueError(f"SHA-256 mismatch: {path}")
                files.append({"path": remote.rfilename, "bytes": remote.size, "sha256": digest})
            reports_root.mkdir(parents=True, exist_ok=True)
            report = {"repo_id": repo_id, "revision": revision, "directory": str(destination), "files": files}
            (reports_root / f"asset_{name}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Verified {name}: {sum(entry['bytes'] for entry in files)} bytes", flush=True)
            return
        except Exception as error:
            print(f"Attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
            if attempt == 8:
                raise
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", choices=tuple(ASSETS), nargs="+", default=list(ASSETS))
    parser.add_argument("--weights-root", type=Path, default=Path("weights"))
    parser.add_argument("--reports-root", type=Path, default=Path("logs"))
    parser.add_argument("--wait-for-pid", type=int)
    arguments = parser.parse_args()
    if arguments.wait_for_pid:
        process = Path(f"/proc/{arguments.wait_for_pid}/stat")
        try:
            start_time = process.read_text().rsplit(")", 1)[1].split()[19]
            while process.read_text().rsplit(")", 1)[1].split()[19] == start_time:
                time.sleep(10)
        except FileNotFoundError:
            pass
    for asset in arguments.assets:
        download(asset, arguments.weights_root, arguments.reports_root)
