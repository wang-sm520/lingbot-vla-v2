import argparse
import hashlib
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
DESTINATION = Path("wheelhouse/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
PART_SIZE = 8 * 1024 * 1024


def main(url=URL, destination=DESTINATION):
    with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(20, 30)) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Server did not honor HTTP byte ranges.")
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    parts = destination.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size < total:
        with destination.open("rb") as source:
            for index in range((total + PART_SIZE - 1) // PART_SIZE):
                content = source.read(PART_SIZE)
                if not content:
                    break
                part = parts / f"{index:04d}"
                if not part.exists() or part.stat().st_size < len(content):
                    part.write_bytes(content)

    def download(index):
        start = index * PART_SIZE
        end = min(total, start + PART_SIZE) - 1
        part = parts / f"{index:04d}"
        expected_size = end - start + 1
        for attempt in range(1, 11):
            existing = part.stat().st_size if part.exists() else 0
            if existing == expected_size:
                return part
            if existing > expected_size:
                raise ValueError(f"Oversized partial file: {part}")
            try:
                with requests.get(
                    url, headers={"Range": f"bytes={start + existing}-{end}"}, stream=True, timeout=(20, 45)
                ) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start + existing}-{end}/{total}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"Unexpected byte range for part {index}")
                    with part.open("ab") as target:
                        for content in response.iter_content(1024 * 1024):
                            target.write(content)
                if part.stat().st_size != expected_size:
                    raise ValueError(f"Incomplete part {index}")
                print(f"Part {index + 1}/{(total + PART_SIZE - 1) // PART_SIZE} complete", flush=True)
                return part
            except Exception as error:
                print(f"Part {index}, attempt {attempt}: {type(error).__name__}: {error}", flush=True)
                if attempt == 10:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(download, range((total + PART_SIZE - 1) // PART_SIZE)))
    assembled = destination.with_suffix(".assembled")
    with assembled.open("wb") as output:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output)
    if assembled.stat().st_size != total:
        raise ValueError("Wheel size mismatch.")
    if destination.suffix == ".whl":
        with zipfile.ZipFile(assembled) as archive:
            damaged = archive.testzip()
            if damaged is not None:
                raise ValueError(f"Wheel CRC mismatch: {damaged}")
    assembled.replace(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    destination.with_suffix(".sha256").write_text(f"{digest}  {destination.name}\n")
    print(f"File assembled: {total} bytes, sha256={digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    arguments = parser.parse_args()
    main(arguments.url, arguments.output)
