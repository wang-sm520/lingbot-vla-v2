#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${LINGBOT_ENV_NAME:-lingbotvla}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/../pi0.5/openpi_v0.2.0}"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

DATA_COLLECTION_ROOT="${OPENPI_ROOT}/../airbot-data-5.1.6.7/data-collection"
HARDWARE_TAR="${OPENPI_ROOT}/airbot_hardware_release.tar"
export OPENPI_ROOT HARDWARE_TAR

python -m pip install --no-deps -e "${DATA_COLLECTION_ROOT}"
python -m pip install -c "${REPO_ROOT}/requirements.txt" \
  'pyrealsense2==2.58.4.10922' linuxpy PyTurboJPEG more-itertools hydra-core
python - <<'PY'
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

supported = set(sys_tags())
with tarfile.open(os.environ["HARDWARE_TAR"]) as archive:
    compatible = []
    for member in archive.getmembers():
        filename = Path(member.name).name
        if not member.isfile() or not filename.startswith("airbot_hardware_py-") or not filename.endswith(".whl"):
            continue
        _, version, _, tags = parse_wheel_filename(filename)
        if str(version) == "0.2.9.2" and supported.intersection(tags):
            compatible.append(member)
    if not compatible:
        raise RuntimeError("No AIRBOT 0.2.9.2 wheel matches this Python, architecture and glibc.")
    member = sorted(compatible, key=lambda value: value.name)[0]
    with tempfile.TemporaryDirectory(prefix="lingbot-airbot-wheel-") as directory:
        wheel_path = Path(directory) / Path(member.name).name
        with archive.extractfile(member) as source, wheel_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel_path)], check=True)

sys.path.insert(0, str(Path(os.environ["OPENPI_ROOT"]) / "examples/airbot"))
import airbot_arm
import play_operator_ah

assert importlib.metadata.version("airbot-hardware-py") == "0.2.9.2"
print("AIRBOT runtime imports verified; no robot or camera was initialized.")
PY
