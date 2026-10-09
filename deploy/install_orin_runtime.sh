#!/usr/bin/env bash
set -euo pipefail

LINGBOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$LINGBOT_ROOT/.venv/bin/python" ]]; then
  /usr/bin/python3 -m venv "$LINGBOT_ROOT/.venv"
fi
cp "$LINGBOT_ROOT/deploy/orin_shared_runtime.pth" "$LINGBOT_ROOT/.venv/lib/python3.10/site-packages/orin_shared_runtime.pth"
export PIP_CACHE_DIR="$LINGBOT_ROOT/cache/pip" TMPDIR="$LINGBOT_ROOT/cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
"$LINGBOT_ROOT/.venv/bin/python" -m pip install --no-deps \
  'transformers==4.57.3' 'qwen-vl-utils==0.0.11' 'timm==1.0.20' 'torchdata==0.11.0' 'lerobot==0.4.2'
