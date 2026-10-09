#!/usr/bin/env bash
set -euo pipefail

LINGBOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LINGBOT_NATIVE="${LINGBOT_NATIVE:-/mnt/nvme/groot/env/native-cu126}"
LINGBOT_CUDA="$LINGBOT_NATIVE/usr/local/cuda-12.6"
LINGBOT_TORCH_SITE="${LINGBOT_TORCH_SITE:-/mnt/nvme/groot/env/orin/lib/python3.10/site-packages}"
OPENPI_AIRBOT_ROOT="${OPENPI_AIRBOT_ROOT:-/mnt/nvme/pi05/runtime/official-20260910/openpi_v0.2.0}"
export OPENPI_AIRBOT_ROOT
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
export PATH="$LINGBOT_ROOT/.venv/bin:$LINGBOT_CUDA/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$LINGBOT_ROOT/cache/hf" TORCH_HOME="$LINGBOT_ROOT/cache/torch"
export TMPDIR="$LINGBOT_ROOT/cache/tmp" TRITON_CACHE_DIR="$LINGBOT_ROOT/cache/triton"
export CUDA_CACHE_PATH="$LINGBOT_ROOT/cache/cuda" XDG_CACHE_HOME="$LINGBOT_ROOT/cache/xdg"
export CUDA_HOME="$LINGBOT_CUDA" CUDA_PATH="$LINGBOT_CUDA" TRITON_PTXAS_PATH="$LINGBOT_CUDA/bin/ptxas"
export CPATH="$LINGBOT_CUDA/targets/aarch64-linux/include"
export LD_LIBRARY_PATH="$LINGBOT_CUDA/targets/aarch64-linux/lib:$LINGBOT_NATIVE/usr/lib/aarch64-linux-gnu:$LINGBOT_TORCH_SITE/torch/lib:$OPENPI_AIRBOT_ROOT/../native/lib"
for lingbot_native_python in "$LINGBOT_TORCH_SITE/nvidia" /mnt/nvme/groot/env/native-python/nvidia; do
  if [[ -d "$lingbot_native_python" ]]; then
    while IFS= read -r lingbot_library; do
      export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$lingbot_library"
    done < <(find "$lingbot_native_python" -type d -name lib | sort)
  fi
done
export QWEN3VL_PATH="$LINGBOT_ROOT/weights/Qwen3-VL-4B-Instruct"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$LINGBOT_ROOT/cache/"{hf,torch,tmp,triton,cuda,xdg}
cd "$LINGBOT_ROOT"
exec "$@"
