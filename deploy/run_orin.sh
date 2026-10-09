#!/usr/bin/env bash
set -euo pipefail

LINGBOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LINGBOT_TASK="${1:?Usage: bash deploy/run_orin.sh can|red-paperbag|watermelon [inference options]}"
shift
case "$LINGBOT_TASK" in
  can) LINGBOT_PROMPT='Pick up the can, return to the starting position, and keep holding it.' ;;
  red-paperbag) LINGBOT_PROMPT='Pick up the red paper bag and hold in the reset position.' ;;
  watermelon) LINGBOT_PROMPT='Pick up the watermelon peel, return to the starting position, and keep holding it.' ;;
  *) printf 'Unknown task: %s\n' "$LINGBOT_TASK" >&2; exit 2 ;;
esac
exec bash "$LINGBOT_ROOT/deploy/orin_env.sh" "$LINGBOT_ROOT/.venv/bin/python" \
  -m deploy.airbot_play_lingbot_inference \
  --model-path "$LINGBOT_ROOT/deployments/multitask-52500/checkpoints/global_step_52500/hf_ckpt" \
  --prompt "$LINGBOT_PROMPT" --robot-interface can2 --control-frequency 100 \
  --reset-action 0 0 0 0 0 0 0.07 --max-steps 1000 --chunk-size-execute 30 "$@"
