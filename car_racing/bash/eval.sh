car_racing_rl_config#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Pass --build to build, otherwise just source
BUILD_ARG=""
for arg in "$@"; do
  if [[ "$arg" == "--build" ]]; then
    BUILD_ARG="--build"
  fi
done

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/colcon_build_and_source.sh" $BUILD_ARG


# Evaluation
ros2 run rl_trainers rl_trainers \
  --ros-args \
  --params-file /app/src/car_racing/config/car_racing_rl_config.yaml \
  -r __ns:=/car_racing \
  -r __node:=rl_trainers \
  -p metadata.mode:=eval \
  -p eval.model_path:=/app/src/car_racing/data/model/PPO/car_racing_20250911_17_32