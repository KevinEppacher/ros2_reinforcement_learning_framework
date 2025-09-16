#!/usr/bin/env bash
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

# Training
ros2 run rl_trainers rl_trainers \
  --ros-args \
  --params-file /app/src/lunar_lander/config/lunar_lander_rl_config.yaml \
  -r __ns:=/lunar_lander \
  -r __node:=rl_trainers \
  -p metadata.mode:=train