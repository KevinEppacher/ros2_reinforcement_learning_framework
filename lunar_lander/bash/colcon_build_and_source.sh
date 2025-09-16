#!/usr/bin/env bash
# Build and source the ROS 2 workspace, and cd into it.
# Usage:
#   source ./colcon_build_and_source.sh [--build]
# If --build is given, build the workspace. Otherwise, only source.

set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Find workspace root by walking up until a directory containing 'src' exists
find_ws() {
  local cur="$1"
  while [ "$cur" != "/" ]; do
    if [ -d "$cur/src" ]; then echo "$cur"; return 0; fi
    cur="$(dirname "$cur")"
  done
  return 1
}

WS_DIR="$(find_ws "$SCRIPT_DIR")"
if [ -z "${WS_DIR:-}" ]; then
  echo "ERROR: Could not locate a colcon workspace (a folder containing 'src') above: $SCRIPT_DIR" >&2
  return 1 2>/dev/null || exit 1
fi

# Safe-source helper to avoid 'unbound variable' issues in setup scripts
source_safe() { set +u; # shellcheck disable=SC1090
  source "$1"; set -u; }

# Source ROS 2
source_safe /opt/ros/humble/setup.bash

cd "$WS_DIR"

# Only build if --build is given
if [[ "${1:-}" == "--build" ]]; then
  colcon build --symlink-install
fi

source_safe "$WS_DIR/install/setup.bash"

echo "Workspace ready at: $WS_DIR"