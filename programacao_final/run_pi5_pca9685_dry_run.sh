#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export FZ_ROBOT_BACKEND=pca9685
export FZ_ROBOT_DRY_RUN=1

exec "${REPO_ROOT}/New_AI/obr_overengineering_v1/scripts/run_pi_headless.sh" \
    --env-file "${SCRIPT_DIR}/fusionzero_pi5_pca9685.env" \
    --robot-dry-run \
    "$@"
