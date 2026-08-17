#!/usr/bin/env bash

set -euo pipefail

SERVICE_NAME="${1:-fusionzero-live-dashboard.service}"
LOG_LINES="${FZ_LOG_LINES:-80}"

sudo systemctl status --no-pager "${SERVICE_NAME}" || true
echo
sudo journalctl -u "${SERVICE_NAME}" -n "${LOG_LINES}" --no-pager || true
