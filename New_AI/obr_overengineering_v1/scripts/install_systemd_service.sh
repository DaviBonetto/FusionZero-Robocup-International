#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime_common.sh
. "${SCRIPT_DIR}/runtime_common.sh"

SERVICE_NAME="fusionzero-live-dashboard"
SERVICE_USER="$(id -un)"
ENV_FILE="${FZ_DEFAULT_ENV_FILE}"
ENABLE_NOW=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --user)
            SERVICE_USER="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --enable-now)
            ENABLE_NOW=1
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

fz_require_file "${FZ_PROJECT_ROOT}/deploy/fusionzero-live-dashboard.service.template"
fz_require_file "${FZ_PROJECT_ROOT}/scripts/run_pi_headless.sh"
fz_require_file "${ENV_FILE}"

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

sed \
    -e "s|__FZ_RUN_USER__|${SERVICE_USER}|g" \
    -e "s|__FZ_RUN_SCRIPT__|${FZ_PROJECT_ROOT}/scripts/run_pi_headless.sh|g" \
    -e "s|__FZ_ENV_FILE__|${ENV_FILE}|g" \
    "${FZ_PROJECT_ROOT}/deploy/fusionzero-live-dashboard.service.template" > "${TMP_FILE}"

sudo cp "${TMP_FILE}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload

if [[ "${ENABLE_NOW}" -eq 1 ]]; then
    sudo systemctl enable --now "${SERVICE_NAME}.service"
else
    sudo systemctl enable "${SERVICE_NAME}.service"
fi

cat <<EOF
Systemd unit installed: /etc/systemd/system/${SERVICE_NAME}.service
- Start:   sudo systemctl start ${SERVICE_NAME}.service
- Stop:    sudo systemctl stop ${SERVICE_NAME}.service
- Disable: sudo systemctl disable --now ${SERVICE_NAME}.service
- Logs:    sudo journalctl -u ${SERVICE_NAME}.service -f
EOF
