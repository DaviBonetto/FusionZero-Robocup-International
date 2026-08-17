#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime_common.sh
. "${SCRIPT_DIR}/runtime_common.sh"

ENV_FILE="${FZ_DEFAULT_ENV_FILE}"
SKIP_APT=0
INSTALL_OPTIONAL_ML=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --skip-apt)
            SKIP_APT=1
            shift
            ;;
        --with-optional-ml)
            INSTALL_OPTIONAL_ML=1
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

APT_PACKAGES=(
    python3
    python3-pip
    python3-venv
    python3-opencv
    python3-numpy
    python3-psutil
    python3-serial
    python3-pil
    python3-smbus
    i2c-tools
)

if [[ "${SKIP_APT}" -eq 0 ]]; then
    sudo apt-get update
    sudo apt-get install -y "${APT_PACKAGES[@]}"
fi

python3 -m venv --system-site-packages "${FZ_PROJECT_ROOT}/.venv"
"${FZ_PROJECT_ROOT}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"${FZ_PROJECT_ROOT}/.venv/bin/python" -m pip install -r "${FZ_PROJECT_ROOT}/deploy/requirements-pi-base.txt"

if [[ "${INSTALL_OPTIONAL_ML}" -eq 1 ]]; then
    "${FZ_PROJECT_ROOT}/.venv/bin/python" -m pip install -r "${FZ_PROJECT_ROOT}/deploy/requirements-pi-optional-ml.txt"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    cp "${FZ_PROJECT_ROOT}/deploy/fusionzero.env.example" "${ENV_FILE}"
fi

cat <<EOF
Pi runtime install complete.
- Venv: ${FZ_PROJECT_ROOT}/.venv
- Env file: ${ENV_FILE}
- Headless run: bash "${FZ_PROJECT_ROOT}/scripts/run_pi_headless.sh" --env-file "${ENV_FILE}"
- Systemd install: sudo bash "${FZ_PROJECT_ROOT}/scripts/install_systemd_service.sh" --env-file "${ENV_FILE}" --enable-now
EOF
