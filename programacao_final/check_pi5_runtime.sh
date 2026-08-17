#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../New_AI/obr_overengineering_v1" && pwd)"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

echo "[1/4] Python"
"${PYTHON_BIN}" --version

echo "[2/4] I2C devices"
if command -v i2cdetect >/dev/null 2>&1; then
    i2cdetect -y 1
else
    echo "i2cdetect not installed"
fi

echo "[3/4] PCA9685 Python imports"
"${PYTHON_BIN}" - <<'PY'
import board
import busio
from adafruit_pca9685 import PCA9685
print("pca9685_imports_ok")
PY

echo "[4/4] Power status"
if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd get_throttled
else
    echo "vcgencmd unavailable"
fi
