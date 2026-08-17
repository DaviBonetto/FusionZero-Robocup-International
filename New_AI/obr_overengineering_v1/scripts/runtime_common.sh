#!/usr/bin/env bash

set -euo pipefail

FZ_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FZ_PROJECT_ROOT="$(cd "${FZ_SCRIPT_DIR}/.." && pwd)"
FZ_REPO_ROOT="$(cd "${FZ_PROJECT_ROOT}/../.." && pwd)"
FZ_DEFAULT_ENV_FILE="${FZ_PROJECT_ROOT}/deploy/fusionzero.env"

fz_load_env() {
    local env_file="${1:-${FZ_DEFAULT_ENV_FILE}}"
    if [[ -f "${env_file}" ]]; then
        set -a
        # shellcheck disable=SC1090
        . "${env_file}"
        set +a
    fi
    FZ_ENV_FILE="${env_file}"
}

fz_python_bin() {
    if [[ -n "${FZ_PYTHON:-}" ]]; then
        printf '%s\n' "${FZ_PYTHON}"
        return
    fi
    if [[ -x "${FZ_PROJECT_ROOT}/.venv/bin/python" ]]; then
        printf '%s\n' "${FZ_PROJECT_ROOT}/.venv/bin/python"
        return
    fi
    command -v python3
}

fz_bool_enabled() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

fz_require_file() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "missing required file: ${path}" >&2
        exit 1
    fi
}
