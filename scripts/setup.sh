#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD="$PYTHON_BIN"
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_CMD="python3.11"
else
    PYTHON_CMD="python3"
fi

EXTRAS=(dev)
for arg in "$@"; do
    case "$arg" in
        --hardware) EXTRAS+=(hardware) ;;
        --labeling) EXTRAS+=(labeling) ;;
        -h|--help)
            echo "Usage: ./scripts/setup.sh [--hardware] [--labeling]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

"$PYTHON_CMD" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Python 3.11 or 3.12 is required"'
"$PYTHON_CMD" -m venv "$REPO_DIR/.venv"

VENV_PYTHON="$REPO_DIR/.venv/bin/python"
EXTRAS_CSV="$(IFS=,; echo "${EXTRAS[*]}")"

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -e "$REPO_DIR[$EXTRAS_CSV]"
"$VENV_PYTHON" -m pytest -q

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
