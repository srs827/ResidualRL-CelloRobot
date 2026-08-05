#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/label_studio_data"
VENV="$REPO_ROOT/.venv_labelstudio311"

export DEBUG=false
export LATEST_VERSION_CHECK=false
export SENTRY_DSN=
export LABEL_STUDIO_BASE_DATA_DIR="$DATA_DIR"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$SCRIPT_DIR"

exec "$VENV/bin/label-studio" start --no-browser --port 8080
