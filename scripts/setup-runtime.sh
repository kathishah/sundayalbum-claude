#!/bin/bash
# setup-runtime.sh — Bootstrap the Sunday Album production Python venv.
#
# Called by the app on first launch via /bin/bash setup-runtime.sh <venv_dir> <requirements_file>
# Both arguments are absolute paths supplied by RuntimeManager.swift.
#
# Exit 0 on success; non-zero on any failure (the caller will delete the partial venv).

set -euo pipefail

VENV_DIR="$1"
REQUIREMENTS="$2"

echo "==> Creating Python virtual environment at: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo "==> Installing Sunday Album dependencies (~200 MB, please wait)..."
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"

echo "==> Setup complete."
