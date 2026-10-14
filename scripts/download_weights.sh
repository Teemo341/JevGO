#!/usr/bin/env bash
# Download or verify the portable, pinned model files in this project.
set -euo pipefail
JEVGO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python_bin="$JEVGO_ROOT/.tools/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
if [[ -x "$JEVGO_ROOT/.venv/bin/python" && -f "$JEVGO_ROOT/.venv/.jevgo-location" ]] &&
   [[ "$(cat "$JEVGO_ROOT/.venv/.jevgo-location")" == "$JEVGO_ROOT" ]]; then
    python_bin="$JEVGO_ROOT/.venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
    echo 'Missing bundled Python: copy the .tools directory with this project.' >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" -B "$JEVGO_ROOT/scripts/model_weights.py" "$@"
