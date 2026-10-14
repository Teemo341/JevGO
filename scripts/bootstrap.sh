#!/usr/bin/env bash
# Run after copying the project to a different absolute path.
set -euo pipefail
JEVGO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
bash "$JEVGO_ROOT/scripts/env.sh" JevGO --sync
bash "$JEVGO_ROOT/scripts/env.sh" JevGO --check
