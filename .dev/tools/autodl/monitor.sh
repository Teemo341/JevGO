#!/usr/bin/env bash
set -euo pipefail
control="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$control/remote_monitor.sh" "$@"
