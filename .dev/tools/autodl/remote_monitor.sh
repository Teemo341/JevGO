#!/usr/bin/env bash
# Run only after SSH has entered the AutoDL instance.
set -euo pipefail
case "$(hostname)" in
    autodl-container-*) ;;
    *) echo 'monitor must run on AutoDL; refusing the local machine.' >&2; exit 2 ;;
esac
control="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec /root/JevGO/.venv/bin/python -B "$control/remote_lifecycle.py" monitor "$@"
