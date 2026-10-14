#!/usr/bin/env bash
# One offline uv environment; old component names remain aliases.
set -euo pipefail
JEVGO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export UV_CACHE_DIR="$JEVGO_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$JEVGO_ROOT/.tools/python"
export UV_PYTHON_BIN_DIR="$JEVGO_ROOT/.tools/bin"
export UV_PYTHON_DOWNLOADS=never
export UV_OFFLINE="${UV_OFFLINE:-1}"
export UV_LINK_MODE=hardlink
export UV_LOCK_TIMEOUT=1800
export TMPDIR="$JEVGO_ROOT/.cache/tmp"
export XDG_CACHE_HOME="$JEVGO_ROOT/.cache"
export CUDA_CACHE_PATH="$JEVGO_ROOT/.cache/nvidia"
export TRITON_CACHE_DIR="$JEVGO_ROOT/.cache/triton"
export TORCH_HOME="$JEVGO_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$JEVGO_ROOT/.cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$JEVGO_ROOT/.cache/torchinductor"
export NUMBA_CACHE_DIR="$JEVGO_ROOT/.cache/numba"
export DG_JIT_CACHE_DIR="$JEVGO_ROOT/.cache/deep_gemm"
export FLASHINFER_WORKSPACE_BASE="$JEVGO_ROOT"
export TILELANG_CACHE_DIR="$JEVGO_ROOT/.cache/tilelang"
export HF_HOME="${HF_HOME:-$JEVGO_ROOT/.cache/huggingface}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$JEVGO_ROOT/data/alfworld}"
export MPLCONFIGDIR="$JEVGO_ROOT/.cache/matplotlib"
export NLTK_DATA="${NLTK_DATA:-$JEVGO_ROOT/data/nltk}"
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export JAVA_HOME="$JEVGO_ROOT/.tools/java"
export PATH="$JAVA_HOME/bin:$JEVGO_ROOT/.tools/uv:$PATH"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$MPLCONFIGDIR"
uv_bin="${JEVGO_UV:-$JEVGO_ROOT/.tools/uv/uv}"
python_bin="$JEVGO_ROOT/.tools/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
if [[ ! -x "$uv_bin" || ! -x "$python_bin" ]]; then
    echo 'Missing bundled uv/Python: copy the .tools directory with this project.' >&2
    exit 2
fi
name="${1:-}"
case "$name" in
    JevGO|jevgo|core|autotrust|semif|minigrid|alfworld|scienceworld|discoveryworld) shift ;;
    *) echo 'Usage: bash scripts/env.sh {JevGO|jevgo|core|autotrust|semif|minigrid|alfworld|scienceworld|discoveryworld} {--sync|--check|--uv UV_ARGS...|command [args...]}' >&2; exit 2 ;;
esac
cd "$JEVGO_ROOT"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV

sync_environment() {
    "$python_bin" -B "$JEVGO_ROOT/scripts/verify_offline.py"
    # Copied venvs retain absolute paths. Rebuild this generated directory.
    if [[ -e .venv || -L .venv ]] && [[ "$(cat .venv/.jevgo-location 2>/dev/null || true)" != "$JEVGO_ROOT" ]]; then
        "$python_bin" -B - <<'PY'
from pathlib import Path
import shutil
import uuid
p = Path('.venv')
if p.is_symlink() or not (p / 'pyvenv.cfg').is_file():
    raise SystemExit('Refusing to replace .venv: not a generated local environment')
if (p / '.jevgo-location').is_file():
    shutil.rmtree(p)
else:
    backup = Path('.cache/venv-backups') / uuid.uuid4().hex
    backup.parent.mkdir(parents=True, exist_ok=True)
    p.rename(backup)
    print(f'Preserved unmarked .venv at {backup}', flush=True)
PY
    fi
    # Disposable extraction cache; the durable cache is the wheelhouse.
    "$uv_bin" sync --project "$JEVGO_ROOT" --locked --offline --no-cache --python "$python_bin"
    printf '%s\n' "$JEVGO_ROOT" > .venv/.jevgo-location
    "$uv_bin" pip check --python .venv/bin/python
}

case "${1:-}" in
    --sync) sync_environment ;;
    --uv)
        shift
        exec "$uv_bin" --project "$JEVGO_ROOT" "$@"
        ;;
    --check)
        if [[ "$(cat .venv/.jevgo-location 2>/dev/null || true)" != "$JEVGO_ROOT" ]]; then
            echo 'Environment missing or moved: run bash scripts/bootstrap.sh first.' >&2
            exit 2
        fi
        "$python_bin" -B scripts/verify_offline.py --quick
        "$uv_bin" lock --project "$JEVGO_ROOT" --check --offline --python "$python_bin"
        "$uv_bin" sync --project "$JEVGO_ROOT" --locked --check --offline --python "$python_bin"
        exec "$uv_bin" pip check --python .venv/bin/python
        ;;
    '') echo 'Supply --sync, --check, --uv UV_ARGS..., or a command.' >&2; exit 2 ;;
    *)
        if [[ ! -x .venv/bin/python || "$(cat .venv/.jevgo-location 2>/dev/null || true)" != "$JEVGO_ROOT" ]]; then
            echo 'Creating or relocating JevGO from local offline files...' >&2
            sync_environment
        fi
        # ALFWorld may be used under any alias. Keep dlopen/unlink scratch off NFS.
        runtime_tmp="$(mktemp -d /tmp/jevgo-runtime.XXXXXXXX)"
        trap 'rm -rf -- "$runtime_tmp"' EXIT
        "$uv_bin" run --project "$JEVGO_ROOT" --locked --no-sync --offline env TMPDIR="$runtime_tmp" "$@"
        ;;
esac
