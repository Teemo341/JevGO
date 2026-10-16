#!/usr/bin/env bash
# AutoDL-only runtime environment. Source this before running JevGO.
export JEVGO_ROOT="${JEVGO_ROOT:-/root/JevGO}"
export JEVGO_STORAGE="${JEVGO_STORAGE:-/root/autodl-tmp/jevgo}"
export JEVGO_SHARED_MODELS="${JEVGO_SHARED_MODELS:-/root/autodl-fs/JevGO/models}"
export UV_CACHE_DIR="$JEVGO_STORAGE/install/uv-cache"
export UV_PYTHON_INSTALL_DIR="$JEVGO_ROOT/.tools/python"
export UV_PYTHON_BIN_DIR="$JEVGO_ROOT/.tools/bin"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
export UV_NO_CACHE=1
export UV_NO_SYNC=1
export TMPDIR="$JEVGO_STORAGE/tmp"
export XDG_CACHE_HOME="$JEVGO_STORAGE/cache"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export DG_JIT_CACHE_DIR="$XDG_CACHE_HOME/deep_gemm"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$JEVGO_STORAGE/install/pip-cache"
export ALFWORLD_DATA="$JEVGO_STORAGE/data/alfworld"
export NLTK_DATA="$JEVGO_STORAGE/data/nltk"
export JAVA_HOME="$JEVGO_ROOT/.tools/java"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export PATH="$JEVGO_ROOT/.venv/bin:$JAVA_HOME/bin:$JEVGO_ROOT/.tools/uv:$PATH"
# Prefer the validated image toolchain over an older base-image CUDA symlink.
if [[ -x /usr/local/cuda-12.9/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.9
    export CUDA_PATH="$CUDA_HOME"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
# Refuse a missing data disk instead of silently filling the system disk.
if ! mountpoint -q /root/autodl-tmp; then
    echo 'AutoDL fast storage is not mounted at /root/autodl-tmp.' >&2
    return 2
fi
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
    "$FLASHINFER_WORKSPACE_BASE" "$JEVGO_STORAGE/runs" \
    "$JEVGO_STORAGE/install/wheels" "$JEVGO_STORAGE/cache/offline"
if [[ -L "$JEVGO_ROOT/.cache" ]]; then
    [[ "$(readlink -m "$JEVGO_ROOT/.cache")" == "$XDG_CACHE_HOME" ]] || return 2
elif [[ -e "$JEVGO_ROOT/.cache" ]]; then
    echo 'Refusing to replace an existing .cache directory.' >&2
    return 2
else
    ln -s "$XDG_CACHE_HOME" "$JEVGO_ROOT/.cache"
fi
if [[ ! -e "$XDG_CACHE_HOME/offline/wheels" && ! -L "$XDG_CACHE_HOME/offline/wheels" ]]; then
    ln -s "$JEVGO_STORAGE/install/wheels" "$XDG_CACHE_HOME/offline/wheels"
fi
