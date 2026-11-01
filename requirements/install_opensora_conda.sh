#! /bin/bash
# Conda counterpart of:
#   bash requirements/install.sh embodied --model openvla-oft --env opensora
# for NVIDIA GPUs. Packages are still installed with `uv pip` (into the conda
# env) so the torch overrides and wheel resolution match the uv venv setup.
#
# Usage:
#   bash requirements/install_opensora_conda.sh [--name ENV] [--use-mirror] [--no-root]
#       [--no-flash-attn] [--no-apex]
#
# Env vars: OPENSORA_PATH (reuse an existing RLinf/opensora checkout),
# UV_TORCH_BACKEND (default auto: pick the CUDA wheel the driver supports).

set -eo pipefail

CONDA_ENV_NAME="rlinf-opensora"
PYTHON_VERSION="3.11"
USE_MIRRORS=0
NO_ROOT=0
DISABLE_FLASH_ATTN=0
DISABLE_APEX=0
GITHUB_PREFIX=""
# Same as RAY_COMPAT_DEPS in install.sh for NVIDIA.
RAY_COMPAT_DEPS=("protobuf>=6.33.5,<7")
SAPIEN_SPEC="sapien==3.0.1; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version < '3.14'"

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

while [ $# -gt 0 ]; do
    case "$1" in
        --name) CONDA_ENV_NAME="$2"; shift 2 ;;
        --use-mirror) USE_MIRRORS=1; shift ;;
        --no-root) NO_ROOT=1; shift ;;
        --no-flash-attn) DISABLE_FLASH_ATTN=1; shift ;;
        --no-apex) DISABLE_APEX=1; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[install_opensora_conda] $*"; }

retry_cmd() {
    local max=5 delay=15 attempt=1
    until "$@"; do
        if [ "$attempt" -ge "$max" ]; then
            echo "'$*' failed after ${max} attempts" >&2
            return 1
        fi
        log "'$*' failed (attempt ${attempt}/${max}); retrying in ${delay}s"
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

py_tag() { python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")'; }
torch_mm() { python -c 'import torch; print(".".join(torch.__version__.split("+")[0].split(".")[:2]))'; }

#=======================Conda env + uv=======================
setup_mirror() {
    [ "$USE_MIRRORS" -eq 1 ] || return 0
    export GITHUB_PREFIX="${GITHUB_PREFIX:-https://gh-proxy.com/}"
    export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
    export HF_ENDPOINT=https://hf-mirror.com
    local idx="${GIT_CONFIG_COUNT:-0}"
    export "GIT_CONFIG_KEY_${idx}=url.${GITHUB_PREFIX}github.com/.insteadOf"
    export "GIT_CONFIG_VALUE_${idx}=https://github.com/"
    export GIT_CONFIG_COUNT=$((idx + 1))
}

create_conda_env() {
    command -v conda >/dev/null || { echo "conda not found on PATH." >&2; exit 1; }
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        log "Reusing existing conda env '$CONDA_ENV_NAME'."
    else
        conda create -y -n "$CONDA_ENV_NAME" "python=${PYTHON_VERSION}" pip
    fi
    conda activate "$CONDA_ENV_NAME"
    local active_mm
    active_mm=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$active_mm" != "$PYTHON_VERSION" ]; then
        echo "Conda env '$CONDA_ENV_NAME' has Python ${active_mm}, expected ${PYTHON_VERSION}. Remove it or pass --name." >&2
        exit 1
    fi
    command -v uv >/dev/null || pip install uv
    # Point every `uv pip` call at this conda env.
    export UV_PYTHON="$CONDA_PREFIX/bin/python"
    export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
    ACTIVATE_HOOK="$CONDA_PREFIX/etc/conda/activate.d/rlinf_opensora.sh"
    mkdir -p "$(dirname "$ACTIVATE_HOOK")"
    : > "$ACTIVATE_HOOK"
}

# Replaces `uv sync --extra embodied`: installs pyproject.toml's core and
# embodied deps (not rlinf itself; run from the repo or set PYTHONPATH) under
# its [tool.uv] override-dependencies.
install_project_deps() {
    OVERRIDE_FILE="$CONDA_PREFIX/rlinf-overrides.txt"
    python - "$REPO_ROOT/pyproject.toml" "${RAY_COMPAT_DEPS[@]}" > "$OVERRIDE_FILE" <<'EOF'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    overrides = tomllib.load(f)["tool"]["uv"]["override-dependencies"]
overrides = [o for o in overrides if not o.replace(" ", "").startswith("protobuf")]
print("\n".join(sys.argv[2:] + overrides))
EOF
    local torch_pins
    torch_pins=$(grep -E '^(torch|torchvision|torchaudio)==' "$OVERRIDE_FILE")
    # Install torch first so UV_TORCH_BACKEND picks the driver-compatible CUDA wheel.
    # shellcheck disable=SC2086
    uv pip install $torch_pins
    (cd "$REPO_ROOT" && uv pip install -r pyproject.toml --extra embodied --override "$OVERRIDE_FILE")
}

install_common_embodied_deps() {
    install_project_deps
    uv pip install -r "$SCRIPT_DIR/embodied/envs/common.txt"
    if [ "$NO_ROOT" -eq 0 ]; then
        bash "$SCRIPT_DIR/sys_deps.sh" nvidia
    fi
    cat >> "$ACTIVATE_HOOK" <<'EOF'
export NVIDIA_DRIVER_CAPABILITIES=all
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
EOF
}

#=======================maniskill_libero env=======================
install_libero_env() {
    uv pip install rlinf-libero
    # Copy package files out of uv's cache so LIBERO can write its assets and config.
    local ver
    ver=$(uv pip show rlinf-libero 2>/dev/null | sed -n 's/^Version: //p')
    [ -z "$ver" ] || UV_LINK_MODE=copy uv pip install --force-reinstall --no-deps "rlinf-libero==${ver}"
    retry_cmd libero-download-assets --skip-existing
    python - <<'EOF'
import importlib

module = importlib.import_module("libero.libero")
module.set_libero_default_path()
EOF
}

install_maniskill_libero_env() {
    install_libero_env
    retry_cmd uv pip install "git+${GITHUB_PREFIX}https://github.com/haosulab/ManiSkill.git@v3.0.0b22" "$SAPIEN_SPEC"
    GITHUB_PREFIX="$GITHUB_PREFIX" USE_MIRRORS="$USE_MIRRORS" \
        bash "$SCRIPT_DIR/embodied/download_assets.sh" --assets maniskill
}

#=======================OpenSora world model=======================
install_tensornvme() {
    local url="git+${GITHUB_PREFIX}https://github.com/fangqi-Zhu/TensorNVMe.git"
    uv pip install "$url" --no-build-isolation
    local tnvme_env=(env "LD_LIBRARY_PATH=$HOME/.tensornvme/lib:${LD_LIBRARY_PATH:-}")
    if ! "${tnvme_env[@]}" python -c "import tensornvme._C" >/dev/null 2>&1; then
        log "tensornvme does not load against this torch; rebuilding."
        uv pip install "$url" --no-build-isolation --reinstall-package tensornvme --no-cache
        "${tnvme_env[@]}" python -c "import tensornvme._C" >/dev/null 2>&1 \
            || log "WARNING: tensornvme still does not import; expected without a GPU, otherwise check the torch ABI."
    fi
}

install_apex() {
    if [ "$DISABLE_APEX" -eq 1 ]; then
        log "--no-apex given; skipping apex."
        return 0
    fi
    local base_url="${GITHUB_PREFIX}https://github.com/RLinf/apex/releases/download/25.09"
    local tag torch_tag torch_cu
    tag=$(py_tag)
    torch_tag="torch$(torch_mm)"
    torch_cu=$(python -c 'import torch; v=(torch.version.cuda or "").split("."); print("cu"+v[0] if v[0] else "")')
    uv pip uninstall apex || true
    export NUM_THREADS=$(nproc)
    export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-"--threads ${NUM_THREADS}"}
    export APEX_PARALLEL_BUILD=${APEX_PARALLEL_BUILD:-${NUM_THREADS}}
    if [ -n "$torch_cu" ] && uv pip install "${base_url}/apex-0.1+${torch_cu}${torch_tag}-${tag}-${tag}-linux_x86_64.whl"; then
        :
    elif uv pip install "${base_url}/apex-0.1+${torch_tag}-${tag}-${tag}-linux_x86_64.whl"; then
        :
    else
        log "apex wheel unavailable; building from source."
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install "git+${GITHUB_PREFIX}https://github.com/RLinf/apex.git" --no-build-isolation
    fi
}

install_opensora_world_model() {
    local opensora_dir="${OPENSORA_PATH:-$CONDA_PREFIX/opensora}"
    if [ ! -d "$opensora_dir" ]; then
        retry_cmd git clone "${GITHUB_PREFIX}https://github.com/RLinf/opensora.git" "$opensora_dir"
    fi
    uv pip install -e "$opensora_dir"
    # xformers embeds the torch version in its local label; 0.0.35 matches torch 2.11.
    uv pip install "xformers==0.0.35"
    uv pip install -r "$SCRIPT_DIR/embodied/models/opensora.txt"
    install_tensornvme
    echo "export LD_LIBRARY_PATH=\$HOME/.tensornvme/lib:\$LD_LIBRARY_PATH" >> "$ACTIVATE_HOOK"
    install_apex
}

install_flash_attn() {
    if [ "$DISABLE_FLASH_ATTN" -eq 1 ]; then
        log "--no-flash-attn given; skipping flash-attn."
        return 0
    fi
    local flash_ver="2.7.4.post1"
    if python -c 'import sys, torch; v=tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); sys.exit(v < (2, 8))'; then
        flash_ver="2.8.3"
    fi
    local tag cu_major cxx_abi wheel host
    tag=$(py_tag)
    cu_major=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')
    cxx_abi=$(python -c 'import torch; print("cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")')
    wheel="flash_attn-${flash_ver}+cu${cu_major}torch$(torch_mm)${cxx_abi}-${tag}-${tag}-linux_x86_64.whl"
    uv pip uninstall flash-attn || true
    for host in https://github.com/Dao-AILab/flash-attention https://github.com/RLinf/flash-attention; do
        if uv pip install "${GITHUB_PREFIX}${host}/releases/download/v${flash_ver}/${wheel}"; then
            return 0
        fi
    done
    log "No prebuilt flash-attn wheel ${wheel}; building from source."
    FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install "flash-attn==${flash_ver}" --no-build-isolation
}

#=======================Main=======================
setup_mirror
create_conda_env
install_common_embodied_deps
install_maniskill_libero_env
install_opensora_world_model
install_flash_attn
uv pip install "git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git"
# Env/model installs may have changed the protobuf runtime.
uv pip install "${RAY_COMPAT_DEPS[@]}"

log "Done. Activate with: conda activate ${CONDA_ENV_NAME}"
