#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HUMANOIDARENA_ROOT_DEFAULT="$(cd "${ISAACLAB_ROOT_DEFAULT}/.." && pwd)"

HUMANOIDARENA_ROOT="${HUMANOIDARENA_ROOT:-${HUMANOIDARENA_ROOT_DEFAULT}}"
CONDA_BASE="${CONDA_BASE:-}"
ISAACLAB_CONDA_ENV_NAME="${ISAACLAB_CONDA_ENV_NAME:-unitree_sim_env}"
LEROBOT_CONDA_ENV_NAME="${LEROBOT_CONDA_ENV_NAME:-lerobot}"
ISAACLAB_PYTHON_VERSION="${ISAACLAB_PYTHON_VERSION:-3.11}"
LEROBOT_PYTHON_VERSION="${LEROBOT_PYTHON_VERSION:-3.12}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.0.0}"
ISAACLAB_GIT_URL="${ISAACLAB_GIT_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_GIT_REF="${ISAACLAB_GIT_REF:-release/2.2.0}"
UNITREE_SDK_GIT_URL="${UNITREE_SDK_GIT_URL:-https://github.com/unitreerobotics/unitree_sdk2_python}"
GMR_GIT_URL="${GMR_GIT_URL:-https://github.com/YanjieZe/GMR.git}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-${HUMANOIDARENA_ROOT}/external}"
GMR_DIR="${GMR_DIR:-${HUMANOIDARENA_ROOT}/GMR}"
SONIC_POLICY_REPO="${SONIC_POLICY_REPO:-nvidia/GEAR-SONIC}"
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT:-${HUMANOIDARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release}"
LEROBOT_EXTRA="${LEROBOT_EXTRA:-pi}"
ISAACSIM_CONSTRAINT_FILE="${ISAACSIM_CONSTRAINT_FILE:-${EXTERNAL_ROOT}/humanoidarena_isaacsim5_constraints.txt}"
DRY_RUN=1
INSTALL_SYSTEM_PACKAGES=0

usage() {
  cat <<'EOF'
Usage: bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh [options]

Default mode is dry-run: commands are printed but not executed.

Options:
  --execute                    Actually run installation commands.
  --dry-run                    Print commands only. This is the default.
  --conda-base PATH            Conda installation root. Auto-detected if omitted.
  --external-root PATH         Where IsaacLab and unitree_sdk2_python are cloned.
  --gmr-dir PATH               Where GMR is cloned. Default: $HUMANOIDARENA_ROOT/GMR.
  --sonic-policy-root PATH     Where GEAR-SONIC release files are downloaded.
  --isaac-env NAME             Isaac Sim/Isaac Lab conda env name. Default: unitree_sim_env.
  --lerobot-env NAME           LeRobot conda env name. Default: lerobot.
  --lerobot-extra NAME         LeRobot extra installed as -e ".[NAME]". Default: pi.
  --install-system-packages    Also run apt-get install for cmake/build-essential/git/git-lfs/cyclonedds-dev.
  -h, --help                   Show this help.

Useful environment overrides:
  PYTORCH_INDEX_URL            Default: https://download.pytorch.org/whl/cu128
  ISAACSIM_VERSION             Default: 5.0.0
  ISAACLAB_GIT_REF             Default: release/2.2.0
  GMR_GIT_URL                  Default: https://github.com/YanjieZe/GMR.git
  SONIC_POLICY_REPO            Default: nvidia/GEAR-SONIC
  SONIC_POLICY_ROOT            Default: $HUMANOIDARENA_ROOT/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
  ISAACSIM_CONSTRAINT_FILE     Default: $EXTERNAL_ROOT/humanoidarena_isaacsim5_constraints.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      DRY_RUN=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --conda-base)
      CONDA_BASE="$2"
      shift 2
      ;;
    --external-root)
      EXTERNAL_ROOT="$2"
      shift 2
      ;;
    --gmr-dir)
      GMR_DIR="$2"
      shift 2
      ;;
    --sonic-policy-root)
      SONIC_POLICY_ROOT="$2"
      shift 2
      ;;
    --isaac-env)
      ISAACLAB_CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --lerobot-env)
      LEROBOT_CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --lerobot-extra)
      LEROBOT_EXTRA="$2"
      shift 2
      ;;
    --install-system-packages)
      INSTALL_SYSTEM_PACKAGES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

detect_conda_base() {
  local candidate
  local -a candidates=()
  if [[ -n "${CONDA_EXE:-}" ]]; then
    candidate="$(cd "$(dirname "${CONDA_EXE}")/.." 2>/dev/null && pwd || true)"
    [[ -n "${candidate}" ]] && candidates+=("${candidate}")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidate="$(cd "${CONDA_PREFIX}/../.." 2>/dev/null && pwd || true)"
    [[ -n "${candidate}" ]] && candidates+=("${candidate}")
  fi
  candidates+=("${HOME:-}/miniconda3" "/opt/conda" "/root/miniconda3" "/ai/Yichi/0_Systems/miniconda3")

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      printf "%s" "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${CONDA_BASE}" ]]; then
  CONDA_BASE="$(detect_conda_base || true)"
fi

if [[ -z "${CONDA_BASE}" ]]; then
  echo "Could not detect CONDA_BASE. Pass --conda-base /path/to/miniconda3." >&2
  exit 1
fi

CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ "${DRY_RUN}" == "0" && ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

run_cmd() {
  echo "+ $(quote_cmd "$@")"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

run_shell() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    bash -lc "$*"
  fi
}

run_in_conda() {
  local env_name="$1"
  shift
  local quoted=""
  local arg
  for arg in "$@"; do
    quoted+=" $(printf "%q" "${arg}")"
  done
  run_shell "source $(printf "%q" "${CONDA_SH}") && conda activate $(printf "%q" "${env_name}") &&${quoted}"
}

run_in_conda_shell() {
  local env_name="$1"
  local command="$2"
  run_shell "source $(printf "%q" "${CONDA_SH}") && conda activate $(printf "%q" "${env_name}") && ${command}"
}

write_isaacsim_constraints() {
  echo "+ write ${ISAACSIM_CONSTRAINT_FILE}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "$(dirname "${ISAACSIM_CONSTRAINT_FILE}")"
    cat > "${ISAACSIM_CONSTRAINT_FILE}" <<'EOF'
# Isaac Sim 5.0.0 pins. Keep these constraints active while installing
# IsaacLab/project packages so transitive deps do not upgrade the runtime stack.
torch==2.7.0
torchvision==0.22.0
torchaudio==2.7.0
numpy==1.26.0
packaging==23.0
click==8.1.7
typing_extensions==4.12.2
psutil==5.9.8
sentry-sdk==1.43.0
Pillow==11.2.1
filelock==3.13.1
fsspec==2024.6.1
MarkupSafe==2.1.3
networkx==3.3
PyYAML==6.0.2
requests==2.32.3
charset-normalizer==3.3.2
idna==3.10
sympy==1.13.3
wheel==0.45.1
transformers<5
huggingface-hub<1
EOF
  fi
}

run_in_conda_constrained_shell() {
  local env_name="$1"
  local command="$2"
  run_in_conda_shell "${env_name}" "export PIP_CONFIG_FILE=/dev/null; export PIP_CONSTRAINT=$(printf "%q" "${ISAACSIM_CONSTRAINT_FILE}"); ${command}"
}

repair_isaacsim_runtime_pins() {
  run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "python -m pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url $(printf "%q" "${PYTORCH_INDEX_URL}")"
  run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "python -m pip install --force-reinstall numpy==1.26.0 packaging==23.0 click==8.1.7 typing_extensions==4.12.2 psutil==5.9.8 sentry-sdk==1.43.0 Pillow==11.2.1 filelock==3.13.1 fsspec==2024.6.1 MarkupSafe==2.1.3 networkx==3.3 PyYAML==6.0.2 requests==2.32.3 charset-normalizer==3.3.2 idna==3.10 sympy==1.13.3 wheel==0.45.1 'transformers<5' 'huggingface-hub<1'"
  run_in_conda_shell "${ISAACLAB_CONDA_ENV_NAME}" "python -m pip uninstall -y typer || true"
}

ensure_conda_env() {
  local env_name="$1"
  local python_version="$2"
  run_shell "source $(printf "%q" "${CONDA_SH}") && conda env list | awk '{print \$1}' | grep -qx $(printf "%q" "${env_name}") || conda create -n $(printf "%q" "${env_name}") python=$(printf "%q" "${python_version}") -y"
}

echo "[setup] mode=$([[ "${DRY_RUN}" == "1" ]] && echo dry-run || echo execute)"
echo "[setup] HUMANOIDARENA_ROOT=${HUMANOIDARENA_ROOT}"
echo "[setup] CONDA_BASE=${CONDA_BASE}"
echo "[setup] EXTERNAL_ROOT=${EXTERNAL_ROOT}"
echo "[setup] GMR_DIR=${GMR_DIR}"
echo "[setup] SONIC_POLICY_ROOT=${SONIC_POLICY_ROOT}"
echo "[setup] ISAACLAB_CONDA_ENV_NAME=${ISAACLAB_CONDA_ENV_NAME}"
echo "[setup] LEROBOT_CONDA_ENV_NAME=${LEROBOT_CONDA_ENV_NAME}"

if [[ "${INSTALL_SYSTEM_PACKAGES}" == "1" ]]; then
  run_cmd sudo apt-get update
  run_cmd sudo apt-get install -y cmake build-essential git git-lfs cyclonedds-dev
else
  echo "[setup] skip system packages; pass --install-system-packages to install cmake/build-essential/git/git-lfs/cyclonedds-dev"
fi

run_cmd mkdir -p "${EXTERNAL_ROOT}"
write_isaacsim_constraints

ensure_conda_env "${ISAACLAB_CONDA_ENV_NAME}" "${ISAACLAB_PYTHON_VERSION}"
run_in_conda "${ISAACLAB_CONDA_ENV_NAME}" python -m pip install --upgrade pip
run_in_conda_shell "${ISAACLAB_CONDA_ENV_NAME}" "PIP_CONFIG_FILE=/dev/null python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url $(printf "%q" "${PYTORCH_INDEX_URL}")"
run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "python -m pip install 'isaacsim[all,extscache]==${ISAACSIM_VERSION}' --extra-index-url https://pypi.nvidia.com"
run_shell "source $(printf "%q" "${CONDA_SH}") && conda install -n $(printf "%q" "${ISAACLAB_CONDA_ENV_NAME}") -c conda-forge cyclonedds=0.10.5 cmake make pkg-config -y"

ISAACLAB_DIR="${EXTERNAL_ROOT}/IsaacLab"
if [[ -d "${ISAACLAB_DIR}/.git" ]]; then
  run_shell "cd $(printf "%q" "${ISAACLAB_DIR}") && git fetch --all --prune && git checkout $(printf "%q" "${ISAACLAB_GIT_REF}")"
else
  run_cmd git clone "${ISAACLAB_GIT_URL}" "${ISAACLAB_DIR}"
  run_shell "cd $(printf "%q" "${ISAACLAB_DIR}") && git checkout $(printf "%q" "${ISAACLAB_GIT_REF}")"
fi
run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "cd $(printf "%q" "${ISAACLAB_DIR}") && ./isaaclab.sh --install"

UNITREE_SDK_DIR="${EXTERNAL_ROOT}/unitree_sdk2_python"
if [[ -d "${UNITREE_SDK_DIR}/.git" ]]; then
  run_shell "cd $(printf "%q" "${UNITREE_SDK_DIR}") && git pull --ff-only"
else
  run_cmd git clone "${UNITREE_SDK_GIT_URL}" "${UNITREE_SDK_DIR}"
fi
run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "CMAKE_PREFIX_PATH=\"\${CONDA_PREFIX}:\${CMAKE_PREFIX_PATH:-}\" python -m pip install -e $(printf "%q" "${UNITREE_SDK_DIR}")"
run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "python -m pip install -r $(printf "%q" "${HUMANOIDARENA_ROOT}/isaaclab_twist2_g1/requirements.txt")"
repair_isaacsim_runtime_pins

sonic_policy_complete() {
  [[ -f "${SONIC_POLICY_ROOT}/model_encoder.onnx" ]] || return 1
  [[ -f "${SONIC_POLICY_ROOT}/model_decoder.onnx" ]] || return 1
  [[ -f "${SONIC_POLICY_ROOT}/observation_config.yaml" ]] || return 1
}

if [[ -d "${GMR_DIR}/.git" ]]; then
  run_shell "cd $(printf "%q" "${GMR_DIR}") && git pull --ff-only"
else
  run_cmd git clone "${GMR_GIT_URL}" "${GMR_DIR}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "+ if missing SONIC policy files, download ${SONIC_POLICY_REPO} to ${SONIC_POLICY_ROOT}"
  run_cmd mkdir -p "${SONIC_POLICY_ROOT}"
  run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "python -m pip install 'huggingface-hub<1'"
  run_in_conda_shell "${ISAACLAB_CONDA_ENV_NAME}" "hf download $(printf "%q" "${SONIC_POLICY_REPO}") model_encoder.onnx model_decoder.onnx observation_config.yaml --local-dir $(printf "%q" "${SONIC_POLICY_ROOT}")"
elif sonic_policy_complete; then
  echo "[setup] SONIC policy files already present: ${SONIC_POLICY_ROOT}"
else
  run_cmd mkdir -p "${SONIC_POLICY_ROOT}"
  run_in_conda_constrained_shell "${ISAACLAB_CONDA_ENV_NAME}" "python -m pip install 'huggingface-hub<1'"
  run_in_conda_shell "${ISAACLAB_CONDA_ENV_NAME}" "hf download $(printf "%q" "${SONIC_POLICY_REPO}") model_encoder.onnx model_decoder.onnx observation_config.yaml --local-dir $(printf "%q" "${SONIC_POLICY_ROOT}")"
fi
for sonic_file in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ test -f ${SONIC_POLICY_ROOT}/${sonic_file}"
  elif [[ ! -f "${SONIC_POLICY_ROOT}/${sonic_file}" ]]; then
    echo "Missing SONIC policy release file: ${SONIC_POLICY_ROOT}/${sonic_file}" >&2
    echo "The setup helper downloads these files from HuggingFace repo ${SONIC_POLICY_REPO}." >&2
    exit 1
  fi
done

ensure_conda_env "${LEROBOT_CONDA_ENV_NAME}" "${LEROBOT_PYTHON_VERSION}"
run_in_conda "${LEROBOT_CONDA_ENV_NAME}" python -m pip install --upgrade pip
if [[ -n "${LEROBOT_EXTRA}" ]]; then
  run_in_conda "${LEROBOT_CONDA_ENV_NAME}" python -m pip install -e "${HUMANOIDARENA_ROOT}/lerobot[${LEROBOT_EXTRA}]"
else
  run_in_conda "${LEROBOT_CONDA_ENV_NAME}" python -m pip install -e "${HUMANOIDARENA_ROOT}/lerobot"
fi

echo "[setup] finished command generation"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[setup] dry-run only; rerun with --execute to install"
fi
