#!/usr/bin/env bash

if [[ -n "${_HUMANOIDARENA_RUNTIME_PATHS_SH:-}" ]]; then
  return 0
fi
readonly _HUMANOIDARENA_RUNTIME_PATHS_SH=1

RUNTIME_PATHS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "${RUNTIME_PATHS_DIR}/../.." && pwd)}"
HUMANOIDARENA_ROOT="${HUMANOIDARENA_ROOT:-$(cd "${ISAACLAB_ROOT}/.." && pwd)}"
TWIST2_ROOT="${TWIST2_ROOT:-${HUMANOIDARENA_ROOT}/TWIST2}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${HUMANOIDARENA_ROOT}/lerobot}"
GROOT_ROOT="${GROOT_ROOT:-${HUMANOIDARENA_ROOT}/GR00T-WholeBodyControl}"
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT:-${GROOT_ROOT}/gear_sonic_deploy/policy/release}"

CONDA_BASE="${CONDA_BASE:-}"
ISAACLAB_CONDA_ENV_NAME="${ISAACLAB_CONDA_ENV_NAME:-unitree_sim_env}"
LEROBOT_CONDA_ENV_NAME="${LEROBOT_CONDA_ENV_NAME:-lerobot}"

conda_base_has_required_envs() {
  local base="$1"
  [[ -n "${base}" ]] || return 1
  [[ -x "${base}/envs/${ISAACLAB_CONDA_ENV_NAME}/bin/python" ]] || return 1
  [[ -x "${base}/envs/${LEROBOT_CONDA_ENV_NAME}/bin/python" ]] || return 1
}

detect_conda_base() {
  local candidate=""
  local -a candidates=()

  if [[ -n "${CONDA_EXE:-}" ]]; then
    candidate="$(cd "$(dirname "${CONDA_EXE}")/.." 2>/dev/null && pwd || true)"
    [[ -n "${candidate}" ]] && candidates+=("${candidate}")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidate="$(cd "${CONDA_PREFIX}/../.." 2>/dev/null && pwd || true)"
    [[ -n "${candidate}" ]] && candidates+=("${candidate}")
  fi

  candidates+=(
    "/ai/Yichi/0_Systems/miniconda3"
    "${HOME:-}/miniconda3"
    "/opt/conda"
    "/root/miniconda3"
  )

  local seen=""
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    [[ " ${seen} " != *" ${candidate} "* ]] || continue
    seen="${seen} ${candidate}"
    if conda_base_has_required_envs "${candidate}"; then
      printf "%s" "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${CONDA_BASE}" ]]; then
  CONDA_BASE="$(detect_conda_base || true)"
fi

if [[ -z "${ISAACLAB_PYTHON:-}" && -n "${CONDA_BASE}" ]]; then
  ISAACLAB_PYTHON="${CONDA_BASE}/envs/${ISAACLAB_CONDA_ENV_NAME}/bin/python"
fi
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-python}"

if [[ -z "${SERVER_PYTHON:-}" && -n "${CONDA_BASE}" ]]; then
  SERVER_PYTHON="${CONDA_BASE}/envs/${LEROBOT_CONDA_ENV_NAME}/bin/python"
fi
SERVER_PYTHON="${SERVER_PYTHON:-python}"

export ISAACLAB_ROOT HUMANOIDARENA_ROOT TWIST2_ROOT LEROBOT_ROOT GROOT_ROOT SONIC_POLICY_ROOT
export CONDA_BASE ISAACLAB_PYTHON SERVER_PYTHON
