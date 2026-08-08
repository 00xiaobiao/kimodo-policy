#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_INPUT_ROOT="${REPO_ROOT}/isaaclab_twist2_g1/recording_data"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/lerobot/outputs/HumanoidArena_datasets_v3_1"

INPUT_ROOT="${DEFAULT_INPUT_ROOT}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE=0
USE_IMAGES=0
LIMIT=""
DRY_RUN=0
ALLOW_MISSING_VIDEO_FRAMES=0
SKIP_CONVERT=0
SKIP_INSTRUCTIONS=0
SKIP_GRIPPER_FIX=0
TASKS_CSV=""

usage() {
    cat <<EOF
Usage: bash ${0} [options]

Batch convert first-level HOI_*/HSI_* IsaacLab recordings into refpose v3.1 LeRobot datasets.

Options:
  --input-root PATH              Source recording root. Default: ${DEFAULT_INPUT_ROOT}
  --output-root PATH             Output dataset root. Default: ${DEFAULT_OUTPUT_ROOT}
  --python PATH                  Python executable for converters/postprocess. Default: \$PYTHON_BIN or python
  --tasks A,B                    Only convert selected task dirs or canonical task names.
  --overwrite                    Replace existing output dataset directories.
  --use-images                   Store images in parquet instead of LeRobot videos.
  --limit N                      Pass --limit N to each converter.
  --allow-missing-video-frames   Pass through to converters.
  --skip-convert                 Only run postprocess steps.
  --skip-instructions            Do not write language instructions.
  --skip-gripper-fix             Do not patch hand-binary quantile stats.
  --dry-run                      Print commands without executing.
  -h, --help                     Show this help.

Default backend directories under each task:
  sonic, twist2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root)
            INPUT_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --tasks)
            TASKS_CSV="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --use-images)
            USE_IMAGES=1
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --allow-missing-video-frames)
            ALLOW_MISSING_VIDEO_FRAMES=1
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=1
            shift
            ;;
        --skip-instructions)
            SKIP_INSTRUCTIONS=1
            shift
            ;;
        --skip-gripper-fix)
            SKIP_GRIPPER_FIX=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

INPUT_ROOT="$(cd "${INPUT_ROOT}" && pwd)"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

SONIC_CONVERTER="${REPO_ROOT}/isaaclab_twist2_g1/tools/data_tools/sonic2lerobot_rotlocal_v3.py"
TWIST2_CONVERTER="${REPO_ROOT}/isaaclab_twist2_g1/tools/data_tools/twist2lerobot_rotlocal_v3.py"
GRIPPER_FIX_SCRIPT="${REPO_ROOT}/lerobot/src/lerobot/scripts/fix_gripper_quantile_stats.py"
INSTRUCTION_SCRIPT="${REPO_ROOT}/lerobot/scripts/write_dataset_instructions.py"

PYTHONPATH_VALUE="${REPO_ROOT}/lerobot/src:${REPO_ROOT}/isaaclab_twist2_g1:${REPO_ROOT}/isaaclab_twist2_g1/tools/data_tools"

declare -A INSTRUCTIONS=(
    ["HOI_double_desk"]="Put the hammer from the right table into the basket on the left table."
    ["HOI_football"]="Kick the football into the goal."
    ["HOI_pp_box"]="Move the box from the table onto the shelf."
    ["HSI_vision_navi"]="Avoid obstacles and move to the yellow marked area."
    ["HSI_open_door"]="Open the door."
    ["HSI_sit_sofa"]="Sit on the sofa."
    ["HOI_grap_cup"]="Place the orange drink can on the table into the basket on the tea table."
    ["HSI_boxing"]="Strike the green markers on the punching bag."
)

BACKEND_DIRS=(
    sonic
    twist2
)

declare -A TASK_FILTER=()
declare -A TOUCHED_TASKS=()

canonical_task_name() {
    local raw_name="$1"
    case "${raw_name}" in
        HOI_football_v2)
            echo "HOI_football"
            ;;
        HOI_grapcup)
            echo "HOI_grap_cup"
            ;;
        *)
            echo "${raw_name}"
            ;;
    esac
}

if [[ -n "${TASKS_CSV}" ]]; then
    IFS=',' read -r -a TASK_LIST <<< "${TASKS_CSV}"
    for task_name in "${TASK_LIST[@]}"; do
        task_name="${task_name//[[:space:]]/}"
        [[ -n "${task_name}" ]] || continue
        TASK_FILTER["${task_name}"]=1
        TASK_FILTER["$(canonical_task_name "${task_name}")"]=1
    done
fi

run_cmd() {
    echo "+ $*"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        PYTHONPATH="${PYTHONPATH_VALUE}${PYTHONPATH:+:${PYTHONPATH}}" "$@"
    fi
}

backend_kind() {
    local backend_dir="$1"
    case "${backend_dir}" in
        twist2)
            echo "twist2"
            ;;
        sonic)
            echo "sonic"
            ;;
        *)
            echo ""
            ;;
    esac
}

has_npz_files() {
    local dir="$1"
    [[ -n "$(find "${dir}" -type f -name '*.npz' -print -quit)" ]]
}

task_selected() {
    local raw_task="$1"
    local canonical_task="$2"
    if [[ "${#TASK_FILTER[@]}" -eq 0 ]]; then
        return 0
    fi
    [[ -n "${TASK_FILTER[${raw_task}]:-}" || -n "${TASK_FILTER[${canonical_task}]:-}" ]]
}

convert_backend() {
    local task_dir="$1"
    local backend_dir="$2"
    local raw_task
    local canonical_task
    local kind
    local src_dir
    local dataset_name
    local output_dir
    local repo_id
    local converter

    raw_task="$(basename "${task_dir}")"
    canonical_task="$(canonical_task_name "${raw_task}")"
    kind="$(backend_kind "${backend_dir}")"
    src_dir="${task_dir}/${backend_dir}"
    dataset_name="${backend_dir}_refpose_v3_1"
    output_dir="${OUTPUT_ROOT}/${canonical_task}/${dataset_name}"
    repo_id="local/${canonical_task}_${dataset_name}"

    if [[ -z "${kind}" || ! -d "${src_dir}" ]]; then
        return
    fi
    if ! has_npz_files "${src_dir}"; then
        echo "SKIP empty backend: ${src_dir}"
        return
    fi

    if [[ "${kind}" == "sonic" ]]; then
        converter="${SONIC_CONVERTER}"
    else
        converter="${TWIST2_CONVERTER}"
    fi

    local cmd=(
        "${PYTHON_BIN}"
        "${converter}"
        --input-root "${src_dir}"
        --output-root "${output_dir}"
        --repo-id "${repo_id}"
    )
    if [[ "${OVERWRITE}" -eq 1 ]]; then
        cmd+=(--overwrite)
    fi
    if [[ "${USE_IMAGES}" -eq 1 ]]; then
        cmd+=(--use-images)
    fi
    if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
    fi
    if [[ "${ALLOW_MISSING_VIDEO_FRAMES}" -eq 1 ]]; then
        cmd+=(--allow-missing-video-frames)
    fi

    run_cmd "${cmd[@]}"
    TOUCHED_TASKS["${canonical_task}"]=1
}

if [[ "${SKIP_CONVERT}" -eq 0 ]]; then
    for task_dir in "${INPUT_ROOT}"/HOI_* "${INPUT_ROOT}"/HSI_*; do
        [[ -d "${task_dir}" ]] || continue
        raw_task="$(basename "${task_dir}")"
        canonical_task="$(canonical_task_name "${raw_task}")"
        if ! task_selected "${raw_task}" "${canonical_task}"; then
            continue
        fi
        echo "== ${raw_task} -> ${canonical_task} =="
        for backend_dir in "${BACKEND_DIRS[@]}"; do
            convert_backend "${task_dir}" "${backend_dir}"
        done
    done
else
    for task_root in "${OUTPUT_ROOT}"/HOI_* "${OUTPUT_ROOT}"/HSI_*; do
        [[ -d "${task_root}" ]] || continue
        raw_task="$(basename "${task_root}")"
        canonical_task="$(canonical_task_name "${raw_task}")"
        if task_selected "${raw_task}" "${canonical_task}"; then
            TOUCHED_TASKS["${canonical_task}"]=1
        fi
    done
fi

if [[ "${SKIP_INSTRUCTIONS}" -eq 0 ]]; then
    for canonical_task in "${!TOUCHED_TASKS[@]}"; do
        task_root="${OUTPUT_ROOT}/${canonical_task}"
        instruction="${INSTRUCTIONS[${canonical_task}]:-}"
        if [[ -z "${instruction}" ]]; then
            echo "WARN no instruction configured for ${canonical_task}; skip ${task_root}"
            continue
        fi
        if [[ "${DRY_RUN}" -eq 0 && ! -d "${task_root}" ]]; then
            echo "WARN task output missing: ${task_root}"
            continue
        fi
        run_cmd "${PYTHON_BIN}" "${INSTRUCTION_SCRIPT}" "${task_root}" "${instruction}"
    done
fi

if [[ "${SKIP_GRIPPER_FIX}" -eq 0 ]]; then
    run_cmd "${PYTHON_BIN}" "${GRIPPER_FIX_SCRIPT}" --root "${OUTPUT_ROOT}"
fi

echo "Done. Output root: ${OUTPUT_ROOT}"
