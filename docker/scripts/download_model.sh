#!/bin/bash
set -e

# Find project root by looking for api directory
find_project_root() {
    local current_dir="$PWD"
    local max_steps=5
    local steps=0

    while [ $steps -lt $max_steps ]; do
        if [ -d "$current_dir/api" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
        ((steps++))
    done

    echo "Error: Could not find project root (no api directory found)" >&2
    exit 1
}

resolve_project_path() {
    local raw_path="$1"
    local project_root="$2"

    case "$raw_path" in
        /*) echo "$raw_path" ;;
        src/*) echo "$project_root/api/$raw_path" ;;
        *) echo "$project_root/$raw_path" ;;
    esac
}

PROJECT_ROOT=$(find_project_root)
KOKORO_FILE="${KOKORO_V1_FILE:-v1_1_zh/kokoro-v1_1-zh.pth}"
MODEL_SUBDIR="$(dirname "$KOKORO_FILE")"
MODEL_ROOT=$(resolve_project_path "${MODEL_DIR:-api/src/models}" "$PROJECT_ROOT")

# 根据 KOKORO_V1_FILE 推导模型子目录，避免脚本硬编码具体版本。
if [ "$MODEL_SUBDIR" = "." ]; then
    MODEL_DIR_RESOLVED="$MODEL_ROOT"
    DEFAULT_VOICES_DIR="$PROJECT_ROOT/api/src/voices"
else
    MODEL_DIR_RESOLVED="$MODEL_ROOT/$MODEL_SUBDIR"
    DEFAULT_VOICES_DIR="$PROJECT_ROOT/api/src/voices/$MODEL_SUBDIR"
fi

VOICES_DIR_RESOLVED=$(resolve_project_path "${VOICES_DIR:-$DEFAULT_VOICES_DIR}" "$PROJECT_ROOT")

echo "Model directory: $MODEL_DIR_RESOLVED"
echo "Voices directory: $VOICES_DIR_RESOLVED"

python "$PROJECT_ROOT/docker/scripts/download_model.py" \
    --output "$MODEL_DIR_RESOLVED" \
    --voices-output "$VOICES_DIR_RESOLVED"
