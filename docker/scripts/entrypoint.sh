#!/bin/bash
set -e

if [ "$DOWNLOAD_MODEL" = "true" ]; then
    # 根据环境变量推导下载目录，支持 v1.1-zh 与 v1.0 等模型版本切换。
    MODEL_ROOT="${MODEL_DIR:-api/src/models}"
    KOKORO_FILE="${KOKORO_V1_FILE:-v1_1_zh/kokoro-v1_1-zh.pth}"
    MODEL_SUBDIR="$(dirname "$KOKORO_FILE")"

    if [ "$MODEL_SUBDIR" = "." ]; then
        MODEL_OUTPUT="$MODEL_ROOT"
        DEFAULT_VOICES_OUTPUT="api/src/voices"
    else
        MODEL_OUTPUT="$MODEL_ROOT/$MODEL_SUBDIR"
        DEFAULT_VOICES_OUTPUT="api/src/voices/$MODEL_SUBDIR"
    fi

    VOICES_OUTPUT="${VOICES_DIR:-$DEFAULT_VOICES_OUTPUT}"

    python download_model.py --output "$MODEL_OUTPUT" --voices-output "$VOICES_OUTPUT"
fi

exec uv run --extra "$DEVICE" --no-sync python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug
