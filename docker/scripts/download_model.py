#!/usr/bin/env python3
"""下载并准备 Kokoro 模型与语音文件。

默认下载 Kokoro-82M-v1.1-zh；也可通过环境变量或 CLI 参数切换到 v1.0 等仓库，
避免 entrypoint、compose 与下载脚本各自硬编码不同的模型版本。
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger

DEFAULT_REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
DEFAULT_KOKORO_V1_FILE = "v1_1_zh/kokoro-v1_1-zh.pth"
DEFAULT_CONFIG_FILE = "config.json"
VOICE_PATTERN = "voices/*.pt"


def _model_filename(kokoro_v1_file: str) -> str:
    """从配置中的模型相对路径提取仓库内模型文件名。"""
    return Path(kokoro_v1_file).name


def _default_voices_output(output_dir: str) -> str:
    """根据模型输出目录推导对应 voices 目录。"""
    output_path = Path(output_dir)
    return str(output_path.parent.parent / "voices" / output_path.name)


def verify_model_files(model_dir: str, model_file: str, config_file: str) -> bool:
    """Verify required model files are present and valid."""
    model_path = Path(model_dir) / model_file
    config_path = Path(model_dir) / config_file

    try:
        if not model_path.exists() or not config_path.exists():
            return False
        if model_path.stat().st_size == 0:
            return False
        with config_path.open(encoding="utf-8") as file_handle:
            json.load(file_handle)
        return True
    except Exception:
        return False


def verify_voice_files(voices_dir: str, required_voice: str | None = None) -> bool:
    """Verify voice files exist; if DEFAULT_VOICE is set, require that exact file."""
    voice_dir_path = Path(voices_dir)
    if not voice_dir_path.exists():
        return False

    # 关键校验：默认语音必须存在，否则启动 warmup 会在后续失败。
    if required_voice:
        return (voice_dir_path / f"{required_voice}.pt").is_file()

    return any(voice_dir_path.glob("*.pt"))


def _copy_if_needed(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Copy a file when missing or when caller asks to repair an invalid target."""
    if not src.is_file():
        raise FileNotFoundError(f"Expected file not found in downloaded snapshot: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not dst.exists():
        shutil.copy2(src, dst)


def download_model(
    output_dir: str,
    voices_output_dir: str,
    repo_id: str,
    model_file: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    required_voice: str | None = None,
) -> None:
    """Download model files and voices from HuggingFace.

    Args:
        output_dir: Directory for model files (`*.pth`, `config.json`).
        voices_output_dir: Directory for voice `.pt` files.
        repo_id: HuggingFace repository id.
        model_file: Model filename inside the repository.
        config_file: Config filename inside the repository.
        required_voice: Optional voice name that must exist after download.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        logger.error(
            "huggingface_hub is required to download model files. Install via: uv pip install huggingface_hub"
        )
        raise RuntimeError("huggingface_hub not available") from error

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(voices_output_dir, exist_ok=True)

    has_model = verify_model_files(output_dir, model_file, config_file)
    has_voices = verify_voice_files(voices_output_dir, required_voice)
    if has_model and has_voices:
        logger.info("Model and voice files already exist and are valid")
        return

    logger.info(
        f"Downloading resources from HuggingFace repo={repo_id}, model={model_file}"
    )

    try:
        with TemporaryDirectory() as tmp_dir:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                local_dir=tmp_dir,
                allow_patterns=[model_file, config_file, VOICE_PATTERN],
            )

            snapshot_root = Path(snapshot_path)
            # 如果现有模型或 config 无效，覆盖以修复半下载文件。
            _copy_if_needed(
                snapshot_root / model_file,
                Path(output_dir) / model_file,
                overwrite=not has_model,
            )
            _copy_if_needed(
                snapshot_root / config_file,
                Path(output_dir) / config_file,
                overwrite=not has_model,
            )

            voices_root = snapshot_root / "voices"
            for voice_file in voices_root.glob("*.pt"):
                _copy_if_needed(voice_file, Path(voices_output_dir) / voice_file.name)

    except Exception as error:
        logger.error(f"Failed to download model from HuggingFace: {error}")
        logger.error(
            f"Manual fallback: git clone https://huggingface.co/{repo_id} and copy model + voices into your mounted directories."
        )
        raise

    if not verify_model_files(output_dir, model_file, config_file):
        raise RuntimeError("Downloaded model files failed verification")
    if not verify_voice_files(voices_output_dir, required_voice):
        voice_hint = f" required voice '{required_voice}'" if required_voice else ""
        raise RuntimeError(f"Downloaded voice files failed verification:{voice_hint}")

    logger.info(f"✓ Model files prepared in {output_dir}")
    logger.info(f"✓ Voice files prepared in {voices_output_dir}")


def main() -> None:
    """Main entry point."""
    kokoro_v1_file = os.getenv("KOKORO_V1_FILE", DEFAULT_KOKORO_V1_FILE)

    parser = argparse.ArgumentParser(description="Download Kokoro model and voices")
    parser.add_argument(
        "--output", required=True, help="Output directory for model files"
    )
    parser.add_argument(
        "--voices-output",
        required=False,
        help="Output directory for voice files (defaults to sibling voices/<model-version>)",
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("REPO_ID", DEFAULT_REPO_ID),
        help="HuggingFace repository id",
    )
    parser.add_argument(
        "--model-file",
        default=_model_filename(kokoro_v1_file),
        help="Model filename inside the HuggingFace repository",
    )
    parser.add_argument(
        "--config-file",
        default=os.getenv("KOKORO_CONFIG_FILE", DEFAULT_CONFIG_FILE),
        help="Config filename inside the HuggingFace repository",
    )
    parser.add_argument(
        "--required-voice",
        default=os.getenv("DEFAULT_VOICE"),
        help="Voice name that must exist after download",
    )

    args = parser.parse_args()
    voices_output = args.voices_output or _default_voices_output(args.output)

    download_model(
        output_dir=args.output,
        voices_output_dir=voices_output,
        repo_id=args.repo_id,
        model_file=args.model_file,
        config_file=args.config_file,
        required_voice=args.required_voice,
    )


if __name__ == "__main__":
    main()
