"""Install Video Audio Fix runtime dependencies and portable media tools.

Run this file with the same Python executable used by ComfyUI. For the Windows
portable build, that is usually ``python_embeded\\python.exe``.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
BIN_DIR = ROOT / "bin"


def _pip(*arguments: str) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            *arguments,
        ]
    )


def _is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = subprocess.run(
            [str(path), "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _local_media_tools() -> tuple[Path, Path]:
    suffix = ".exe" if os.name == "nt" else ""
    return BIN_DIR / f"ffmpeg{suffix}", BIN_DIR / f"ffprobe{suffix}"


def _install_portable_media_tools() -> tuple[Path, Path]:
    local_ffmpeg, local_ffprobe = _local_media_tools()
    if _is_executable(local_ffmpeg) and _is_executable(local_ffprobe):
        print(f"Portable FFmpeg is already installed in: {BIN_DIR}")
        return local_ffmpeg, local_ffprobe

    system_ffmpeg = shutil.which("ffmpeg")
    system_ffprobe = shutil.which("ffprobe")
    if system_ffmpeg and system_ffprobe:
        print("System FFmpeg and FFprobe are already available on PATH.")
        return Path(system_ffmpeg), Path(system_ffprobe)

    print("FFmpeg/FFprobe were not found. Downloading portable binaries...")
    try:
        from portable_ffmpeg import get_ffmpeg

        downloaded_ffmpeg, downloaded_ffprobe = get_ffmpeg()
    except Exception as exc:
        raise RuntimeError(
            "Could not download portable FFmpeg and FFprobe. Check the internet "
            "connection, then run install.py again with ComfyUI's Python."
        ) from exc

    source_ffmpeg = Path(downloaded_ffmpeg)
    source_ffprobe = Path(downloaded_ffprobe)
    if not _is_executable(source_ffmpeg) or not _is_executable(source_ffprobe):
        raise RuntimeError(
            "portable-ffmpeg returned invalid FFmpeg/FFprobe executables: "
            f"{source_ffmpeg}, {source_ffprobe}"
        )

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_ffmpeg, local_ffmpeg)
    shutil.copy2(source_ffprobe, local_ffprobe)
    if os.name != "nt":
        local_ffmpeg.chmod(local_ffmpeg.stat().st_mode | 0o111)
        local_ffprobe.chmod(local_ffprobe.stat().st_mode | 0o111)

    if not _is_executable(local_ffmpeg) or not _is_executable(local_ffprobe):
        raise RuntimeError(f"Portable FFmpeg validation failed in: {BIN_DIR}")

    print(f"Portable FFmpeg and FFprobe installed in: {BIN_DIR}")
    return local_ffmpeg, local_ffprobe


def main() -> None:
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS}")

    print(f"Installing Video Audio Fix dependencies with: {sys.executable}")
    _pip("install", "-r", str(REQUIREMENTS))

    # AudioSR 0.0.7 declares obsolete dependency pins. Its compatible runtime
    # dependencies are installed above, so install the package without allowing
    # it to downgrade ComfyUI's NumPy, Librosa, Transformers, or Torch stack.
    if importlib.util.find_spec("audiosr") is None:
        _pip("install", "--no-deps", "audiosr==0.0.7")
        print("AudioSR installed.")
    else:
        print("AudioSR is already installed.")

    ffmpeg, ffprobe = _install_portable_media_tools()
    print(f"FFmpeg:  {ffmpeg}")
    print(f"FFprobe: {ffprobe}")
    print("Installation complete. Restart ComfyUI.")


if __name__ == "__main__":
    main()
