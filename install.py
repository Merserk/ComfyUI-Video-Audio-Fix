"""Install Video Audio Fix into a private repository-local virtual environment.

Run this file with the same Python executable used by ComfyUI. The installer
creates ``.venv`` beside this file, installs all AudioSR dependencies there,
and never installs node dependencies into ComfyUI's Python environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_REQUIREMENTS = ROOT / "requirements-venv.txt"
WORKER = ROOT / "audiosr_worker.py"
BIN_DIR = ROOT / "bin"
INSTALL_MARKER = VENV_DIR / ".video_audio_fix_install.json"
AUDIOSR_VERSION = "0.0.7"
INSTALL_SCHEMA = "3"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(command: list[str], *, label: str) -> None:
    print(f"\n[{label}]", flush=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.check_call(command, cwd=str(ROOT), env=environment)


def _create_venv() -> Path:
    python = _venv_python()
    config = VENV_DIR / "pyvenv.cfg"
    if python.is_file() and config.is_file():
        try:
            config_text = config.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            config_text = ""
        expected_version = f"version = {sys.version_info.major}.{sys.version_info.minor}"
        uses_host_packages = "include-system-site-packages = true" in config_text
        version_matches = expected_version in config_text
        if uses_host_packages and version_matches:
            return python
        print("Existing .venv is incompatible with the current ComfyUI Python; rebuilding it.")

    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    print(f"Creating private virtual environment: {VENV_DIR}")
    command = [
        sys.executable,
        "-m",
        "venv",
        "--system-site-packages",
        str(VENV_DIR),
    ]
    try:
        subprocess.check_call(command, cwd=str(ROOT))
    except (subprocess.CalledProcessError, OSError) as first_error:
        # Some Windows embeddable Python distributions omit a usable venv
        # module. virtualenv.pyz creates the environment without installing
        # virtualenv into ComfyUI's Python.
        print("Built-in venv creation failed; trying standalone virtualenv.pyz...")
        with tempfile.TemporaryDirectory(prefix="video_audio_fix_virtualenv_") as temporary:
            pyz = Path(temporary) / "virtualenv.pyz"
            try:
                urllib.request.urlretrieve(
                    "https://bootstrap.pypa.io/virtualenv.pyz",
                    pyz,
                )
                subprocess.check_call(
                    [
                        sys.executable,
                        str(pyz),
                        "--system-site-packages",
                        str(VENV_DIR),
                    ],
                    cwd=str(ROOT),
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "Could not create the node's private .venv with ComfyUI's Python. "
                    "Make sure the portable installation is writable and internet access "
                    "is available, then run install.py again."
                ) from fallback_error
        if not python.is_file():
            raise RuntimeError("Private .venv creation did not produce a Python executable.") from first_error

    return python


def _probe_host_torch() -> dict[str, str | None]:
    """Read the Torch build used by the ComfyUI interpreter running installer."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "ComfyUI's Python cannot import Torch. Start ComfyUI once to confirm its "
            "Torch installation works, then run this installer with that same Python."
        ) from exc

    return {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "hip": str(getattr(torch.version, "hip", None) or "") or None,
        # These packages may be absent or broken in ComfyUI's interpreter. Their
        # private versions are derived from Torch instead of imported globally.
        "torchvision": None,
        "torchaudio": None,
    }


def _public_version(version: str) -> str:
    return version.split("+", 1)[0]


def _torchvision_version(torch_version: str, detected: str | None) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", _public_version(torch_version))
    if not match:
        raise RuntimeError(f"Cannot determine a matching TorchVision version for Torch {torch_version}.")
    major, minor, patch = (int(value) for value in match.groups())
    if major == 2:
        return f"0.{minor + 15}.{patch}"
    if major == 1 and minor == 13:
        return f"0.14.{patch}"
    if detected:
        return _public_version(detected)
    raise RuntimeError(
        f"Torch {torch_version} is outside the supported automatic compatibility map. "
        "Install a current official ComfyUI build, then rerun install.py."
    )


def _torchaudio_version(torch_version: str, detected: str | None) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", _public_version(torch_version))
    if not match:
        raise RuntimeError(f"Cannot determine a matching TorchAudio version for Torch {torch_version}.")
    major, minor, patch = (int(value) for value in match.groups())
    if major == 2:
        return f"{major}.{minor}.{patch}"
    if major == 1 and minor == 13:
        return f"0.13.{patch}"
    if detected:
        return _public_version(detected)
    raise RuntimeError(
        f"Torch {torch_version} is outside the supported automatic compatibility map. "
        "Install a current official ComfyUI build, then rerun install.py."
    )


def _pytorch_wheel_index(torch_info: dict[str, str | None]) -> str | None:
    if sys.platform == "darwin":
        return None

    torch_version = torch_info["torch"] or ""
    local = torch_version.split("+", 1)[1].lower() if "+" in torch_version else ""
    for prefix in ("cu", "rocm", "xpu", "cpu"):
        if local.startswith(prefix):
            # Nightly/local metadata may append extra dot-separated text. Official
            # wheel indexes use the first build token, e.g. cu130 or rocm6.4.
            if prefix == "rocm":
                match = re.match(r"^(rocm\d+(?:\.\d+)?)", local)
            else:
                match = re.match(rf"^({prefix}\d*)", local)
            if match:
                return f"https://download.pytorch.org/whl/{match.group(1)}"

    cuda = torch_info.get("cuda")
    if cuda:
        parts = re.findall(r"\d+", cuda)
        if len(parts) >= 2:
            return f"https://download.pytorch.org/whl/cu{parts[0]}{parts[1]}"

    hip = torch_info.get("hip")
    if hip:
        match = re.match(r"(\d+\.\d+)", hip)
        if match:
            return f"https://download.pytorch.org/whl/rocm{match.group(1)}"

    return "https://download.pytorch.org/whl/cpu"


def _requirements_fingerprint(torch_info: dict[str, str | None]) -> str:
    digest = hashlib.sha256()
    digest.update(VENV_REQUIREMENTS.read_bytes())
    digest.update(WORKER.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(AUDIOSR_VERSION.encode("utf-8"))
    digest.update(INSTALL_SCHEMA.encode("ascii"))
    digest.update(json.dumps(torch_info, sort_keys=True).encode("utf-8"))
    digest.update(f"{sys.version_info.major}.{sys.version_info.minor}".encode("ascii"))
    return digest.hexdigest()


def _read_marker() -> dict[str, object]:
    try:
        return json.loads(INSTALL_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_marker(fingerprint: str, torch_info: dict[str, str | None]) -> None:
    INSTALL_MARKER.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "host_python": sys.executable,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "audiosr": AUDIOSR_VERSION,
                "install_schema": INSTALL_SCHEMA,
                "torch": torch_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _verify_runtime(venv_python: Path) -> bool:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(venv_python), str(WORKER), "--check"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=environment,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    return result.returncode == 0


def _install_pytorch_companions(
    venv_python: Path,
    torch_info: dict[str, str | None],
) -> None:
    """Install binary packages compiled for the exact host Torch release.

    Torch itself remains provided by ComfyUI through --system-site-packages.
    TorchAudio and TorchVision are force-installed into this repository's .venv,
    so no package is added to or changed in ComfyUI's Python environment.
    """
    torch_version = torch_info["torch"]
    if not torch_version:
        raise RuntimeError("Could not determine ComfyUI's Torch version.")

    audio_version = _torchaudio_version(torch_version, torch_info.get("torchaudio"))
    vision_version = _torchvision_version(torch_version, torch_info.get("torchvision"))
    command = [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
    ]
    index_url = _pytorch_wheel_index(torch_info)
    if index_url:
        command.extend(["--index-url", index_url])
    command.extend(
        [
            f"torchaudio=={audio_version}",
            f"torchvision=={vision_version}",
        ]
    )

    build_label = index_url.rsplit("/", 1)[-1] if index_url else "macOS/PyPI"
    print(
        "Matching private PyTorch companions: "
        f"Torch {torch_version}, TorchAudio {audio_version}, "
        f"TorchVision {vision_version}, build {build_label}"
    )
    try:
        _run(command, label="Installing matching TorchAudio and TorchVision in private .venv")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Could not install TorchAudio/TorchVision wheels matching ComfyUI's Torch build. "
            f"Detected Torch {torch_version}; requested TorchAudio {audio_version} and "
            f"TorchVision {vision_version} from {index_url or 'PyPI'}. "
            "No global packages were changed. Update ComfyUI to an official supported "
            "Torch build and rerun install.py."
        ) from exc


def _install_python_runtime(venv_python: Path, torch_info: dict[str, str | None]) -> None:
    if not VENV_REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Missing private requirements file: {VENV_REQUIREMENTS}")

    fingerprint = _requirements_fingerprint(torch_info)
    marker = _read_marker()
    if marker.get("fingerprint") == fingerprint and _verify_runtime(venv_python):
        print("Private AudioSR environment is already complete.")
        return

    _run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        label="Updating private pip tooling",
    )
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(VENV_REQUIREMENTS),
        ],
        label="Resolving private AudioSR dependencies",
    )
    # A --system-site-packages venv reuses ComfyUI's large Torch/CUDA build.
    # Force local copies of every explicitly listed non-Torch package so their
    # versions are selected from .venv before ComfyUI's site-packages.
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            "-r",
            str(VENV_REQUIREMENTS),
        ],
        label="Pinning dependencies inside private .venv",
    )
    _install_pytorch_companions(venv_python, torch_info)
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            f"audiosr=={AUDIOSR_VERSION}",
        ],
        label="Installing AudioSR in private environment",
    )

    if not _verify_runtime(venv_python):
        raise RuntimeError(
            "The private AudioSR environment was created, but its import check failed. "
            "Review the messages above. TorchAudio and TorchVision are installed inside "
            "the node's .venv and must exactly match ComfyUI's Torch release/build."
        )
    _write_marker(fingerprint, torch_info)


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


def _download_portable_media_tools(venv_python: Path) -> tuple[Path, Path]:
    script = r'''
import json
from portable_ffmpeg import get_ffmpeg
ffmpeg, ffprobe = get_ffmpeg()
print("VIDEO_AUDIO_FIX_FFMPEG=" + json.dumps([ffmpeg, ffprobe]))
'''
    result = subprocess.run(
        [str(venv_python), "-c", script],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not download portable FFmpeg/FFprobe from the private .venv.\n"
            + result.stdout[-6000:]
        )
    marker = "VIDEO_AUDIO_FIX_FFMPEG="
    payload = next(
        (line[len(marker):] for line in reversed(result.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if payload is None:
        raise RuntimeError("Portable FFmpeg downloader returned no executable paths.")
    ffmpeg, ffprobe = json.loads(payload)
    return Path(ffmpeg), Path(ffprobe)


def _install_portable_media_tools(venv_python: Path) -> tuple[Path, Path]:
    local_ffmpeg, local_ffprobe = _local_media_tools()
    if _is_executable(local_ffmpeg) and _is_executable(local_ffprobe):
        print(f"Portable FFmpeg is already installed in: {BIN_DIR}")
        return local_ffmpeg, local_ffprobe

    print("Downloading private FFmpeg and FFprobe through the node's .venv...")
    source_ffmpeg, source_ffprobe = _download_portable_media_tools(venv_python)
    if not _is_executable(source_ffmpeg) or not _is_executable(source_ffprobe):
        raise RuntimeError(
            "The portable downloader returned invalid FFmpeg/FFprobe executables: "
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
    print(f"ComfyUI host Python: {sys.executable}")
    print("Node dependencies will be installed only inside the repository .venv.")
    torch_info = _probe_host_torch()
    print(
        "Detected ComfyUI Torch: "
        f"{torch_info['torch']} (CUDA={torch_info['cuda'] or 'none'}, "
        f"ROCm={torch_info['hip'] or 'none'})"
    )
    venv_python = _create_venv()
    print(f"Private Python: {venv_python}")
    _install_python_runtime(venv_python, torch_info)
    ffmpeg, ffprobe = _install_portable_media_tools(venv_python)
    print(f"FFmpeg:  {ffmpeg}")
    print(f"FFprobe: {ffprobe}")
    print("Installation complete. Restart ComfyUI.")


if __name__ == "__main__":
    main()
