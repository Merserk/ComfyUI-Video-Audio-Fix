"""Install Video Audio Fix into a private repository-local virtual environment.

Run this file with the same Python executable used by ComfyUI. The installer
creates ``.venv`` beside this file, installs all FlashSR dependencies there,
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
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_REQUIREMENTS = ROOT / "requirements-venv.txt"
WORKER = ROOT / "flashsr_worker.py"
BIN_DIR = ROOT / "bin"
INSTALL_MARKER = VENV_DIR / ".video_audio_fix_install.json"
FLASHSR_SOURCE_URL = "https://github.com/jakeoneijk/FlashSR_Inference/archive/refs/heads/main.zip"
FLASHSR_VERSION = "main"
INSTALL_SCHEMA = "7"
REQUIRED_ARTIFACT_FILTERS = (
    "adeclick",
    "apad",
    "asetpts",
    "atrim",
    "highpass",
    "loudnorm",
    "lowpass",
    "volume",
)


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(command: list[str], *, label: str) -> None:
    print(f"\n[{label}]", flush=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
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
        marker_schema = None
        try:
            marker_schema = json.loads(INSTALL_MARKER.read_text(encoding="utf-8")).get("install_schema")
        except (OSError, json.JSONDecodeError):
            pass
        if uses_host_packages and version_matches and marker_schema == INSTALL_SCHEMA:
            return python
        print("Existing .venv is incomplete or uses an older install schema; rebuilding it.")

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


def _host_site_packages() -> Path:
    if os.name == "nt":
        return Path(sys.base_prefix) / "Lib" / "site-packages"
    return (
        Path(sys.base_prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _ensure_system_site_packages(venv_python: Path) -> None:
    """Make the private .venv see the host's site-packages (Torch/CUDA build).

    Some embedded Python distributions create venvs whose sys.base_prefix
    resolves to the venv itself, so include-system-site-packages=true has no
    effect and ``import torch`` fails inside the worker. When the host
    site-packages is missing from the venv path, expose it through a .pth file
    inside the venv; entries added by .pth come after the venv's own
    site-packages, so private package pins still win.
    """
    host = _host_site_packages()
    if not host.is_dir():
        return
    probe = subprocess.run(
        [
            str(venv_python),
            "-c",
            "import sys; sys.exit(0 if sys.argv[1] in sys.path else 1)",
            str(host),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        return
    site_packages = VENV_DIR / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "_video_audio_fix_system.pth").write_text(
        str(host) + "\n",
        encoding="utf-8",
    )
    print(f"Linked host site-packages into the private .venv: {host}")


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
    }



def _requirements_fingerprint(torch_info: dict[str, str | None]) -> str:
    digest = hashlib.sha256()
    digest.update(VENV_REQUIREMENTS.read_bytes())
    digest.update(WORKER.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(FLASHSR_SOURCE_URL.encode("utf-8"))
    digest.update(FLASHSR_VERSION.encode("utf-8"))
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
                "flashsr": FLASHSR_VERSION,
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
    environment["PYTHONIOENCODING"] = "utf-8"
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



def _venv_site_packages() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Lib" / "site-packages"
    return (
        VENV_DIR
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _install_flashsr_source() -> None:
    """Vendor the upstream FlashSR_Inference source into the private .venv.

    The repository's setup.py declares no package data, so building it with pip
    drops the runtime YAML configs and other assets inside the FlashSR package.
    Copying the FlashSR and TorchJaekwon trees directly into site-packages
    preserves every file the inference path needs.
    """
    target = _venv_site_packages()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".flashsr_source_version"
    if marker.is_file() and marker.read_text(encoding="utf-8") == FLASHSR_VERSION:
        if (target / "FlashSR").is_dir() and (target / "TorchJaekwon").is_dir():
            print("FlashSR_Inference source is already installed.")
            return

    print(f"Downloading FlashSR_Inference source: {FLASHSR_SOURCE_URL}")
    with tempfile.TemporaryDirectory(prefix="flashsr_source_") as temporary:
        temporary_dir = Path(temporary)
        archive = temporary_dir / "flashsr_inference.zip"
        urllib.request.urlretrieve(FLASHSR_SOURCE_URL, archive)
        with zipfile.ZipFile(archive) as archive_zip:
            archive_zip.extractall(temporary_dir)
        root = next(temporary_dir.iterdir(), None)
        if root is None or not root.is_dir():
            raise RuntimeError("FlashSR_Inference archive contained no top-level folder.")
        for package in ("FlashSR", "TorchJaekwon"):
            source = root / package
            if not source.is_dir():
                raise RuntimeError(f"FlashSR_Inference archive is missing the {package} package.")
            destination = target / package
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination)
    marker.write_text(FLASHSR_VERSION, encoding="utf-8")


def _install_python_runtime(venv_python: Path, torch_info: dict[str, str | None]) -> None:
    if not VENV_REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Missing private requirements file: {VENV_REQUIREMENTS}")

    fingerprint = _requirements_fingerprint(torch_info)
    marker = _read_marker()
    if marker.get("fingerprint") == fingerprint and _verify_runtime(venv_python):
        print("Private FlashSR environment is already complete.")
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
        label="Resolving private FlashSR dependencies",
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
    _install_flashsr_source()

    if not _verify_runtime(venv_python):
        raise RuntimeError(
            "The private FlashSR environment was created, but its import check failed. "
            "Review the messages above. No TorchAudio, TorchVision, or AudioSR package is "
            "required; the worker uses ComfyUI's existing Torch and the vendored "
            "FlashSR_Inference source."
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


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _parse_ffmpeg_filter_names(listing: str) -> set[str]:
    names: set[str] = set()
    clean_listing = _ANSI_ESCAPE_RE.sub("", listing)
    for raw_line in clean_listing.splitlines():
        fields = raw_line.split()
        if len(fields) < 3 or "->" not in fields[2]:
            continue
        candidate = fields[1]
        if re.fullmatch(r"[A-Za-z0-9_]+", candidate):
            names.add(candidate)
    return names


def _filter_help_available(ffmpeg: Path, filter_name: str) -> bool:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-h", f"filter={filter_name}"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    output = _ANSI_ESCAPE_RE.sub("", result.stdout)
    if result.returncode != 0 or re.search(
        r"^\s*Unknown filter\b", output, re.MULTILINE | re.IGNORECASE
    ):
        return False
    return re.search(
        rf"^\s*Filter\s+{re.escape(filter_name)}(?:\s|$)",
        output,
        re.MULTILINE | re.IGNORECASE,
    ) is not None


def _ffmpeg_version_summary(ffmpeg: Path) -> str:
    result = subprocess.run(
        [str(ffmpeg), "-version"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else "version unavailable"


def _verify_artifact_filters(ffmpeg: Path) -> None:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-filters"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Portable FFmpeg could not report its available filters.\n"
            + result.stdout[-6000:]
        )

    names = _parse_ffmpeg_filter_names(result.stdout)
    missing = [
        name
        for name in REQUIRED_ARTIFACT_FILTERS
        if name not in names and not _filter_help_available(ffmpeg, name)
    ]
    if missing:
        raise RuntimeError(
            "The installed FFmpeg build is incompatible with artifacts_reductions. "
            "Missing required audio filters: " + ", ".join(missing) + ".\n"
            f"Executable: {ffmpeg}\n"
            f"Version: {_ffmpeg_version_summary(ffmpeg)}"
        )
    print(
        "FFmpeg artifacts_reductions filter set verified: "
        + ", ".join(REQUIRED_ARTIFACT_FILTERS)
    )
    print("FFmpeg build: " + _ffmpeg_version_summary(ffmpeg))


def _local_media_tools() -> tuple[Path, Path]:
    suffix = ".exe" if os.name == "nt" else ""
    return BIN_DIR / f"ffmpeg{suffix}", BIN_DIR / f"ffprobe{suffix}"


def _download_portable_media_tools(
    venv_python: Path,
    *,
    refresh_cache: bool = False,
) -> tuple[Path, Path]:
    script = r'''
import json
import sys
from portable_ffmpeg import clear_cache, get_ffmpeg
if sys.argv[1] == "1":
    clear_cache()
ffmpeg, ffprobe = get_ffmpeg()
print("VIDEO_AUDIO_FIX_FFMPEG=" + json.dumps([str(ffmpeg), str(ffprobe)]))
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [str(venv_python), "-c", script, "1" if refresh_cache else "0"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=environment,
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
    refresh_cache = False
    if _is_executable(local_ffmpeg) and _is_executable(local_ffprobe):
        try:
            _verify_artifact_filters(local_ffmpeg)
        except RuntimeError as exc:
            print("Existing portable FFmpeg is incompatible; downloading a fresh build.")
            print(str(exc))
            local_ffmpeg.unlink(missing_ok=True)
            local_ffprobe.unlink(missing_ok=True)
            refresh_cache = True
        else:
            print(f"Portable FFmpeg is already installed in: {BIN_DIR}")
            return local_ffmpeg, local_ffprobe

    print("Downloading private FFmpeg and FFprobe through the node's .venv...")
    source_ffmpeg, source_ffprobe = _download_portable_media_tools(
        venv_python,
        refresh_cache=refresh_cache,
    )
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
    _verify_artifact_filters(local_ffmpeg)
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
    _ensure_system_site_packages(venv_python)
    print(f"Private Python: {venv_python}")
    _install_python_runtime(venv_python, torch_info)
    ffmpeg, ffprobe = _install_portable_media_tools(venv_python)
    print(f"FFmpeg:  {ffmpeg}")
    print(f"FFprobe: {ffprobe}")
    print("Installation complete. Restart ComfyUI.")


if __name__ == "__main__":
    main()
