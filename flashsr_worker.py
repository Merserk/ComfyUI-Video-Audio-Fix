"""Private FlashSR worker executed by the repository-local .venv.

The ComfyUI node launches this file as a subprocess so FlashSR's dependency set
never enters ComfyUI's interpreter. Communication is file-based WAV input/output
plus line-oriented progress messages on stdout.

The engine is FlashSR (one-step versatile audio super-resolution via diffusion
distillation, https://github.com/jakeoneijk/FlashSR_Inference), which runs its
fixed-size 5.12 s diffusion model over overlapping chunks of each channel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 48_000
CHUNK_SECONDS = 5.12
CHUNK_SAMPLES = int(round(CHUNK_SECONDS * SAMPLE_RATE))  # 245760, fixed model input
OVERLAP_RATIO = 0.04
BASE_SEED = 42
MODEL_DATASET = "jakeoneijk/FlashSR_weights"
MODEL_FILES = ("student_ldm.pth", "sr_vocoder.pth", "vae.pth")
ARTIFACT_TRUE_PEAK_TARGET_DBTP = -1.0
ARTIFACT_HIGHPASS_HZ = 10.0
ARTIFACT_LOWPASS_HZ = 23_000.0
ARTIFACT_DECLICK_WINDOW = 10.0
ARTIFACT_DECLICK_OVERLAP = 75.0
ARTIFACT_DECLICK_ARORDER = 8.0
ARTIFACT_DECLICK_THRESHOLD = 20.0
ARTIFACT_DECLICK_BURST = 1.0
ARTIFACT_TRUE_PEAK_TOLERANCE_DB = 0.01
ARTIFACT_REQUIRED_FILTERS = (
    "adeclick",
    "apad",
    "asetpts",
    "atrim",
    "highpass",
    "loudnorm",
    "lowpass",
    "volume",
)


def _print(message: str) -> None:
    print(message, flush=True)


def _patch_numpy_aliases() -> None:
    # Patch aliases before importing FlashSR/TorchJaekwon code written for
    # NumPy 1.x (several modules are copied from the AudioSR codebase).
    aliases = {
        "complex": np.complex128,
        "float": float,
        "int": int,
        "bool": bool,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _check_runtime() -> None:
    _patch_numpy_aliases()
    import soundfile  # noqa: F401
    import scipy  # noqa: F401
    import librosa  # noqa: F401
    import sklearn  # noqa: F401
    import yaml  # noqa: F401
    import einops  # noqa: F401
    import torch  # noqa: F401

    from FlashSR.FlashSR import FlashSR  # noqa: F401
    from TorchJaekwon.Model.Diffusion.DDPM.BetaSchedule import BetaSchedule
    from TorchJaekwon.Model.Diffusion.External.diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )

    betas = BetaSchedule.cosine(timesteps=1000)
    if tuple(np.asarray(betas).shape) != (1000,):
        raise RuntimeError("Internal FlashSR beta schedule failed its self-test.")
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        trained_betas=betas,
        prediction_type="v_prediction",
    )
    scheduler.set_timesteps(1)
    if len(scheduler.timesteps) != 1:
        raise RuntimeError("Internal FlashSR scheduler failed its self-test.")

    _print(
        "Private runtime OK: "
        f"Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"NumPy {np.__version__}, Torch {torch.__version__}, "
        "FlashSR import chain verified"
    )


def _run(command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Unknown process error"
        raise RuntimeError(f"{label} failed.\n{detail[-6000:]}")
    return result


def _ensure_models(model_dir: Path, auto_download: bool) -> dict[str, Path]:
    model_dir = model_dir.resolve()
    existing = {
        name: model_dir / name
        for name in MODEL_FILES
        if (model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
    }
    if len(existing) == len(MODEL_FILES):
        return existing
    if not auto_download:
        raise RuntimeError(
            "FlashSR models are missing and auto_download is disabled.\n"
            f"Place {', '.join(MODEL_FILES)} in:\n{model_dir}"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import snapshot_download

    _print(f"Downloading FlashSR models to {model_dir} ...")
    snapshot_download(
        repo_id=MODEL_DATASET,
        repo_type="dataset",
        local_dir=str(model_dir),
        allow_patterns=[f"*/{name}" for name in MODEL_FILES]
        + [name for name in MODEL_FILES],
    )
    downloaded: dict[str, Path] = {}
    for name in MODEL_FILES:
        candidate = model_dir / name
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded FlashSR checkpoint is invalid: {candidate}")
        downloaded[name] = candidate
    return downloaded


def _load_model(
    model_dir: Path,
    auto_download: bool,
    fixed_cutoff_hz: int = 0,
    cfg_scale: float = 0.0,
) -> tuple[Any, str]:
    _patch_numpy_aliases()
    import torch

    original_load = torch.load

    def _safe_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    # Upstream FlashSR calls torch.load without map_location, which crashes when
    # the checkpoints were saved from CUDA and this machine has no GPU, and
    # without weights_only, which raises on newer Torch builds.
    torch.load = _safe_load  # type: ignore[assignment]

    from FlashSR.FlashSR import FlashSR

    class _FixedCutoffFlashSR(FlashSR):
        """FlashSR with a file-wide lowpass cutoff instead of the upstream
        per-chunk detection, which is inconsistent across chunks."""

        def __init__(self, *args: Any, cutoff: int = 0, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._fixed_cutoff = int(cutoff)

        def forward(
            self,
            lr_audio: Any,
            num_steps: int = 1,
            lowpass_input: bool = True,
            lowpass_cutoff_freq: Any = None,
        ) -> Any:
            return super().forward(
                lr_audio,
                num_steps=num_steps,
                lowpass_input=lowpass_input,
                lowpass_cutoff_freq=(self._fixed_cutoff if self._fixed_cutoff > 0 else None),
            )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    models = _ensure_models(model_dir, bool(auto_download))
    _print(
        f"Loading FlashSR one-step model on {device} "
        f"(student {models['student_ldm.pth'].name}, "
        f"vocoder {models['sr_vocoder.pth'].name}, vae {models['vae.pth'].name}) ..."
    )
    if fixed_cutoff_hz > 0:
        flashsr = _FixedCutoffFlashSR(
            str(models["student_ldm.pth"]),
            str(models["sr_vocoder.pth"]),
            str(models["vae.pth"]),
            cutoff=fixed_cutoff_hz,
        )
    else:
        flashsr = FlashSR(
            str(models["student_ldm.pth"]),
            str(models["sr_vocoder.pth"]),
            str(models["vae.pth"]),
        )
    if cfg_scale > 1.0:
        flashsr.cfg_scale = float(cfg_scale)
    flashsr.eval()
    flashsr = flashsr.to(device)
    torch.set_grad_enabled(False)
    return flashsr, str(device)


def _chunk_window(length: int, overlap: int, has_left: bool, has_right: bool) -> np.ndarray:
    window = np.ones(length, dtype=np.float32)
    actual = min(overlap, length)
    if actual <= 1:
        return window
    fade = np.sin(np.linspace(0.0, math.pi / 2.0, actual, dtype=np.float32)) ** 2
    if has_left:
        window[:actual] *= fade
    if has_right:
        window[-actual:] *= fade[::-1]
    return window


def _match_chunk_level(processed: np.ndarray, original: np.ndarray) -> np.ndarray:
    original = original.astype(np.float32, copy=False)
    processed = processed.astype(np.float32, copy=False)
    original_rms = float(np.sqrt(np.mean(np.square(original, dtype=np.float64)) + 1e-12))
    processed_rms = float(np.sqrt(np.mean(np.square(processed, dtype=np.float64)) + 1e-12))
    if original_rms < 1e-6 or processed_rms < 1e-8:
        return original.copy() if original_rms < 1e-6 else processed
    gain = float(np.clip(original_rms / processed_rms, 0.25, 4.0))
    return processed * gain


def _run_chunk(
    flashsr: Any,
    chunk: np.ndarray,
    seed: int,
    lowpass_input: bool,
    steps: int,
    use_fp16: bool,
) -> tuple[np.ndarray, bool]:
    """Run the one-step FlashSR model on one padded 5.12 s chunk."""
    import torch

    device = getattr(flashsr, "device", None)
    if device is None:
        device = next(flashsr.parameters()).device
    input_tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    kwargs = {"lowpass_input": lowpass_input, "num_steps": max(1, int(steps))}
    use_autocast = use_fp16 and device.type == "cuda"
    fp32_fallback = False
    try:
        with torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    output = flashsr(input_tensor, **kwargs)
            else:
                output = flashsr(input_tensor, **kwargs)
    except RuntimeError as exc:
        if not (use_autocast and "out of memory" in str(exc).lower()):
            raise
        _print("fp16 chunk ran out of memory, retrying in fp32")
        fp32_fallback = True
        with torch.no_grad():
            output = flashsr(input_tensor, **kwargs)

    if hasattr(output, "detach"):
        output = output.detach().float().cpu().numpy()
    array = np.asarray(output, dtype=np.float32).reshape(-1)
    if array.size < chunk.size:
        array = np.pad(array, (0, chunk.size - array.size))
    return array[: chunk.size], fp32_fallback


def _lfe_channels(channel_layout: str, channel_count: int) -> set[int]:
    layout = channel_layout.lower()
    if ".1" not in layout and "lfe" not in layout:
        return set()
    if channel_count == 3:
        return {2}
    if channel_count >= 4:
        return {3}
    return set()


def _detect_cutoff(extracted: Path, percentile: float = 0.983) -> int:
    """Whole-file estimate of the input's high-frequency cutoff, mirroring the
    upstream per-chunk detection but computed once over the full file so every
    chunk is lowpassed consistently."""
    import librosa
    import soundfile as sf

    info = sf.info(str(extracted))
    total = int(info.frames)
    segment = min(CHUNK_SAMPLES, total)
    cutoffs: list[float] = []
    with sf.SoundFile(str(extracted), mode="r") as source:
        for position in range(0, max(1, total - segment), CHUNK_SAMPLES):
            source.seek(position)
            block = source.read(segment, dtype="float32", always_2d=True)
            mono = np.mean(block, axis=1).astype(np.float64)
            mono = mono - float(np.mean(mono))
            if float(np.max(np.abs(mono))) < 1e-5:
                continue
            stft = np.abs(librosa.stft(mono, n_fft=2048, hop_length=480, win_length=2048, center=False, window="hann"))
            energy = np.cumsum(np.sum(stft, axis=1))
            index = int(np.searchsorted(energy, energy[-1] * percentile))
            cutoffs.append((min(index, len(energy) - 1) / 1024) * 24000)
    if not cutoffs:
        return 24000
    cutoff = float(np.median(cutoffs))
    return int(24000 if cutoff < 1000 else cutoff)


def _process_audio(
    extracted: Path,
    restored: Path,
    model_dir: Path,
    auto_download: bool,
    channel_layout: str,
    lowpass_input: bool = True,
    steps: int = 1,
    use_fp16: bool = False,
    fixed_cutoff: int = 0,
    cfg_scale: float = 0.0,
    preloaded: tuple[Any, str] | None = None,
) -> tuple[int, int, str]:
    import soundfile as sf

    info = sf.info(str(extracted))
    if info.samplerate != SAMPLE_RATE:
        raise RuntimeError("Extracted audio is not 48 kHz.")
    total_frames = int(info.frames)
    channels = int(info.channels)
    if total_frames <= 100:
        shutil.copy2(extracted, restored)
        return total_frames, channels, "BYPASS"

    _ensure_models(model_dir, bool(auto_download))
    if preloaded is not None:
        flashsr, device = preloaded
    else:
        flashsr, device = _load_model(model_dir, bool(auto_download), fixed_cutoff, cfg_scale)
    actual_precision = "FP16" if use_fp16 and str(device).startswith("cuda") else "FP32"
    requested_precision = "FP16" if use_fp16 else "FP32"
    if requested_precision == "FP16" and actual_precision == "FP32":
        _print(f"Requested FP16, but device {device} uses FP32 inference")
    else:
        _print(f"Inference precision: {actual_precision}")
    _print(
        f"Restoring {channels} channel(s), {total_frames / SAMPLE_RATE:.2f} seconds, "
        f"FlashSR device {device}"
    )

    chunk_samples = CHUNK_SAMPLES
    overlap_samples = int(round(chunk_samples * OVERLAP_RATIO))
    step_samples = chunk_samples - overlap_samples
    starts = list(range(0, total_frames, step_samples))
    lfe = _lfe_channels(channel_layout, channels)
    total_tasks = len(starts) * max(1, channels - len(lfe))
    _print(f"TOTAL {total_tasks}")
    completed = 0
    fp32_fallback_used = False

    with tempfile.TemporaryDirectory(prefix="flashsr_private_work_") as work:
        work_dir = Path(work)
        data_path = work_dir / "restored.float32"
        weights_path = work_dir / "weights.float32"
        output_map = np.memmap(
            data_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, channels),
        )
        weights = np.memmap(weights_path, mode="w+", dtype=np.float32, shape=(total_frames,))
        try:
            output_map[:] = 0.0
            weights[:] = 0.0

            for start in starts:
                length = min(chunk_samples, total_frames - start)
                weights[start : start + length] += _chunk_window(
                    length,
                    overlap_samples,
                    has_left=start > 0,
                    has_right=start + length < total_frames,
                )
            weights.flush()

            with sf.SoundFile(str(extracted), mode="r") as source:
                for channel in range(channels):
                    if channel in lfe:
                        source.seek(0)
                        write_at = 0
                        while write_at < total_frames:
                            block = source.read(
                                min(262_144, total_frames - write_at),
                                dtype="float32",
                                always_2d=True,
                            )
                            output_map[write_at : write_at + len(block), channel] = block[:, channel]
                            write_at += len(block)
                        _print(f"Passed through LFE channel {channel + 1}/{channels}")
                        continue

                    for chunk_index, start in enumerate(starts):
                        length = min(chunk_samples, total_frames - start)
                        source.seek(start)
                        block = source.read(length, dtype="float32", always_2d=True)
                        original = np.asarray(block[:, channel], dtype=np.float32)
                        padded = np.zeros(chunk_samples, dtype=np.float32)
                        padded[:length] = original

                        if float(np.max(np.abs(original))) < 1e-5:
                            processed = padded
                        else:
                            processed, chunk_fp32_fallback = _run_chunk(
                                flashsr,
                                padded,
                                BASE_SEED + chunk_index,
                                lowpass_input,
                                steps,
                                use_fp16,
                            )
                            fp32_fallback_used = fp32_fallback_used or chunk_fp32_fallback
                            processed = _match_chunk_level(processed, padded)

                        window = _chunk_window(
                            length,
                            overlap_samples,
                            has_left=start > 0,
                            has_right=start + length < total_frames,
                        )
                        output_map[start : start + length, channel] += processed[:length] * window
                        completed += 1
                        _print(f"PROGRESS {completed}")

                    for block_start in range(0, total_frames, 262_144):
                        block_end = min(total_frames, block_start + 262_144)
                        denominator = np.maximum(np.asarray(weights[block_start:block_end]), 1e-8)
                        output_map[block_start:block_end, channel] /= denominator

            output_map.flush()
            with sf.SoundFile(
                str(restored),
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=channels,
                format="WAV",
                subtype="FLOAT",
            ) as destination:
                for block_start in range(0, total_frames, 262_144):
                    block_end = min(total_frames, block_start + 262_144)
                    destination.write(np.asarray(output_map[block_start:block_end]))
        finally:
            output_map._mmap.close()
            weights._mmap.close()

    if fp32_fallback_used:
        actual_precision = "MIXED_FP16_FP32"
    return total_frames, channels, actual_precision


def _estimate_crossover(extracted: Path) -> int:
    import soundfile as sf

    info = sf.info(str(extracted))
    total = int(info.frames)
    if total < 2048:
        return 8_000
    sample_length = min(SAMPLE_RATE, total)
    count = min(8, max(1, total // sample_length))
    positions = np.linspace(0, max(0, total - sample_length), num=count, dtype=int)
    spectra: list[np.ndarray] = []
    with sf.SoundFile(str(extracted), mode="r") as source:
        for position in positions:
            source.seek(int(position))
            block = source.read(sample_length, dtype="float32", always_2d=True)
            if len(block) < 2048:
                continue
            mono = np.mean(block, axis=1)
            mono = mono - float(np.mean(mono))
            if float(np.max(np.abs(mono))) < 1e-5:
                continue
            window = np.hanning(len(mono)).astype(np.float32)
            spectra.append(np.abs(np.fft.rfft(mono * window)) ** 2)
    if not spectra:
        return 8_000
    power = np.mean(np.stack(spectra), axis=0)
    frequencies = np.fft.rfftfreq((len(power) - 1) * 2, d=1.0 / SAMPLE_RATE)
    usable = frequencies <= 22_000
    power = power[usable]
    frequencies = frequencies[usable]
    cumulative = np.cumsum(power)
    if cumulative[-1] <= 0:
        return 8_000
    rolloff_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.997))
    rolloff = float(frequencies[min(rolloff_index, len(frequencies) - 1)])
    return int(np.clip(round(rolloff * 0.9 / 250.0) * 250, 4_000, 18_000))


def _blend_audio(
    ffmpeg: str,
    original: Path,
    restored: Path,
    output: Path,
    total_frames: int,
    channels: int,
    channel_layout: str,
    crossover: int,
) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(original)]
    if channel_layout and channel_layout.lower() != "unknown":
        command += ["-channel_layout", channel_layout]
    command += ["-i", str(restored)]
    filter_graph = (
        f"[0:a]acrossover=split={crossover}:order=4th[lo][odisc];[odisc]anullsink;"
        f"[1:a]acrossover=split={crossover}:order=4th[idisc][hi];[idisc]anullsink;"
        f"[lo][hi]amix=inputs=2:normalize=0,"
        f"atrim=end_sample={total_frames},apad=whole_len={total_frames},asetpts=N/SR/TB[out]"
    )
    command += [
        "-filter_complex",
        filter_graph,
        "-map",
        "[out]",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_f32le",
        str(output),
    ]
    _run(command, label="audio reconstruction")


def _format_filter_number(value: float) -> str:
    return f"{value:g}"


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _parse_ffmpeg_filter_names(listing: str) -> frozenset[str]:
    """Parse ``ffmpeg -filters`` without assuming a fixed flag-column width.

    FFmpeg has added filter capability flags over time.  The old parser expected
    exactly three characters (T/S/C), which made newer builds look as though
    they contained no filters at all.  The stable part of each data row is the
    I/O signature token (for example ``A->A``), so use that instead.
    """
    names: set[str] = set()
    clean_listing = _ANSI_ESCAPE_RE.sub("", listing)
    for raw_line in clean_listing.splitlines():
        fields = raw_line.split()
        if len(fields) < 3 or "->" not in fields[2]:
            continue
        candidate = fields[1]
        if re.fullmatch(r"[A-Za-z0-9_]+", candidate):
            names.add(candidate)
    return frozenset(names)


@lru_cache(maxsize=8)
def _ffmpeg_filter_names(ffmpeg: str) -> frozenset[str]:
    result = _run(
        [ffmpeg, "-hide_banner", "-filters"],
        label="FFmpeg filter inspection",
    )
    return _parse_ffmpeg_filter_names(result.stdout + "\n" + result.stderr)


@lru_cache(maxsize=64)
def _ffmpeg_filter_help_available(ffmpeg: str, filter_name: str) -> bool:
    """Use FFmpeg's per-filter help as a format-independent fallback probe."""
    try:
        result = _run(
            [ffmpeg, "-hide_banner", "-h", f"filter={filter_name}"],
            label=f"FFmpeg {filter_name} filter inspection",
        )
    except RuntimeError:
        return False
    output = _ANSI_ESCAPE_RE.sub("", result.stdout + "\n" + result.stderr)
    if re.search(r"^\s*Unknown filter\b", output, re.MULTILINE | re.IGNORECASE):
        return False
    return re.search(
        rf"^\s*Filter\s+{re.escape(filter_name)}(?:\s|$)",
        output,
        re.MULTILINE | re.IGNORECASE,
    ) is not None


@lru_cache(maxsize=8)
def _ffmpeg_version_summary(ffmpeg: str) -> str:
    try:
        result = _run([ffmpeg, "-version"], label="FFmpeg version inspection")
    except RuntimeError:
        return "version unavailable"
    output = (result.stdout + "\n" + result.stderr).strip().splitlines()
    return output[0].strip() if output else "version unavailable"


def _require_artifact_filters(ffmpeg: str) -> None:
    available = set(_ffmpeg_filter_names(ffmpeg))
    missing: list[str] = []
    for name in ARTIFACT_REQUIRED_FILTERS:
        if name in available or _ffmpeg_filter_help_available(ffmpeg, name):
            continue
        missing.append(name)

    if missing:
        raise RuntimeError(
            "The selected FFmpeg build cannot run artifacts_reductions. "
            "Missing required audio filters: " + ", ".join(missing) + ".\n"
            f"Selected executable: {ffmpeg}\n"
            f"Detected version: {_ffmpeg_version_summary(ffmpeg)}\n"
            "Run install.py to replace an incompatible local build with the "
            "node's supported portable FFmpeg build."
        )


def _build_artifact_filtergraph(total_frames: int, gain_db: float = 0.0) -> str:
    if total_frames < 0:
        raise ValueError("Artifact-reduction frame count cannot be negative.")
    filters = [
        (
            "adeclick="
            f"window={_format_filter_number(ARTIFACT_DECLICK_WINDOW)}:"
            f"overlap={_format_filter_number(ARTIFACT_DECLICK_OVERLAP)}:"
            f"arorder={_format_filter_number(ARTIFACT_DECLICK_ARORDER)}:"
            f"threshold={_format_filter_number(ARTIFACT_DECLICK_THRESHOLD)}:"
            f"burst={_format_filter_number(ARTIFACT_DECLICK_BURST)}:"
            "method=save"
        ),
        (
            "highpass="
            f"frequency={_format_filter_number(ARTIFACT_HIGHPASS_HZ)}:"
            "poles=2:width_type=q:width=0.707:precision=f64"
        ),
        (
            "lowpass="
            f"frequency={_format_filter_number(ARTIFACT_LOWPASS_HZ)}:"
            "poles=2:width_type=q:width=0.707:precision=f64"
        ),
    ]
    if gain_db < -1e-9:
        filters.append(f"volume={gain_db:.8f}dB:precision=double")
    filters.extend(
        [
            f"atrim=end_sample={total_frames}",
            f"apad=whole_len={total_frames}",
            f"atrim=end_sample={total_frames}",
            "asetpts=N/SR/TB",
        ]
    )
    return ",".join(filters)


def _channel_layout_input_args(channel_layout: str) -> list[str]:
    if channel_layout and channel_layout.lower() != "unknown":
        return ["-channel_layout", channel_layout]
    return []


def _parse_loudnorm_true_peak(output: str) -> float:
    candidates = re.findall(r'\{\s*"input_i".*?\}', output, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        value = payload.get("input_tp")
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"-inf", "-infinity"}:
            return -240.0
        try:
            parsed = float(normalized)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return parsed
    raise RuntimeError(
        "FFmpeg loudnorm did not return readable true-peak statistics.\n"
        + output[-4000:]
    )


def _measure_true_peak(
    ffmpeg: str,
    source_path: Path,
    filtergraph: str,
    channel_layout: str,
) -> float:
    loudnorm = (
        "loudnorm=I=-24:LRA=7:"
        f"TP={_format_filter_number(ARTIFACT_TRUE_PEAK_TARGET_DBTP)}:"
        "linear=true:print_format=json"
    )
    # loudnorm reports -inf for clips shorter than its analysis window. Appending
    # silence only inside the meter keeps the true peak unchanged while making
    # very short non-empty inputs measurable.
    meter_tail = "apad=pad_dur=1"
    audio_filter = (
        f"{filtergraph},{meter_tail},{loudnorm}"
        if filtergraph
        else f"{meter_tail},{loudnorm}"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        *_channel_layout_input_args(channel_layout),
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-af",
        audio_filter,
        "-f",
        "null",
        "-",
    ]
    result = _run(command, label="artifact true-peak analysis")
    return _parse_loudnorm_true_peak(result.stderr + "\n" + result.stdout)


def _render_artifact_reduction(
    ffmpeg: str,
    source_path: Path,
    output_path: Path,
    total_frames: int,
    channels: int,
    channel_layout: str,
    gain_db: float,
) -> None:
    filtergraph = _build_artifact_filtergraph(total_frames, gain_db)
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        *_channel_layout_input_args(channel_layout),
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-af",
        filtergraph,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_f32le",
        str(output_path),
    ]
    _run(command, label="FFmpeg artifact reduction")


def _verify_artifact_output(
    output_path: Path,
    expected_frames: int,
    expected_channels: int,
) -> int:
    import soundfile as sf

    info = sf.info(str(output_path))
    actual_frames = int(info.frames)
    if int(info.samplerate) != SAMPLE_RATE:
        raise RuntimeError(
            "Artifact reduction changed the sample rate: "
            f"expected {SAMPLE_RATE}, received {info.samplerate}."
        )
    if int(info.channels) != expected_channels:
        raise RuntimeError(
            "Artifact reduction changed the channel count: "
            f"expected {expected_channels}, received {info.channels}."
        )
    if actual_frames != expected_frames:
        raise RuntimeError(
            "Artifact reduction changed the audio duration: "
            f"expected {expected_frames} frames, received {actual_frames}."
        )
    return actual_frames


def _reduce_artifacts(
    ffmpeg: str,
    source_path: Path,
    output_path: Path,
    channel_layout: str,
) -> dict[str, Any]:
    """Apply a preservation-first FFmpeg artifact guard.

    The always-on chain is deliberately narrow: conservative overlap-save
    de-clicking, a 10 Hz high-pass, a 23 kHz low-pass, exact duration repair,
    and file-wide attenuation only when the cleaned signal exceeds -1 dBTP.
    It performs no denoising, gating, declipping, EQ shaping, compression,
    excitation, stereo decorrelation, silence removal, or loudness matching.
    """
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_info = sf.info(str(source_path))
    total_frames = int(source_info.frames)
    channels = int(source_info.channels)
    source_rate = int(source_info.samplerate)
    if source_rate != SAMPLE_RATE:
        raise RuntimeError(
            "Artifact reduction requires the reconstructed 48 kHz WAV: "
            f"received {source_rate} Hz."
        )
    normalized_layout = (
        channel_layout
        if channel_layout and channel_layout.lower() != "unknown"
        else ""
    )

    if total_frames <= 0 or channels <= 0:
        shutil.copy2(source_path, output_path)
        return {
            "enabled": True,
            "status": "empty_passthrough",
            "filters_applied": [],
            "declick_settings": {
                "window": ARTIFACT_DECLICK_WINDOW,
                "overlap": ARTIFACT_DECLICK_OVERLAP,
                "arorder": ARTIFACT_DECLICK_ARORDER,
                "threshold": ARTIFACT_DECLICK_THRESHOLD,
                "burst": ARTIFACT_DECLICK_BURST,
                "method": "save",
            },
            "highpass_hz": ARTIFACT_HIGHPASS_HZ,
            "lowpass_hz": ARTIFACT_LOWPASS_HZ,
            "true_peak_before_dbtp": None,
            "true_peak_target_dbtp": ARTIFACT_TRUE_PEAK_TARGET_DBTP,
            "peak_attenuation_db": 0.0,
            "true_peak_after_dbtp": None,
            "input_frames": total_frames,
            "output_frames": total_frames,
            "duration_preserved": True,
            "channel_layout": normalized_layout or f"{channels} channels",
        }

    _require_artifact_filters(ffmpeg)
    base_filtergraph = _build_artifact_filtergraph(total_frames)
    _print(
        "FFmpeg artifact guard: conservative overlap-save de-click, "
        f"{ARTIFACT_HIGHPASS_HZ:g} Hz high-pass, {ARTIFACT_LOWPASS_HZ:g} Hz low-pass, "
        f"and {ARTIFACT_TRUE_PEAK_TARGET_DBTP:g} dBTP protection"
    )
    true_peak_before = _measure_true_peak(
        ffmpeg,
        source_path,
        base_filtergraph,
        normalized_layout,
    )
    gain_db = min(0.0, ARTIFACT_TRUE_PEAK_TARGET_DBTP - true_peak_before)
    _render_artifact_reduction(
        ffmpeg,
        source_path,
        output_path,
        total_frames,
        channels,
        normalized_layout,
        gain_db,
    )
    output_frames = _verify_artifact_output(output_path, total_frames, channels)
    true_peak_after = _measure_true_peak(ffmpeg, output_path, "", normalized_layout)

    if true_peak_after > ARTIFACT_TRUE_PEAK_TARGET_DBTP + ARTIFACT_TRUE_PEAK_TOLERANCE_DB:
        gain_db += ARTIFACT_TRUE_PEAK_TARGET_DBTP - true_peak_after
        _render_artifact_reduction(
            ffmpeg,
            source_path,
            output_path,
            total_frames,
            channels,
            normalized_layout,
            gain_db,
        )
        output_frames = _verify_artifact_output(output_path, total_frames, channels)
        true_peak_after = _measure_true_peak(ffmpeg, output_path, "", normalized_layout)

    if true_peak_after > ARTIFACT_TRUE_PEAK_TARGET_DBTP + 0.05:
        raise RuntimeError(
            "Artifact reduction could not satisfy the true-peak ceiling: "
            f"measured {true_peak_after:.2f} dBTP after rendering."
        )

    filters_applied = ["adeclick", "highpass", "lowpass"]
    if gain_db < -1e-9:
        filters_applied.append("volume")
    _print(
        "FFmpeg artifact guard complete: "
        f"true peak {true_peak_before:.2f} dBTP -> {true_peak_after:.2f} dBTP, "
        f"file-wide attenuation {max(0.0, -gain_db):.2f} dB, "
        f"duration preserved at {output_frames} frames"
    )
    return {
        "enabled": True,
        "status": "ffmpeg_artifact_guard",
        "filters_applied": filters_applied,
        "declick_settings": {
            "window": ARTIFACT_DECLICK_WINDOW,
            "overlap": ARTIFACT_DECLICK_OVERLAP,
            "arorder": ARTIFACT_DECLICK_ARORDER,
            "threshold": ARTIFACT_DECLICK_THRESHOLD,
            "burst": ARTIFACT_DECLICK_BURST,
            "method": "save",
        },
        "highpass_hz": ARTIFACT_HIGHPASS_HZ,
        "lowpass_hz": ARTIFACT_LOWPASS_HZ,
        "true_peak_before_dbtp": round(true_peak_before, 4),
        "true_peak_target_dbtp": ARTIFACT_TRUE_PEAK_TARGET_DBTP,
        "peak_attenuation_db": round(max(0.0, -gain_db), 4),
        "true_peak_after_dbtp": round(true_peak_after, 4),
        "input_frames": total_frames,
        "output_frames": output_frames,
        "duration_preserved": output_frames == total_frames,
        "channel_layout": normalized_layout or f"{channels} channels",
    }


def _detect_effective_bandwidth(
    frequencies: np.ndarray,
    decibels: np.ndarray,
    sample_rate: int,
) -> tuple[int, int]:
    """Estimate the highest sustained, meaningful frequency in the signal.

    The detector uses active time frames, a high percentile per frequency bin,
    and minimum time occupancy. That avoids treating one isolated FFT spike or
    the codec noise floor as useful bandwidth while still retaining short
    high-frequency transients.
    """
    if decibels.ndim != 2 or not decibels.size or not len(frequencies):
        maximum = max(1_000, int(sample_rate // 2 // 1_000) * 1_000)
        return maximum, maximum

    frame_peaks = np.max(decibels, axis=0)
    reference_peak = float(np.percentile(frame_peaks, 95.0))
    active_threshold = max(reference_peak - 50.0, -105.0)
    active_frames = frame_peaks >= active_threshold
    if int(np.count_nonzero(active_frames)) < 2:
        active_frames = np.ones_like(frame_peaks, dtype=bool)

    active_db = decibels[:, active_frames]
    profile = np.percentile(active_db, 90.0, axis=1)
    meaningful_threshold = max(reference_peak - 65.0, -108.0)
    occupancy = np.mean(active_db >= meaningful_threshold, axis=1)

    bin_width = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 1.0
    neighborhood = max(1, int(round(180.0 / max(bin_width, 1e-9))))
    if neighborhood > 1:
        from scipy.ndimage import maximum_filter1d

        profile = maximum_filter1d(profile, size=neighborhood, mode="nearest")
        occupancy = maximum_filter1d(occupancy, size=neighborhood, mode="nearest")

    minimum_occupancy = max(0.005, 2.0 / max(1, active_db.shape[1]))
    meaningful = (profile >= meaningful_threshold) & (occupancy >= minimum_occupancy)
    meaningful &= frequencies >= 80.0

    if np.any(meaningful):
        detected = float(frequencies[np.flatnonzero(meaningful)[-1]])
    else:
        # Fallback for very quiet or unusual material: use a conservative
        # energy rolloff rather than claiming full Nyquist bandwidth.
        linear_power = np.mean(np.power(10.0, active_db / 10.0), axis=1)
        cumulative = np.cumsum(linear_power)
        if cumulative.size and cumulative[-1] > 0:
            index = int(np.searchsorted(cumulative, cumulative[-1] * 0.9995))
            detected = float(frequencies[min(index, len(frequencies) - 1)])
        else:
            detected = min(8_000.0, sample_rate / 2.0)

    nyquist = float(sample_rate) / 2.0
    detected = float(np.clip(detected, 1_000.0, nyquist))
    maximum_display = max(1_000, int(math.floor(nyquist / 1_000.0)) * 1_000)
    display_limit = int(round(detected / 1_000.0)) * 1_000
    display_limit = int(np.clip(display_limit, 1_000, maximum_display))
    return int(round(detected)), display_limit


def _write_spectrogram(wav_path: Path, output_png: Path, title: str) -> dict[str, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    from scipy.signal import spectrogram as spectrogram_stft

    data, sample_rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    sample_rate = int(sample_rate)
    mono = data.mean(axis=1)
    if len(mono) < 4096:
        raise ValueError("audio too short for a spectrogram")

    frequencies, times, power = spectrogram_stft(
        mono,
        fs=sample_rate,
        nperseg=2048,
        noverlap=1536,
        window="hann",
        mode="magnitude",
    )
    decibels = 20.0 * np.log10(power + 1e-12)
    detected_hz, display_hz = _detect_effective_bandwidth(
        frequencies,
        decibels,
        sample_rate,
    )

    fig, axis = plt.subplots(figsize=(9.6, 3.2), dpi=120)
    image = axis.imshow(
        decibels,
        aspect="auto",
        origin="lower",
        extent=[0.0, len(mono) / sample_rate, 0.0, sample_rate / 2.0],
        cmap="magma",
        vmin=-120.0,
        vmax=-20.0,
    )
    axis.set_ylim(0, display_hz)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Hz")
    axis.set_title(title, fontsize=11)
    fig.colorbar(image, ax=axis, label="dBFS")
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
    _print(
        f"{title}: detected bandwidth {detected_hz} Hz, "
        f"display ceiling {display_hz} Hz (sample rate {sample_rate} Hz)"
    )
    return {
        "sample_rate": sample_rate,
        "detected_hz": detected_hz,
        "display_hz": display_hz,
    }


def _render_spectrogram_pair(before: Path, after: Path, output_dir: Path) -> dict[str, Any]:
    if not before.is_file():
        raise FileNotFoundError(f"Before WAV does not exist: {before}")
    if not after.is_file():
        raise FileNotFoundError(f"After WAV does not exist: {after}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {}
    for side, source, label in (
        ("before", before, "Spectrogram (before)"),
        ("after", after, "Spectrogram (after)"),
    ):
        destination = output_dir / f"spectrogram_{side}.png"
        metrics = _write_spectrogram(source, destination, label)
        result[f"spectrogram_{side}"] = str(destination)
        result[f"{side}_bandwidth"] = metrics
    return result


_STRENGTH_PRESETS = {
    "subtle": {"crossover": "auto", "lowpass_input": 1, "cfg_scale": 1.0},
    "balanced": {"crossover": "3000", "lowpass_input": 1, "cfg_scale": 1.0},
    "full": {"crossover": "full", "lowpass_input": 1, "cfg_scale": 1.0},
}


def _resolve_params(args: argparse.Namespace) -> dict[str, Any]:
    strength = args.strength or "balanced"
    preset = _STRENGTH_PRESETS[strength]
    return {
        "strength": strength,
        "crossover": preset["crossover"] if args.crossover is None else args.crossover,
        "lowpass_input": bool(preset["lowpass_input"] if args.lowpass_input is None else args.lowpass_input),
        "cfg_scale": float(preset["cfg_scale"] if args.cfg_scale is None else args.cfg_scale),
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input WAV does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = _resolve_params(args)
    crossover_setting = params["crossover"]
    _print(
        f"Enhancement strength: {params['strength']} (crossover={crossover_setting}, "
        f"cfg_scale={params['cfg_scale']:g})"
    )

    with tempfile.TemporaryDirectory(prefix="flashsr_private_result_") as temporary:
        raw = Path(temporary) / "flashsr_raw.wav"
        reconstructed = Path(temporary) / "flashsr_reconstructed.wav"
        total_frames, channels, precision_used = _process_audio(
            input_path,
            raw,
            model_dir,
            bool(args.auto_download),
            args.channel_layout or "",
            params["lowpass_input"],
            max(1, int(args.steps)),
            bool(args.fp16),
            int(args.lowpass_cutoff or 0),
            params["cfg_scale"],
        )
        if crossover_setting == "full":
            shutil.copy2(raw, reconstructed)
        else:
            if crossover_setting == "auto":
                crossover = _estimate_crossover(input_path)
            else:
                crossover = int(crossover_setting)
            _print(f"Using FlashSR crossover at {crossover} Hz")
            _blend_audio(
                args.ffmpeg,
                input_path,
                raw,
                reconstructed,
                total_frames,
                channels,
                args.channel_layout or "",
                crossover,
            )

        if bool(args.artifact_reduction):
            artifact_metrics = _reduce_artifacts(
                args.ffmpeg,
                reconstructed,
                output_path,
                args.channel_layout or "",
            )
            artifact_status = "ON"
        else:
            shutil.copy2(reconstructed, output_path)
            artifact_metrics = {
                "enabled": False,
                "status": "disabled",
                "filters_applied": [],
                "declick_settings": {
                    "window": ARTIFACT_DECLICK_WINDOW,
                    "overlap": ARTIFACT_DECLICK_OVERLAP,
                    "arorder": ARTIFACT_DECLICK_ARORDER,
                    "threshold": ARTIFACT_DECLICK_THRESHOLD,
                    "burst": ARTIFACT_DECLICK_BURST,
                    "method": "save",
                },
                "highpass_hz": ARTIFACT_HIGHPASS_HZ,
                "lowpass_hz": ARTIFACT_LOWPASS_HZ,
                "true_peak_before_dbtp": None,
                "true_peak_target_dbtp": ARTIFACT_TRUE_PEAK_TARGET_DBTP,
                "peak_attenuation_db": 0.0,
                "true_peak_after_dbtp": None,
                "input_frames": total_frames,
                "output_frames": total_frames,
                "duration_preserved": True,
                "channel_layout": (args.channel_layout or "") or f"{channels} channels",
            }
            artifact_status = "OFF"

    result: dict[str, Any] = {
        "frames": total_frames,
        "channels": channels,
        "precision_requested": "FP16" if args.fp16 else "FP32",
        "precision_used": precision_used,
        "artifact_reduction": artifact_status,
        "artifact_reduction_metrics": artifact_metrics,
    }
    if args.spectrogram_dir:
        spec_dir = Path(args.spectrogram_dir).resolve()
        result.update(_render_spectrogram_pair(input_path, output_path, spec_dir))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Private FlashSR worker")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render-spectrograms", action="store_true")
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--model-dir")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--channel-layout", default="")
    parser.add_argument("--auto-download", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--strength",
        choices=tuple(_STRENGTH_PRESETS),
        default=None,
        help="subtle | balanced | full; maps to crossover/cfg defaults unless overridden",
    )
    parser.add_argument(
        "--crossover",
        default=None,
        help="auto | full | fixed low-pass/high-pass crossover in Hz (default: strength preset)",
    )
    parser.add_argument(
        "--lowpass-input", type=int, choices=(0, 1), default=None,
        help="1 = let FlashSR lowpass its input (default: strength preset)",
    )
    parser.add_argument(
        "--lowpass-cutoff",
        type=int,
        default=0,
        help="0 = automatic per-chunk cutoff detection inside FlashSR",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="classifier-free guidance scale (default: strength preset)",
    )
    parser.add_argument("--steps", type=int, default=1)
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--fp16",
        action="store_true",
        help="use CUDA FP16 autocast (FP32 is the default)",
    )
    precision_group.add_argument(
        "--fp32",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--artifact-reduction",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 = preservation-first FFmpeg artifact guard after FlashSR (default)",
    )
    parser.add_argument(
        "--spectrogram-dir",
        default=None,
        help="write spectrogram_before.png and spectrogram_after.png into this directory",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.check:
            _check_runtime()
            return 0
        if args.render_spectrograms:
            missing = [name for name in ("before", "after", "spectrogram_dir") if not getattr(args, name)]
            if missing:
                raise ValueError(
                    "Missing required spectrogram arguments: " + ", ".join(missing)
                )
            result = _render_spectrogram_pair(
                Path(args.before).resolve(),
                Path(args.after).resolve(),
                Path(args.spectrogram_dir).resolve(),
            )
            _print("RESULT " + json.dumps(result, sort_keys=True))
            return 0
        missing = [name for name in ("input", "output", "model_dir", "ffmpeg") if not getattr(args, name)]
        if missing:
            raise ValueError("Missing required worker arguments: " + ", ".join(missing))
        result = _execute(args)
        _print("RESULT " + json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        _print(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
