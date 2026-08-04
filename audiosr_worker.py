"""Private AudioSR worker executed by the repository-local .venv.

The ComfyUI node launches this file as a subprocess so AudioSR's dependency set
never enters ComfyUI's interpreter. Communication is file-based WAV input/output
plus line-oriented progress messages on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 48_000
CHUNK_SECONDS = 10.24
OVERLAP_RATIO = 0.04
DDIM_STEPS = 50
GUIDANCE_SCALE = 3.5
BASE_SEED = 42
MODEL_REPO = "haoheliu/audiosr_basic"
MODEL_FILE = "pytorch_model.bin"


def _print(message: str) -> None:
    print(message, flush=True)


def _check_runtime() -> None:
    # Patch aliases before importing AudioSR/torchlibrosa code written for NumPy 1.x.
    aliases = {
        "complex": np.complex128,
        "float": float,
        "int": int,
        "bool": bool,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    import soundfile  # noqa: F401
    import scipy  # noqa: F401
    import librosa  # noqa: F401
    import torch
    import torchaudio
    import torchvision
    import audiosr.pipeline  # noqa: F401

    private_root = Path(sys.prefix).resolve()
    for name, module in (("TorchAudio", torchaudio), ("TorchVision", torchvision)):
        module_file = Path(module.__file__).resolve()
        try:
            module_file.relative_to(private_root)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} was loaded outside the node's private .venv: {module_file}"
            ) from exc

    torch_public = str(torch.__version__).split("+", 1)[0]
    audio_public = str(torchaudio.__version__).split("+", 1)[0]
    if torch_public != audio_public:
        raise RuntimeError(
            f"TorchAudio {torchaudio.__version__} does not match Torch {torch.__version__}."
        )

    _print(
        "Private runtime OK: "
        f"Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"NumPy {np.__version__}, Torch {torch.__version__}, "
        f"TorchAudio {torchaudio.__version__}, TorchVision {torchvision.__version__}"
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


def _ensure_checkpoint(checkpoint: Path, auto_download: bool) -> Path:
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return checkpoint
    if not auto_download:
        raise RuntimeError(
            "AudioSR model is missing and auto_download is disabled.\n"
            f"Place {MODEL_FILE} at:\n{checkpoint}"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download

    _print(f"Downloading AudioSR model to {checkpoint.parent} ...")
    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(checkpoint.parent),
        )
    )
    if downloaded.resolve() != checkpoint.resolve():
        shutil.copy2(downloaded, checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise RuntimeError("Downloaded AudioSR checkpoint is invalid.")
    return checkpoint


def _load_model(checkpoint: Path) -> tuple[Any, Any, str]:
    aliases = {
        "complex": np.complex128,
        "float": float,
        "int": int,
        "bool": bool,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    import torch
    import audiosr.pipeline as pipeline
    import audiosr.latent_diffusion.models.ddpm as ddpm_module

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _print(f"Loading AudioSR basic model on {device} ...")
    config = pipeline.default_audioldm_config("basic")
    config["model"]["params"]["device"] = device

    # The super-resolution path does not use the CLAP evaluator. Avoiding its
    # construction prevents an unnecessary model download and allocation.
    class _UnusedClap(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__()

    original_clap = ddpm_module.CLAPAudioEmbeddingClassifierFreev2
    ddpm_module.CLAPAudioEmbeddingClassifierFreev2 = _UnusedClap
    try:
        model = pipeline.LatentDiffusion(**config["model"]["params"])
    finally:
        ddpm_module.CLAPAudioEmbeddingClassifierFreev2 = original_clap

    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    torch.set_grad_enabled(False)
    return model, pipeline, str(device)


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
    model: Any,
    pipeline: Any,
    chunk: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Run AudioSR directly from an in-memory waveform.

    This bypasses ``torchaudio.load`` so current TorchAudio releases do not need
    TorchCodec merely to reopen the temporary WAV that this worker already read.
    """
    import torch

    pipeline.seed_everything(int(seed))
    waveform = np.asarray(chunk, dtype=np.float32)[None, :]
    batch, duration = pipeline.make_batch_for_super_resolution(None, waveform=waveform)
    with torch.no_grad():
        output = model.generate_batch(
            batch,
            unconditional_guidance_scale=GUIDANCE_SCALE,
            ddim_steps=DDIM_STEPS,
            duration=duration,
        )
    if hasattr(output, "detach"):
        output = output.detach().float().cpu().numpy()
    array = np.asarray(output, dtype=np.float32).reshape(-1)
    if array.size < chunk.size:
        array = np.pad(array, (0, chunk.size - array.size))
    return array[: chunk.size]


def _lfe_channels(channel_layout: str, channel_count: int) -> set[int]:
    layout = channel_layout.lower()
    if ".1" not in layout and "lfe" not in layout:
        return set()
    if channel_count == 3:
        return {2}
    if channel_count >= 4:
        return {3}
    return set()


def _process_audio(
    extracted: Path,
    restored: Path,
    checkpoint: Path,
    auto_download: bool,
    channel_layout: str,
) -> tuple[int, int]:
    import soundfile as sf

    info = sf.info(str(extracted))
    if info.samplerate != SAMPLE_RATE:
        raise RuntimeError("Extracted audio is not 48 kHz.")
    total_frames = int(info.frames)
    channels = int(info.channels)
    if total_frames <= 100:
        shutil.copy2(extracted, restored)
        return total_frames, channels

    checkpoint = _ensure_checkpoint(checkpoint, auto_download)
    model, pipeline, device = _load_model(checkpoint)
    _print(
        f"Restoring {channels} channel(s), {total_frames / SAMPLE_RATE:.2f} seconds, "
        f"AudioSR device {device}"
    )

    chunk_samples = int(round(CHUNK_SECONDS * SAMPLE_RATE))
    overlap_samples = int(round(chunk_samples * OVERLAP_RATIO))
    step_samples = chunk_samples - overlap_samples
    starts = list(range(0, total_frames, step_samples))
    lfe = _lfe_channels(channel_layout, channels)
    total_tasks = len(starts) * max(1, channels - len(lfe))
    _print(f"TOTAL {total_tasks}")
    completed = 0

    with tempfile.TemporaryDirectory(prefix="audiosr_private_work_") as work:
        work_dir = Path(work)
        data_path = work_dir / "restored.float32"
        weights_path = work_dir / "weights.float32"
        output_map = np.memmap(
            data_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, channels),
        )
        output_map[:] = 0.0
        weights = np.memmap(weights_path, mode="w+", dtype=np.float32, shape=(total_frames,))
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
                        processed = _run_chunk(
                            model,
                            pipeline,
                            padded,
                            BASE_SEED + chunk_index,
                        )
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

        del output_map
        del weights

    return total_frames, channels


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
) -> None:
    crossover = _estimate_crossover(original)
    _print(f"Using automatic AudioSR crossover at {crossover} Hz")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(original)]
    if channel_layout and channel_layout.lower() != "unknown":
        command += ["-channel_layout", channel_layout]
    command += ["-i", str(restored)]
    filter_graph = (
        f"[0:a]lowpass=f={crossover}:p=2[low];"
        f"[1:a]highpass=f={crossover}:p=2,lowpass=f=23000:p=1[high];"
        f"[low][high]amix=inputs=2:normalize=0,"
        f"alimiter=limit=0.98:attack=5:release=50,"
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


def _execute(args: argparse.Namespace) -> dict[str, int]:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    checkpoint = Path(args.model).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input WAV does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="audiosr_private_result_") as temporary:
        raw = Path(temporary) / "audiosr_raw.wav"
        total_frames, channels = _process_audio(
            input_path,
            raw,
            checkpoint,
            bool(args.auto_download),
            args.channel_layout or "",
        )
        _blend_audio(
            args.ffmpeg,
            input_path,
            raw,
            output_path,
            total_frames,
            channels,
            args.channel_layout or "",
        )
    return {"frames": total_frames, "channels": channels}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Private AudioSR worker")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--channel-layout", default="")
    parser.add_argument("--auto-download", type=int, choices=(0, 1), default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.check:
            _check_runtime()
            return 0
        missing = [name for name in ("input", "output", "model", "ffmpeg") if not getattr(args, name)]
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
