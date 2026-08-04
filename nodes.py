from __future__ import annotations

import io as pyio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, io, ui
from typing_extensions import override

LOGGER = logging.getLogger("ComfyUI-Video-Audio-Fix")

SAMPLE_RATE = 48_000
CHUNK_SECONDS = 10.24
OVERLAP_RATIO = 0.04
DDIM_STEPS = 50
GUIDANCE_SCALE = 3.5
BASE_SEED = 42
MODEL_REPO = "haoheliu/audiosr_basic"
MODEL_FILE = "pytorch_model.bin"

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.RLock()
_MEDIA_TOOL_LOCK = threading.RLock()
_ENCODER_CACHE: set[str] | None = None


def _soundfile():
    """Import SoundFile with an actionable ComfyUI-Python error message."""
    try:
        import soundfile as sf
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        requirements = Path(__file__).with_name("requirements.txt")
        raise RuntimeError(
            "Video Audio Fix: the Python package 'SoundFile' is missing or failed to load.\n"
            "Install this node's dependencies with the same Python executable used by ComfyUI:\n"
            f'"{sys.executable}" -m pip install -r "{requirements}"\n'
            "Then restart ComfyUI."
        ) from exc
    return sf


def _run(command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "Unknown process error"
        raise RuntimeError(f"Video Audio Fix: {label} failed.\n{detail[-6000:]}")
    return process


def _media_tool_works(path: str | Path | None) -> bool:
    if not path:
        return False
    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        result = subprocess.run(
            [str(candidate), "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _find_media_tools() -> tuple[str, str]:
    suffix = ".exe" if os.name == "nt" else ""
    local_bin = Path(__file__).resolve().parent / "bin"
    local_ffmpeg = local_bin / f"ffmpeg{suffix}"
    local_ffprobe = local_bin / f"ffprobe{suffix}"
    if _media_tool_works(local_ffmpeg) and _media_tool_works(local_ffprobe):
        return str(local_ffmpeg), str(local_ffprobe)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe

    # ComfyUI Manager normally runs install.py. This fallback also makes a
    # manual repository copy self-healing on the first execution.
    try:
        with _MEDIA_TOOL_LOCK:
            # Another execution may have completed the download while waiting.
            if _media_tool_works(local_ffmpeg) and _media_tool_works(local_ffprobe):
                return str(local_ffmpeg), str(local_ffprobe)

            from portable_ffmpeg import get_ffmpeg

            downloaded_ffmpeg, downloaded_ffprobe = get_ffmpeg()
            if _media_tool_works(downloaded_ffmpeg) and _media_tool_works(downloaded_ffprobe):
                try:
                    local_bin.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(downloaded_ffmpeg, local_ffmpeg)
                    shutil.copy2(downloaded_ffprobe, local_ffprobe)
                    if os.name != "nt":
                        local_ffmpeg.chmod(local_ffmpeg.stat().st_mode | 0o111)
                        local_ffprobe.chmod(local_ffprobe.stat().st_mode | 0o111)
                    if _media_tool_works(local_ffmpeg) and _media_tool_works(local_ffprobe):
                        return str(local_ffmpeg), str(local_ffprobe)
                except OSError as copy_exc:
                    LOGGER.warning("Could not cache portable FFmpeg in the node folder: %s", copy_exc)
                return str(downloaded_ffmpeg), str(downloaded_ffprobe)
    except Exception as exc:
        LOGGER.warning("Portable FFmpeg resolution failed: %s", exc)

    installer = Path(__file__).with_name("install.py")
    raise RuntimeError(
        "Video Audio Fix: FFmpeg and FFprobe are unavailable.\n"
        "Run the node installer with the same Python used by ComfyUI:\n"
        f'"{sys.executable}" "{installer}"\n'
        "The installer downloads portable binaries automatically; system PATH is not required."
    )


def _probe(ffprobe: str, path: Path) -> dict[str, Any]:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        label="media inspection",
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Video Audio Fix: FFprobe returned invalid JSON.") from exc


def _first_video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise RuntimeError("Video Audio Fix: the input does not contain a video stream.")
    return stream


def _selected_audio_stream(probe: dict[str, Any]) -> dict[str, Any]:
    audio = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise RuntimeError(
            "Video Audio Fix: the input video does not contain a decodable audio stream."
        )
    return next((s for s in audio if s.get("disposition", {}).get("default") == 1), audio[0])


def _container_extension(path: Path | None, video: Input.Video) -> str:
    if path and path.suffix:
        return path.suffix.lower().lstrip(".")
    try:
        container = str(video.get_container_format()).split(",")[0].lower()
    except Exception:
        container = "mp4"
    return {
        "matroska": "mkv",
        "webm": "webm",
        "mov": "mov",
        "mp4": "mp4",
        "avi": "avi",
        "mpeg": "mpg",
        "mpegts": "ts",
        "ogg": "ogv",
        "flv": "flv",
    }.get(container, container if re.fullmatch(r"[a-z0-9]{2,5}", container) else "mp4")


def _has_active_trim(video: Input.Video) -> bool:
    try:
        start, duration = video.get_active_trim_window()
        return abs(float(start)) > 1e-9 or abs(float(duration)) > 1e-9
    except Exception:
        return False


def _materialize_video(video: Input.Video, temp_dir: Path) -> tuple[Path, str]:
    source: Any = None
    try:
        source = video.get_stream_source()
    except Exception:
        source = None

    source_path = Path(source).resolve() if isinstance(source, (str, os.PathLike)) else None
    extension = _container_extension(source_path, video)

    if source_path and source_path.is_file() and not _has_active_trim(video):
        return source_path, extension

    materialized = temp_dir / f"source.{extension}"
    if isinstance(source, pyio.BytesIO) and not _has_active_trim(video):
        source.seek(0)
        materialized.write_bytes(source.read())
        source.seek(0)
        return materialized, extension

    try:
        video.save_to(str(materialized))
    except Exception as exc:
        if extension != "mp4":
            extension = "mp4"
            materialized = temp_dir / "source.mp4"
            try:
                video.save_to(str(materialized))
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Video Audio Fix: ComfyUI could not materialize the input VIDEO."
                ) from fallback_exc
        else:
            raise RuntimeError(
                "Video Audio Fix: ComfyUI could not materialize the input VIDEO."
            ) from exc
    return materialized, extension


def _extract_audio(
    ffmpeg: str,
    source: Path,
    stream_index: int,
    output: Path,
) -> None:
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            f"0:{stream_index}",
            "-vn",
            "-sn",
            "-dn",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            str(output),
        ],
        label="audio extraction",
    )


def _model_path() -> Path:
    return Path(folder_paths.models_dir) / "audiosr" / "audiosr_basic" / MODEL_FILE


def _ensure_checkpoint(auto_download: bool) -> Path:
    checkpoint = _model_path()
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return checkpoint
    if not auto_download:
        raise RuntimeError(
            "Video Audio Fix: AudioSR model is missing and auto_download is disabled.\n"
            f"Place {MODEL_FILE} at:\n{checkpoint}"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(checkpoint.parent),
        )
    except Exception as exc:
        raise RuntimeError(
            "Video Audio Fix: failed to download the AudioSR basic model from Hugging Face."
        ) from exc
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != checkpoint.resolve():
        shutil.copy2(downloaded_path, checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("Video Audio Fix: the downloaded AudioSR checkpoint is invalid.")
    return checkpoint


def _load_audiosr_model(auto_download: bool) -> tuple[Any, Any, str]:
    checkpoint = _ensure_checkpoint(auto_download)
    # AudioSR 0.0.7 predates NumPy 2.x. Supply removed aliases only when absent.
    legacy_numpy_aliases = {
        "complex": np.complex128,
        "float": float,
        "int": int,
        "bool": bool,
    }
    for alias, value in legacy_numpy_aliases.items():
        if alias not in np.__dict__:
            setattr(np, alias, value)

    try:
        import torch
        import comfy.model_management as model_management
        import audiosr.pipeline as pipeline
        import audiosr.latent_diffusion.models.ddpm as ddpm_module
    except Exception as exc:
        raise RuntimeError(
            "Video Audio Fix: AudioSR dependencies are missing or incompatible. "
            "Run the repository install.py with ComfyUI's Python, then restart ComfyUI."
        ) from exc

    try:
        device = model_management.get_torch_device()
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    cache_key = f"{checkpoint.resolve()}::{device}"
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key], pipeline, str(device)

        LOGGER.info("Loading AudioSR basic model on %s", device)
        config = pipeline.default_audioldm_config("basic")
        config["model"]["params"]["device"] = device

        # AudioSR constructs a CLAP evaluator that is never used by its
        # super-resolution inference path. Skipping it avoids an unnecessary
        # Roberta download and a large extra model allocation.
        class _UnusedClap(torch.nn.Module):
            def __init__(self, *args, **kwargs):
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
        _MODEL_CACHE[cache_key] = model
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


def _run_audiosr_chunk(
    model: Any,
    pipeline: Any,
    chunk: np.ndarray,
    seed: int,
    temp_file: Path,
) -> np.ndarray:
    sf = _soundfile()

    sf.write(str(temp_file), chunk, SAMPLE_RATE, subtype="FLOAT")
    with _INFERENCE_LOCK:
        output = pipeline.super_resolution(
            model,
            str(temp_file),
            seed=seed,
            ddim_steps=DDIM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
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
    auto_download: bool,
    channel_layout: str,
) -> tuple[int, int]:
    sf = _soundfile()

    info = sf.info(str(extracted))
    if info.samplerate != SAMPLE_RATE:
        raise RuntimeError("Video Audio Fix: extracted audio is not 48 kHz.")
    total_frames = int(info.frames)
    channels = int(info.channels)
    if total_frames <= 100:
        shutil.copy2(extracted, restored)
        return total_frames, channels

    model, pipeline, device = _load_audiosr_model(auto_download)
    LOGGER.info(
        "Restoring %d channel(s), %.2f seconds, AudioSR device %s",
        channels,
        total_frames / SAMPLE_RATE,
        device,
    )

    chunk_samples = int(round(CHUNK_SECONDS * SAMPLE_RATE))
    overlap_samples = int(round(chunk_samples * OVERLAP_RATIO))
    step_samples = chunk_samples - overlap_samples
    starts = list(range(0, total_frames, step_samples))
    lfe = _lfe_channels(channel_layout, channels)
    progress = None
    try:
        from comfy.utils import ProgressBar

        progress = ProgressBar(max(1, len(starts) * max(1, channels - len(lfe))))
    except Exception:
        progress = None

    with tempfile.TemporaryDirectory(prefix="audiosr_work_") as work:
        work_dir = Path(work)
        data_path = work_dir / "restored.float32"
        weights_path = work_dir / "weights.float32"
        output_map = np.memmap(
            data_path, mode="w+", dtype=np.float32, shape=(total_frames, channels)
        )
        output_map[:] = 0.0
        weights = np.memmap(weights_path, mode="w+", dtype=np.float32, shape=(total_frames,))
        weights[:] = 0.0

        for start in starts:
            length = min(chunk_samples, total_frames - start)
            window = _chunk_window(
                length,
                overlap_samples,
                has_left=start > 0,
                has_right=start + length < total_frames,
            )
            weights[start : start + length] += window
        weights.flush()

        temp_chunk = work_dir / "chunk.wav"
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
                        processed = _run_audiosr_chunk(
                            model,
                            pipeline,
                            padded,
                            BASE_SEED + chunk_index,
                            temp_chunk,
                        )
                        processed = _match_chunk_level(processed, padded)

                    window = _chunk_window(
                        length,
                        overlap_samples,
                        has_left=start > 0,
                        has_right=start + length < total_frames,
                    )
                    output_map[start : start + length, channel] += processed[:length] * window
                    if progress is not None:
                        progress.update(1)
                    LOGGER.info(
                        "AudioSR channel %d/%d, chunk %d/%d",
                        channel + 1,
                        channels,
                        chunk_index + 1,
                        len(starts),
                    )

                for block_start in range(0, total_frames, 262_144):
                    block_end = min(total_frames, block_start + 262_144)
                    denom = np.maximum(np.asarray(weights[block_start:block_end]), 1e-8)
                    output_map[block_start:block_end, channel] /= denom

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
    sf = _soundfile()

    info = sf.info(str(extracted))
    total = int(info.frames)
    if total < 2048:
        return 8_000
    sample_length = min(SAMPLE_RATE, total)
    positions = np.linspace(0, max(0, total - sample_length), num=min(8, max(1, total // sample_length)), dtype=int)
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
    LOGGER.info("Using automatic AudioSR crossover at %d Hz", crossover)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(original)]
    if channel_layout and channel_layout.lower() != "unknown" and re.fullmatch(
        r"[A-Za-z0-9_().+\-]+", channel_layout
    ):
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


def _available_encoders(ffmpeg: str) -> set[str]:
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None:
        return _ENCODER_CACHE
    result = _run([ffmpeg, "-hide_banner", "-encoders"], label="FFmpeg encoder query")
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.match(r"\s*[A-Z.]{6}\s+([A-Za-z0-9_\-]+)", line)
        if match:
            encoders.add(match.group(1))
    _ENCODER_CACHE = encoders
    return encoders


def _audio_encoder(
    ffmpeg: str,
    extension: str,
    source_codec: str,
    channels: int,
) -> tuple[str, list[str]]:
    available = _available_encoders(ffmpeg)
    source_candidates = {
        "aac": ["aac"],
        "opus": ["libopus", "opus"],
        "vorbis": ["libvorbis", "vorbis"],
        "mp3": ["libmp3lame", "mp3"],
        "flac": ["flac"],
        "alac": ["alac"],
        "ac3": ["ac3"],
        "eac3": ["eac3"],
        "pcm_s16le": ["pcm_s16le"],
        "pcm_s24le": ["pcm_s24le"],
        "pcm_s32le": ["pcm_s32le"],
        "pcm_f32le": ["pcm_f32le"],
    }
    container_candidates = {
        "mp4": ["aac"],
        "m4v": ["aac"],
        "mov": ["alac", "aac"],
        "3gp": ["aac"],
        "webm": ["libopus", "opus"],
        "mkv": ["flac", "libopus", "aac"],
        "mka": ["flac", "libopus", "aac"],
        "avi": ["pcm_s16le", "mp3"],
        "wav": ["pcm_f32le", "pcm_s24le"],
        "flv": ["aac"],
        "ogg": ["libvorbis", "vorbis", "libopus"],
        "ogv": ["libvorbis", "vorbis", "libopus"],
        "ts": ["aac", "ac3"],
        "m2ts": ["aac", "ac3"],
        "mts": ["aac", "ac3"],
        "mpg": ["ac3", "mp2"],
        "mpeg": ["ac3", "mp2"],
    }
    candidates = source_candidates.get(source_codec, []) + container_candidates.get(extension, ["aac"])
    encoder = next((candidate for candidate in candidates if candidate in available), None)
    if encoder is None:
        raise RuntimeError(
            f"Video Audio Fix: no compatible FFmpeg audio encoder is available for .{extension}."
        )

    bitrate = "320k" if channels <= 2 else ("640k" if channels <= 6 else "960k")
    options: list[str] = []
    if encoder == "aac":
        options = ["-b:a", bitrate, "-aac_coder", "twoloop"]
    elif encoder in {"libopus", "opus"}:
        options = ["-b:a", "256k" if channels <= 2 else "512k", "-vbr", "on"]
        if encoder == "libopus":
            options += ["-compression_level", "10"]
    elif encoder in {"libmp3lame", "mp3"}:
        options = ["-b:a", "320k"]
    elif encoder == "flac":
        options = ["-compression_level", "8"]
    elif encoder == "ac3":
        options = ["-b:a", "640k"]
    elif encoder == "eac3":
        options = ["-b:a", "1024k"]
    elif encoder in {"libvorbis", "vorbis"}:
        options = ["-q:a", "8"]
    return encoder, options


def _stream_disposition(stream: dict[str, Any]) -> str:
    disposition = stream.get("disposition", {})
    flags = [
        name
        for name in ("default", "dub", "original", "comment", "lyrics", "karaoke", "forced")
        if disposition.get(name) == 1
    ]
    return "+".join(flags) if flags else "0"


def _mux(
    ffmpeg: str,
    source: Path,
    restored_audio: Path,
    output: Path,
    extension: str,
    source_probe: dict[str, Any],
    selected_audio: dict[str, Any],
    channels: int,
) -> None:
    audio_streams = [s for s in source_probe.get("streams", []) if s.get("codec_type") == "audio"]
    selected_global_index = int(selected_audio["index"])
    output_audio_index = max(0, len(audio_streams) - 1)
    encoder, codec_options = _audio_encoder(
        ffmpeg,
        extension,
        str(selected_audio.get("codec_name", "")),
        channels,
    )
    try:
        start_time = float(selected_audio.get("start_time") or 0.0)
    except (TypeError, ValueError):
        start_time = 0.0

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if abs(start_time) > 1e-9:
        command += ["-itsoffset", f"{start_time:.9f}"]
    command += [
        "-i",
        str(restored_audio),
        "-map",
        "0",
        "-map",
        f"-0:{selected_global_index}",
        "-map",
        "1:a:0",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c",
        "copy",
        f"-c:a:{output_audio_index}",
        encoder,
    ]

    for option, value in zip(codec_options[0::2], codec_options[1::2]):
        if option in {"-b:a", "-q:a"}:
            command += [f"{option}:{output_audio_index}", value]
        else:
            command += [option, value]

    tags = selected_audio.get("tags", {}) or {}
    for key in ("language", "title", "handler_name"):
        value = tags.get(key)
        if value:
            command += [f"-metadata:s:a:{output_audio_index}", f"{key}={value}"]
    command += [
        f"-disposition:a:{output_audio_index}",
        _stream_disposition(selected_audio),
        "-max_muxing_queue_size",
        "4096",
    ]
    if extension in {"mp4", "mov", "m4v"}:
        command += ["-movflags", "+faststart"]
    command.append(str(output))
    _run(command, label="final video merge")


def _validate_output(
    ffprobe: str,
    source_probe: dict[str, Any],
    output: Path,
) -> None:
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("Video Audio Fix: the output video was not created correctly.")
    output_probe = _probe(ffprobe, output)
    source_video = _first_video_stream(source_probe)
    output_video = _first_video_stream(output_probe)
    if source_video.get("codec_name") != output_video.get("codec_name"):
        raise RuntimeError("Video Audio Fix: validation detected an unexpected video codec change.")
    for key in ("width", "height", "pix_fmt"):
        if source_video.get(key) != output_video.get(key):
            raise RuntimeError(f"Video Audio Fix: validation detected a video {key} change.")
    for key in ("avg_frame_rate", "r_frame_rate"):
        source_rate = source_video.get(key)
        output_rate = output_video.get(key)
        if source_rate and output_rate and source_rate != output_rate:
            raise RuntimeError("Video Audio Fix: validation detected a frame-rate change.")
    audio = [s for s in output_probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise RuntimeError("Video Audio Fix: validation found no output audio stream.")
    if int(audio[-1].get("sample_rate") or 0) != SAMPLE_RATE:
        raise RuntimeError("Video Audio Fix: validation found an invalid restored audio sample rate.")


class AudioVideoFix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioVideoFix",
            display_name="Video Audio Fix",
            category="video/audio",
            description=(
                "Restores the default audio track with AudioSR, copies the video stream "
                "without re-encoding, saves the result, and shows a native video preview."
            ),
            is_output_node=True,
            inputs=[
                io.Video.Input("video"),
                io.String.Input("filename_prefix", default="video/AudioVideoFix"),
                io.Boolean.Input("auto_download", default=True),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        filename_prefix: str,
        auto_download: bool,
    ) -> io.NodeOutput:
        ffmpeg, ffprobe = _find_media_tools()
        with tempfile.TemporaryDirectory(prefix="comfy_video_audio_fix_") as temporary:
            temp_dir = Path(temporary)
            source, extension = _materialize_video(video, temp_dir)
            source_probe = _probe(ffprobe, source)
            _first_video_stream(source_probe)
            selected_audio = _selected_audio_stream(source_probe)
            channels = int(selected_audio.get("channels") or 1)
            channel_layout = str(selected_audio.get("channel_layout") or "")

            extracted = temp_dir / "original_48k.wav"
            restored_raw = temp_dir / "audiosr_raw.wav"
            restored_final = temp_dir / "audiosr_final.wav"
            _extract_audio(ffmpeg, source, int(selected_audio["index"]), extracted)
            total_frames, processed_channels = _process_audio(
                extracted,
                restored_raw,
                bool(auto_download),
                channel_layout,
            )
            _blend_audio(
                ffmpeg,
                extracted,
                restored_raw,
                restored_final,
                total_frames,
                processed_channels,
                channel_layout,
            )

            width, height = video.get_dimensions()
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                width,
                height,
            )
            final_name = f"{filename}_{counter:05}_.{extension}"
            final_path = Path(full_output_folder) / final_name
            partial_path = Path(full_output_folder) / f".{filename}_{counter:05}_partial.{extension}"
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _mux(
                    ffmpeg,
                    source,
                    restored_final,
                    partial_path,
                    extension,
                    source_probe,
                    selected_audio,
                    processed_channels,
                )
                _validate_output(ffprobe, source_probe, partial_path)
                os.replace(partial_path, final_path)
            finally:
                partial_path.unlink(missing_ok=True)

        output_video = InputImpl.VideoFromFile(str(final_path))
        preview = ui.PreviewVideo(
            [ui.SavedResult(final_name, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(output_video, ui=preview)


class VideoAudioFixExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AudioVideoFix]


async def comfy_entrypoint() -> VideoAudioFixExtension:
    return VideoAudioFixExtension()
