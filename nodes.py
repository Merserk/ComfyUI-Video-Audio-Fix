from __future__ import annotations

import io as pyio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, io, ui
from typing_extensions import override

LOGGER = logging.getLogger("ComfyUI-Video-Audio-Fix")

SAMPLE_RATE = 48_000
OUTPUT_FORMATS = ("MP4", "MKV")
PRECISION_OPTIONS = ("FP32", "FP16")
ARTIFACT_REDUCTION_OPTIONS = ("On", "Off")


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

    installer = Path(__file__).with_name("install.py")
    raise RuntimeError(
        "Video Audio Fix: FFmpeg and FFprobe are unavailable.\n"
        "Run the node installer; it downloads private portable binaries automatically:\n"
        f'"{sys.executable}" "{installer}"\n'
        "Then restart ComfyUI. A system PATH entry is not required."
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
    return Path(folder_paths.models_dir) / "flashsr"


def _private_python() -> Path:
    root = Path(__file__).resolve().parent
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        installer = root / "install.py"
        raise RuntimeError(
            "Video Audio Fix: the private FlashSR environment is not installed.\n"
            "Run the installer with ComfyUI's Python:\n"
            f'"{sys.executable}" "{installer}"\n'
            "All FlashSR packages will be installed only inside this node's .venv."
        )
    return python


def _run_flashsr_worker(
    extracted: Path,
    restored: Path,
    auto_download: bool,
    channel_layout: str,
    ffmpeg: str,
    strength: str,
    precision: str,
    artifacts_reductions: bool,
) -> dict[str, Any]:
    worker = Path(__file__).with_name("flashsr_worker.py")
    if not worker.is_file():
        raise RuntimeError(f"Video Audio Fix: missing private worker: {worker}")

    command = [
        str(_private_python()),
        str(worker),
        "--input",
        str(extracted),
        "--output",
        str(restored),
        "--model-dir",
        str(_model_path()),
        "--ffmpeg",
        ffmpeg,
        "--channel-layout",
        channel_layout,
        "--auto-download",
        "1" if auto_download else "0",
        "--strength",
        strength,
        "--artifact-reduction",
        "1" if artifacts_reductions else "0",
    ]
    if precision == "FP16":
        command.append("--fp16")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        env=environment,
        bufsize=1,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("Video Audio Fix: could not read the private FlashSR worker output.")

    progress = None
    progress_value = 0
    result: dict[str, Any] | None = None
    output_tail: list[str] = []
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        output_tail.append(line)
        if len(output_tail) > 120:
            output_tail.pop(0)
        if line.startswith("TOTAL "):
            try:
                from comfy.utils import ProgressBar

                progress = ProgressBar(max(1, int(line.split(maxsplit=1)[1])))
            except Exception:
                progress = None
            continue
        if line.startswith("PROGRESS "):
            try:
                current = int(line.split(maxsplit=1)[1])
                if progress is not None and current > progress_value:
                    progress.update(current - progress_value)
                progress_value = current
            except (TypeError, ValueError):
                pass
            continue
        if line.startswith("RESULT "):
            try:
                result = json.loads(line[len("RESULT "):])
            except json.JSONDecodeError:
                result = None
            continue
        LOGGER.info("FlashSR worker: %s", line)

    return_code = process.wait()
    if return_code != 0 or result is None:
        installer = Path(__file__).with_name("install.py")
        detail = "\n".join(output_tail[-80:]) or "No worker diagnostic was returned."
        raise RuntimeError(
            "Video Audio Fix: private FlashSR processing failed.\n"
            f"{detail}\n\n"
            "To rebuild the isolated runtime, delete the node's .venv folder and run:\n"
            f'"{sys.executable}" "{installer}"'
        )

    try:
        frames = int(result["frames"])
        channels = int(result["channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Video Audio Fix: private worker returned an invalid result.") from exc
    if frames <= 0 or channels <= 0 or not restored.is_file():
        raise RuntimeError("Video Audio Fix: private worker did not create valid restored audio.")
    return result


def _render_spectrograms(before: Path, after: Path, output_dir: Path) -> dict[str, Any]:
    worker = Path(__file__).with_name("flashsr_worker.py")
    result = _run(
        [
            str(_private_python()),
            str(worker),
            "--render-spectrograms",
            "--before",
            str(before),
            "--after",
            str(after),
            "--spectrogram-dir",
            str(output_dir),
        ],
        label="spectrogram rendering",
    )
    payload: dict[str, Any] | None = None
    for line in result.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                payload = json.loads(line[len("RESULT "):])
            except json.JSONDecodeError:
                payload = None
        elif line.strip():
            LOGGER.info("FlashSR worker: %s", line.strip())
    if payload is None:
        raise RuntimeError("Video Audio Fix: spectrogram worker returned no valid result.")
    return payload


def _output_profile(output_format: str) -> tuple[str, str]:
    if output_format == "MP4":
        return "mp4", "aac"
    if output_format == "MKV":
        return "mkv", "pcm_s16le"
    raise RuntimeError(
        f"Video Audio Fix: unknown output format '{output_format}' "
        f"(expected {', '.join(OUTPUT_FORMATS)})."
    )


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
    output_format: str,
    source_probe: dict[str, Any],
    selected_audio: dict[str, Any],
) -> None:
    audio_streams = [s for s in source_probe.get("streams", []) if s.get("codec_type") == "audio"]
    selected_global_index = int(selected_audio["index"])
    output_audio_index = max(0, len(audio_streams) - 1)
    extension, encoder = _output_profile(output_format)
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
        f"-ar:a:{output_audio_index}",
        str(SAMPLE_RATE),
    ]
    if output_format == "MP4":
        command += [f"-b:a:{output_audio_index}", "320k", "-aac_coder", "twoloop"]

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
    try:
        _run(command, label="final video merge")
    except RuntimeError as exc:
        if output_format == "MP4":
            raise RuntimeError(
                "Video Audio Fix: MP4 could not copy one or more source streams. "
                "Choose MKV for broader stream compatibility.\n"
                f"{exc}"
            ) from exc
        raise


def _validate_output(
    ffprobe: str,
    source_probe: dict[str, Any],
    output: Path,
    output_format: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    restored_stream = audio[-1]
    if int(restored_stream.get("sample_rate") or 0) != SAMPLE_RATE:
        raise RuntimeError("Video Audio Fix: validation found an invalid restored audio sample rate.")
    format_name = str(output_probe.get("format", {}).get("format_name") or "")
    expected_codec = "aac" if output_format == "MP4" else "pcm_s16le"
    if str(restored_stream.get("codec_name") or "") != expected_codec:
        raise RuntimeError(
            f"Video Audio Fix: expected {expected_codec} audio but FFmpeg produced "
            f"{restored_stream.get('codec_name') or 'an unknown codec'}."
        )
    if output_format == "MP4":
        if "mp4" not in format_name and "mov" not in format_name:
            raise RuntimeError("Video Audio Fix: validation found a non-MP4 output container.")
    else:
        if "matroska" not in format_name:
            raise RuntimeError("Video Audio Fix: validation found a non-Matroska output container.")
        bits = int(
            restored_stream.get("bits_per_raw_sample")
            or restored_stream.get("bits_per_sample")
            or 0
        )
        if bits and bits != 16:
            raise RuntimeError(
                f"Video Audio Fix: expected 16-bit PCM audio but FFprobe reported {bits}-bit."
            )
    return output_probe, restored_stream


def _stack_spectrogram_images(paths: list[Path]) -> torch.Tensor | None:
    images = [Image.open(path).convert("RGB") for path in paths if path.is_file()]
    if not images:
        return None

    resampling = getattr(Image, "Resampling", Image)
    max_width = max(image.width for image in images)
    prepared: list[Image.Image] = []
    total_height = 0
    for image in images:
        current = image
        if image.width != max_width and image.width > 0:
            scaled_height = max(1, round(image.height * (max_width / image.width)))
            current = image.resize((max_width, scaled_height), resampling.LANCZOS)
        prepared.append(current)
        total_height += current.height

    combined = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    offset_y = 0
    for image in prepared:
        combined.paste(image, (0, offset_y))
        offset_y += image.height

    array = np.asarray(combined, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class AudioVideoFix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioVideoFix",
            display_name="Video Audio Fix",
            category="video/audio",
            description=(
                "Restores the default audio track with FlashSR (one-step audio "
                "super-resolution), copies the video stream without re-encoding, "
                "saves the result, previews the video inline, and outputs a stacked "
                "before/after spectrogram image."
            ),
            is_output_node=True,
            inputs=[
                io.Video.Input("video"),
                io.String.Input("filename_prefix", default="video/AudioVideoFix"),
                io.Boolean.Input("auto_download", default=True),
                io.Combo.Input(
                    "strength",
                    options=["Subtle", "Balanced", "Full band"],
                    default="Full band",
                    tooltip=(
                        "How much of the FlashSR-restored spectrum replaces the original audio. "
                        "Subtle keeps the original up to an automatic crossover; Balanced blends "
                        "from 3 kHz up; Full band uses the restored audio across the full spectrum."
                    ),
                ),
                io.Combo.Input(
                    "output_format",
                    options=list(OUTPUT_FORMATS),
                    default="MP4",
                    tooltip=(
                        "MP4 uses 48 kHz AAC at 320 kb/s. MKV uses uncompressed "
                        "48 kHz signed 16-bit PCM audio."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=list(PRECISION_OPTIONS),
                    default="FP32",
                    tooltip=(
                        "FP32 is the default for maximum numerical precision. FP16 uses "
                        "CUDA autocast for faster, lower-memory inference. CPU and MPS use FP32."
                    ),
                ),
                io.Combo.Input(
                    "artifacts_reductions",
                    options=list(ARTIFACT_REDUCTION_OPTIONS),
                    default="On",
                    tooltip=(
                        "Preservation-first FFmpeg cleanup after FlashSR: conservative overlap-save "
                        "de-clicking, 10 Hz high-pass and 23 kHz low-pass filters, exact duration "
                        "preservation, and file-wide attenuation only above -1 dBTP."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.Image.Output("image_spectrogram"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        filename_prefix: str,
        auto_download: bool,
        strength: str,
        output_format: str = "MP4",
        precision: str = "FP32",
        artifacts_reductions: str = "On",
    ) -> io.NodeOutput:
        strength = (strength or "Full band").strip()
        if strength not in ("Subtle", "Balanced", "Full band"):
            raise RuntimeError(
                f"Video Audio Fix: unknown strength '{strength}' (expected Subtle, Balanced, Full band)."
            )
        output_format = (output_format or "MP4").strip().upper()
        if output_format not in OUTPUT_FORMATS:
            raise RuntimeError(
                f"Video Audio Fix: unknown output format '{output_format}' "
                f"(expected {', '.join(OUTPUT_FORMATS)})."
            )
        precision = (precision or "FP32").strip().upper()
        if precision not in PRECISION_OPTIONS:
            raise RuntimeError(
                f"Video Audio Fix: unknown precision '{precision}' "
                f"(expected {', '.join(PRECISION_OPTIONS)})."
            )
        artifacts_reductions = (artifacts_reductions or "On").strip().title()
        if artifacts_reductions not in ARTIFACT_REDUCTION_OPTIONS:
            raise RuntimeError(
                f"Video Audio Fix: unknown Artifacts reductions setting "
                f"'{artifacts_reductions}' (expected {', '.join(ARTIFACT_REDUCTION_OPTIONS)})."
            )
        extension, _ = _output_profile(output_format)
        ffmpeg, ffprobe = _find_media_tools()
        with tempfile.TemporaryDirectory(prefix="comfy_video_audio_fix_") as temporary:
            temp_dir = Path(temporary)
            source, _ = _materialize_video(video, temp_dir)
            source_probe = _probe(ffprobe, source)
            _first_video_stream(source_probe)
            selected_audio = _selected_audio_stream(source_probe)
            channel_layout = str(selected_audio.get("channel_layout") or "")

            extracted = temp_dir / "original_48k.wav"
            restored_final = temp_dir / "flashsr_final.wav"
            _extract_audio(ffmpeg, source, int(selected_audio["index"]), extracted)
            worker_result = _run_flashsr_worker(
                extracted,
                restored_final,
                bool(auto_download),
                channel_layout,
                ffmpeg,
                {"Subtle": "subtle", "Balanced": "balanced", "Full band": "full"}[strength],
                precision,
                artifacts_reductions == "On",
            )
            LOGGER.info(
                "FlashSR precision requested=%s used=%s",
                precision,
                worker_result.get("precision_used", "unknown"),
            )
            LOGGER.info(
                "FlashSR artifact guard requested=%s used=%s",
                artifacts_reductions,
                worker_result.get("artifact_reduction", "unknown"),
            )
            if artifacts_reductions == "On":
                LOGGER.info(
                    "FlashSR artifact guard metrics: %s",
                    worker_result.get("artifact_reduction_metrics", {}),
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
                    output_format,
                    source_probe,
                    selected_audio,
                )
                _, restored_stream = _validate_output(
                    ffprobe,
                    source_probe,
                    partial_path,
                    output_format,
                )

                # Render the "after" graph from the decoded final container,
                # not from the pre-encode FlashSR WAV. This makes AAC bandwidth
                # changes visible while MKV PCM reflects the lossless mux result.
                final_decoded = temp_dir / "final_container_audio.wav"
                _extract_audio(
                    ffmpeg,
                    partial_path,
                    int(restored_stream["index"]),
                    final_decoded,
                )
                _render_spectrograms(extracted, final_decoded, temp_dir)
                os.replace(partial_path, final_path)
            finally:
                partial_path.unlink(missing_ok=True)

            spec_paths: list[Path] = []
            for side in ("before", "after"):
                spec_source = temp_dir / f"spectrogram_{side}.png"
                if not spec_source.is_file():
                    continue
                spec_name = f"{filename}_{counter:05}__spectrogram_{side}.png"
                output_spec_path = Path(full_output_folder) / spec_name
                shutil.move(spec_source, output_spec_path)
                spec_paths.append(output_spec_path)
        output_video = InputImpl.VideoFromFile(str(final_path))
        spectrogram_image = _stack_spectrogram_images(spec_paths)
        media_ui: dict[str, Any] = {
            "video_preview": [ui.SavedResult(final_name, subfolder, io.FolderType.output)],
        }
        if spectrogram_image is None:
            spectrogram_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return io.NodeOutput(output_video, spectrogram_image, ui=media_ui)


class VideoAudioFixExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AudioVideoFix]


async def comfy_entrypoint() -> VideoAudioFixExtension:
    return VideoAudioFixExtension()
