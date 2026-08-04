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


import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, io, ui
from typing_extensions import override

LOGGER = logging.getLogger("ComfyUI-Video-Audio-Fix")

SAMPLE_RATE = 48_000
MODEL_FILE = "pytorch_model.bin"

_ENCODER_CACHE: set[str] | None = None


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
    return Path(folder_paths.models_dir) / "audiosr" / "audiosr_basic" / MODEL_FILE


def _private_python() -> Path:
    root = Path(__file__).resolve().parent
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        installer = root / "install.py"
        raise RuntimeError(
            "Video Audio Fix: the private AudioSR environment is not installed.\n"
            "Run the installer with ComfyUI's Python:\n"
            f'"{sys.executable}" "{installer}"\n'
            "All AudioSR packages will be installed only inside this node's .venv."
        )
    return python


def _run_audiosr_worker(
    extracted: Path,
    restored: Path,
    auto_download: bool,
    channel_layout: str,
    ffmpeg: str,
) -> tuple[int, int]:
    worker = Path(__file__).with_name("audiosr_worker.py")
    if not worker.is_file():
        raise RuntimeError(f"Video Audio Fix: missing private worker: {worker}")

    command = [
        str(_private_python()),
        str(worker),
        "--input",
        str(extracted),
        "--output",
        str(restored),
        "--model",
        str(_model_path()),
        "--ffmpeg",
        ffmpeg,
        "--channel-layout",
        channel_layout,
        "--auto-download",
        "1" if auto_download else "0",
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

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
        raise RuntimeError("Video Audio Fix: could not read the private AudioSR worker output.")

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
        LOGGER.info("AudioSR worker: %s", line)

    return_code = process.wait()
    if return_code != 0 or result is None:
        installer = Path(__file__).with_name("install.py")
        detail = "\n".join(output_tail[-80:]) or "No worker diagnostic was returned."
        raise RuntimeError(
            "Video Audio Fix: private AudioSR processing failed.\n"
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
    return frames, channels


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
            restored_final = temp_dir / "audiosr_final.wav"
            _extract_audio(ffmpeg, source, int(selected_audio["index"]), extracted)
            _, processed_channels = _run_audiosr_worker(
                extracted,
                restored_final,
                bool(auto_download),
                channel_layout,
                ffmpeg,
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
