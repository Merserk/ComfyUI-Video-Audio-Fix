# ComfyUI Video Audio Fix

A compact ComfyUI custom node that restores a video's default audio track with
[AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution), merges it
back into the original container, saves the result, and shows the native ComfyUI
video preview.

## Node

**Display name:** `Video Audio Fix`  
**Internal class:** `AudioVideoFix`

Inputs/widgets:

- `video` — ComfyUI `VIDEO`
- `filename_prefix` — output filename prefix
- `auto_download` — download the official AudioSR basic checkpoint when missing

Output:

- `video` — the processed file-backed ComfyUI `VIDEO`

The node has no quality controls. It automatically uses the AudioSR basic model,
48 kHz output, 50 DDIM steps, guidance scale 3.5, 10.24-second chunks, and smooth
4% overlap for short and long videos.

## Dependency isolation

AudioSR 0.0.7 declares dependency versions that conflict with current ComfyUI
installations. This repository therefore runs AudioSR in a separate Python
process using:

```text
ComfyUI/custom_nodes/ComfyUI-Video-Audio-Fix/.venv/
```

`requirements.txt` is intentionally empty so ComfyUI Manager does not install
AudioSR packages into ComfyUI's Python. `install.py` installs the private package
set from `requirements-venv.txt` only into `.venv`.

The private environment is created with `--system-site-packages` solely to reuse
ComfyUI's existing Torch, TorchAudio, TorchVision, and CUDA build. AudioSR,
SoundFile, NumPy, Librosa, Transformers, and the remaining node dependencies are
resolved and pinned inside `.venv`, and inference runs in a subprocess. They are
never imported into ComfyUI's running interpreter.

## What it preserves

- The original container/extension for file-backed inputs
- The encoded video stream through FFmpeg stream copy (`-c:v copy` via `-c copy`)
- Resolution, frame rate, pixel format, chapters, metadata, subtitles, and other
  audio tracks where the original container supports them
- The selected audio track's language/title/default disposition
- Audio duration and start offset

The restored audio is a newly generated waveform, so it must be encoded into a
codec accepted by the original container. The video itself is not re-encoded.

## Installation

1. Place or clone the repository at:

   ```text
   ComfyUI/custom_nodes/ComfyUI-Video-Audio-Fix
   ```

2. Run `install.py` with the Python used by ComfyUI.

   Windows Portable example:

   ```bat
   C:\Portable\AI\ComfyUI_windows_portable\python_embeded\python.exe C:\Portable\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Video-Audio-Fix\install.py
   ```

   Standard installation example:

   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-Video-Audio-Fix
   python install.py
   ```

   The installer will:

   1. Create `.venv` inside this repository.
   2. Install all AudioSR/node packages into that `.venv` only.
   3. Verify that the private worker can import AudioSR and reuse ComfyUI's Torch.
   4. Download and validate a private portable FFmpeg and FFprobe pair.

   Portable binaries are stored at:

   ```text
   ComfyUI/custom_nodes/ComfyUI-Video-Audio-Fix/bin/
   ```

   No administrator access or system `PATH` change is required.

3. Restart ComfyUI.

## Rebuilding the private environment

After updating the repository, or when the private runtime becomes damaged:

1. Close ComfyUI.
2. Delete only this folder:

   ```text
   ComfyUI/custom_nodes/ComfyUI-Video-Audio-Fix/.venv
   ```

3. Run `install.py` again with ComfyUI's Python.
4. Restart ComfyUI.

Do not run `pip install -r requirements-venv.txt` with ComfyUI's Python directly.
The installer deliberately targets the repository-local `.venv`.

## Model

With `auto_download = true`, the official basic checkpoint is downloaded from
`haoheliu/audiosr_basic` to:

```text
ComfyUI/models/audiosr/audiosr_basic/pytorch_model.bin
```

With `auto_download = false`, no network request is made and the file must
already exist at that path.

## Processing

1. Resolve or materialize the input `VIDEO`.
2. Select the default audio track, or the first audio track when no default is set.
3. Extract it as 48 kHz floating-point WAV.
4. Launch the isolated `.venv` AudioSR worker.
5. Restore each channel in overlapping chunks; LFE is passed through.
6. Preserve original low-frequency content and use AudioSR for restored highs with
   an automatically detected crossover.
7. Match the exact source audio sample length.
8. Replace only the selected audio track and copy all video packets unchanged.
9. Validate the result and return a native `PreviewVideo` result.

## Notes

- The first run downloads a large model and loads it into a separate process.
- The AudioSR model is loaded once per node execution, then the private worker exits.
- AudioSR inference is computationally expensive, especially for long or
  multichannel videos.
- If the incoming ComfyUI `VIDEO` exists only as frames/components, ComfyUI must
  encode it once before this node can work with a media stream. The final merge
  still copies that materialized video stream without another video encode.
- Audio super-resolution cannot reliably repair every kind of clipping,
  distortion, or compression artifact.
- A same-container merge can fail when FFmpeg has no compatible audio encoder.
  The node reports the error instead of silently changing the container or
  re-encoding the video.

## Troubleshooting

### Private environment is missing or processing fails

Close ComfyUI, delete the node's `.venv`, and run:

```bat
C:\path\to\ComfyUI_windows_portable\python_embeded\python.exe C:\path\to\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Video-Audio-Fix\install.py
```

The installer performs a final `audiosr_worker.py --check` import test. Read the
last displayed error if verification fails.

### `No module named soundfile`

This error should not occur in the updated architecture because ComfyUI no longer
imports SoundFile. Delete `.venv`, rerun `install.py`, and confirm the private
runtime check succeeds.

### FFmpeg and FFprobe are unavailable

Rerun `install.py`. It downloads both executables through the private `.venv` and
copies them to the node's `bin` folder. Delete `bin` first to force a fresh download.

### Packages were already installed into ComfyUI's Python

Earlier repository versions installed dependencies into the host interpreter.
The updated version does not uninstall them automatically because other custom
nodes may rely on those packages. The new worker ignores those host versions when
a private copy exists and does not import AudioSR into ComfyUI.

## License and attribution

This custom node is MIT licensed. AudioSR is a separate upstream project by
Haohe Liu and collaborators; its code, model, and licenses remain governed by
the upstream project and model repository. Portable FFmpeg binaries are downloaded
at install time rather than distributed in this repository and remain governed by
their respective FFmpeg/build-provider licenses.
