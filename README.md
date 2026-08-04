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

1. Clone this repository into `ComfyUI/custom_nodes`:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Merserk/ComfyUI-Video-Audio-Fix.git
   ```

2. Install the dependencies with the same Python used by ComfyUI:

   ```bash
   cd ComfyUI-Video-Audio-Fix
   python -m pip install -r requirements.txt
   python install.py
   ```

   `install.py` installs `audiosr==0.0.7` with `--no-deps` because the upstream
   package metadata pins old NumPy/Librosa/Transformers versions that can damage a
   modern ComfyUI environment. The compatible runtime dependencies are listed
   separately in this repository's `requirements.txt`.

3. Ensure `ffmpeg` and `ffprobe` are available on `PATH`.

4. Restart ComfyUI.

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
4. Restore each channel with AudioSR in overlapping chunks. LFE is passed through.
5. Preserve original low-frequency content and use AudioSR for restored highs with
   an automatically detected crossover.
6. Match the exact source audio sample length.
7. Replace only the selected audio track and copy all video packets unchanged.
8. Validate the result and return a native `PreviewVideo` result.

## Notes

- The first run downloads a large model and loads it into memory.
- AudioSR inference is computationally expensive, especially for long or
  multichannel videos.
- If the incoming ComfyUI `VIDEO` exists only as frames/components, ComfyUI must
  encode it once before this node can work with a media stream. The final merge
  still copies that materialized video stream without another video encode.
- Audio super-resolution cannot reliably repair every kind of clipping,
  distortion, or compression artifact.
- A same-container merge can fail when the local FFmpeg build has no compatible
  audio encoder. The node reports the error rather than silently changing the
  container or re-encoding the video.

## License and attribution

This custom node is MIT licensed. AudioSR is a separate upstream project by
Haohe Liu and collaborators; its code, model, and licenses remain governed by
the upstream project and model repository.
