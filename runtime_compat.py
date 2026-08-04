"""Small import adapters for AudioSR's unused legacy optional dependencies.

AudioSR 0.0.7 imports TorchAudio, TorchVision and its large CLAP/AudioMAE encoder
module at import time, even though the basic audio-super-resolution inference path
used by this node does not need those components. Official companion wheels may
not exist for ComfyUI nightly Torch builds, so the private worker installs these
minimal adapters before importing AudioSR.
"""
from __future__ import annotations

import importlib.machinery
import math
import sys
import types
from pathlib import Path
from typing import Any


def _module(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _remove_modules(prefixes: tuple[str, ...]) -> None:
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def _install_torchaudio() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as functional_nn

    _remove_modules(("torchaudio",))

    torchaudio = _module("torchaudio", package=True)
    functional = _module("torchaudio.functional")
    transforms = _module("torchaudio.transforms")
    compliance = _module("torchaudio.compliance", package=True)
    kaldi = _module("torchaudio.compliance.kaldi")

    def resample(
        waveform: torch.Tensor,
        orig_freq: float,
        new_freq: float,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        if float(orig_freq) <= 0 or float(new_freq) <= 0:
            raise ValueError("Sample rates must be positive.")
        if float(orig_freq) == float(new_freq):
            return waveform.clone()
        original_shape = waveform.shape
        if not original_shape:
            raise ValueError("Waveform must have at least one dimension.")
        output_length = max(1, int(round(original_shape[-1] * float(new_freq) / float(orig_freq))))
        input_dtype = waveform.dtype
        working = waveform.reshape(-1, 1, original_shape[-1])
        if not working.is_floating_point():
            working = working.float()
        output = functional_nn.interpolate(
            working,
            size=output_length,
            mode="linear",
            align_corners=False,
        ).reshape(*original_shape[:-1], output_length)
        return output.to(dtype=input_dtype) if waveform.is_floating_point() else output

    class Resample(torch.nn.Module):
        def __init__(self, orig_freq: float, new_freq: float, *args: Any, **kwargs: Any):
            super().__init__()
            del args, kwargs
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return resample(waveform, self.orig_freq, self.new_freq)

    def load(path: str | Path, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, int]:
        del args
        import soundfile as sf

        frame_offset = int(kwargs.pop("frame_offset", 0) or 0)
        num_frames = int(kwargs.pop("num_frames", -1) or -1)
        normalize = bool(kwargs.pop("normalize", True))
        channels_first = bool(kwargs.pop("channels_first", True))
        if kwargs:
            # Ignore backend/format options accepted by TorchAudio but irrelevant to SoundFile.
            kwargs.clear()
        with sf.SoundFile(str(path), mode="r") as source:
            if frame_offset:
                source.seek(frame_offset)
            frames = -1 if num_frames < 0 else num_frames
            data = source.read(frames=frames, dtype="float32", always_2d=True)
            sample_rate = int(source.samplerate)
        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
        if channels_first:
            tensor = tensor.transpose(0, 1).contiguous()
        if not normalize:
            tensor = tensor * 32768.0
        return tensor, sample_rate

    def save(
        path: str | Path,
        source: torch.Tensor,
        sample_rate: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        import soundfile as sf

        tensor = source.detach().cpu()
        if tensor.ndim == 1:
            data = tensor.numpy()
        elif tensor.ndim == 2:
            data = tensor.transpose(0, 1).numpy()
        else:
            raise ValueError("Audio tensor must be one- or two-dimensional.")
        sf.write(str(path), data, int(sample_rate))

    def fbank(
        waveform: torch.Tensor,
        *,
        sample_frequency: float = 16_000,
        num_mel_bins: int = 128,
        frame_shift: float = 10.0,
        frame_length: float = 25.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        import librosa

        audio = waveform
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        elif audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = audio.float()
        win_length = max(16, int(round(float(sample_frequency) * frame_length / 1000.0)))
        hop_length = max(1, int(round(float(sample_frequency) * frame_shift / 1000.0)))
        n_fft = 1 << max(1, int(math.ceil(math.log2(win_length))))
        window = torch.hann_window(win_length, device=audio.device, dtype=audio.dtype)
        spectrum = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs().pow(2.0)
        mel = librosa.filters.mel(
            sr=int(sample_frequency),
            n_fft=n_fft,
            n_mels=int(num_mel_bins),
            fmin=20.0,
            fmax=float(sample_frequency) / 2.0,
        )
        mel_basis = torch.as_tensor(mel, device=spectrum.device, dtype=spectrum.dtype)
        return torch.log(torch.clamp(mel_basis @ spectrum, min=1e-10)).transpose(0, 1)

    functional.resample = resample  # type: ignore[attr-defined]
    transforms.Resample = Resample  # type: ignore[attr-defined]
    kaldi.fbank = fbank  # type: ignore[attr-defined]
    compliance.kaldi = kaldi  # type: ignore[attr-defined]

    torchaudio.functional = functional  # type: ignore[attr-defined]
    torchaudio.transforms = transforms  # type: ignore[attr-defined]
    torchaudio.compliance = compliance  # type: ignore[attr-defined]
    torchaudio.load = load  # type: ignore[attr-defined]
    torchaudio.save = save  # type: ignore[attr-defined]
    torchaudio.set_audio_backend = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    torchaudio.__version__ = "video-audio-fix-compat"
    torchaudio.__video_audio_fix_compat__ = True

    sys.modules.update(
        {
            "torchaudio": torchaudio,
            "torchaudio.functional": functional,
            "torchaudio.transforms": transforms,
            "torchaudio.compliance": compliance,
            "torchaudio.compliance.kaldi": kaldi,
        }
    )


def _install_torchvision() -> None:
    import torch

    _remove_modules(("torchvision",))
    torchvision = _module("torchvision", package=True)
    utils = _module("torchvision.utils")

    def make_grid(tensor: torch.Tensor, nrow: int = 8, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            return tensor
        batch, channels, height, width = tensor.shape
        columns = max(1, min(int(nrow), batch))
        rows = int(math.ceil(batch / columns))
        grid = tensor.new_zeros((channels, rows * height, columns * width))
        for index in range(batch):
            row, column = divmod(index, columns)
            grid[:, row * height : (row + 1) * height, column * width : (column + 1) * width] = tensor[index]
        return grid

    utils.make_grid = make_grid  # type: ignore[attr-defined]
    torchvision.utils = utils  # type: ignore[attr-defined]
    torchvision.__version__ = "video-audio-fix-compat"
    torchvision.__video_audio_fix_compat__ = True
    sys.modules.update({"torchvision": torchvision, "torchvision.utils": utils})


def _install_encoder_module() -> None:
    """Replace AudioSR's unused all-purpose encoder module with its basic SR subset."""
    import torch

    name = "audiosr.latent_diffusion.modules.encoders.modules"
    sys.modules.pop(name, None)
    module = _module(name)

    def disabled_train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        del mode
        return self

    class VAEFeatureExtract(torch.nn.Module):
        def __init__(self, first_stage_config: Any):
            super().__init__()
            from audiosr.latent_diffusion.util import instantiate_from_config

            self.vae = instantiate_from_config(first_stage_config)
            self.vae.eval()
            for parameter in self.vae.parameters():
                parameter.requires_grad = False
            self.vae.train = disabled_train.__get__(self.vae, type(self.vae))
            self.device = None
            self.unconditional_cond = None

        def get_unconditional_condition(self, batchsize: int) -> torch.Tensor:
            if self.unconditional_cond is None:
                raise RuntimeError("AudioSR conditioning has not been initialized.")
            return self.unconditional_cond.unsqueeze(0).expand(batchsize, -1, -1, -1)

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            if self.device is None:
                self.device = next(self.vae.parameters()).device
            with torch.no_grad():
                vae_embed = self.vae.encode(batch.unsqueeze(1)).sample()
            self.unconditional_cond = -11.4981 + vae_embed[0].detach().clone() * 0.0
            return vae_embed.detach()

    class CLAPAudioEmbeddingClassifierFreev2(torch.nn.Module):
        """No-op evaluator: AudioSR basic inference never uses CLAP embeddings."""

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__()
            del args, kwargs

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            del args, kwargs
            return torch.empty(0)

    VAEFeatureExtract.__module__ = name
    CLAPAudioEmbeddingClassifierFreev2.__module__ = name
    module.disabled_train = disabled_train  # type: ignore[attr-defined]
    module.VAEFeatureExtract = VAEFeatureExtract  # type: ignore[attr-defined]
    module.CLAPAudioEmbeddingClassifierFreev2 = CLAPAudioEmbeddingClassifierFreev2  # type: ignore[attr-defined]
    module.__all__ = ["VAEFeatureExtract", "CLAPAudioEmbeddingClassifierFreev2"]
    module.__video_audio_fix_compat__ = True
    sys.modules[name] = module


def install_audio_sr_compatibility() -> None:
    """Install import adapters before the first AudioSR import in this process."""
    _install_torchaudio()
    _install_torchvision()
    _install_encoder_module()
