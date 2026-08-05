from __future__ import annotations

import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import soundfile as sf

import flashsr_worker as worker


class ArtifactReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = shutil.which("ffmpeg")
        if not cls.ffmpeg:
            raise unittest.SkipTest("FFmpeg is not installed")
        try:
            worker._require_artifact_filters(cls.ffmpeg)
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="artifact_guard_test_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, name: str, data: np.ndarray) -> Path:
        path = self.root / name
        sf.write(path, np.asarray(data, dtype=np.float32), worker.SAMPLE_RATE, subtype="FLOAT")
        return path

    def _render(
        self,
        data: np.ndarray,
        *,
        layout: str,
        gain_db: float = 0.0,
        name: str = "source.wav",
    ) -> tuple[Path, np.ndarray]:
        source = self._write(name, data)
        destination = self.root / (Path(name).stem + "_rendered.wav")
        channels = 1 if data.ndim == 1 else int(data.shape[1])
        frames = int(data.shape[0])
        worker._render_artifact_reduction(
            self.ffmpeg,
            source,
            destination,
            frames,
            channels,
            layout,
            gain_db,
        )
        rendered, sample_rate = sf.read(destination, dtype="float32", always_2d=True)
        self.assertEqual(sample_rate, worker.SAMPLE_RATE)
        return destination, rendered

    def test_filter_listing_parser_accepts_variable_flag_width(self) -> None:
        listing = (
            "Filters:\r\n"
            " TS. adeclick          A->A       Remove impulsive noise.\r\n"
            " TSCN highpass         A->A       Apply a high-pass filter.\r\n"
            " .... volume           A->A       Change input volume.\r\n"
            "  A = Audio input/output\r\n"
        )
        names = worker._parse_ffmpeg_filter_names(listing)
        self.assertEqual(names, frozenset({"adeclick", "highpass", "volume"}))

    def test_filter_requirement_uses_help_fallback_when_table_format_is_unknown(self) -> None:
        with (
            mock.patch.object(worker, "_ffmpeg_filter_names", return_value=frozenset()),
            mock.patch.object(worker, "_ffmpeg_filter_help_available", return_value=True),
        ):
            worker._require_artifact_filters("ffmpeg-format-changed")

    def test_filtergraph_is_preservation_first(self) -> None:
        graph = worker._build_artifact_filtergraph(worker.SAMPLE_RATE)
        for required in ("adeclick", "highpass", "lowpass", "atrim", "apad", "asetpts"):
            self.assertIn(required, graph)
        for excluded in (
            "afftdn",
            "anlmdn",
            "afwtdn",
            "arnndn",
            "adeclip",
            "agate",
            "anequalizer",
            "deesser",
            "aexciter",
            "asoftclip",
            "alimiter",
            "acompressor",
            "silenceremove",
        ):
            self.assertNotIn(excluded, graph)

    def test_clean_midband_signal_is_not_peak_attenuated(self) -> None:
        frames = worker.SAMPLE_RATE
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE
        source_data = (0.25 * np.sin(2.0 * np.pi * 1000.0 * time)).astype(np.float32)
        source = self._write("clean.wav", source_data)
        destination = self.root / "cleaned.wav"

        metrics = worker._reduce_artifacts(
            self.ffmpeg,
            source,
            destination,
            "mono",
        )
        rendered, _ = sf.read(destination, dtype="float32")
        center = slice(4800, -4800)
        source_rms = float(np.sqrt(np.mean(np.square(source_data[center], dtype=np.float64))))
        output_rms = float(np.sqrt(np.mean(np.square(rendered[center], dtype=np.float64))))

        self.assertEqual(metrics["status"], "ffmpeg_artifact_guard")
        self.assertEqual(metrics["peak_attenuation_db"], 0.0)
        self.assertNotIn("volume", metrics["filters_applied"])
        self.assertAlmostEqual(output_rms / source_rms, 1.0, delta=0.01)
        self.assertEqual(len(rendered), frames)

    def test_legitimate_transients_are_not_rewritten_by_declick(self) -> None:
        frames = worker.SAMPLE_RATE
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE
        signal = 0.03 * np.sin(2.0 * np.pi * 220.0 * time)
        for position in (8000, 16000, 26000, 36000):
            length = min(3000, frames - position)
            local_time = np.arange(length, dtype=np.float64) / worker.SAMPLE_RATE
            signal[position : position + length] += (
                0.8
                * np.exp(-local_time * 40.0)
                * np.sin(2.0 * np.pi * (1800.0 + 700.0 * local_time) * local_time)
            )
        stereo = np.stack((signal, signal * 0.9), axis=1).astype(np.float32)
        source = self._write("transients.wav", stereo)

        full_output = self.root / "transients_full.wav"
        worker._render_artifact_reduction(
            self.ffmpeg,
            source,
            full_output,
            frames,
            2,
            "stereo",
            0.0,
        )

        reference_output = self.root / "transients_reference.wav"
        reference_graph = (
            "highpass=frequency=10:poles=2:width_type=q:width=0.707:precision=f64,"
            "lowpass=frequency=23000:poles=2:width_type=q:width=0.707:precision=f64,"
            f"atrim=end_sample={frames},apad=whole_len={frames},"
            f"atrim=end_sample={frames},asetpts=N/SR/TB"
        )
        worker._run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-af",
                reference_graph,
                "-ar",
                str(worker.SAMPLE_RATE),
                "-ac",
                "2",
                "-c:a",
                "pcm_f32le",
                str(reference_output),
            ],
            label="transient reference render",
        )

        full, _ = sf.read(full_output, dtype="float32", always_2d=True)
        reference, _ = sf.read(reference_output, dtype="float32", always_2d=True)
        self.assertEqual(full.shape, reference.shape)
        self.assertLess(float(np.max(np.abs(full - reference))), 1e-7)

    def test_isolated_impulse_is_repaired_without_changing_clean_reference(self) -> None:
        frames = worker.SAMPLE_RATE
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE
        clean = (0.10 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
        clicked = clean.copy()
        impulse_index = frames // 2
        clicked[impulse_index] = 1.0

        _, clean_rendered = self._render(clean, layout="mono", name="reference.wav")
        _, click_rendered = self._render(clicked, layout="mono", name="clicked.wav")
        residual = np.abs(click_rendered[:, 0] - clean_rendered[:, 0])

        self.assertLess(float(residual[impulse_index]), 1e-6)
        self.assertLess(float(np.max(residual)), 1e-5)

    def test_boundary_filters_preserve_midband_and_reject_extremes(self) -> None:
        frames = worker.SAMPLE_RATE * 2
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE

        ratios: dict[float, float] = {}
        for frequency in (5.0, 1000.0, 23900.0):
            signal = (0.10 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
            _, rendered = self._render(signal, layout="mono", name=f"tone_{frequency:g}.wav")
            center = slice(12000, -12000)
            input_rms = float(np.sqrt(np.mean(np.square(signal[center], dtype=np.float64))))
            output_rms = float(
                np.sqrt(np.mean(np.square(rendered[center, 0], dtype=np.float64)))
            )
            ratios[frequency] = output_rms / max(input_rms, 1e-12)

        self.assertLess(ratios[5.0], 0.35)
        self.assertAlmostEqual(ratios[1000.0], 1.0, delta=0.02)
        self.assertLess(ratios[23900.0], 0.35)

    def test_very_short_audio_has_a_real_true_peak_measurement(self) -> None:
        signal = np.asarray([0.0, 0.25, -0.5, 0.25, 0.0], dtype=np.float32)
        source = self._write("very_short.wav", signal)
        measured = worker._measure_true_peak(self.ffmpeg, source, "", "mono")
        self.assertGreater(measured, -20.0)
        self.assertLess(measured, 0.0)

    def test_silence_remains_silence_and_preserves_length(self) -> None:
        frames = worker.SAMPLE_RATE // 2
        source = self._write("silence.wav", np.zeros(frames, dtype=np.float32))
        destination = self.root / "silence_out.wav"
        metrics = worker._reduce_artifacts(
            self.ffmpeg,
            source,
            destination,
            "mono",
        )
        rendered, sample_rate = sf.read(destination, dtype="float32")

        self.assertEqual(sample_rate, worker.SAMPLE_RATE)
        self.assertEqual(len(rendered), frames)
        self.assertEqual(float(np.max(np.abs(rendered))), 0.0)
        self.assertEqual(metrics["true_peak_before_dbtp"], -240.0)
        self.assertEqual(metrics["true_peak_after_dbtp"], -240.0)
        self.assertEqual(metrics["peak_attenuation_db"], 0.0)

    def test_hot_signal_is_reduced_to_true_peak_target(self) -> None:
        frames = worker.SAMPLE_RATE
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE
        signal = (1.15 * np.sin(2.0 * np.pi * 997.0 * time)).astype(np.float32)
        source = self._write("hot.wav", signal)
        destination = self.root / "hot_cleaned.wav"

        metrics = worker._reduce_artifacts(
            self.ffmpeg,
            source,
            destination,
            "mono",
        )
        info = sf.info(destination)

        self.assertGreater(metrics["peak_attenuation_db"], 0.0)
        self.assertIn("volume", metrics["filters_applied"])
        self.assertLessEqual(metrics["true_peak_after_dbtp"], -0.95)
        self.assertEqual(info.frames, frames)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.samplerate, worker.SAMPLE_RATE)

    def test_long_stereo_file_preserves_exact_frame_count(self) -> None:
        frames = worker.SAMPLE_RATE * 12 + 137
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE
        stereo = np.stack(
            (
                0.12 * np.sin(2.0 * np.pi * 173.0 * time),
                0.10 * np.sin(2.0 * np.pi * 997.0 * time),
            ),
            axis=1,
        ).astype(np.float32)
        source = self._write("long_stereo.wav", stereo)
        destination = self.root / "long_stereo_out.wav"

        metrics = worker._reduce_artifacts(
            self.ffmpeg,
            source,
            destination,
            "stereo",
        )
        info = sf.info(destination)

        self.assertEqual(info.frames, frames)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.samplerate, worker.SAMPLE_RATE)
        self.assertTrue(metrics["duration_preserved"])
        self.assertEqual(metrics["input_frames"], frames)
        self.assertEqual(metrics["output_frames"], frames)

    def test_duration_and_channel_isolation_for_common_layouts(self) -> None:
        frames = 12000
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE
        for channels, layout in ((1, "mono"), (2, "stereo"), (6, "5.1"), (8, "7.1")):
            data = np.zeros((frames, channels), dtype=np.float32)
            data[:, 0] = 0.1 * np.sin(2.0 * np.pi * 600.0 * time)
            destination, rendered = self._render(
                data,
                layout=layout,
                name=f"layout_{channels}.wav",
            )
            info = sf.info(destination)
            self.assertEqual(info.frames, frames)
            self.assertEqual(info.channels, channels)
            if channels > 1:
                self.assertLess(float(np.max(np.abs(rendered[:, 1:]))), 1e-7)

    def test_full_guard_handles_7_1_without_cross_channel_leakage(self) -> None:
        frames = 12000
        time = np.arange(frames, dtype=np.float64) / worker.SAMPLE_RATE
        data = np.zeros((frames, 8), dtype=np.float32)
        data[:, 0] = 0.1 * np.sin(2.0 * np.pi * 600.0 * time)
        source = self._write("full_7_1.wav", data)
        destination = self.root / "full_7_1_out.wav"

        metrics = worker._reduce_artifacts(
            self.ffmpeg,
            source,
            destination,
            "7.1",
        )
        rendered, sample_rate = sf.read(destination, dtype="float32", always_2d=True)

        self.assertEqual(sample_rate, worker.SAMPLE_RATE)
        self.assertEqual(rendered.shape, data.shape)
        self.assertLess(float(np.max(np.abs(rendered[:, 1:]))), 1e-7)
        self.assertEqual(metrics["channel_layout"], "7.1")
        self.assertTrue(metrics["duration_preserved"])

    def test_empty_audio_is_a_lossless_passthrough(self) -> None:
        source = self._write("empty.wav", np.zeros((0, 1), dtype=np.float32))
        destination = self.root / "empty_out.wav"
        metrics = worker._reduce_artifacts(
            self.ffmpeg,
            source,
            destination,
            "mono",
        )
        info = sf.info(destination)

        self.assertEqual(metrics["status"], "empty_passthrough")
        self.assertEqual(info.frames, 0)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.samplerate, worker.SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
