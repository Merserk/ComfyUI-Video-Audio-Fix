from __future__ import annotations

import inspect
import unittest

import flashsr_worker as worker


class PrecisionDefaultTests(unittest.TestCase):
    def test_worker_parser_defaults_to_fp32(self) -> None:
        args = worker._parser().parse_args([])
        self.assertFalse(args.fp16)
        self.assertFalse(args.fp32)

    def test_fp16_requires_explicit_opt_in(self) -> None:
        args = worker._parser().parse_args(["--fp16"])
        self.assertTrue(args.fp16)
        self.assertFalse(args.fp32)

    def test_legacy_fp32_flag_remains_accepted(self) -> None:
        args = worker._parser().parse_args(["--fp32"])
        self.assertFalse(args.fp16)
        self.assertTrue(args.fp32)

    def test_process_audio_internal_default_is_fp32(self) -> None:
        parameter = inspect.signature(worker._process_audio).parameters["use_fp16"]
        self.assertIs(parameter.default, False)


if __name__ == "__main__":
    unittest.main()
