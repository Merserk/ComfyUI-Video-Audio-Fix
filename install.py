"""Install AudioSR itself without its obsolete transitive dependency pins.

Run with the same Python executable used by ComfyUI:
    python install.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys


def main() -> None:
    if importlib.util.find_spec("audiosr") is not None:
        print("AudioSR is already installed.")
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "audiosr==0.0.7"]
    )
    print("AudioSR installed. Install requirements.txt and restart ComfyUI.")


if __name__ == "__main__":
    main()
