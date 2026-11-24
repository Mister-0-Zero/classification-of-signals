"""Spectrum scanner. Hardware and inference are replaceable boundaries."""

import sys
from pathlib import Path

# Direct launches from desktop/ also need the repository's models package.
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

__version__ = "0.1.0"
