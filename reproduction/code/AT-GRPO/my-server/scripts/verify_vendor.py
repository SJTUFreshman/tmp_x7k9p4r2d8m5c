#!/usr/bin/env python3
"""Thin alias for `vendor_sync.py --check` (drift report, writes nothing)."""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--check", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).with_name("vendor_sync.py")), run_name="__main__")
