"""Stable research-root paths independent of experiment package depth."""

from __future__ import annotations

from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
