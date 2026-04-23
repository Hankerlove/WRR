"""Benchmark adapters for WRR."""

from .ovo import convert_ovo_annotations
from .river import convert_river_annotations

__all__ = ["convert_ovo_annotations", "convert_river_annotations"]
