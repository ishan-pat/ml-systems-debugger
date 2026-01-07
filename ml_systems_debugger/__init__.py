"""
ML Systems Debugger: Production-grade failure detection and attribution for ML pipelines.

This package provides a modular framework for detecting, attributing, and analyzing
failure modes in distributed training and inference pipelines.

Design Philosophy:
- Failures are first-class signals, not exceptions
- Framework-agnostic (works with JAX, TensorFlow, PyTorch)
- Stateless detectors for reproducibility
- Extensible architecture for new failure modes
"""

__version__ = "0.1.0"

from ml_systems_debugger.core.types import (
    Signal,
    FailureMode,
    DetectionResult,
    AttributionHypothesis,
)
from ml_systems_debugger.core.debugger import MLSystemsDebugger

__all__ = [
    "Signal",
    "FailureMode",
    "DetectionResult",
    "AttributionHypothesis",
    "MLSystemsDebugger",
]

