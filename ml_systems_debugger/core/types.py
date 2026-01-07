"""
Core type definitions for the ML Systems Debugger.

These types define the contract between components and ensure type safety
across the debugging pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class FailureMode(Enum):
    """Enumeration of known failure modes in ML systems."""
    
    # Optimization Pathologies
    EXPLODING_GRADIENTS = "exploding_gradients"
    VANISHING_GRADIENTS = "vanishing_gradients"
    LOSS_PLATEAU = "loss_plateau"
    DIVERGENT_TRAINING = "divergent_training"
    LR_INSTABILITY = "learning_rate_instability"
    
    # Data Pathologies
    TRAINING_SERVING_SKEW = "training_serving_skew"
    DISTRIBUTION_DRIFT = "distribution_drift"
    LABEL_LEAKAGE = "label_leakage"
    FEATURE_COLLAPSE = "feature_collapse"
    
    # Numerical & Systems Issues
    NAN_DETECTED = "nan_detected"
    INF_DETECTED = "inf_detected"
    MIXED_PRECISION_INSTABILITY = "mixed_precision_instability"
    PRECISION_LOSS = "precision_loss"
    MEMORY_FRAGMENTATION = "memory_fragmentation"
    OOM_PRECURSOR = "oom_precursor"
    
    # Distributed Systems Failures
    NON_DETERMINISM = "non_determinism"
    GRADIENT_DESYNC = "gradient_desynchronization"
    STRAGGLER_EFFECT = "straggler_effect"
    PARTIAL_WORKER_FAILURE = "partial_worker_failure"


@dataclass
class Signal:
    """
    A structured signal extracted from raw logs/metrics.
    
    Signals are the atomic units of information that detectors operate on.
    They represent normalized, framework-agnostic representations of ML system state.
    
    Design: Signals are intentionally framework-agnostic to allow detectors
    to work across JAX/TF/PyTorch without modification.
    """
    name: str
    value: Union[float, int, bool, str, List[float], Dict[str, float]]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional context for distributed systems
    replica_id: Optional[int] = None
    step: Optional[int] = None
    layer_name: Optional[str] = None


@dataclass
class DetectionResult:
    """
    Result from a failure detector.
    
    Each detector produces a DetectionResult indicating whether a failure
    mode was detected, with associated confidence and evidence.
    
    Design: Stateless detectors produce these results independently,
    allowing for parallel execution and caching.
    """
    failure_mode: FailureMode
    detected: bool
    confidence: float  # [0.0, 1.0]
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    detector_name: str = ""
    
    # Signals that contributed to this detection
    contributing_signals: List[Signal] = field(default_factory=list)


@dataclass
class AttributionHypothesis:
    """
    A ranked hypothesis about the root cause of detected failures.
    
    The attribution engine produces multiple hypotheses with confidence scores,
    allowing for triage and prioritization of remediation efforts.
    """
    hypothesis: str
    confidence: float  # [0.0, 1.0]
    supporting_detections: List[DetectionResult] = field(default_factory=list)
    remediation_hints: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class DebugReport:
    """
    Complete debugging report from a full analysis run.
    
    This is the primary output of the debugger, containing all detections,
    attributions, and actionable insights.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    detections: List[DetectionResult] = field(default_factory=list)
    attributions: List[AttributionHypothesis] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

