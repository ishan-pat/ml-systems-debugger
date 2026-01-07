"""
Gradient pathology detectors.

Detects:
- Exploding gradients (sudden large increases)
- Vanishing gradients (gradients becoming too small)
- Gradient instability (high variance)

Design: These detectors are stateless and operate on signal sequences.
They use configurable thresholds to balance false positives vs. false negatives.
"""

from ml_systems_debugger.core.base import BaseDetector
from ml_systems_debugger.core.types import Signal, DetectionResult, FailureMode
from typing import List, Dict, Optional
import numpy as np


class GradientExplosionDetector(BaseDetector):
    """
    Detects exploding gradients.
    
    Detection logic:
    1. Monitors gradient norms over time
    2. Flags if gradient norm exceeds threshold
    3. Flags if gradient norm increases rapidly (sudden spike)
    
    Tradeoff: Lower threshold = more sensitive but more false positives.
    Production systems typically use adaptive thresholds based on historical norms.
    """
    
    def __init__(
        self,
        threshold: float = 100.0,
        spike_factor: float = 10.0,
        window_size: int = 10
    ):
        super().__init__("gradient_explosion", FailureMode.EXPLODING_GRADIENTS)
        self.threshold = threshold
        self.spike_factor = spike_factor
        self.window_size = window_size
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """
        Detect exploding gradients from gradient norm signals.
        
        Args:
            signals: List of signals containing grad_norm values
            context: Optional context (e.g., previous detections)
        
        Returns:
            DetectionResult with detection status and confidence
        """
        # Filter for gradient norm signals
        grad_signals = [s for s in signals if s.name == "grad_norm"]
        
        if not grad_signals:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_gradient_signals"},
                detector_name=self.name
            )
        
        # Extract gradient norm values (sorted by step)
        grad_norms = []
        for signal in sorted(grad_signals, key=lambda s: s.step or 0):
            if isinstance(signal.value, (int, float)):
                grad_norms.append(float(signal.value))
        
        if not grad_norms:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_valid_gradient_values"},
                detector_name=self.name
            )
        
        # Detection logic
        current_norm = grad_norms[-1]
        detected = False
        confidence = 0.0
        evidence = {
            "current_norm": current_norm,
            "threshold": self.threshold,
            "max_norm": max(grad_norms),
        }
        
        # Check absolute threshold
        if current_norm > self.threshold:
            detected = True
            confidence = min(1.0, current_norm / (self.threshold * 2))
            evidence["trigger"] = "absolute_threshold"
        
        # Check for sudden spike (relative to recent history)
        if len(grad_norms) >= 2:
            recent_avg = np.mean(grad_norms[-self.window_size:])
            if recent_avg > 0:
                spike_ratio = current_norm / recent_avg
                if spike_ratio > self.spike_factor:
                    detected = True
                    # Confidence increases with spike magnitude
                    confidence = max(confidence, min(1.0, spike_ratio / (self.spike_factor * 2)))
                    evidence["trigger"] = "spike_detected"
                    evidence["spike_ratio"] = float(spike_ratio)
                    evidence["recent_avg"] = float(recent_avg)
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
            contributing_signals=grad_signals[-self.window_size:] if detected else [],
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["grad_norm"]


class GradientVanishingDetector(BaseDetector):
    """
    Detects vanishing gradients.
    
    Detection logic:
    1. Monitors gradient norms over time
    2. Flags if gradient norm becomes too small
    3. Flags if gradient norm decreases rapidly (sudden drop)
    
    Tradeoff: Too sensitive threshold may flag normal convergence.
    Should be used in conjunction with loss monitoring.
    """
    
    def __init__(
        self,
        threshold: float = 1e-7,
        vanishing_factor: float = 0.1,
        window_size: int = 10
    ):
        super().__init__("gradient_vanishing", FailureMode.VANISHING_GRADIENTS)
        self.threshold = threshold
        self.vanishing_factor = vanishing_factor
        self.window_size = window_size
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """Detect vanishing gradients from gradient norm signals."""
        grad_signals = [s for s in signals if s.name == "grad_norm"]
        
        if not grad_signals:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_gradient_signals"},
                detector_name=self.name
            )
        
        grad_norms = []
        for signal in sorted(grad_signals, key=lambda s: s.step or 0):
            if isinstance(signal.value, (int, float)):
                grad_norms.append(float(signal.value))
        
        if not grad_norms:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_valid_gradient_values"},
                detector_name=self.name
            )
        
        current_norm = grad_norms[-1]
        detected = False
        confidence = 0.0
        evidence = {
            "current_norm": current_norm,
            "threshold": self.threshold,
            "min_norm": min(grad_norms),
        }
        
        # Check absolute threshold
        if current_norm < self.threshold:
            detected = True
            # Confidence increases as norm approaches zero
            confidence = min(1.0, (self.threshold - current_norm) / self.threshold)
            evidence["trigger"] = "absolute_threshold"
        
        # Check for sudden drop
        if len(grad_norms) >= 2:
            recent_avg = np.mean(grad_norms[-self.window_size:])
            if recent_avg > 0:
                drop_ratio = current_norm / recent_avg
                if drop_ratio < self.vanishing_factor:
                    detected = True
                    confidence = max(confidence, min(1.0, (self.vanishing_factor - drop_ratio) / self.vanishing_factor))
                    evidence["trigger"] = "sudden_drop"
                    evidence["drop_ratio"] = float(drop_ratio)
                    evidence["recent_avg"] = float(recent_avg)
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
            contributing_signals=grad_signals[-self.window_size:] if detected else [],
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["grad_norm"]

