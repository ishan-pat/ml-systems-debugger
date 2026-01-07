"""
Numerical stability detectors.

Detects:
- NaN values in computations
- Inf values in computations
- Numerical precision loss
- Mixed-precision instability

Design: These are critical detectors that should run early in the pipeline.
They operate on direct signals (has_nan, has_inf) rather than derived metrics.
"""

from ml_systems_debugger.core.base import BaseDetector
from ml_systems_debugger.core.types import Signal, DetectionResult, FailureMode
from typing import List, Dict, Optional


class NaNDetectionDetector(BaseDetector):
    """
    Detects NaN (Not a Number) values in computations.
    
    Detection logic:
    - Direct detection from has_nan signals
    - Any NaN is considered a failure (high confidence)
    
    Tradeoff: Zero tolerance for NaNs is appropriate in production ML systems.
    NaNs typically indicate numerical instability or bugs.
    """
    
    def __init__(self):
        super().__init__("nan_detection", FailureMode.NAN_DETECTED)
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """Detect NaN values from has_nan signals."""
        nan_signals = [s for s in signals if s.name == "has_nan"]
        
        if not nan_signals:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_nan_signals"},
                detector_name=self.name
            )
        
        # Check for any True values
        detected_nans = [s for s in nan_signals if s.value is True]
        
        if detected_nans:
            # Aggregate evidence
            total_nans = len(detected_nans)
            evidence = {
                "nan_count": total_nans,
                "affected_layers": list(set(s.layer_name for s in detected_nans if s.layer_name)),
                "steps_with_nan": list(set(s.step for s in detected_nans if s.step is not None)),
            }
            
            # Add metadata from signals
            nan_details = []
            for signal in detected_nans:
                if "nan_count" in signal.metadata:
                    nan_details.append({
                        "layer": signal.layer_name,
                        "count": signal.metadata.get("nan_count"),
                        "total": signal.metadata.get("total"),
                    })
            if nan_details:
                evidence["nan_details"] = nan_details
            
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=True,
                confidence=1.0,  # High confidence for direct NaN detection
                evidence=evidence,
                contributing_signals=detected_nans,
                detector_name=self.name
            )
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=False,
            confidence=0.0,
            evidence={"reason": "no_nans_detected"},
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["has_nan"]


class InfDetectionDetector(BaseDetector):
    """
    Detects Inf (Infinity) values in computations.
    
    Similar to NaN detection but for infinity values.
    Infs can occur from overflow, division by zero, or extreme values.
    """
    
    def __init__(self):
        super().__init__("inf_detection", FailureMode.INF_DETECTED)
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """Detect Inf values from has_inf signals."""
        inf_signals = [s for s in signals if s.name == "has_inf"]
        
        if not inf_signals:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_inf_signals"},
                detector_name=self.name
            )
        
        detected_infs = [s for s in inf_signals if s.value is True]
        
        if detected_infs:
            evidence = {
                "inf_count": len(detected_infs),
                "affected_layers": list(set(s.layer_name for s in detected_infs if s.layer_name)),
                "steps_with_inf": list(set(s.step for s in detected_infs if s.step is not None)),
            }
            
            inf_details = []
            for signal in detected_infs:
                if "inf_count" in signal.metadata:
                    inf_details.append({
                        "layer": signal.layer_name,
                        "count": signal.metadata.get("inf_count"),
                        "total": signal.metadata.get("total"),
                    })
            if inf_details:
                evidence["inf_details"] = inf_details
            
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=True,
                confidence=1.0,
                evidence=evidence,
                contributing_signals=detected_infs,
                detector_name=self.name
            )
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=False,
            confidence=0.0,
            evidence={"reason": "no_infs_detected"},
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["has_inf"]

