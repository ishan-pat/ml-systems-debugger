"""
Optimization pathology detectors.

Detects:
- Loss plateaus (training stagnation)
- Divergent training (loss increasing)
- Learning rate instability

Design: These detectors require historical context to distinguish
normal convergence from pathological behavior.
"""

from ml_systems_debugger.core.base import BaseDetector
from ml_systems_debugger.core.types import Signal, DetectionResult, FailureMode
from typing import List, Dict, Optional
import numpy as np


class LossPlateauDetector(BaseDetector):
    """
    Detects loss plateaus (training stagnation).
    
    Detection logic:
    1. Monitors loss over a window
    2. Computes relative improvement
    3. Flags if improvement is below threshold for extended period
    
    Tradeoff: Must distinguish between:
    - Normal convergence (small improvements)
    - True plateau (no improvement)
    - Early stopping (intentional)
    
    Uses relative improvement to handle different loss scales.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        improvement_threshold: float = 0.01,  # 1% relative improvement
        min_steps: int = 20  # Minimum steps before flagging
    ):
        super().__init__("loss_plateau", FailureMode.LOSS_PLATEAU)
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.min_steps = min_steps
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """Detect loss plateaus from loss signals."""
        loss_signals = [s for s in signals if s.name == "loss"]
        
        if not loss_signals:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_loss_signals"},
                detector_name=self.name
            )
        
        # Sort by step
        sorted_signals = sorted(loss_signals, key=lambda s: s.step or 0)
        loss_values = []
        for signal in sorted_signals:
            if isinstance(signal.value, (int, float)):
                loss_values.append(float(signal.value))
        
        if len(loss_values) < self.min_steps:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "insufficient_history", "steps": len(loss_values)},
                detector_name=self.name
            )
        
        # Analyze recent window
        recent_losses = loss_values[-self.window_size:]
        if len(recent_losses) < 2:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "insufficient_window_data"},
                detector_name=self.name
            )
        
        # Compute relative improvement
        initial_loss = recent_losses[0]
        final_loss = recent_losses[-1]
        
        if initial_loss <= 0:
            # Handle edge case (loss should be positive)
            relative_improvement = 0.0
        else:
            relative_improvement = (initial_loss - final_loss) / initial_loss
        
        detected = False
        confidence = 0.0
        evidence = {
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "relative_improvement": float(relative_improvement),
            "threshold": self.improvement_threshold,
            "window_size": len(recent_losses),
        }
        
        # Check if improvement is below threshold
        if relative_improvement < self.improvement_threshold:
            detected = True
            # Confidence increases as improvement approaches zero
            confidence = min(1.0, (self.improvement_threshold - relative_improvement) / self.improvement_threshold)
            evidence["trigger"] = "low_improvement"
        
        # Additional check: variance in recent losses (true plateau vs. noisy)
        if len(recent_losses) >= 10:
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            if loss_mean > 0:
                cv = loss_std / loss_mean  # Coefficient of variation
                evidence["coefficient_of_variation"] = float(cv)
                # Low variance + low improvement = high confidence plateau
                if detected and cv < 0.05:
                    confidence = min(1.0, confidence * 1.2)
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
            contributing_signals=sorted_signals[-self.window_size:] if detected else [],
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["loss"]


class DivergentTrainingDetector(BaseDetector):
    """
    Detects divergent training (loss increasing significantly).
    
    Detection logic:
    1. Monitors loss trend
    2. Flags if loss increases beyond threshold
    3. Distinguishes from normal training noise
    
    Tradeoff: Must be robust to temporary spikes while catching
    true divergence early.
    """
    
    def __init__(
        self,
        increase_threshold: float = 0.5,  # 50% increase
        window_size: int = 20,
        min_steps: int = 10
    ):
        super().__init__("divergent_training", FailureMode.DIVERGENT_TRAINING)
        self.increase_threshold = increase_threshold
        self.window_size = window_size
        self.min_steps = min_steps
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """Detect divergent training from loss signals."""
        loss_signals = [s for s in signals if s.name == "loss"]
        
        if not loss_signals:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_loss_signals"},
                detector_name=self.name
            )
        
        sorted_signals = sorted(loss_signals, key=lambda s: s.step or 0)
        loss_values = []
        for signal in sorted_signals:
            if isinstance(signal.value, (int, float)):
                loss_values.append(float(signal.value))
        
        if len(loss_values) < self.min_steps:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "insufficient_history"},
                detector_name=self.name
            )
        
        # Analyze recent window
        recent_losses = loss_values[-self.window_size:]
        if len(recent_losses) < 2:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "insufficient_window_data"},
                detector_name=self.name
            )
        
        # Compute relative increase
        baseline_loss = recent_losses[0]
        current_loss = recent_losses[-1]
        
        if baseline_loss <= 0:
            relative_increase = 0.0
        else:
            relative_increase = (current_loss - baseline_loss) / baseline_loss
        
        detected = False
        confidence = 0.0
        evidence = {
            "baseline_loss": float(baseline_loss),
            "current_loss": float(current_loss),
            "relative_increase": float(relative_increase),
            "threshold": self.increase_threshold,
        }
        
        # Check if loss increased significantly
        if relative_increase > self.increase_threshold:
            detected = True
            # Confidence increases with magnitude of increase
            confidence = min(1.0, relative_increase / (self.increase_threshold * 2))
            evidence["trigger"] = "significant_increase"
        
        # Check trend (not just endpoint comparison)
        if len(recent_losses) >= 5:
            # Linear regression to check trend
            x = np.arange(len(recent_losses))
            slope = np.polyfit(x, recent_losses, 1)[0]
            evidence["slope"] = float(slope)
            
            if slope > 0 and relative_increase > 0.2:  # Positive trend
                detected = True
                confidence = max(confidence, min(1.0, slope / (np.mean(recent_losses) * 0.1)))
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
            contributing_signals=sorted_signals[-self.window_size:] if detected else [],
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["loss"]

