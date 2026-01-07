"""
Distributed systems failure detectors.

Detects:
- Non-determinism across replicas
- Gradient desynchronization
- Straggler effects
- Partial worker failure

Design: These detectors require signals from multiple replicas/workers.
They compare signals across replicas to detect inconsistencies.
"""

from ml_systems_debugger.core.base import BaseDetector
from ml_systems_debugger.core.types import Signal, DetectionResult, FailureMode
from typing import List, Dict, Optional
import numpy as np


class DistributedDesyncDetector(BaseDetector):
    """
    Detects gradient desynchronization across replicas.
    
    Detection logic:
    1. Collects gradient norms from all replicas at same step
    2. Computes variance across replicas
    3. Flags if variance exceeds threshold (indicating desync)
    
    Tradeoff: Some variance is expected due to:
    - Numerical precision differences
    - Non-deterministic operations (dropout, etc.)
    - Communication delays
    
    Threshold must account for these factors.
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.1,  # Coefficient of variation threshold
        min_replicas: int = 2
    ):
        super().__init__("distributed_desync", FailureMode.GRADIENT_DESYNC)
        self.variance_threshold = variance_threshold
        self.min_replicas = min_replicas
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """
        Detect gradient desynchronization from multi-replica signals.
        
        Expects signals with replica_id metadata to group by replica.
        """
        # Group signals by step and replica
        grad_signals = [s for s in signals if s.name == "grad_norm" and s.replica_id is not None]
        
        if not grad_signals:
            return DetectionResult(
                failure_mode=self.failure_mode,
                detected=False,
                confidence=0.0,
                evidence={"reason": "no_replica_signals"},
                detector_name=self.name
            )
        
        # Group by step
        by_step = {}
        for signal in grad_signals:
            step = signal.step or 0
            if step not in by_step:
                by_step[step] = []
            by_step[step].append(signal)
        
        detected = False
        confidence = 0.0
        evidence = {
            "steps_analyzed": len(by_step),
            "desync_steps": [],
        }
        contributing_signals = []
        
        # Analyze each step
        for step, step_signals in by_step.items():
            if len(step_signals) < self.min_replicas:
                continue
            
            # Extract gradient norms per replica
            grad_norms = []
            for signal in step_signals:
                if isinstance(signal.value, (int, float)):
                    grad_norms.append(float(signal.value))
            
            if len(grad_norms) < self.min_replicas:
                continue
            
            # Compute coefficient of variation
            mean_norm = np.mean(grad_norms)
            std_norm = np.std(grad_norms)
            
            if mean_norm > 0:
                cv = std_norm / mean_norm  # Coefficient of variation
                
                if cv > self.variance_threshold:
                    detected = True
                    step_confidence = min(1.0, cv / (self.variance_threshold * 2))
                    confidence = max(confidence, step_confidence)
                    
                    evidence["desync_steps"].append({
                        "step": step,
                        "mean_norm": float(mean_norm),
                        "std_norm": float(std_norm),
                        "coefficient_of_variation": float(cv),
                        "replica_norms": grad_norms,
                    })
                    contributing_signals.extend(step_signals)
        
        if not detected:
            evidence["reason"] = "no_desync_detected"
            evidence["max_cv"] = 0.0
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
            contributing_signals=contributing_signals,
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["grad_norm"]

