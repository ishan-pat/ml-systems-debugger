"""
Root cause attribution engine.

Maps detected failure modes to ranked hypotheses about root causes.
Supports confidence scoring and explainability.

Design: Attribution is separate from detection to allow:
1. Multiple hypotheses per detection
2. Cross-detection correlation
3. Confidence calibration
4. Remediation hints

Tradeoff: Attribution can be heuristic-based (fast, interpretable) or
ML-based (more accurate but less interpretable). This implementation
uses rule-based heuristics for interpretability.
"""

from ml_systems_debugger.core.types import (
    DetectionResult,
    AttributionHypothesis,
    FailureMode,
)
from typing import List, Dict
import heapq


class AttributionEngine:
    """
    Generates root cause hypotheses from detection results.
    
    Uses rule-based heuristics to map failure modes to likely causes.
    Can be extended with ML-based attribution for more sophisticated analysis.
    """
    
    def __init__(self):
        # Failure mode -> root cause mappings
        self.attribution_rules = self._build_attribution_rules()
    
    def attribute(self, detections: List[DetectionResult]) -> List[AttributionHypothesis]:
        """
        Generate ranked hypotheses from detections.
        
        Args:
            detections: List of detection results
        
        Returns:
            Ranked list of attribution hypotheses (highest confidence first)
        """
        hypotheses = []
        
        # Group detections by failure mode
        by_mode = {}
        for detection in detections:
            if detection.detected:
                mode = detection.failure_mode
                if mode not in by_mode:
                    by_mode[mode] = []
                by_mode[mode].append(detection)
        
        # Generate hypotheses for each failure mode
        for mode, mode_detections in by_mode.items():
            if mode in self.attribution_rules:
                rule = self.attribution_rules[mode]
                hypothesis = self._generate_hypothesis(
                    mode, mode_detections, rule
                )
                if hypothesis:
                    hypotheses.append(hypothesis)
        
        # Generate cross-mode hypotheses (correlations)
        cross_hypotheses = self._generate_cross_mode_hypotheses(by_mode)
        hypotheses.extend(cross_hypotheses)
        
        # Sort by confidence (descending)
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return hypotheses
    
    def _generate_hypothesis(
        self,
        mode: FailureMode,
        detections: List[DetectionResult],
        rule: Dict
    ) -> AttributionHypothesis:
        """Generate a hypothesis from a single failure mode."""
        # Aggregate confidence from all detections
        avg_confidence = sum(d.confidence for d in detections) / len(detections)
        
        # Determine severity
        severity = "medium"
        if avg_confidence > 0.8:
            severity = "critical"
        elif avg_confidence > 0.6:
            severity = "high"
        elif avg_confidence < 0.3:
            severity = "low"
        
        # Generate remediation hints
        hints = rule.get("remediation_hints", [])
        
        # Add evidence-based hints
        for detection in detections:
            evidence = detection.evidence
            if "trigger" in evidence:
                trigger = evidence["trigger"]
                if trigger in rule.get("trigger_hints", {}):
                    hints.extend(rule["trigger_hints"][trigger])
        
        return AttributionHypothesis(
            hypothesis=rule["hypothesis"],
            confidence=avg_confidence,
            supporting_detections=detections,
            remediation_hints=list(set(hints)),  # Deduplicate
            severity=severity
        )
    
    def _generate_cross_mode_hypotheses(
        self,
        by_mode: Dict[FailureMode, List[DetectionResult]]
    ) -> List[AttributionHypothesis]:
        """
        Generate hypotheses from correlated failure modes.
        
        Some failure modes often co-occur, indicating a common root cause.
        """
        hypotheses = []
        
        # NaN + Exploding gradients -> numerical instability
        if FailureMode.NAN_DETECTED in by_mode and FailureMode.EXPLODING_GRADIENTS in by_mode:
            nan_detections = by_mode[FailureMode.NAN_DETECTED]
            grad_detections = by_mode[FailureMode.EXPLODING_GRADIENTS]
            
            # Check if they occurred at similar steps
            nan_steps = set()
            for d in nan_detections:
                for s in d.contributing_signals:
                    if s.step is not None:
                        nan_steps.add(s.step)
            
            grad_steps = set()
            for d in grad_detections:
                for s in d.contributing_signals:
                    if s.step is not None:
                        grad_steps.add(s.step)
            
            if nan_steps & grad_steps:  # Overlapping steps
                combined_confidence = (
                    sum(d.confidence for d in nan_detections) / len(nan_detections) +
                    sum(d.confidence for d in grad_detections) / len(grad_detections)
                ) / 2
                
                hypotheses.append(AttributionHypothesis(
                    hypothesis="Numerical instability causing both NaN values and gradient explosion. Likely due to: learning rate too high, unstable loss function, or numerical precision issues.",
                    confidence=min(1.0, combined_confidence * 1.2),  # Boost confidence for correlation
                    supporting_detections=nan_detections + grad_detections,
                    remediation_hints=[
                        "Reduce learning rate",
                        "Add gradient clipping",
                        "Check for numerical precision issues in loss computation",
                        "Consider using mixed precision with loss scaling"
                    ],
                    severity="critical"
                ))
        
        # Loss plateau + Vanishing gradients -> optimization stuck
        if FailureMode.LOSS_PLATEAU in by_mode and FailureMode.VANISHING_GRADIENTS in by_mode:
            plateau_detections = by_mode[FailureMode.LOSS_PLATEAU]
            vanishing_detections = by_mode[FailureMode.VANISHING_GRADIENTS]
            
            combined_confidence = (
                sum(d.confidence for d in plateau_detections) / len(plateau_detections) +
                sum(d.confidence for d in vanishing_detections) / len(vanishing_detections)
            ) / 2
            
            hypotheses.append(AttributionHypothesis(
                hypothesis="Training optimization stuck: loss plateau combined with vanishing gradients suggests the model is in a local minimum or dead zone.",
                confidence=combined_confidence,
                supporting_detections=plateau_detections + vanishing_detections,
                remediation_hints=[
                    "Increase learning rate (if too small)",
                    "Try different optimizer (e.g., Adam instead of SGD)",
                    "Add batch normalization or layer normalization",
                    "Check for dead ReLU units",
                    "Consider learning rate scheduling"
                ],
                severity="high"
            ))
        
        return hypotheses
    
    def _build_attribution_rules(self) -> Dict[FailureMode, Dict]:
        """Build rule-based attribution mappings."""
        return {
            FailureMode.EXPLODING_GRADIENTS: {
                "hypothesis": "Gradient explosion detected. Likely causes: learning rate too high, loss function instability, or numerical overflow.",
                "remediation_hints": [
                    "Reduce learning rate",
                    "Add gradient clipping (e.g., clip_by_norm)",
                    "Check loss function for numerical stability",
                    "Consider gradient accumulation for stability"
                ],
                "trigger_hints": {
                    "absolute_threshold": ["Gradient norm exceeded absolute threshold - immediate action required"],
                    "spike_detected": ["Sudden gradient spike detected - check recent changes to model or data"]
                }
            },
            FailureMode.VANISHING_GRADIENTS: {
                "hypothesis": "Gradient vanishing detected. Likely causes: deep network without proper initialization, activation function saturation, or learning rate too small.",
                "remediation_hints": [
                    "Check weight initialization (e.g., He/Xavier initialization)",
                    "Use residual connections or skip connections",
                    "Replace sigmoid/tanh with ReLU variants",
                    "Increase learning rate (if too small)",
                    "Add batch normalization"
                ],
                "trigger_hints": {
                    "absolute_threshold": ["Gradient norm below absolute threshold - optimization may be stuck"],
                    "sudden_drop": ["Sudden gradient drop detected - check for dead neurons or activation saturation"]
                }
            },
            FailureMode.NAN_DETECTED: {
                "hypothesis": "NaN values detected in computation. Critical issue requiring immediate attention. Likely causes: numerical overflow, division by zero, or uninitialized variables.",
                "remediation_hints": [
                    "Immediate action required - NaN propagation will corrupt training",
                    "Check for division by zero operations",
                    "Add numerical stability checks (e.g., epsilon in denominators)",
                    "Reduce learning rate",
                    "Check input data for invalid values",
                    "Enable NaN checking in framework (e.g., tf.debugging.enable_check_numerics)",
                    "Check loss function for numerical stability",
                    "Verify input data preprocessing"
                ]
            },
            FailureMode.INF_DETECTED: {
                "hypothesis": "Infinity values detected. Likely causes: numerical overflow, extreme values, or division by very small numbers.",
                "remediation_hints": [
                    "Check for numerical overflow in loss computation",
                    "Add value clipping",
                    "Check input data for extreme values",
                    "Use gradient clipping to prevent overflow"
                ]
            },
            FailureMode.LOSS_PLATEAU: {
                "hypothesis": "Loss plateau detected - training progress has stalled. May indicate: local minimum, learning rate too small, or insufficient model capacity.",
                "remediation_hints": [
                    "Try learning rate scheduling (e.g., reduce on plateau)",
                    "Increase model capacity if underfitting",
                    "Check if early stopping is appropriate",
                    "Try different optimizer or optimizer settings"
                ]
            },
            FailureMode.DIVERGENT_TRAINING: {
                "hypothesis": "Training divergence detected - loss is increasing. Likely causes: learning rate too high, unstable optimization, or data issues.",
                "remediation_hints": [
                    "Immediately reduce learning rate",
                    "Add gradient clipping",
                    "Check data quality and preprocessing",
                    "Verify loss function is correct",
                    "Consider restarting training from checkpoint"
                ]
            },
            FailureMode.TRAINING_SERVING_SKEW: {
                "hypothesis": "Training-serving skew detected - distribution mismatch between training and serving data. This can cause significant performance degradation in production.",
                "remediation_hints": [
                    "Audit data preprocessing pipelines (training vs serving)",
                    "Check for data leakage or time-based features",
                    "Implement distribution monitoring in production",
                    "Consider domain adaptation techniques",
                    "Verify feature engineering consistency"
                ]
            },
            FailureMode.GRADIENT_DESYNC: {
                "hypothesis": "Gradient desynchronization detected across replicas. Indicates non-determinism or communication issues in distributed training.",
                "remediation_hints": [
                    "Check for non-deterministic operations (dropout, etc.)",
                    "Verify all-reduce operations are working correctly",
                    "Check network connectivity between workers",
                    "Ensure same random seeds across replicas (where appropriate)",
                    "Monitor straggler workers"
                ]
            },
        }

