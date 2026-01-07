"""
Data pathology detectors.

Detects:
- Training-serving skew (distribution differences)
- Distribution drift over time
- Label leakage
- Feature collapse

Design: These detectors compare distributions across datasets or time.
Requires careful handling of statistical significance and false positives.
"""

from ml_systems_debugger.core.base import BaseDetector
from ml_systems_debugger.core.types import Signal, DetectionResult, FailureMode
from typing import List, Dict, Optional
import numpy as np


class TrainingServingSkewDetector(BaseDetector):
    """
    Detects training-serving skew (distribution differences).
    
    Detection logic:
    1. Compares feature distributions between training and serving
    2. Uses KL divergence or statistical tests
    3. Flags if difference exceeds threshold
    
    Tradeoff: Must balance sensitivity (catch real skew) vs. specificity
    (avoid false positives from normal variation). Thresholds should be
    calibrated per feature type.
    """
    
    def __init__(
        self,
        kl_threshold: float = 0.1,  # KL divergence threshold
        wasserstein_threshold: float = 0.5,  # Normalized Wasserstein distance
        min_samples: int = 100
    ):
        super().__init__("training_serving_skew", FailureMode.TRAINING_SERVING_SKEW)
        self.kl_threshold = kl_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.min_samples = min_samples
    
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """
        Detect training-serving skew from distribution signals.
        
        Expects signals with:
        - kl_divergence or wasserstein_distance signals
        - feature_mean/feature_std signals with dataset metadata
        """
        # Check for direct distribution distance signals
        kl_signals = [s for s in signals if s.name == "kl_divergence"]
        wass_signals = [s for s in signals if s.name == "wasserstein_distance"]
        
        detected = False
        confidence = 0.0
        evidence = {}
        contributing_signals = []
        
        # Check KL divergence
        for signal in kl_signals:
            if isinstance(signal.value, (int, float)):
                kl_val = float(signal.value)
                dataset = signal.metadata.get("dataset", "unknown")
                
                if dataset == "serving" and kl_val > self.kl_threshold:
                    detected = True
                    confidence = max(confidence, min(1.0, kl_val / (self.kl_threshold * 2)))
                    evidence["kl_divergence"] = kl_val
                    evidence["dataset"] = dataset
                    contributing_signals.append(signal)
        
        # Check Wasserstein distance
        for signal in wass_signals:
            if isinstance(signal.value, (int, float)):
                wass_val = float(signal.value)
                dataset = signal.metadata.get("dataset", "unknown")
                
                if dataset == "serving" and wass_val > self.wasserstein_threshold:
                    detected = True
                    confidence = max(confidence, min(1.0, wass_val / (self.wasserstein_threshold * 2)))
                    evidence["wasserstein_distance"] = wass_val
                    evidence["dataset"] = dataset
                    contributing_signals.append(signal)
        
        # Check feature statistics (mean/std differences)
        training_means = {}
        serving_means = {}
        training_stds = {}
        serving_stds = {}
        
        for signal in signals:
            if signal.name in ["feature_mean", "feature_std"]:
                dataset = signal.metadata.get("dataset", "unknown")
                feature_name = signal.layer_name or "unknown"
                
                if isinstance(signal.value, (int, float)):
                    if signal.name == "feature_mean":
                        if dataset == "training":
                            training_means[feature_name] = float(signal.value)
                        elif dataset == "serving":
                            serving_means[feature_name] = float(signal.value)
                    elif signal.name == "feature_std":
                        if dataset == "training":
                            training_stds[feature_name] = float(signal.value)
                        elif dataset == "serving":
                            serving_stds[feature_name] = float(signal.value)
        
        # Compare overlapping features
        common_features = set(training_means.keys()) & set(serving_means.keys())
        if common_features:
            skew_features = []
            for feat in common_features:
                train_mean = training_means[feat]
                serve_mean = serving_means[feat]
                
                # Relative difference
                if train_mean != 0:
                    rel_diff = abs(serve_mean - train_mean) / abs(train_mean)
                    if rel_diff > 0.2:  # 20% difference threshold
                        skew_features.append({
                            "feature": feat,
                            "training_mean": train_mean,
                            "serving_mean": serve_mean,
                            "relative_diff": rel_diff
                        })
            
            if skew_features:
                detected = True
                max_diff = max(f["relative_diff"] for f in skew_features)
                confidence = max(confidence, min(1.0, max_diff / 0.5))
                evidence["skewed_features"] = skew_features
                evidence["max_relative_diff"] = max_diff
        
        if not detected:
            evidence["reason"] = "no_skew_detected"
        
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
            contributing_signals=contributing_signals,
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["kl_divergence", "wasserstein_distance", "feature_mean", "feature_std"]

