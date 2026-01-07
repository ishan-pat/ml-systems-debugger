"""
Distribution signal extraction.

Extracts distribution-related signals for detecting data pathologies:
- Feature distribution statistics
- Training-serving distribution differences
- Label distribution shifts
- Feature correlation changes
"""

from ml_systems_debugger.core.base import BaseSignalExtractor
from ml_systems_debugger.core.types import Signal
from typing import Any, List, Dict
from datetime import datetime
import numpy as np


class DistributionSignalExtractor(BaseSignalExtractor):
    """
    Extracts distribution-related signals.
    
    Handles:
    - Feature statistics (mean, std, quantiles)
    - Distribution comparisons (KL divergence, Wasserstein distance)
    - Label distributions
    """
    
    def extract(self, raw_data: Any) -> List[Signal]:
        """
        Extract distribution signals from raw data.
        
        Expected formats:
        - Feature statistics: {"feature_mean": {...}, "feature_std": {...}}
        - Distribution comparisons: {"kl_div": 0.5, "dataset": "serving"}
        - Feature arrays: {"features": np.array, "dataset": "training"}
        """
        signals = []
        timestamp = datetime.now()
        
        if isinstance(raw_data, dict):
            signals.extend(self._extract_from_dict(raw_data, timestamp))
        elif isinstance(raw_data, list):
            for entry in raw_data:
                signals.extend(self._extract_from_dict(entry, timestamp))
        
        return signals
    
    def _extract_from_dict(self, data: Dict, timestamp: datetime) -> List[Signal]:
        """Extract signals from a dict entry."""
        signals = []
        step = data.get("step", 0)
        dataset = data.get("dataset", "unknown")
        
        # Direct distribution metrics
        if "kl_divergence" in data:
            signals.append(Signal(
                name="kl_divergence",
                value=float(data["kl_divergence"]),
                timestamp=timestamp,
                step=step,
                metadata={"dataset": dataset, "type": "distribution_distance"}
            ))
        
        if "wasserstein_distance" in data:
            signals.append(Signal(
                name="wasserstein_distance",
                value=float(data["wasserstein_distance"]),
                timestamp=timestamp,
                step=step,
                metadata={"dataset": dataset, "type": "distribution_distance"}
            ))
        
        # Feature statistics
        if "feature_mean" in data:
            means = data["feature_mean"]
            if isinstance(means, dict):
                for feature_name, mean_val in means.items():
                    signals.append(Signal(
                        name="feature_mean",
                        value=float(mean_val),
                        timestamp=timestamp,
                        step=step,
                        layer_name=feature_name,
                        metadata={"dataset": dataset}
                    ))
        
        if "feature_std" in data:
            stds = data["feature_std"]
            if isinstance(stds, dict):
                for feature_name, std_val in stds.items():
                    signals.append(Signal(
                        name="feature_std",
                        value=float(std_val),
                        timestamp=timestamp,
                        step=step,
                        layer_name=feature_name,
                        metadata={"dataset": dataset}
                    ))
        
        # Extract from feature arrays
        if "features" in data:
            features = data["features"]
            feature_array = self._to_numpy(features)
            if feature_array is not None:
                signals.extend(self._compute_feature_stats(
                    feature_array, step, timestamp, dataset
                ))
        
        return signals
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert to numpy array (framework-agnostic)."""
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, "numpy"):
            return data.numpy()
        elif hasattr(data, "__array__"):
            return np.array(data)
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        return None
    
    def _compute_feature_stats(
        self,
        features: np.ndarray,
        step: int,
        timestamp: datetime,
        dataset: str
    ) -> List[Signal]:
        """Compute feature distribution statistics."""
        signals = []
        
        # Per-feature statistics
        if features.ndim == 2:
            for feat_idx in range(features.shape[1]):
                feat_col = features[:, feat_idx]
                mean = float(np.mean(feat_col))
                std = float(np.std(feat_col))
                
                signals.append(Signal(
                    name="feature_mean",
                    value=mean,
                    timestamp=timestamp,
                    step=step,
                    layer_name=f"feature_{feat_idx}",
                    metadata={"dataset": dataset}
                ))
                signals.append(Signal(
                    name="feature_std",
                    value=std,
                    timestamp=timestamp,
                    step=step,
                    layer_name=f"feature_{feat_idx}",
                    metadata={"dataset": dataset}
                ))
        
        return signals

