"""
Numerical signal extraction.

Extracts signals related to numerical stability:
- NaN/Inf detection
- Precision loss indicators
- Numerical range checks
- Mixed-precision instability signals
"""

from ml_systems_debugger.core.base import BaseSignalExtractor
from ml_systems_debugger.core.types import Signal
from typing import Any, List, Dict
from datetime import datetime
import numpy as np
import math


class NumericalSignalExtractor(BaseSignalExtractor):
    """
    Extracts numerical stability signals.
    
    Handles:
    - Direct NaN/Inf flags
    - Value arrays (checks for NaNs/Infs)
    - Precision metrics
    - Numerical range violations
    """
    
    def extract(self, raw_data: Any) -> List[Signal]:
        """
        Extract numerical signals from raw data.
        
        Expected formats:
        - Direct flags: {"has_nan": True, "has_inf": False}
        - Value arrays: {"values": np.array, "step": 1}
        - Precision metrics: {"precision_loss": 0.1}
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
        
        # Direct flags
        if "has_nan" in data:
            signals.append(Signal(
                name="has_nan",
                value=bool(data["has_nan"]),
                timestamp=timestamp,
                step=step,
                metadata={"type": "numerical_stability"}
            ))
        
        if "has_inf" in data:
            signals.append(Signal(
                name="has_inf",
                value=bool(data["has_inf"]),
                timestamp=timestamp,
                step=step,
                metadata={"type": "numerical_stability"}
            ))
        
        # Check arrays for NaNs/Infs
        for key in ["values", "weights", "activations", "loss"]:
            if key in data:
                value_array = self._to_numpy(data[key])
                if value_array is not None:
                    signals.extend(self._check_numerical_stability(
                        value_array, key, step, timestamp
                    ))
        
        # Precision loss
        if "precision_loss" in data:
            signals.append(Signal(
                name="precision_loss",
                value=float(data["precision_loss"]),
                timestamp=timestamp,
                step=step,
                metadata={"type": "precision"}
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
    
    def _check_numerical_stability(
        self,
        values: np.ndarray,
        name: str,
        step: int,
        timestamp: datetime
    ) -> List[Signal]:
        """Check array for numerical issues."""
        signals = []
        
        # Check for NaNs
        nan_count = int(np.isnan(values).sum())
        if nan_count > 0:
            signals.append(Signal(
                name="has_nan",
                value=True,
                timestamp=timestamp,
                step=step,
                layer_name=name,
                metadata={"nan_count": nan_count, "total": values.size}
            ))
        
        # Check for Infs
        inf_count = int(np.isinf(values).sum())
        if inf_count > 0:
            signals.append(Signal(
                name="has_inf",
                value=True,
                timestamp=timestamp,
                step=step,
                layer_name=name,
                metadata={"inf_count": inf_count, "total": values.size}
            ))
        
        # Check for extreme values (potential precision loss)
        if values.size > 0:
            finite_values = values[np.isfinite(values)]
            if len(finite_values) > 0:
                max_abs = float(np.max(np.abs(finite_values)))
                if max_abs > 1e10:  # Threshold for potential precision issues
                    signals.append(Signal(
                        name="extreme_value",
                        value=max_abs,
                        timestamp=timestamp,
                        step=step,
                        layer_name=name,
                        metadata={"type": "precision_warning"}
                    ))
        
        return signals

