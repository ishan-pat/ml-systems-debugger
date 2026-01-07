"""
Gradient signal extraction.

Extracts gradient statistics from raw metrics/logs:
- Gradient norms (L2, L∞)
- Gradient mean/std per layer
- Gradient-to-parameter ratios
- Gradient curvature proxies

Design: These signals are framework-agnostic and can be computed
from any gradient representation (numpy arrays, framework tensors, etc.).
"""

from ml_systems_debugger.core.base import BaseSignalExtractor
from ml_systems_debugger.core.types import Signal
from typing import Any, List, Dict
from datetime import datetime
import numpy as np


class GradientSignalExtractor(BaseSignalExtractor):
    """
    Extracts gradient-related signals from metrics/logs.
    
    Handles:
    - Direct gradient metrics (grad_norm, grad_mean, etc.)
    - Gradient arrays (computes statistics)
    - Framework-specific gradient objects (via adapters)
    """
    
    def extract(self, raw_data: Any) -> List[Signal]:
        """
        Extract gradient signals from raw data.
        
        Expected formats:
        - Dict with gradient metrics: {"grad_norm": 1.5, "step": 1}
        - Dict with gradient arrays: {"gradients": {...}, "step": 1}
        - List of gradient metric entries
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
        
        # Direct gradient metrics
        if "grad_norm" in data:
            signals.append(Signal(
                name="grad_norm",
                value=float(data["grad_norm"]),
                timestamp=timestamp,
                step=step,
                metadata={"type": "l2_norm"}
            ))
        
        if "grad_max" in data:
            signals.append(Signal(
                name="grad_max",
                value=float(data["grad_max"]),
                timestamp=timestamp,
                step=step,
                metadata={"type": "linf_norm"}
            ))
        
        # Extract from gradient arrays
        if "gradients" in data:
            gradients = data["gradients"]
            if isinstance(gradients, dict):
                # Per-layer gradients
                for layer_name, grad in gradients.items():
                    grad_array = self._to_numpy(grad)
                    if grad_array is not None:
                        signals.extend(self._compute_gradient_stats(
                            grad_array, layer_name, step, timestamp
                        ))
            elif isinstance(gradients, (list, np.ndarray)):
                # Single gradient array
                grad_array = self._to_numpy(gradients)
                if grad_array is not None:
                    signals.extend(self._compute_gradient_stats(
                        grad_array, "global", step, timestamp
                    ))
        
        return signals
    
    def _to_numpy(self, grad: Any) -> np.ndarray:
        """
        Convert gradient to numpy array.
        
        Framework-agnostic conversion. In production, you'd have
        framework-specific adapters here (JAX, TF, PyTorch).
        """
        if isinstance(grad, np.ndarray):
            return grad
        elif hasattr(grad, "numpy"):  # TensorFlow/PyTorch
            return grad.numpy()
        elif hasattr(grad, "__array__"):  # JAX
            return np.array(grad)
        elif isinstance(grad, (list, tuple)):
            return np.array(grad)
        return None
    
    def _compute_gradient_stats(
        self,
        grad: np.ndarray,
        layer_name: str,
        step: int,
        timestamp: datetime
    ) -> List[Signal]:
        """Compute gradient statistics from array."""
        signals = []
        
        # L2 norm
        l2_norm = float(np.linalg.norm(grad))
        signals.append(Signal(
            name="grad_norm",
            value=l2_norm,
            timestamp=timestamp,
            step=step,
            layer_name=layer_name,
            metadata={"type": "l2_norm"}
        ))
        
        # L∞ norm (max absolute value)
        linf_norm = float(np.max(np.abs(grad)))
        signals.append(Signal(
            name="grad_max",
            value=linf_norm,
            timestamp=timestamp,
            step=step,
            layer_name=layer_name,
            metadata={"type": "linf_norm"}
        ))
        
        # Mean and std
        mean = float(np.mean(grad))
        std = float(np.std(grad))
        signals.append(Signal(
            name="grad_mean",
            value=mean,
            timestamp=timestamp,
            step=step,
            layer_name=layer_name,
            metadata={"type": "mean"}
        ))
        signals.append(Signal(
            name="grad_std",
            value=std,
            timestamp=timestamp,
            step=step,
            layer_name=layer_name,
            metadata={"type": "std"}
        ))
        
        # Gradient-to-parameter ratio (if parameters provided)
        # This is a proxy for learning rate sensitivity
        
        return signals

