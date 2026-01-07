"""
Metric ingestor for training/inference metrics.

Supports common formats:
- Dict-based metrics (e.g., from TensorBoard, WandB exports)
- List of metric entries
- Framework-specific metric objects (via adapters)

Design: Pluggable adapters allow framework-specific ingestion without
polluting the core interface.
"""

from ml_systems_debugger.core.base import BaseIngestor
from ml_systems_debugger.core.types import Signal
from ml_systems_debugger.ingestion.base import SchemaValidator
from typing import Any, List, Dict, Union, Optional
from datetime import datetime


class MetricIngestor(BaseIngestor):
    """
    Ingests training and inference metrics.
    
    Expected input formats:
    1. List of dicts: [{"name": "loss", "value": 0.5, "step": 1}, ...]
    2. Dict of lists: {"loss": [0.5, 0.4, ...], "step": [1, 2, ...]}
    3. Framework-specific objects (via adapters)
    """
    
    def __init__(self, default_timestamp: Optional[datetime] = None):
        self.default_timestamp = default_timestamp or datetime.now()
        self.validator = SchemaValidator()
    
    def ingest(self, source: Any) -> List[Signal]:
        """
        Ingest metrics from source.
        
        Handles multiple input formats for flexibility.
        """
        if isinstance(source, list):
            return self._ingest_list_format(source)
        elif isinstance(source, dict):
            return self._ingest_dict_format(source)
        else:
            # Try framework-specific adapter
            return self._ingest_framework_object(source)
    
    def _ingest_list_format(self, entries: List[Dict]) -> List[Signal]:
        """Ingest from list of metric entries."""
        signals = []
        for entry in entries:
            if not self.validator.validate_metric_entry(entry):
                continue
            
            timestamp = entry.get("timestamp", self.default_timestamp)
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            signal = Signal(
                name=entry["name"],
                value=entry["value"],
                timestamp=timestamp,
                step=entry.get("step"),
                metadata=entry.get("metadata", {})
            )
            signals.append(signal)
        
        return signals
    
    def _ingest_dict_format(self, data: Dict[str, List]) -> List[Signal]:
        """Ingest from dict of metric arrays."""
        signals = []
        steps = data.get("step", list(range(len(data.get("loss", [])))))
        
        # Extract all metric names (excluding step)
        metric_names = [k for k in data.keys() if k != "step"]
        
        for idx, step in enumerate(steps):
            timestamp = self.default_timestamp  # Default if not provided
            for metric_name in metric_names:
                if idx < len(data[metric_name]):
                    signal = Signal(
                        name=metric_name,
                        value=data[metric_name][idx],
                        timestamp=timestamp,
                        step=int(step),
                        metadata={}
                    )
                    signals.append(signal)
        
        return signals
    
    def _ingest_framework_object(self, obj: Any) -> List[Signal]:
        """
        Attempt to ingest framework-specific metric objects.
        
        This is a placeholder for framework adapters. In production,
        you'd have JAX/TF/PyTorch-specific adapters here.
        """
        # For now, raise informative error
        raise ValueError(
            f"Unsupported metric format: {type(obj)}. "
            "Use list of dicts or dict of lists format, or implement a framework adapter."
        )
    
    def validate_schema(self, data: Any) -> bool:
        """Validate metric data schema."""
        if isinstance(data, list):
            return all(self.validator.validate_metric_entry(entry) for entry in data)
        elif isinstance(data, dict):
            # Dict format should have at least one metric array
            return "step" in data or any(isinstance(v, list) for v in data.values())
        return False

