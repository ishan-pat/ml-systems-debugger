"""Base ingestor implementation with common utilities."""

from ml_systems_debugger.core.base import BaseIngestor
from ml_systems_debugger.core.types import Signal
from typing import Any, List
from datetime import datetime


class SchemaValidator:
    """
    Validates schema for ingested data.
    
    Design: Centralized validation allows for consistent error handling
    and clear error messages when data doesn't match expected format.
    """
    
    @staticmethod
    def validate_metric_entry(entry: dict) -> bool:
        """Validate a metric entry has required fields."""
        required = ["name", "value", "step"]
        return all(key in entry for key in required)
    
    @staticmethod
    def validate_log_entry(entry: dict) -> bool:
        """Validate a log entry has required fields."""
        required = ["message", "timestamp"]
        return all(key in entry for key in required)

