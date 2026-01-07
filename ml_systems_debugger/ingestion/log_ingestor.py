"""
Log ingestor for training/inference logs.

Parses structured and unstructured logs to extract relevant signals.
Supports common log formats and can be extended for custom formats.
"""

from ml_systems_debugger.core.base import BaseIngestor
from ml_systems_debugger.core.types import Signal
from ml_systems_debugger.ingestion.base import SchemaValidator
from typing import Any, List, Dict
from datetime import datetime
import re


class LogIngestor(BaseIngestor):
    """
    Ingests logs and extracts structured signals.
    
    Handles:
    - Structured JSON logs
    - Unstructured text logs with pattern matching
    - Error/warning extraction
    """
    
    def __init__(self):
        self.validator = SchemaValidator()
        # Patterns for common log signals
        self.patterns = {
            "nan": re.compile(r"(?i)(nan|not a number)"),
            "inf": re.compile(r"(?i)(inf|infinity)"),
            "oom": re.compile(r"(?i)(out of memory|oom|cuda.*memory)"),
            "error": re.compile(r"(?i)(error|exception|failed)"),
        }
    
    def ingest(self, source: Any) -> List[Signal]:
        """Ingest logs from source (file path, list of log entries, or raw text)."""
        if isinstance(source, str):
            # Assume file path or raw log text
            if "\n" in source or len(source) > 200:
                # Likely raw log text
                return self._ingest_text(source)
            else:
                # Likely file path
                return self._ingest_file(source)
        elif isinstance(source, list):
            return self._ingest_list_format(source)
        else:
            raise ValueError(f"Unsupported log source type: {type(source)}")
    
    def _ingest_file(self, filepath: str) -> List[Signal]:
        """Ingest logs from a file."""
        with open(filepath, "r") as f:
            content = f.read()
        return self._ingest_text(content)
    
    def _ingest_text(self, text: str) -> List[Signal]:
        """Ingest from raw log text."""
        signals = []
        lines = text.split("\n")
        timestamp = datetime.now()
        
        for line_num, line in enumerate(lines):
            # Check for patterns
            for signal_name, pattern in self.patterns.items():
                if pattern.search(line):
                    signal = Signal(
                        name=f"log_{signal_name}",
                        value=True,
                        timestamp=timestamp,
                        metadata={"line": line_num, "content": line.strip()}
                    )
                    signals.append(signal)
        
        return signals
    
    def _ingest_list_format(self, entries: List[Dict]) -> List[Signal]:
        """Ingest from list of structured log entries."""
        signals = []
        for entry in entries:
            if not self.validator.validate_log_entry(entry):
                continue
            
            timestamp = entry.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Extract signals from log level, message, etc.
            level = entry.get("level", "").lower()
            message = entry.get("message", "")
            
            # Create signal for log entry
            signal = Signal(
                name=f"log_{level}",
                value=message,
                timestamp=timestamp,
                metadata={
                    "level": level,
                    "message": message,
                    **entry.get("metadata", {})
                }
            )
            signals.append(signal)
            
            # Check for patterns in message
            for signal_name, pattern in self.patterns.items():
                if pattern.search(message):
                    pattern_signal = Signal(
                        name=f"log_{signal_name}",
                        value=True,
                        timestamp=timestamp,
                        metadata={"source": "structured_log", "message": message}
                    )
                    signals.append(pattern_signal)
        
        return signals
    
    def validate_schema(self, data: Any) -> bool:
        """Validate log data schema."""
        if isinstance(data, list):
            return all(self.validator.validate_log_entry(entry) for entry in data)
        elif isinstance(data, str):
            return True  # Text logs are always valid
        return False

