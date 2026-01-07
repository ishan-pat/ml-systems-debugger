"""
Base classes and interfaces for extensible components.

These abstract base classes define the contracts that all detectors,
ingestors, and signal extractors must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ml_systems_debugger.core.types import Signal, DetectionResult, FailureMode


class BaseDetector(ABC):
    """
    Abstract base class for all failure detectors.
    
    Design Principles:
    1. Stateless: Detectors should be stateless to enable parallel execution
       and caching. Stateful detectors must explicitly manage their own state.
    2. Composable: Detectors can be chained or run in parallel
    3. Framework-agnostic: Operate on Signal objects, not framework-specific types
    
    Tradeoff: Statelessness requires passing context (e.g., historical signals)
    explicitly, which can be verbose but enables better testability and parallelism.
    """
    
    def __init__(self, name: str, failure_mode: FailureMode):
        self.name = name
        self.failure_mode = failure_mode
    
    @abstractmethod
    def detect(
        self,
        signals: List[Signal],
        context: Optional[Dict] = None
    ) -> DetectionResult:
        """
        Detect the failure mode from a list of signals.
        
        Args:
            signals: List of relevant signals for this detector
            context: Optional context dict (e.g., previous detections, config)
        
        Returns:
            DetectionResult indicating whether failure was detected
        """
        pass
    
    def get_required_signals(self) -> List[str]:
        """
        Return list of signal names this detector requires.
        
        Used by the debugger to route signals efficiently.
        """
        return []


class BaseIngestor(ABC):
    """
    Abstract base class for data ingestion.
    
    Supports both batch and streaming ingestion patterns.
    Framework-agnostic: converts framework-specific formats to Signals.
    """
    
    @abstractmethod
    def ingest(self, source: Any) -> List[Signal]:
        """
        Ingest data from a source and convert to Signals.
        
        Args:
            source: Framework-specific data source (logs, metrics, traces)
        
        Returns:
            List of normalized Signal objects
        """
        pass
    
    @abstractmethod
    def validate_schema(self, data: Any) -> bool:
        """Validate that the input data matches expected schema."""
        pass


class BaseSignalExtractor(ABC):
    """
    Abstract base class for signal extraction.
    
    Converts raw ingested data into structured signals that detectors
    can operate on. Handles normalization and framework-specific parsing.
    """
    
    @abstractmethod
    def extract(self, raw_data: Any) -> List[Signal]:
        """
        Extract signals from raw ingested data.
        
        Args:
            raw_data: Raw data from ingestor
        
        Returns:
            List of extracted Signal objects
        """
        pass

