"""
Main ML Systems Debugger class.

Orchestrates the full debugging pipeline:
1. Ingestion
2. Signal extraction
3. Detection
4. Attribution
5. Reporting

Design: This is the primary entry point for users. It manages
the lifecycle of all components and provides a clean API.
"""

from ml_systems_debugger.core.types import (
    Signal,
    DetectionResult,
    DebugReport,
)
from ml_systems_debugger.core.base import BaseIngestor, BaseSignalExtractor, BaseDetector
from ml_systems_debugger.attribution.attribution_engine import AttributionEngine
from ml_systems_debugger.reporting.reporter import Reporter, ReportFormat
from typing import List, Optional, Dict, Any
from datetime import datetime


class MLSystemsDebugger:
    """
    Main debugger class that orchestrates the debugging pipeline.
    
    Usage:
        debugger = MLSystemsDebugger()
        debugger.add_detector(GradientExplosionDetector())
        debugger.add_ingestor(MetricIngestor())
        
        report = debugger.debug(metrics_data)
        print(debugger.generate_report(report))
    
    Design: Composable architecture allows users to:
    1. Add custom detectors
    2. Use different ingestors for different data sources
    3. Chain signal extractors
    4. Customize attribution and reporting
    """
    
    def __init__(self):
        self.ingestors: List[BaseIngestor] = []
        self.signal_extractors: List[BaseSignalExtractor] = []
        self.detectors: List[BaseDetector] = []
        self.attribution_engine = AttributionEngine()
        self.reporter = Reporter(ReportFormat.TEXT)
        
        # Signal routing cache (maps signal names to detectors)
        self._signal_routing: Dict[str, List[BaseDetector]] = {}
        self._routing_dirty = True
    
    def add_ingestor(self, ingestor: BaseIngestor):
        """Add a data ingestor."""
        self.ingestors.append(ingestor)
    
    def add_signal_extractor(self, extractor: BaseSignalExtractor):
        """Add a signal extractor."""
        self.signal_extractors.append(extractor)
    
    def add_detector(self, detector: BaseDetector):
        """Add a failure detector."""
        self.detectors.append(detector)
        self._routing_dirty = True
    
    def debug(
        self,
        data: Any,
        context: Optional[Dict] = None
    ) -> DebugReport:
        """
        Run the full debugging pipeline.
        
        Args:
            data: Raw data to debug (logs, metrics, traces)
            context: Optional context dict (e.g., config, previous results)
        
        Returns:
            DebugReport with all detections and attributions
        """
        # Step 1: Ingestion
        all_signals = []
        for ingestor in self.ingestors:
            try:
                if ingestor.validate_schema(data):
                    signals = ingestor.ingest(data)
                    all_signals.extend(signals)
            except Exception as e:
                # Log error but continue with other ingestors
                # In production, you'd want proper logging here
                pass
        
        # Step 2: Signal extraction
        for extractor in self.signal_extractors:
            try:
                extracted = extractor.extract(data)
                all_signals.extend(extracted)
            except Exception as e:
                pass
        
        # Step 3: Detection
        detections = self._run_detectors(all_signals, context or {})
        
        # Step 4: Attribution
        attributions = self.attribution_engine.attribute(detections)
        
        # Step 5: Generate report
        report = DebugReport(
            timestamp=datetime.now(),
            detections=detections,
            attributions=attributions,
            summary=self._generate_summary(detections, attributions),
            metadata={
                "total_signals": len(all_signals),
                "detectors_run": len(self.detectors),
                "context": context or {},
            }
        )
        
        return report
    
    def _run_detectors(
        self,
        signals: List[Signal],
        context: Dict
    ) -> List[DetectionResult]:
        """Run all detectors on signals."""
        # Update signal routing if needed
        if self._routing_dirty:
            self._update_signal_routing()
        
        detections = []
        
        for detector in self.detectors:
            # Route relevant signals to detector
            required_signals = detector.get_required_signals()
            
            if required_signals:
                # Get signals that match detector requirements
                relevant_signals = [
                    s for s in signals
                    if s.name in required_signals
                ]
            else:
                # If no requirements specified, use all signals
                relevant_signals = signals
            
            if relevant_signals:
                try:
                    result = detector.detect(relevant_signals, context)
                    detections.append(result)
                except Exception as e:
                    # Log error but continue
                    # In production, you'd want proper error handling
                    pass
        
        return detections
    
    def _update_signal_routing(self):
        """Update signal-to-detector routing cache."""
        self._signal_routing = {}
        for detector in self.detectors:
            for signal_name in detector.get_required_signals():
                if signal_name not in self._signal_routing:
                    self._signal_routing[signal_name] = []
                self._signal_routing[signal_name].append(detector)
        self._routing_dirty = False
    
    def _generate_summary(
        self,
        detections: List[DetectionResult],
        attributions: List[AttributionHypothesis]
    ) -> str:
        """Generate a human-readable summary."""
        detected_count = sum(1 for d in detections if d.detected)
        
        if detected_count == 0:
            return "No failure modes detected. Training appears healthy."
        
        critical = [a for a in attributions if a.severity == "critical"]
        high = [a for a in attributions if a.severity == "high"]
        
        summary_parts = [
            f"Detected {detected_count} failure mode(s)."
        ]
        
        if critical:
            summary_parts.append(f"{len(critical)} critical issue(s) identified.")
        if high:
            summary_parts.append(f"{len(high)} high-severity issue(s) identified.")
        
        if attributions:
            top_hypothesis = attributions[0]
            summary_parts.append(
                f"Top hypothesis: {top_hypothesis.hypothesis[:100]}..."
            )
        
        return " ".join(summary_parts)
    
    def generate_report(
        self,
        report: DebugReport,
        format: Optional[ReportFormat] = None
    ) -> str:
        """
        Generate a formatted report.
        
        Args:
            report: DebugReport to format
            format: Optional format override (defaults to reporter's format)
        
        Returns:
            Formatted report string
        """
        if format:
            original_format = self.reporter.format
            self.reporter.format = format
            result = self.reporter.generate(report)
            self.reporter.format = original_format
            return result
        else:
            return self.reporter.generate(report)

