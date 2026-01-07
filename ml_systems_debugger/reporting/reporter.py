"""
Reporting layer for debugging output.

Supports multiple output formats:
- Human-readable text summaries
- Machine-readable JSON
- Structured markdown

Design: Separates formatting from content generation to allow
easy extension to new formats (HTML, PDF, etc.).
"""

from ml_systems_debugger.core.types import DebugReport, DetectionResult, AttributionHypothesis
from enum import Enum
from typing import Dict, Any
import json
from datetime import datetime


class ReportFormat(Enum):
    """Supported report formats."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class Reporter:
    """
    Generates reports from debug results.
    
    Supports multiple output formats for different use cases:
    - Text: Human-readable console output
    - JSON: Machine-readable for CI/CD integration
    - Markdown: Documentation-friendly format
    """
    
    def __init__(self, format: ReportFormat = ReportFormat.TEXT):
        self.format = format
    
    def generate(self, report: DebugReport) -> str:
        """Generate report in the specified format."""
        if self.format == ReportFormat.TEXT:
            return self._generate_text(report)
        elif self.format == ReportFormat.JSON:
            return self._generate_json(report)
        elif self.format == ReportFormat.MARKDOWN:
            return self._generate_markdown(report)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _generate_text(self, report: DebugReport) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("ML Systems Debugger Report")
        lines.append("=" * 80)
        lines.append(f"Generated: {report.timestamp.isoformat()}")
        lines.append("")
        
        # Summary
        detected_count = sum(1 for d in report.detections if d.detected)
        lines.append(f"Summary: {detected_count} failure mode(s) detected")
        lines.append("")
        
        # Detections
        if report.detections:
            lines.append("-" * 80)
            lines.append("DETECTIONS")
            lines.append("-" * 80)
            
            for detection in report.detections:
                if detection.detected:
                    lines.append(f"\n[{detection.failure_mode.value.upper()}]")
                    lines.append(f"  Detector: {detection.detector_name}")
                    lines.append(f"  Confidence: {detection.confidence:.2%}")
                    lines.append(f"  Timestamp: {detection.timestamp.isoformat()}")
                    
                    if detection.evidence:
                        lines.append("  Evidence:")
                        for key, value in detection.evidence.items():
                            if isinstance(value, (int, float)):
                                lines.append(f"    {key}: {value:.4f}")
                            else:
                                lines.append(f"    {key}: {value}")
        
        # Attributions
        if report.attributions:
            lines.append("")
            lines.append("-" * 80)
            lines.append("ROOT CAUSE ATTRIBUTIONS")
            lines.append("-" * 80)
            
            for i, attribution in enumerate(report.attributions, 1):
                lines.append(f"\n[{i}] {attribution.hypothesis}")
                lines.append(f"    Confidence: {attribution.confidence:.2%}")
                lines.append(f"    Severity: {attribution.severity.upper()}")
                
                if attribution.remediation_hints:
                    lines.append("    Remediation Hints:")
                    for hint in attribution.remediation_hints:
                        lines.append(f"      - {hint}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_json(self, report: DebugReport) -> str:
        """Generate machine-readable JSON report."""
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "summary": {
                "total_detections": len(report.detections),
                "detected_failures": sum(1 for d in report.detections if d.detected),
                "total_attributions": len(report.attributions),
            },
            "detections": [
                {
                    "failure_mode": d.failure_mode.value,
                    "detected": d.detected,
                    "confidence": d.confidence,
                    "detector_name": d.detector_name,
                    "timestamp": d.timestamp.isoformat(),
                    "evidence": d.evidence,
                    "contributing_signals_count": len(d.contributing_signals),
                }
                for d in report.detections
            ],
            "attributions": [
                {
                    "hypothesis": a.hypothesis,
                    "confidence": a.confidence,
                    "severity": a.severity,
                    "remediation_hints": a.remediation_hints,
                    "supporting_detections": [
                        {
                            "failure_mode": d.failure_mode.value,
                            "confidence": d.confidence,
                        }
                        for d in a.supporting_detections
                    ],
                }
                for a in report.attributions
            ],
            "metadata": report.metadata,
        }
        
        return json.dumps(report_dict, indent=2)
    
    def _generate_markdown(self, report: DebugReport) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# ML Systems Debugger Report")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp.isoformat()}")
        lines.append("")
        
        # Summary
        detected_count = sum(1 for d in report.detections if d.detected)
        lines.append(f"## Summary")
        lines.append("")
        lines.append(f"- **Total Detections:** {len(report.detections)}")
        lines.append(f"- **Detected Failures:** {detected_count}")
        lines.append(f"- **Root Cause Hypotheses:** {len(report.attributions)}")
        lines.append("")
        
        # Detections
        if report.detections:
            lines.append("## Detections")
            lines.append("")
            
            for detection in report.detections:
                if detection.detected:
                    lines.append(f"### {detection.failure_mode.value.replace('_', ' ').title()}")
                    lines.append("")
                    lines.append(f"- **Detector:** `{detection.detector_name}`")
                    lines.append(f"- **Confidence:** {detection.confidence:.2%}")
                    lines.append(f"- **Timestamp:** {detection.timestamp.isoformat()}")
                    lines.append("")
                    
                    if detection.evidence:
                        lines.append("**Evidence:**")
                        lines.append("```json")
                        lines.append(json.dumps(detection.evidence, indent=2))
                        lines.append("```")
                        lines.append("")
        
        # Attributions
        if report.attributions:
            lines.append("## Root Cause Attributions")
            lines.append("")
            
            for i, attribution in enumerate(report.attributions, 1):
                lines.append(f"### Hypothesis {i}")
                lines.append("")
                lines.append(f"**Confidence:** {attribution.confidence:.2%} | **Severity:** {attribution.severity.upper()}")
                lines.append("")
                lines.append(attribution.hypothesis)
                lines.append("")
                
                if attribution.remediation_hints:
                    lines.append("**Remediation Hints:**")
                    lines.append("")
                    for hint in attribution.remediation_hints:
                        lines.append(f"- {hint}")
                    lines.append("")
        
        return "\n".join(lines)

