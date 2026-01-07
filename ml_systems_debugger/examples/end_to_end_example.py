"""
End-to-end example demonstrating the ML Systems Debugger.

This example shows:
1. Setting up the debugger with multiple detectors
2. Ingesting synthetic training metrics
3. Running detection and attribution
4. Generating reports in multiple formats

This simulates a real-world debugging scenario with multiple failure modes.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ml_systems_debugger.core.debugger import MLSystemsDebugger
from ml_systems_debugger.ingestion.metric_ingestor import MetricIngestor
from ml_systems_debugger.signals.gradient_extractor import GradientSignalExtractor
from ml_systems_debugger.signals.numerical_extractor import NumericalSignalExtractor
from ml_systems_debugger.detectors.gradient_detector import (
    GradientExplosionDetector,
    GradientVanishingDetector,
)
from ml_systems_debugger.detectors.numerical_detector import (
    NaNDetectionDetector,
    InfDetectionDetector,
)
from ml_systems_debugger.detectors.optimization_detector import (
    LossPlateauDetector,
    DivergentTrainingDetector,
)
from ml_systems_debugger.reporting.reporter import ReportFormat
import numpy as np


def generate_synthetic_metrics():
    """
    Generate synthetic training metrics with injected failure modes.
    
    This simulates:
    1. Normal training for first 50 steps
    2. Gradient explosion at step 50
    3. NaN values at step 60
    4. Loss plateau starting at step 80
    """
    metrics = []
    base_time = datetime.now()
    
    # Normal training phase (steps 0-49)
    for step in range(50):
        loss = 2.0 * np.exp(-step / 30) + np.random.normal(0, 0.05)
        grad_norm = 5.0 * np.exp(-step / 40) + np.random.normal(0, 0.5)
        
        metrics.append({
            "name": "loss",
            "value": float(loss),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
        metrics.append({
            "name": "grad_norm",
            "value": float(grad_norm),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
    
    # Gradient explosion phase (steps 50-59)
    for step in range(50, 60):
        loss = 0.5 + np.random.normal(0, 0.1)
        # Sudden gradient explosion
        if step == 50:
            grad_norm = 150.0  # Spike
        else:
            grad_norm = 120.0 + np.random.normal(0, 10)
        
        metrics.append({
            "name": "loss",
            "value": float(loss),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
        metrics.append({
            "name": "grad_norm",
            "value": float(grad_norm),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
    
    # NaN injection phase (steps 60-79)
    for step in range(60, 80):
        loss = 0.5 + np.random.normal(0, 0.1)
        grad_norm = 10.0 + np.random.normal(0, 2)
        
        metrics.append({
            "name": "loss",
            "value": float(loss),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
        metrics.append({
            "name": "grad_norm",
            "value": float(grad_norm),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
        
        # Inject NaN signal
        if step == 60:
            metrics.append({
                "name": "has_nan",
                "value": True,
                "step": step,
                "timestamp": base_time + timedelta(seconds=step),
                "metadata": {
                    "nan_count": 42,
                    "total": 1000,
                    "layer": "dense_2"
                }
            })
    
    # Loss plateau phase (steps 80-100)
    for step in range(80, 101):
        # Loss plateaus (minimal improvement)
        loss = 0.48 + np.random.normal(0, 0.02)
        grad_norm = 0.001 + np.random.normal(0, 0.0001)  # Vanishing gradients
        
        metrics.append({
            "name": "loss",
            "value": float(loss),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
        metrics.append({
            "name": "grad_norm",
            "value": float(grad_norm),
            "step": step,
            "timestamp": base_time + timedelta(seconds=step),
        })
    
    return metrics


def main():
    """Run the end-to-end debugging example."""
    print("=" * 80)
    print("ML Systems Debugger - End-to-End Example")
    print("=" * 80)
    print()
    
    # Step 1: Generate synthetic metrics with failure modes
    print("Step 1: Generating synthetic training metrics...")
    metrics = generate_synthetic_metrics()
    print(f"  Generated {len(metrics)} metric entries")
    print()
    
    # Step 2: Set up the debugger
    print("Step 2: Setting up ML Systems Debugger...")
    debugger = MLSystemsDebugger()
    
    # Add ingestors
    debugger.add_ingestor(MetricIngestor())
    
    # Add signal extractors
    debugger.add_signal_extractor(GradientSignalExtractor())
    debugger.add_signal_extractor(NumericalSignalExtractor())
    
    # Add detectors
    debugger.add_detector(GradientExplosionDetector(threshold=100.0))
    debugger.add_detector(GradientVanishingDetector(threshold=1e-6))
    debugger.add_detector(NaNDetectionDetector())
    debugger.add_detector(InfDetectionDetector())
    debugger.add_detector(LossPlateauDetector(window_size=20))
    debugger.add_detector(DivergentTrainingDetector())
    
    print(f"  Added {len(debugger.detectors)} detectors")
    print()
    
    # Step 3: Run debugging
    print("Step 3: Running debugging pipeline...")
    report = debugger.debug(metrics)
    print(f"  Detections: {len(report.detections)}")
    print(f"  Attributions: {len(report.attributions)}")
    print()
    
    # Step 4: Generate reports
    print("Step 4: Generating reports...")
    print()
    
    # Text report
    print("--- TEXT REPORT ---")
    text_report = debugger.generate_report(report, ReportFormat.TEXT)
    print(text_report)
    print()
    
    # JSON report
    print("--- JSON REPORT (first 500 chars) ---")
    json_report = debugger.generate_report(report, ReportFormat.JSON)
    print(json_report[:500] + "...")
    print()
    
    # Markdown report
    print("--- MARKDOWN REPORT (first 500 chars) ---")
    md_report = debugger.generate_report(report, ReportFormat.MARKDOWN)
    print(md_report[:500] + "...")
    print()
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

