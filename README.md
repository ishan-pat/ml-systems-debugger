# ML Systems Debugger

A production-grade ML Systems Debugger for detecting, attributing, and analyzing failure modes in distributed training and inference pipelines.

## Overview

The ML Systems Debugger treats failures as first-class signals, enabling automated detection, attribution, and triage of hidden failure modes in large-scale ML training and inference pipelines.

### Key Features

- **Modular Architecture**: Pluggable detectors, ingestors, and signal extractors
- **Framework-Agnostic**: Works with JAX, TensorFlow, PyTorch, and other frameworks
- **Comprehensive Failure Detection**: Covers optimization pathologies, data issues, numerical instability, and distributed systems failures
- **Root Cause Attribution**: Maps detections to ranked hypotheses with confidence scores
- **Multiple Output Formats**: Human-readable text, JSON, and Markdown reports

## Architecture

The system is organized into five main layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Reporting Layer                       │
│         (Text, JSON, Markdown output)                   │
└─────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────┐
│              Attribution Engine                          │
│    (Root cause hypotheses, confidence scoring)          │
└─────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────┐
│              Detection Layer                             │
│  (Gradient, Numerical, Optimization, Data, Distributed)  │
└─────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────┐
│              Signal Extraction Layer                     │
│  (Gradient stats, Distribution metrics, Numerical)      │
└─────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────┐
│              Ingestion Layer                             │
│         (Logs, Metrics, Traces)                         │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Stateless Detectors**: Detectors are stateless to enable parallel execution and caching. Stateful detectors must explicitly manage their own state.

2. **Framework-Agnostic Signals**: Signals are normalized, framework-agnostic representations that allow detectors to work across frameworks without modification.

3. **Composable Components**: All components (detectors, ingestors, extractors) are composable and can be extended or replaced independently.

4. **Separation of Concerns**: Detection, attribution, and reporting are separate concerns, allowing for independent optimization and extension.

## Installation

```bash
pip install -r requirements.txt
# Or for development:
pip install -e .
```

## Quick Start

```python
from ml_systems_debugger import MLSystemsDebugger
from ml_systems_debugger.ingestion import MetricIngestor
from ml_systems_debugger.detectors import (
    GradientExplosionDetector,
    NaNDetectionDetector,
    LossPlateauDetector,
)

# Initialize debugger
debugger = MLSystemsDebugger()

# Add components
debugger.add_ingestor(MetricIngestor())
debugger.add_detector(GradientExplosionDetector(threshold=100.0))
debugger.add_detector(NaNDetectionDetector())
debugger.add_detector(LossPlateauDetector())

# Run debugging on metrics
metrics = [
    {"name": "loss", "value": 0.5, "step": 1},
    {"name": "grad_norm", "value": 150.0, "step": 1},  # Exploding gradient!
    # ... more metrics
]

report = debugger.debug(metrics)
print(debugger.generate_report(report))
```

## Core Components

### Ingestion Layer

The ingestion layer converts raw data (logs, metrics, traces) into normalized `Signal` objects.

**Available Ingestors:**
- `MetricIngestor`: Ingests training/inference metrics
- `LogIngestor`: Parses logs and extracts structured signals

**Design Tradeoff**: Pluggable ingestors allow framework-specific adapters without polluting the core interface. The tradeoff is that users must explicitly choose ingestors, but this provides flexibility.

### Signal Extraction Layer

Converts raw ingested data into structured signals that detectors can operate on.

**Available Extractors:**
- `GradientSignalExtractor`: Extracts gradient statistics (norms, means, stds)
- `DistributionSignalExtractor`: Extracts distribution metrics (KL divergence, feature stats)
- `NumericalSignalExtractor`: Extracts numerical stability signals (NaN/Inf detection)

**Design Tradeoff**: Signal extraction is separate from ingestion to allow multiple extractors to operate on the same data, enabling richer signal sets.

### Detection Layer

Failure detectors identify specific failure modes from signals.

**Available Detectors:**

#### Optimization Pathologies
- `GradientExplosionDetector`: Detects exploding gradients (threshold-based, spike detection)
- `GradientVanishingDetector`: Detects vanishing gradients
- `LossPlateauDetector`: Detects training stagnation
- `DivergentTrainingDetector`: Detects increasing loss

#### Numerical Issues
- `NaNDetectionDetector`: Detects NaN values (zero tolerance)
- `InfDetectionDetector`: Detects infinity values

#### Data Pathologies
- `TrainingServingSkewDetector`: Detects distribution mismatch between training and serving

#### Distributed Systems
- `DistributedDesyncDetector`: Detects gradient desynchronization across replicas

**Design Tradeoff**: Stateless detectors enable parallelism but require explicit context passing. This is a deliberate choice for testability and scalability.

### Attribution Engine

Maps detected failure modes to ranked root cause hypotheses with confidence scores.

**Features:**
- Rule-based heuristics for interpretability
- Cross-mode correlation (e.g., NaN + Exploding Gradients → Numerical Instability)
- Remediation hints for each hypothesis
- Severity classification (low, medium, high, critical)

**Design Tradeoff**: Rule-based attribution is fast and interpretable but may miss complex correlations. ML-based attribution could be more accurate but less interpretable. The current implementation prioritizes interpretability for production debugging.

### Reporting Layer

Generates human-readable and machine-readable reports.

**Formats:**
- **Text**: Console-friendly output
- **JSON**: Machine-readable for CI/CD integration
- **Markdown**: Documentation-friendly format

## Failure Modes

The system detects the following classes of failure modes:

### Optimization Pathologies
- Exploding / vanishing gradients
- Loss plateaus
- Divergent training
- Learning rate instability

### Data Pathologies
- Training–serving skew
- Distribution drift
- Label leakage signals
- Feature collapse

### Numerical & Systems Issues
- NaNs / Infs
- Mixed-precision instability
- Silent precision loss
- Memory fragmentation / OOM precursors

### Distributed Systems Failures
- Non-determinism across replicas
- Gradient desynchronization
- Straggler effects
- Partial worker failure masking

## Extension Points

### Adding a Custom Detector

```python
from ml_systems_debugger.core.base import BaseDetector
from ml_systems_debugger.core.types import Signal, DetectionResult, FailureMode

class CustomDetector(BaseDetector):
    def __init__(self):
        super().__init__("custom_detector", FailureMode.CUSTOM_MODE)
    
    def detect(self, signals: List[Signal], context: Optional[Dict] = None) -> DetectionResult:
        # Your detection logic here
        return DetectionResult(
            failure_mode=self.failure_mode,
            detected=True,
            confidence=0.8,
            evidence={"key": "value"},
            detector_name=self.name
        )
    
    def get_required_signals(self) -> List[str]:
        return ["required_signal_name"]
```

### Adding a Custom Ingestor

```python
from ml_systems_debugger.core.base import BaseIngestor
from ml_systems_debugger.core.types import Signal

class CustomIngestor(BaseIngestor):
    def ingest(self, source: Any) -> List[Signal]:
        # Convert source to Signals
        return signals
    
    def validate_schema(self, data: Any) -> bool:
        # Validate data schema
        return True
```

## Examples

See `ml_systems_debugger/examples/end_to_end_example.py` for a complete example demonstrating:
- Setting up the debugger
- Ingesting synthetic metrics with injected failures
- Running detection and attribution
- Generating reports in multiple formats

Run the example:
```bash
python -m ml_systems_debugger.examples.end_to_end_example
```

## Design Tradeoffs

### Stateless vs. Stateful Detectors

**Choice**: Stateless detectors by default.

**Rationale**: Enables parallel execution, caching, and better testability. Stateful detectors can still be implemented but must manage state explicitly.

**Tradeoff**: Requires explicit context passing, which can be verbose but provides clarity.

### Rule-Based vs. ML-Based Attribution

**Choice**: Rule-based heuristics.

**Rationale**: Interpretability is critical for production debugging. Engineers need to understand why a hypothesis was generated.

**Tradeoff**: May miss complex correlations that ML could capture, but provides actionable, explainable results.

### Framework-Agnostic Signals

**Choice**: Normalize all data to framework-agnostic `Signal` objects.

**Rationale**: Allows detectors to work across frameworks without modification.

**Tradeoff**: Requires conversion overhead, but enables code reuse and framework independence.

### Pluggable Architecture

**Choice**: Explicit component registration (add_ingestor, add_detector, etc.).

**Rationale**: Provides flexibility and clear dependencies.

**Tradeoff**: More verbose than convention-based discovery, but more explicit and testable.

## Integration with ML Pipelines

### CI/CD Integration

The JSON report format is designed for CI/CD integration:

```python
report = debugger.debug(metrics)
json_report = debugger.generate_report(report, ReportFormat.JSON)

# Parse and check for critical issues
import json
report_data = json.loads(json_report)
critical_issues = [
    a for a in report_data["attributions"]
    if a["severity"] == "critical"
]

if critical_issues:
    # Fail CI/CD pipeline
    exit(1)
```

### Training Loop Integration

```python
# In your training loop
for step in range(num_steps):
    # ... training code ...
    
    # Collect metrics
    metrics.append({
        "name": "loss",
        "value": loss.item(),
        "step": step,
    })
    
    # Periodic debugging
    if step % 100 == 0:
        report = debugger.debug(metrics)
        if any(d.detected for d in report.detections):
            # Handle detected failures
            handle_failures(report)
```

## Performance Considerations

- **Signal Routing**: Signals are routed to detectors based on `get_required_signals()`, avoiding unnecessary detector execution.
- **Lazy Evaluation**: Detectors only process signals they require.
- **Parallel Execution**: Stateless detectors can be run in parallel (future enhancement).

## Future Enhancements

- Failure correlation across time
- Detector confidence calibration
- Hooks for distributed tracing (OpenTelemetry, etc.)
- Lightweight visualization outputs
- ML-based attribution for complex correlations
- Streaming detection for real-time monitoring

## Contributing

This is a research- and production-grade system. Contributions should:
- Maintain the modular, extensible architecture
- Include comprehensive documentation
- Explain design tradeoffs
- Include examples for new detectors/ingestors


## Acknowledgments

Designed for production ML systems at scale, with considerations for:
- Distributed training (multi-GPU, multi-node)
- Framework diversity (JAX, TensorFlow, PyTorch)
- Production observability requirements
- CI/CD integration needs

