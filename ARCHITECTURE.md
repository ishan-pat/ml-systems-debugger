# ML Systems Debugger - Architecture Document

## System Overview

The ML Systems Debugger is designed as a modular, extensible framework for detecting and attributing failure modes in ML training and inference pipelines. The architecture emphasizes:

1. **Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **Framework Independence**: Core components work with any ML framework
3. **Extensibility**: New detectors, ingestors, and extractors can be added without modifying core code
4. **Production Readiness**: Designed for integration into CI/CD pipelines and production monitoring

## Component Architecture

### 1. Ingestion Layer (`ml_systems_debugger/ingestion/`)

**Purpose**: Convert raw data (logs, metrics, traces) into normalized `Signal` objects.

**Components**:
- `MetricIngestor`: Handles training/inference metrics
- `LogIngestor`: Parses structured and unstructured logs
- `SchemaValidator`: Validates input data schemas

**Design Decisions**:
- **Pluggable Ingestors**: Allows framework-specific adapters without core changes
- **Multiple Format Support**: Handles list-of-dicts, dict-of-lists, and framework objects
- **Schema Validation**: Early validation prevents downstream errors

**Tradeoffs**:
- Explicit ingestor selection required (more verbose) but provides flexibility
- Framework adapters deferred to production (placeholder implementation)

### 2. Signal Extraction Layer (`ml_systems_debugger/signals/`)

**Purpose**: Extract structured signals from raw data for detector consumption.

**Components**:
- `GradientSignalExtractor`: Gradient norms, means, stds
- `DistributionSignalExtractor`: KL divergence, feature statistics
- `NumericalSignalExtractor`: NaN/Inf detection, precision checks

**Design Decisions**:
- **Separate from Ingestion**: Allows multiple extractors on same data
- **Framework-Agnostic**: Converts to numpy arrays internally
- **Rich Metadata**: Signals include step, layer, replica context

**Tradeoffs**:
- Conversion overhead but enables framework independence
- Metadata richness increases memory but improves attribution

### 3. Detection Layer (`ml_systems_debugger/detectors/`)

**Purpose**: Identify specific failure modes from signals.

**Components**:
- **Optimization**: `GradientExplosionDetector`, `GradientVanishingDetector`, `LossPlateauDetector`, `DivergentTrainingDetector`
- **Numerical**: `NaNDetectionDetector`, `InfDetectionDetector`
- **Data**: `TrainingServingSkewDetector`
- **Distributed**: `DistributedDesyncDetector`

**Design Decisions**:
- **Stateless by Default**: Enables parallelism and caching
- **Configurable Thresholds**: Balance sensitivity vs. false positives
- **Signal Routing**: Detectors declare required signals for efficient routing

**Tradeoffs**:
- Stateless requires explicit context passing (verbose) but enables parallelism
- Threshold tuning required per use case (no universal defaults)

### 4. Attribution Engine (`ml_systems_debugger/attribution/`)

**Purpose**: Map detections to root cause hypotheses with confidence scores.

**Components**:
- `AttributionEngine`: Rule-based hypothesis generation

**Design Decisions**:
- **Rule-Based**: Interpretable, fast, production-ready
- **Cross-Mode Correlation**: Detects correlated failure modes
- **Severity Classification**: Prioritizes critical issues
- **Remediation Hints**: Actionable guidance for engineers

**Tradeoffs**:
- Rule-based may miss complex correlations (ML could help) but provides interpretability
- Heuristic confidence scoring (could be calibrated with historical data)

### 5. Reporting Layer (`ml_systems_debugger/reporting/`)

**Purpose**: Generate human- and machine-readable reports.

**Components**:
- `Reporter`: Multi-format report generation
- `ReportFormat`: Text, JSON, Markdown

**Design Decisions**:
- **Multiple Formats**: Different formats for different use cases
- **Structured Output**: JSON enables CI/CD integration
- **Human-Readable**: Text/Markdown for manual review

**Tradeoffs**:
- Format selection required (could auto-detect) but provides explicit control

## Core Abstractions

### Base Classes

All components inherit from base classes that define contracts:

- `BaseDetector`: `detect(signals, context) -> DetectionResult`
- `BaseIngestor`: `ingest(source) -> List[Signal]`
- `BaseSignalExtractor`: `extract(raw_data) -> List[Signal]`

**Design Rationale**: Clear contracts enable:
- Easy testing (mock implementations)
- Type safety (via type hints)
- Documentation (contracts are self-documenting)

### Type System

Core types (`Signal`, `DetectionResult`, `AttributionHypothesis`, `DebugReport`) are dataclasses with:
- **Immutability**: Dataclasses are hashable and immutable-friendly
- **Rich Metadata**: Support for extensible metadata dicts
- **Type Safety**: Type hints throughout

**Tradeoffs**:
- Dataclasses are Python 3.7+ only (acceptable for modern ML systems)
- Metadata dicts are untyped (flexibility vs. type safety)

## Data Flow

```
Raw Data (logs/metrics/traces)
    ↓
[Ingestion Layer] → List[Signal]
    ↓
[Signal Extraction Layer] → Enhanced List[Signal]
    ↓
[Detection Layer] → List[DetectionResult]
    ↓
[Attribution Engine] → List[AttributionHypothesis]
    ↓
[Reporting Layer] → Formatted Report
```

## Extension Points

### Adding a New Detector

1. Inherit from `BaseDetector`
2. Implement `detect()` method
3. Override `get_required_signals()` if needed
4. Register with `debugger.add_detector()`

**Example**: See `detectors/gradient_detector.py`

### Adding a New Ingestor

1. Inherit from `BaseIngestor`
2. Implement `ingest()` and `validate_schema()`
3. Register with `debugger.add_ingestor()`

**Example**: See `ingestion/metric_ingestor.py`

### Adding a New Signal Extractor

1. Inherit from `BaseSignalExtractor`
2. Implement `extract()` method
3. Register with `debugger.add_signal_extractor()`

**Example**: See `signals/gradient_extractor.py`

## Performance Considerations

### Signal Routing

Signals are routed to detectors based on `get_required_signals()`. This avoids:
- Unnecessary detector execution
- Signal copying overhead
- Memory bloat

**Optimization**: Routing cache is built once and reused.

### Parallel Execution

Stateless detectors can be run in parallel (future enhancement):
- Use `concurrent.futures.ThreadPoolExecutor` or `ProcessPoolExecutor`
- Detectors must be truly stateless
- Context must be thread-safe

### Memory Management

- Signals are lightweight (dataclasses)
- Large arrays should be referenced, not copied
- Consider streaming for very long training runs

## Scalability Considerations

### Distributed Training

- `DistributedDesyncDetector` handles multi-replica scenarios
- Signals include `replica_id` for correlation
- Attribution engine handles cross-replica correlations

### Long Training Runs

- Detectors operate on signal windows (configurable)
- Historical signals can be sampled (not all required)
- Consider checkpointing detection state for very long runs

### High-Frequency Monitoring

- Batch signal processing (current implementation)
- Streaming support (future enhancement)
- Detector execution can be rate-limited

## Testing Strategy

### Unit Tests

Each component should have unit tests:
- Detectors: Test with synthetic signals
- Ingestors: Test with various input formats
- Extractors: Test with framework objects

### Integration Tests

- End-to-end pipeline tests
- Multi-detector scenarios
- Cross-mode correlation tests

### Production Validation

- A/B testing detector thresholds
- Confidence calibration with labeled failures
- False positive/negative tracking

## Future Enhancements

1. **ML-Based Attribution**: Use ML models for complex correlation detection
2. **Streaming Detection**: Real-time failure detection during training
3. **Visualization**: Lightweight plots for failure trends
4. **Distributed Tracing**: Integration with OpenTelemetry, etc.
5. **Confidence Calibration**: Calibrate detector confidence with historical data
6. **Adaptive Thresholds**: Learn optimal thresholds from historical data

## Design Philosophy Summary

1. **Failures as First-Class Signals**: Failures are not exceptions; they're data
2. **Framework Independence**: Core logic works with any framework
3. **Extensibility Over Completeness**: Easy to add new components
4. **Production Readiness**: Designed for real-world constraints
5. **Interpretability**: Engineers must understand why failures were detected
6. **Actionability**: Every detection should lead to actionable remediation

