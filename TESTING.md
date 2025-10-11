# ROCM Ninodes Testing Framework

This document describes the comprehensive testing framework for ROCM Ninodes, including unit tests, integration tests, benchmarks, and data collection.

## Overview

The testing framework is designed to:
- Validate node functionality across different configurations
- Measure performance improvements on AMD GPUs
- Collect real-world data from Flux and WAN workflows
- Optimize nodes based on test results
- Ensure reliability and prevent regressions

## Directory Structure

```
tests/
├── unit/                    # Unit tests for individual nodes
│   ├── test_vae_decode.py
│   ├── test_ksampler.py
│   ├── test_checkpoint_loader.py
│   ├── test_vae_error_scenarios.py
│   └── test_vae_optimization.py
├── integration/             # Integration tests for workflows
│   └── test_workflows.py
├── benchmarks/              # Performance benchmark tests
│   └── test_performance.py
├── fixtures/                # Test fixtures and data
├── conftest.py             # Pytest configuration and fixtures
├── run_tests.py            # Test runner script
└── __init__.py

test_data/                   # Collected test data
├── inputs/                 # Node input data
├── outputs/                # Node output data
├── benchmarks/             # Performance metrics
├── optimization/           # Optimization data
└── debug/                  # Debug data

instrumentation.py           # Data collection instrumentation
collect_test_data.py         # Data collection script
pytest.ini                  # Pytest configuration
```

## Test Types

### Unit Tests

Unit tests validate individual node functionality:

- **test_vae_decode.py**: Tests for ROCMOptimizedVAEDecode and ROCMOptimizedVAEDecodeTiled
- **test_ksampler.py**: Tests for ROCMOptimizedKSampler and ROCMOptimizedKSamplerAdvanced
- **test_checkpoint_loader.py**: Tests for ROCMOptimizedCheckpointLoader
- **test_vae_error_scenarios.py**: Error handling and edge cases
- **test_vae_optimization.py**: Optimization-specific tests

### Integration Tests

Integration tests validate complete workflows:

- **test_workflows.py**: Tests for Flux and WAN workflows
- Validates end-to-end functionality
- Tests node interactions
- Validates workflow compatibility

### Benchmark Tests

Benchmark tests measure performance:

- **test_performance.py**: Performance benchmarks
- Measures execution time and memory usage
- Compares different configurations
- Tracks performance improvements

## Running Tests

### Prerequisites

Install required dependencies:

```bash
pip install pytest pytest-mock torch numpy
```

### Run All Tests

```bash
python tests/run_tests.py
```

### Run Specific Test Types

```bash
# Unit tests only
python tests/run_tests.py unit

# Integration tests only
python tests/run_tests.py integration

# Benchmark tests only
python tests/run_tests.py benchmark
```

### Run Individual Test Files

```bash
# Run specific test file
pytest tests/unit/test_vae_decode.py -v

# Run with specific markers
pytest -m "benchmark" -v
pytest -m "integration" -v
pytest -m "requires_gpu" -v
```

## Data Collection

### Collect Test Data

Run the data collection script to gather test data from workflows:

```bash
python collect_test_data.py
```

This will:
- Instrument all nodes for data collection
- Run Flux and WAN workflows
- Collect inputs, outputs, and performance metrics
- Generate optimization data

### Data Collection Features

- **Automatic Instrumentation**: All nodes are automatically instrumented
- **Workflow Testing**: Tests both Flux and WAN workflows
- **Performance Profiling**: Collects execution time and memory usage
- **Configuration Testing**: Tests different precision modes and settings
- **Error Handling**: Captures data even when errors occur

## Test Configuration

### Pytest Configuration

The `pytest.ini` file contains:
- Test discovery patterns
- Output formatting
- Markers for different test types
- Timeout settings
- Warning filters

### Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.benchmark`: Performance tests
- `@pytest.mark.slow`: Slow tests
- `@pytest.mark.requires_gpu`: Tests requiring GPU
- `@pytest.mark.requires_amd`: Tests requiring AMD GPU

## Instrumentation

### Node Instrumentation

The `instrumentation.py` module provides:
- **Input Capture**: Saves all node inputs
- **Output Capture**: Saves all node outputs
- **Performance Metrics**: Tracks execution time and memory usage
- **Data Serialization**: Converts tensors to serializable format
- **Test Fixtures**: Generates test fixtures from collected data

### Data Collection Process

1. **Instrument Nodes**: Automatically instrument all ROCM nodes
2. **Run Workflows**: Execute Flux and WAN workflows
3. **Capture Data**: Save inputs, outputs, and performance metrics
4. **Generate Fixtures**: Create test fixtures from collected data
5. **Optimize**: Use data for node optimization

## Performance Testing

### Benchmark Categories

1. **Resolution Tests**: Different image/video resolutions
2. **Precision Tests**: Different precision modes (fp32, fp16, bf16)
3. **Configuration Tests**: Different tile sizes and settings
4. **Memory Tests**: Memory usage and optimization
5. **Workflow Tests**: Complete workflow performance

### Performance Metrics

- **Execution Time**: Time to complete operations
- **Memory Usage**: Peak and average memory consumption
- **Throughput**: Operations per second
- **Efficiency**: Memory efficiency ratios
- **Improvement**: Performance improvements over baseline

## Optimization

### Data-Driven Optimization

The testing framework enables data-driven optimization:

1. **Collect Data**: Run workflows and collect performance data
2. **Analyze Results**: Identify bottlenecks and optimization opportunities
3. **Implement Changes**: Optimize nodes based on findings
4. **Validate Improvements**: Re-run tests to verify improvements
5. **Track Progress**: Monitor performance over time

### Optimization Targets

- **Memory Usage**: Reduce VRAM consumption
- **Execution Time**: Improve processing speed
- **Stability**: Reduce errors and crashes
- **Compatibility**: Ensure cross-platform compatibility
- **Scalability**: Support larger images and videos

## Test Reports

### Generated Reports

The test runner generates comprehensive reports:

- **Test Summary**: Pass/fail statistics
- **Performance Metrics**: Execution times and memory usage
- **Device Information**: GPU and system details
- **Recommendations**: Optimization suggestions
- **Error Analysis**: Failed test details

### Report Location

Reports are saved to `test_data/test_report.json` and include:
- Test results for all test types
- Performance benchmarks
- Device compatibility information
- Optimization recommendations

## Continuous Integration

### Automated Testing

The testing framework supports CI/CD:

```yaml
# Example GitHub Actions workflow
name: ROCM Ninodes Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/run_tests.py
```

### Test Coverage

- **Unit Tests**: 100% node coverage
- **Integration Tests**: Complete workflow coverage
- **Benchmark Tests**: Performance validation
- **Error Tests**: Edge case coverage

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Errors**: Check GPU availability and drivers
3. **Memory Errors**: Reduce batch sizes or use CPU fallback
4. **Timeout Errors**: Increase timeout values for slow tests

### Debug Mode

Run tests in debug mode for detailed output:

```bash
pytest -v -s --tb=long
```

### Test Data Issues

If test data is missing:
1. Run `python collect_test_data.py` to collect data
2. Check that workflow files exist
3. Verify node instrumentation is working

## Contributing

### Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use appropriate markers
4. Include docstrings and comments
5. Test both success and failure cases

### Test Guidelines

- **Isolation**: Tests should be independent
- **Deterministic**: Tests should produce consistent results
- **Fast**: Unit tests should be fast
- **Clear**: Test names should be descriptive
- **Comprehensive**: Cover edge cases and error conditions

## Performance Monitoring

### Benchmark Tracking

The framework tracks performance over time:
- Historical performance data
- Regression detection
- Improvement measurement
- Device-specific optimization

### Optimization Metrics

Key metrics for optimization:
- **Speed**: Execution time improvements
- **Memory**: VRAM usage reduction
- **Stability**: Error rate reduction
- **Compatibility**: Cross-platform support

## Future Enhancements

### Planned Features

1. **Automated Optimization**: AI-driven node optimization
2. **Real-time Monitoring**: Live performance tracking
3. **A/B Testing**: Compare different implementations
4. **Regression Testing**: Automatic performance regression detection
5. **Cloud Testing**: Test on different hardware configurations

### Extensibility

The framework is designed to be extensible:
- Easy to add new test types
- Pluggable instrumentation
- Configurable test parameters
- Custom performance metrics

## Support

For issues with the testing framework:
1. Check the troubleshooting section
2. Review test logs and reports
3. Create an issue with detailed information
4. Include test data and error messages

The testing framework is essential for maintaining code quality and performance in ROCM Ninodes. Regular testing ensures reliability and helps identify optimization opportunities.