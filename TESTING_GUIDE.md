# ROCM Ninodes Testing Guide

This guide explains how to test ROCM Ninodes, capture data for testing, and interpret benchmark results.

## Overview

ROCM Ninodes includes a comprehensive test suite designed to:
- Validate performance targets (78% Flux improvement, 5.6% WAN improvement)
- Ensure correctness of tensor shapes and data formats
- Prevent performance regressions
- Support both standalone testing and ComfyUI integration testing

## Test Categories

### 1. Standalone Tests (No ComfyUI Required)
- **Performance Tests**: Timing and memory usage validation
- **Correctness Tests**: Output shape and format validation
- **Mock Data Tests**: Tests using synthetic data when real data unavailable

### 2. Integration Tests (ComfyUI Required)
- **Flux Workflow Tests**: Complete 1024x1024 image generation
- **WAN Workflow Tests**: Complete 320x320 17-frame video generation
- **End-to-End Validation**: Full workflow execution and output verification

## Quick Start

### Run All Tests
```bash
cd tests
./run_tests.sh
```

### Run Specific Test Categories
```bash
# Performance tests only
pytest test_performance.py -v

# Correctness tests only
pytest test_correctness.py -v

# Integration tests (requires ComfyUI)
WITH_COMFYUI=1 ./run_tests.sh
```

## Data Capture

### Enable Debug Mode
```bash
export ROCM_NINODES_DEBUG=1
```

### Run ComfyUI with Debug Mode
```bash
cd /home/nino/ComfyUI
ROCM_NINODES_DEBUG=1 python main.py --use-pytorch-cross-attention --highvram --cache-none
```

### Captured Data Location
Data is automatically saved to `test_data/captured/`:
```
test_data/captured/
├── flux_1024x1024/         # Flux workflow data
├── wan_320x320_17frames/   # WAN workflow data
├── timing/                 # Performance timing data
└── memory/                 # Memory usage data
```

## Test Data Management

### List Available Data
```python
from test_fixtures import list_available_data
available = list_available_data()
print(available)
```

### Load Specific Data
```python
from test_fixtures import load_flux_vae_data, load_wan_vae_data

# Load Flux VAE data
flux_data = load_flux_vae_data()

# Load WAN VAE data
wan_data = load_wan_vae_data()
```

### Clean Up Old Data
```bash
# Remove data older than 7 days
find test_data/captured -name "*.pkl" -mtime +7 -delete

# Remove all captured data
rm -rf test_data/captured/*
```

## Performance Testing

### Performance Targets
- **Flux Checkpoint Load**: <30s
- **Flux Sampling**: <60s
- **Flux VAE Decode**: <10s
- **WAN Sampling**: <100s
- **WAN VAE Decode**: <10s

### Running Performance Tests
```bash
# Run with captured data
pytest test_performance.py -v

# Run with mock data (when no captured data available)
pytest test_performance.py::TestMockDataPerformance -v
```

### Interpreting Results
- ✅ **PASS**: Performance meets target
- ❌ **FAIL**: Performance exceeds target (regression)
- ⚠️ **SKIP**: No captured data available (uses mock data)

## Correctness Testing

### Shape Validation
- **Flux Input**: 4D tensor `[B, C, H, W]`
- **Flux Output**: 4D tensor `[B, H, W, C]` (RGB)
- **WAN Input**: 5D tensor `[B, C, T, H, W]`
- **WAN Output**: 4D tensor `[B*T, H, W, C]` (RGB)

### Running Correctness Tests
```bash
pytest test_correctness.py -v
```

## Integration Testing

### Prerequisites
- ComfyUI installed and accessible
- ROCM Ninodes nodes loaded in ComfyUI
- Debug mode enabled for data capture

### Running Integration Tests
```bash
# Set ComfyUI path (if not default)
export COMFYUI_PATH=/path/to/ComfyUI

# Run integration tests
WITH_COMFYUI=1 ./run_tests.sh
```

### Workflow Testing
1. **Flux Workflow**: Load checkpoint → Sample → VAE decode → Validate output
2. **WAN Workflow**: Sample video → VAE decode → Validate video output

## Benchmarking

### Establish Baseline
1. Run workflows with debug mode enabled
2. Capture timing and memory data
3. Record performance metrics
4. Document baseline performance

### Performance Analysis
```python
from test_fixtures import load_timing_data, load_memory_data

# Load timing data
timing = load_timing_data('vae_decode')
if timing:
    print(f"VAE decode time: {timing['duration']:.2f}s")

# Load memory data
memory = load_memory_data('vae_decode')
if memory:
    print(f"GPU memory used: {memory['gpu_memory_allocated'] / 1024**3:.2f} GB")
```

## Adding New Tests

### Test Structure
```python
def test_new_feature():
    """Test description"""
    # Arrange
    data = load_test_data()
    
    # Act
    result = perform_operation(data)
    
    # Assert
    assert result.meets_requirements()
```

### Performance Test Template
```python
def test_new_feature_performance():
    """Test new feature performance"""
    data = load_test_data()
    
    start_time = time.time()
    result = perform_operation(data)
    elapsed = time.time() - start_time
    
    assert elapsed < TARGET_TIME, f"Too slow: {elapsed:.2f}s"
```

### Correctness Test Template
```python
def test_new_feature_correctness():
    """Test new feature correctness"""
    data = load_test_data()
    result = perform_operation(data)
    
    assert result.shape == EXPECTED_SHAPE
    assert result.dtype == EXPECTED_DTYPE
```

## Troubleshooting

### Common Issues

#### No Captured Data Available
- **Symptom**: Tests skip with "No captured data available"
- **Solution**: Run ComfyUI with `ROCM_NINODES_DEBUG=1` and execute workflows

#### Performance Tests Fail
- **Symptom**: Tests fail with timing errors
- **Solution**: Check if performance targets are realistic, capture new baseline data

#### Integration Tests Fail
- **Symptom**: ComfyUI integration tests fail
- **Solution**: Verify ComfyUI path, ensure nodes are loaded, check debug mode

#### Memory Issues
- **Symptom**: Out of memory errors during testing
- **Solution**: Reduce batch sizes, clear GPU memory, use smaller test data

### Debug Mode
```bash
# Enable verbose output
pytest -v -s

# Run specific test with debug output
pytest test_performance.py::TestFluxPerformance::test_flux_vae_decode_performance -v -s

# Run with Python debugger
pytest --pdb test_performance.py
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: ROCM Ninodes Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install pytest torch
      - name: Run tests
        run: cd tests && ./run_tests.sh
```

## Best Practices

### Test Development
1. **Write tests first** (TDD approach)
2. **Use descriptive test names**
3. **Test edge cases and error conditions**
4. **Keep tests independent and isolated**
5. **Use mock data when real data unavailable**

### Performance Testing
1. **Establish baselines before optimization**
2. **Measure multiple times for statistical significance**
3. **Test with realistic data sizes**
4. **Monitor memory usage alongside timing**
5. **Document performance targets clearly**

### Data Management
1. **Clean up old test data regularly**
2. **Archive important test data**
3. **Use version control for test data when possible**
4. **Document data capture procedures**
5. **Validate data integrity before testing**

## Support

For testing issues or questions:
1. Check this guide first
2. Review test output and error messages
3. Check captured data availability
4. Verify ComfyUI configuration
5. Create issue with test logs and captured data
