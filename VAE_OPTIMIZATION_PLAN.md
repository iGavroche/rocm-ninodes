# ROCM VAE Decoder Optimization Plan
# Generated: $(date)
# Project: rocm-ninodes
# Target: AMD Strix Halo (gfx1151) with ROCm 6.4+

## 🎯 Optimization Objectives

### Primary Goals
1. **Performance**: Achieve <10s VAE decode for WAN video (320x320px, 2-17 frames)
2. **Memory Efficiency**: Minimize VRAM usage for 128GB Unified RAM system
3. **Stability**: Eliminate AttributeError and tuple index errors
4. **Compatibility**: Maintain ComfyUI compatibility while optimizing for ROCm

### Current Status
- ✅ Fixed AttributeError: 'dict' object has no attribute 'shape'
- ✅ Fixed tuple index out of range error
- ✅ Identified correct VAE.decode() input format (tensor directly, not LATENT dict)
- 🔄 Testing and optimization phase

## 📊 Test Data Structure

```
test_data/
├── debug/
│   ├── wan_vae_input_debug_{timestamp}.pkl    # Raw input data from workflow
│   └── wan_vae_test_data_{timestamp}.pkl     # Processed test data
├── optimization/
│   ├── vae_optimization_results_{timestamp}.pkl
│   ├── memory_optimization_results_{timestamp}.pkl
│   └── precision_test_results_{timestamp}.pkl
└── benchmarks/
    ├── wan_workflow_benchmarks_{timestamp}.json
    └── performance_comparison_{timestamp}.json
```

## 🔧 Optimization Strategies

### 1. Memory Management
- **Current**: Basic chunked processing (1 frame chunks)
- **Target**: Optimize chunk size based on VRAM usage
- **Methods**:
  - Dynamic chunk sizing based on available memory
  - Memory cleanup between chunks
  - CPU offloading for large videos
  - Memory fragmentation reduction

### 2. Precision Optimization
- **Current**: fp32 (conservative for ROCm)
- **Target**: Test fp16/bf16 for better performance
- **Methods**:
  - Automatic precision selection based on GPU
  - Mixed precision training compatibility
  - ROCm-specific precision tuning

### 3. Batch Processing
- **Current**: Single frame processing
- **Target**: Optimal batch sizes for gfx1151
- **Methods**:
  - Adaptive batch sizing
  - Temporal batching for video
  - Memory-aware batch limits

### 4. ROCm-Specific Optimizations
- **Current**: Basic ROCm detection
- **Target**: Advanced ROCm tuning
- **Methods**:
  - HIP memory management
  - ROCm attention optimizations
  - AMD-specific kernel tuning

## 🧪 Testing Framework

### Test Scripts
1. **debug_vae_decode.py**: Basic debugging and data collection
2. **test_vae_optimization.py**: Comprehensive optimization testing
3. **benchmark_vae_performance.py**: Performance benchmarking
4. **validate_vae_output.py**: Output quality validation

### Test Cases
1. **Single Frame**: 320x320px latent decode
2. **Short Video**: 2-5 frames, 320x320px
3. **Medium Video**: 10-17 frames, 320x320px
4. **Large Video**: 30+ frames, 480x720px
5. **Memory Stress**: Maximum resolution tests

### Performance Metrics
- **Decode Time**: Target <10s for 17 frames
- **Memory Usage**: Peak VRAM consumption
- **Quality**: Output image/video quality
- **Stability**: Error rate and crash frequency

## 📈 Implementation Phases

### Phase 1: Data Collection (Current)
- ✅ Collect real WAN workflow data
- ✅ Create test data structure
- ✅ Implement debugging framework
- 🔄 Gather baseline performance metrics

### Phase 2: Optimization Testing
- 🔄 Test different chunk sizes (1, 2, 4, 8 frames)
- 🔄 Test precision modes (fp32, fp16, bf16)
- 🔄 Test memory optimization strategies
- 🔄 Test batch processing approaches

### Phase 3: ROCm Tuning
- 🔄 Implement dynamic memory management
- 🔄 Add ROCm-specific optimizations
- 🔄 Optimize attention mechanisms
- 🔄 Tune for gfx1151 architecture

### Phase 4: Integration & Validation
- 🔄 Integrate optimizations into main code
- 🔄 Validate with real WAN workflows
- 🔄 Performance regression testing
- 🔄 Documentation and cleanup

## 🎛️ Configuration Parameters

### Memory Management
```python
MEMORY_OPTIMIZATION_CONFIG = {
    'chunk_size': 'auto',  # 1, 2, 4, 8, or 'auto'
    'memory_threshold': 0.8,  # Use chunking when memory > 80%
    'cleanup_frequency': 1,  # Cleanup after every N chunks
    'cpu_offload': True,  # Offload to CPU when needed
}
```

### Precision Settings
```python
PRECISION_CONFIG = {
    'mode': 'auto',  # 'auto', 'fp32', 'fp16', 'bf16'
    'rocm_preferred': 'fp32',  # Preferred for ROCm
    'fallback': 'fp32',  # Fallback precision
    'mixed_precision': False,  # Enable mixed precision
}
```

### ROCm Optimizations
```python
ROCM_CONFIG = {
    'enable_hip_memory': True,
    'attention_optimization': True,
    'kernel_tuning': True,
    'memory_fraction': 0.9,
    'synchronization': True,
}
```

## 📋 Success Criteria

### Performance Targets
- **WAN 2.2 i2v 320x320px 2s**: <5s decode time
- **WAN 2.2 i2v 320x320px 17 frames**: <10s decode time
- **Memory usage**: <50GB peak VRAM
- **Error rate**: <1% failure rate

### Quality Targets
- **Output quality**: No degradation from standard VAE
- **Compatibility**: Works with all ComfyUI workflows
- **Stability**: No crashes or memory leaks

## 🔍 Monitoring & Debugging

### Debug Output
- Real-time performance metrics
- Memory usage tracking
- Error logging and reporting
- Optimization recommendations

### Logging
- Performance timings
- Memory usage patterns
- Error occurrences
- Optimization effectiveness

## 📚 Documentation

### Code Documentation
- Inline comments for optimization logic
- Performance tuning guidelines
- ROCm-specific notes
- Troubleshooting guide

### User Documentation
- Optimization parameter explanations
- Performance tuning recommendations
- Common issues and solutions
- Best practices for AMD GPUs

## 🚀 Next Steps

1. **Run workflow** to generate test data with new file structure
2. **Execute optimization tests** using test_vae_optimization.py
3. **Analyze results** and identify best strategies
4. **Implement optimizations** based on test results
5. **Validate** with real WAN workflows
6. **Document** findings and recommendations

---

*This plan will be updated as we gather more data and test results.*
