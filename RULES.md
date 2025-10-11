# ROCM Ninodes Development Rules

## Testing Rules

### Mandatory Testing
- **All Changes**: Every code change must have corresponding tests
- **Performance Regression**: No change can regress performance by >5%
- **Coverage**: Critical paths must have >90% test coverage
- **Validation**: All tests must pass before merging

### Test Categories
1. **Unit Tests**: Individual node functionality
2. **Performance Tests**: Timing and memory usage
3. **Integration Tests**: Full workflow execution
4. **Regression Tests**: Prevent performance degradation

### Test Execution
- **Standalone**: Tests must run without ComfyUI when possible
- **Data-Driven**: Use captured pickle data for consistent testing
- **Automated**: CI/CD pipeline integration required
- **Documentation**: All test failures must be documented

## Benchmarking Rules

### Baseline Requirements
- **Before Optimization**: Measure baseline performance first
- **A/B Testing**: Compare before/after for every change
- **Multiple Runs**: Minimum 3 runs for statistical significance
- **Documentation**: Record all benchmark results

### Performance Targets
- **Flux 1024x1024**: ≥78% improvement over baseline
- **WAN 320x320 17-frame**: ≥5.6% improvement over baseline
- **VAE Decode**: <10s for any supported resolution
- **Checkpoint Load**: <30s for Flux models

### Benchmark Process
1. **Capture Data**: Use `ROCM_NINODES_DEBUG=1` to capture test data
2. **Run Baseline**: Measure current performance
3. **Implement Change**: Make optimization
4. **Run Comparison**: Measure new performance
5. **Validate**: Ensure improvement meets targets
6. **Document**: Record results and methodology

## Architecture Constraints

### gfx1151 Specific
- **Precision**: fp32 preferred over fp16 for stability
- **Tile Sizes**: 768-1024 optimal range
- **TF32**: Disabled (not beneficial on gfx1151)
- **Memory**: Conservative allocation (1.5x modifier for attention)
- **Batch Size**: Conservative to prevent OOM

### Memory Management
- **Allocation**: Conservative approach to prevent OOM
- **Cleanup**: Regular `torch.cuda.empty_cache()` calls
- **Monitoring**: Track memory usage in all operations
- **Limits**: Respect 128GB unified RAM constraint

### Data Formats
- **Video**: 5D tensor input `[B, C, T, H, W]` → 4D output `[B*T, H, W, C]`
- **Images**: Standard 4D tensor `[B, H, W, C]`
- **Latents**: ComfyUI format `{"samples": tensor}`
- **Precision**: Maintain dtype consistency throughout pipeline

## Vision and Goals

### Performance Targets
- **Flux Workflow**: 78% improvement (≤110s total time)
- **WAN Workflow**: 5.6% improvement (≤93s sampling time)
- **VAE Decode**: <10s for any resolution
- **Memory Usage**: <80% of available RAM

### Quality Standards
- **Reliability**: Zero crashes during normal operation
- **Consistency**: Reproducible results across runs
- **Compatibility**: Works with standard ComfyUI workflows
- **Maintainability**: Clean, documented, testable code

### User Experience
- **Ease of Use**: Drop-in replacement for standard nodes
- **Performance**: Noticeable speedup without configuration
- **Debugging**: Clear error messages and logging
- **Documentation**: Comprehensive guides and examples

## Best Practices

### Code Quality
- **Style**: Follow PEP 8 and project conventions
- **Documentation**: Docstrings for all public methods
- **Type Hints**: Use type annotations where possible
- **Error Handling**: Comprehensive exception handling

### Performance
- **Profiling**: Profile before and after optimizations
- **Memory**: Monitor memory usage in all operations
- **Caching**: Use appropriate caching strategies
- **Batching**: Optimize batch sizes for gfx1151

### Testing
- **Isolation**: Tests should not depend on external state
- **Data**: Use captured real data for testing
- **Coverage**: Test edge cases and error conditions
- **Automation**: Integrate with CI/CD pipeline

### Debugging
- **Logging**: Use appropriate log levels
- **Debug Mode**: `ROCM_NINODES_DEBUG=1` for data capture
- **Error Messages**: Clear, actionable error messages
- **Documentation**: Document known issues and workarounds

## Research Process

### Methodology
1. **Measure First**: Establish baseline performance
2. **Optimize Second**: Implement targeted improvements
3. **Validate Third**: Ensure improvements are real and stable

### Optimization Strategy
- **Targeted**: Focus on identified bottlenecks
- **Incremental**: Small, testable changes
- **Measurable**: Quantify all improvements
- **Reversible**: Changes must be easily revertible

### Documentation
- **Process**: Document optimization methodology
- **Results**: Record all performance measurements
- **Decisions**: Explain optimization choices
- **Lessons**: Capture learnings for future work

## Quality Gates

### Pre-commit
- **Linting**: Code must pass all linting checks
- **Tests**: All tests must pass
- **Performance**: No performance regression
- **Documentation**: Updated documentation required

### Pre-merge
- **Integration**: Full integration test suite passes
- **Performance**: Meets performance targets
- **Compatibility**: Works with existing workflows
- **Review**: Code review approval required

### Release
- **Stability**: No known critical issues
- **Performance**: Meets all performance targets
- **Documentation**: Complete user documentation
- **Testing**: Full test suite passes

## Error Handling

### Error Categories
1. **Critical**: System crashes or data corruption
2. **Warning**: Performance degradation or unexpected behavior
3. **Info**: Normal operation with additional information

### Response Strategy
- **Critical**: Immediate fix required, no release until resolved
- **Warning**: Fix in next release cycle
- **Info**: Document and monitor

### Logging
- **Levels**: Use appropriate log levels
- **Context**: Include relevant context in error messages
- **Stack Traces**: Include full stack traces for debugging
- **User Action**: Provide actionable next steps

## Memory Management

### Allocation Strategy
- **Conservative**: Prefer under-allocation to over-allocation
- **Monitoring**: Track memory usage in all operations
- **Cleanup**: Explicit cleanup after operations
- **Limits**: Respect system memory constraints

### Optimization
- **Reuse**: Reuse memory where possible
- **Batching**: Optimize batch sizes for memory efficiency
- **Tiling**: Use tiled processing for large operations
- **Streaming**: Process data in streams when possible

## Performance Monitoring

### Metrics
- **Timing**: Measure execution time for all operations
- **Memory**: Track memory usage and allocation
- **Throughput**: Measure data processing rates
- **Quality**: Validate output correctness

### Tools
- **Profiling**: Use PyTorch profiler for detailed analysis
- **Monitoring**: Real-time performance monitoring
- **Logging**: Structured logging for analysis
- **Visualization**: Performance dashboards and reports

## ComfyUI Process Management

### Process Control Scripts
- **Kill Script**: `./kill_comfyui.sh` - Properly terminates ComfyUI processes
- **Start Script**: `/home/nino/ComfyUI/start.sh` - Existing script with ROCm optimizations
- **Restart Script**: `./restart_comfyui.sh` - Combines kill and start for clean restarts

### Process Management Rules
- **NEVER use `pkill -f "python main.py"`** - This method does not work reliably
- **ALWAYS use `ps aux | grep python | grep -v grep`** to find actual process IDs
- **THEN use `kill -9 <PID>`** to forcefully terminate processes
- **VERIFY termination** with another `ps aux` check before restarting

### Debug Mode Control
- **Production Mode**: Use existing `/home/nino/ComfyUI/start.sh` (no debug overhead)
- **Debug Mode**: Set `ROCM_NINODES_DEBUG=1` environment variable before starting
- **Code Changes**: Always restart ComfyUI after code changes to ensure they're loaded

### Workflow Integration
1. **Before Code Changes**: Run `./kill_comfyui.sh` to stop ComfyUI
2. **Make Changes**: Edit code files
3. **After Code Changes**: Use `/home/nino/ComfyUI/start.sh` to restart
4. **For Debug**: Set `ROCM_NINODES_DEBUG=1` and restart
5. **Verify**: Check that changes are loaded and working
