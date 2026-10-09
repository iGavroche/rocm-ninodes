"""
Debug utilities for ROCM Ninodes.

Debug capture is compiled out; the hooks exist so call sites keep a single
shape.
"""

from typing import Any


DEBUG_MODE = False


def save_debug_data(*args: Any, **kwargs: Any) -> None:
    """Save debug data to disk (only active when DEBUG_MODE=True)"""
    if not DEBUG_MODE:
        return
    # Implementation would go here for actual data capture
    pass


def capture_timing(*args: Any, **kwargs: Any) -> None:
    """Capture timing information (only active when DEBUG_MODE=True)"""
    if not DEBUG_MODE:
        return
    # Implementation would go here for timing capture
    pass


def capture_memory_usage(*args: Any, **kwargs: Any) -> None:
    """Capture memory usage information (only active when DEBUG_MODE=True)"""
    if not DEBUG_MODE:
        return
    # Implementation would go here for memory capture
    pass


def log_debug(*args: Any, **kwargs: Any) -> None:
    """Log debug information (only active when DEBUG_MODE=True)"""
    if not DEBUG_MODE:
        return
    # Implementation would go here for debug logging
    pass


__all__ = [
    'DEBUG_MODE',
    'save_debug_data',
    'capture_timing',
    'capture_memory_usage',
    'log_debug',
]

