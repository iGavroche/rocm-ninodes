#!/usr/bin/env python3
"""
DISABLE INSTRUMENTATION - EMERGENCY FIX
The instrumentation system is causing massive performance overhead
"""
import os
import sys

def disable_instrumentation():
    """Disable instrumentation to fix performance regression"""
    
    # Create a no-op instrumentation decorator
    def no_op_instrument_node(node_class):
        """No-op decorator - does nothing"""
        return node_class
    
    # Replace the instrumentation module
    instrumentation_module = sys.modules.get('instrumentation')
    if instrumentation_module:
        instrumentation_module.instrument_node = no_op_instrument_node
        print("✅ Instrumentation disabled - performance should be restored")
    else:
        print("⚠️  Instrumentation module not found")

if __name__ == "__main__":
    disable_instrumentation()
