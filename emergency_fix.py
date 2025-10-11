#!/usr/bin/env python3
"""
EMERGENCY PERFORMANCE FIX
Disable instrumentation system that's causing 173% slowdown
"""
import os
import sys

def emergency_fix():
    """Emergency fix for performance regression"""
    
    print("🚨 EMERGENCY PERFORMANCE FIX")
    print("=" * 50)
    print("Issue: Instrumentation system causing 173% slowdown (55s → 150s)")
    print("Root Cause: Pickle serialization + file I/O on every node execution")
    print("Solution: Disable instrumentation system")
    print()
    
    # Check if instrumentation.py exists
    if os.path.exists("instrumentation.py"):
        print("✅ Found instrumentation.py")
        
        # Create backup
        if not os.path.exists("instrumentation.py.backup"):
            os.system("cp instrumentation.py instrumentation.py.backup")
            print("✅ Created backup: instrumentation.py.backup")
        
        # Create no-op instrumentation
        no_op_content = '''"""
DISABLED INSTRUMENTATION - EMERGENCY PERFORMANCE FIX
The instrumentation system was causing massive performance overhead
"""

def instrument_node(node_class):
    """No-op decorator - instrumentation disabled for performance"""
    return node_class

# Create minimal instrumentation object
class NodeInstrumentation:
    def __init__(self, *args, **kwargs):
        pass
    
    def capture_inputs(self, *args, **kwargs):
        return ""
    
    def capture_outputs(self, *args, **kwargs):
        return ""
    
    def capture_performance(self, *args, **kwargs):
        return ""

# Create global instrumentation object
instrumentation = NodeInstrumentation()
'''
        
        # Write no-op instrumentation
        with open("instrumentation.py", "w") as f:
            f.write(no_op_content)
        
        print("✅ Replaced instrumentation.py with no-op version")
        print("✅ Instrumentation system disabled")
        
    else:
        print("⚠️  instrumentation.py not found")
    
    # Check nodes.py for instrumentation usage
    if os.path.exists("nodes.py"):
        print("✅ Found nodes.py")
        
        # Check if nodes.py imports instrumentation
        with open("nodes.py", "r") as f:
            content = f.read()
            
        if "from instrumentation import instrument_node" in content:
            print("⚠️  nodes.py imports instrumentation")
            print("✅ This will now use the no-op version")
        else:
            print("✅ nodes.py doesn't import instrumentation")
    
    print()
    print("🎉 EMERGENCY FIX COMPLETE")
    print("=" * 50)
    print("Performance should be restored to original levels")
    print("Render time should return from 150s back to ~55s")
    print()
    print("To restore instrumentation later:")
    print("  mv instrumentation.py.backup instrumentation.py")

if __name__ == "__main__":
    emergency_fix()
