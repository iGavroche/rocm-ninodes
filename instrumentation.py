"""
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
