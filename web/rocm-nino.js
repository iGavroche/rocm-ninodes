/**
 * RocM-Nino: ROCM Optimized Nodes for ComfyUI
 * Frontend enhancements for the plugin
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Plugin information
const PLUGIN_NAME = "RocM-Nino";
const PLUGIN_VERSION = "1.0.0";

// Add plugin info to the app
app.registerExtension({
    name: PLUGIN_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add custom styling for ROCM nodes
        if (nodeData.name.startsWith("ROCM")) {
            nodeData.color = "#FF6B35"; // Orange color for ROCM nodes
            nodeData.bgcolor = "#FFF3E0";
        }
    },
    
    async setup() {
        console.log(`${PLUGIN_NAME} v${PLUGIN_VERSION} loaded`);
        
        // Add performance monitoring
        this.addPerformanceMonitoring();
    },
    
    addPerformanceMonitoring() {
        // Monitor node execution times
        const originalExecute = app.graphToPrompt;
        app.graphToPrompt = function(graph, app) {
            const startTime = performance.now();
            const result = originalExecute.call(this, graph, app);
            const endTime = performance.now();
            
            console.log(`RocM-Nino: Graph execution took ${(endTime - startTime).toFixed(2)}ms`);
            return result;
        };
    }
});

// Export for potential external use
window.RocMNino = {
    name: PLUGIN_NAME,
    version: PLUGIN_VERSION,
    nodes: [
        "ROCMOptimizedVAEDecode",
        "ROCMOptimizedVAEDecodeTiled",
        "ROCMOptimizedKSampler", 
        "ROCMOptimizedKSamplerAdvanced",
        "ROCMVAEPerformanceMonitor",
        "ROCMSamplerPerformanceMonitor"
    ]
};
