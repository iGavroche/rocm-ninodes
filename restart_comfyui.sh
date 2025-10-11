#!/bin/bash

# restart_comfyui.sh - Restart ComfyUI with proper cleanup
# This script kills existing ComfyUI processes and starts fresh using existing start.sh

echo "🔄 Restarting ComfyUI..."

# Kill existing processes
echo "1️⃣ Killing existing ComfyUI processes..."
./kill_comfyui.sh

# Wait a moment
sleep 3

# Start fresh using existing start.sh
echo "2️⃣ Starting fresh ComfyUI instance using existing start.sh..."
if [ "$1" = "debug" ]; then
    echo "🐛 Restarting in DEBUG mode"
    cd /home/nino/ComfyUI
    export ROCM_NINODES_DEBUG=1
    ./start.sh
else
    echo "⚡ Restarting in PRODUCTION mode"
    cd /home/nino/ComfyUI
    unset ROCM_NINODES_DEBUG
    ./start.sh
fi

echo "🏁 ComfyUI restart completed"
