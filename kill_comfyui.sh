#!/bin/bash

# kill_comfyui.sh - Properly kill ComfyUI process
# This script finds and kills ComfyUI processes using the correct method

echo "🔍 Searching for ComfyUI processes..."

# Find ComfyUI processes
PIDS=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "✅ No ComfyUI processes found - already stopped"
    exit 0
fi

echo "📋 Found ComfyUI processes: $PIDS"

# Kill each process
for PID in $PIDS; do
    echo "🔪 Killing ComfyUI process $PID..."
    kill -9 $PID
    if [ $? -eq 0 ]; then
        echo "✅ Successfully killed process $PID"
    else
        echo "❌ Failed to kill process $PID"
    fi
done

# Wait a moment for processes to fully terminate
sleep 2

# Verify all processes are killed
REMAINING=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}')
if [ -z "$REMAINING" ]; then
    echo "✅ All ComfyUI processes successfully terminated"
else
    echo "⚠️  Warning: Some processes may still be running: $REMAINING"
fi

echo "🏁 ComfyUI kill script completed"
