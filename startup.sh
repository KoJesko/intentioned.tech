#!/bin/bash
# Kill any existing server instances first
pkill -9 -f "server.py" 2>/dev/null
sleep 2

cd /home/ari-tower-linux/Git/intentioned-tech-voice-assistant
export SERVER_PORT=6942
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/ari-tower-linux/Git/intentioned-tech-voice-assistant/myenv/bin/python /home/ari-tower-linux/Git/intentioned-tech-voice-assistant/server.py