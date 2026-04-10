#!/bin/bash
set -e

rm -f /tmp/.X99-lock /tmp/.X11-unix/X99

Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 2

export DISPLAY=:99

fluxbox &>/dev/null &
sleep 1

x11vnc -display :99 -passwd "${VNC_PASSWORD:-grimmbot}" -forever -shared -rfbport 5900 -q -noxdamage &
sleep 1

websockify 6080 localhost:5900 &
sleep 1

exec python3 -u -m uvicorn grimmbot:app --host 0.0.0.0 --port 5000