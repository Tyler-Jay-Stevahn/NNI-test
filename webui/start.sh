#!/usr/bin/env bash
# start.sh — launch the NNI-Test dashboard on 0.0.0.0:6123 (foreground).
# Zero external dependencies (pure Python stdlib).
#
# For boot auto-start use the systemd user unit instead:
#   mkdir -p ~/.config/systemd/user
#   cp nni-dashboard.service ~/.config/systemd/user/
#   systemctl --user enable --now nni-dashboard.service
set -u
cd "$(dirname "$0")/.."
exec python3 webui/app.py
