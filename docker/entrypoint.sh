#!/usr/bin/env bash
# Container entrypoint: run the autonomous trainer in the background and serve
# the live dashboard in the foreground. Both processes share the ./experiments
# directory (database + status file), which should be a persistent volume so
# training survives restarts.
set -euo pipefail

CONFIG="${MILE_CONFIG:-hosted}"
PORT="${PORT:-8050}"
WEB_WORKERS="${WEB_WORKERS:-1}"
WEB_THREADS="${WEB_THREADS:-4}"

mkdir -p experiments checkpoints

echo "[entrypoint] starting trainer (config=${CONFIG}) in background…"
# --exp/--config drive a continuous, autonomous run. Logs go to the shared dir.
python scripts/run.py --config "${CONFIG}" >> experiments/train.log 2>&1 &
TRAINER_PID=$!

# If the trainer dies, take the container down so the platform restarts it.
trap 'echo "[entrypoint] shutting down…"; kill ${TRAINER_PID} 2>/dev/null || true' TERM INT

echo "[entrypoint] serving dashboard on 0.0.0.0:${PORT} (gunicorn)…"
exec gunicorn wsgi:server \
    --bind "0.0.0.0:${PORT}" \
    --workers "${WEB_WORKERS}" \
    --threads "${WEB_THREADS}" \
    --worker-class gthread \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
