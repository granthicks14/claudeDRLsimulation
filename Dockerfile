# MileRunner — hosted live dashboard + autonomous trainer in one container.
#
# The image installs the CPU build of PyTorch (small, no CUDA) since hosted
# instances are typically CPU-only; on a GPU host you can swap the torch line
# for a CUDA wheel. Physics + the 3D skeleton replay are headless (no OpenGL
# needed), so the image stays slim.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MILE_CONFIG=hosted \
    PORT=8050

WORKDIR /app

# Minimal system libs some wheels dlopen at import time (mujoco/glib, torch/gomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch first (keeps the image ~2GB smaller than the
# default CUDA wheel). This index is reachable from normal build hosts.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then the rest of the stack (torch is already satisfied) + the web server.
COPY requirements.txt ./
RUN pip install -r requirements.txt gunicorn

# Application code.
COPY . .
RUN pip install -e . || true

# Persist training progress here (mount a volume at this path on your host).
VOLUME ["/app/experiments"]

EXPOSE 8050

# tini reaps the background trainer + gunicorn cleanly on shutdown.
ENTRYPOINT ["/usr/bin/tini", "--", "bash", "docker/entrypoint.sh"]
