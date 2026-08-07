"""WSGI entry point for serving the dashboard under a production server.

Run with e.g.::

    gunicorn wsgi:server -b 0.0.0.0:8050

The dashboard is read-only: it renders whatever the (separately running)
trainer process writes to the experiment database and status file, so it is
safe to serve with multiple gunicorn workers/threads.
"""
from __future__ import annotations

import os

from milerunner.dashboard.app import create_app

_DB = os.environ.get("MILE_DB", "experiments/milerunner.db")
_STATUS = os.environ.get("MILE_STATUS", "experiments/status.json")
_EXPERIMENT = os.environ.get("MILE_EXPERIMENT") or None

# Ensure the data directory exists so the first render doesn't error before the
# trainer has written anything.
os.makedirs(os.path.dirname(_DB) or ".", exist_ok=True)

app = create_app(db_path=_DB, status_path=_STATUS, experiment=_EXPERIMENT)
# ``server`` is the Flask WSGI callable gunicorn binds to.
server = app.server

if __name__ == "__main__":  # pragma: no cover - local convenience
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port)
