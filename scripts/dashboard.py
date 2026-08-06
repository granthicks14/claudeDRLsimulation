#!/usr/bin/env python3
"""Launch the live analytics dashboard.

Reads the trainer's status file and experiment database and serves an
auto-refreshing web dashboard (default http://127.0.0.1:8050). Run it alongside
``scripts/run.py``.

    python scripts/dashboard.py
    python scripts/dashboard.py --experiment smoke --port 8060
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from milerunner.dashboard.app import run_dashboard


def main():
    ap = argparse.ArgumentParser(description="MileRunner analytics dashboard")
    ap.add_argument("--db", default="experiments/milerunner.db")
    ap.add_argument("--status", default="experiments/status.json")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"Serving dashboard at http://{args.host}:{args.port}")
    run_dashboard(db_path=args.db, status_path=args.status,
                  experiment=args.experiment, host=args.host, port=args.port,
                  debug=args.debug)


if __name__ == "__main__":
    main()
