# Deploying the hosted live dashboard

MileRunner ships as a single Docker image that runs **both** the autonomous
trainer (in the background) and the **live web dashboard** (foreground, served by
gunicorn). Point it at any platform that runs long‑lived containers and you get a
public URL showing training progress in real time, with the best mile times and
the 3D runner replay.

> Why not Vercel? Vercel only hosts static pages / short serverless functions —
> it can't run a continuous PyTorch training process. Use one of the container
> hosts below instead. (The repo's `index.html` is just a project homepage for
> Vercel.)

## What's in the image

- `Dockerfile` — Python 3.11-slim + CPU PyTorch + the full stack; headless
  (no GPU/OpenGL needed — physics and the 3D skeleton replay are computed).
- `docker/entrypoint.sh` — starts `scripts/run.py` (trainer) in the background,
  then `gunicorn wsgi:server` (dashboard) in the foreground.
- `wsgi.py` — the Flask WSGI app gunicorn serves (read-only view of the DB +
  status file the trainer writes).
- `configs/hosted.yaml` — a small-footprint config (population 6, short
  generations) tuned for a 1–2 GB instance so progress is visible quickly.

Both processes share `/app/experiments` (database, checkpoints, status). Mount a
**persistent volume** there so training survives restarts and redeploys.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `8050` | Port the dashboard binds to (hosts inject this). |
| `MILE_CONFIG` | `hosted` | Which `configs/*.yaml` the trainer uses. |
| `WEB_WORKERS` | `1` | gunicorn worker processes. |
| `WEB_THREADS` | `4` | gunicorn threads per worker. |
| `MILE_DB` | `experiments/milerunner.db` | Database path the dashboard reads. |
| `MILE_STATUS` | `experiments/status.json` | Status file the dashboard reads. |

## Run locally (Docker Compose)

```bash
docker compose up --build
# open http://localhost:8050   (progress persists in the milerunner_data volume)
```

## Render.com (easiest — one click)

1. Push this repo to GitHub (already done).
2. On Render → **New +** → **Blueprint** → select the repo. Render reads
   `render.yaml`, builds the Dockerfile, attaches a 1 GB disk, and gives you an
   `https://…onrender.com` URL.
3. Recommended plan: **Starter** or larger — the free 512 MB tier can run out of
   memory once PyTorch loads.

## Fly.io

```bash
fly launch --no-deploy            # edit the app name in fly.toml first (must be unique)
fly volumes create milerunner_data --size 1 --region iad
fly deploy
fly open                          # opens the dashboard URL
```

`fly.toml` suspends the machine when idle and wakes it on the next request to
save cost; training resumes from the persisted volume.

## Railway

New Project → Deploy from GitHub repo → Railway auto-detects the `Dockerfile`.
Add a **Volume** mounted at `/app/experiments`, set `MILE_CONFIG=hosted`, and
Railway assigns a public domain on the exposed port.

## Any Docker host (VPS, etc.)

```bash
docker build -t milerunner .
docker run -d --name milerunner -p 8050:8050 \
  -v milerunner_data:/app/experiments \
  -e MILE_CONFIG=hosted milerunner
# http://<server-ip>:8050
```

## Scaling up

- More compute → edit `configs/hosted.yaml` (or set `MILE_CONFIG=default`/
  `cluster`) to raise `population.size`, `timesteps_per_gen`, and `n_envs`.
- GPU host → replace the CPU-torch line in the `Dockerfile` with a CUDA wheel and
  set `hardware.device: cuda`.
- Horizontal scale → run several trainer containers pointed at one shared
  database volume; they contribute to the same evolving population.

## Notes & caveats

- The dashboard is **read-only**; the trainer is the single writer of the DB /
  status file. Serving it with multiple gunicorn workers/threads is safe.
- First paint may say "waiting for data…" until the trainer finishes its first
  generation (a minute or two on a small instance).
- The image is CPU-only by default; a full mile takes substantial training time.
  The value is the continuously-improving live view, not an instant result.
