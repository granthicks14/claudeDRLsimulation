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

## Free hosting options

Anything that runs this must give the container **~1–2 GB RAM** (PyTorch alone
needs several hundred MB), so the tiny 512 MB free tiers will OOM. These free
options work:

### Hugging Face Spaces — free hosted URL (recommended)

Free Spaces get **2 vCPU + 16 GB RAM** and run Docker. Steps:

1. Create a Hugging Face account, then **New → Space** → choose **Docker** (blank
   template), name it e.g. `milerunner`.
2. Put this project in the Space repo, with `deploy/huggingface/README.md` as the
   Space's root `README.md` (its YAML header tells HF to build the `Dockerfile`
   and route port 8050):

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/milerunner
   cd milerunner
   git pull https://github.com/granthicks14/claudeDRLsimulation main --allow-unrelated-histories
   cp deploy/huggingface/README.md README.md      # HF front-matter README
   git add -A && git commit -m "MileRunner on Spaces" && git push
   ```
3. The Space builds and serves at `https://<your-username>-milerunner.hf.space`.

Free-tier storage is **ephemeral** (progress resets on rebuild/long pause) — great
for a live demo; add persistent storage or use a volume host for long runs.

### Google Colab — easiest free way to just run it

Open **`notebooks/MileRunner_Colab.ipynb`** in Colab and *Run all*. It installs
everything, starts training in the background, and shows the live dashboard
inline. Free, no account infrastructure — but Colab sessions are temporary.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/granthicks14/claudeDRLsimulation/blob/main/notebooks/MileRunner_Colab.ipynb)

### Oracle Cloud Free Tier — free forever, full control

Oracle's **Always Free** ARM VM (up to 4 cores / 24 GB RAM) can run the Docker
image 24/7 at no cost. Provision the VM, install Docker, then use the "Any Docker
host" commands below. More setup, but genuinely free and persistent.

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
