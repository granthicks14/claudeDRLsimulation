# Run MileRunner free in GitHub Codespaces (no credit card)

**GitHub Codespaces** is the easiest genuinely-free way to run MileRunner as a
browser app — **no credit card required**. Every GitHub account gets free
Codespaces hours each month, an **8 GB RAM** Linux machine (plenty for PyTorch),
and an automatic public-to-you URL for the dashboard. You already have the repo,
so there's nothing new to sign up for.

This repo includes a dev-container that **auto-installs everything and
auto-starts the trainer + dashboard**, and **auto-opens the dashboard in your
browser** — so it's effectively one click.

---

## One click

**[▶ Open MileRunner in Codespaces](https://codespaces.new/granthicks14/claudeDRLsimulation?quickstart=1)**
(sign in to GitHub first). That's it — everything below happens automatically.

---

## What happens after you click

### 1. It builds (~3–5 min, first time only)
A browser VS Code opens and installs PyTorch, MuJoCo, the CPU renderer and the
rest of the stack. You'll see it working in the terminal.

### 2. The dashboard opens itself
When the build finishes, the trainer + dashboard start automatically and a
**"Your application running on port 7860 is available"** popup appears — it opens
the dashboard tab for you (`onAutoForward: openBrowserOnce`).
- If you miss it: click the **Ports** tab (next to the terminal), find port
  **7860** ("MileRunner Dashboard"), and click the 🌐 globe icon.
- To watch the trainer log: in a terminal run `tail -f experiments/app.log`.

That's it — the live dashboard loads. Give the trainer a minute to finish its
first generation, then the fastest-mile number, the speed / heart-rate / cadence
/ oxygen / energy curves, the muscle-fatigue heat-map and the 3D runner fill in
and keep improving. 🎉

---

## Handy things

- **Restart the app** (if needed): in the terminal run
  ```bash
  pkill -f app.py; pkill -f scripts/run.py    # stop
  python app.py                                # start again (Ctrl-C to stop)
  ```
- **See training logs**: `tail -f /tmp/milerunner.log` or `tail -f experiments/train.log`
- **Make it train harder**: edit `configs/hosted.yaml` (raise `population.size`
  and `timesteps_per_gen`) — the 8 GB machine has plenty of headroom — then
  restart the app.
- **Share the URL**: in the **Ports** tab, right-click port 7860 → **Port
  Visibility → Public** to let others open it (otherwise it's just for you).

## Free usage & limits

- GitHub's free plan includes **120 core-hours + 15 GB/month** of Codespaces at
  no cost and **no credit card** (a 2-core machine = ~60 hours/month).
- **Stop the Codespace when you're done** so it doesn't use your free hours:
  <https://github.com/codespaces> → **⋯** → **Stop codespace**. Restarting later
  resumes instantly (deps stay installed).
- Codespaces auto-stop after **30 minutes idle by default** — but you can raise
  that to the **maximum 4 hours** at
  <https://github.com/settings/codespaces> → **Default idle timeout → 240 min**.
- Progress is saved inside the Codespace (`experiments/`), so stopping and
  restarting **resumes training exactly where it left off** — the best mile keeps
  improving across sessions. Deleting the Codespace clears it.

### Want it to run even longer?

No free service with **no credit card** runs this 24/7 forever, but because
MileRunner auto-resumes, you effectively get unlimited cumulative training. See
**[`RUN_LONGER.md`](RUN_LONGER.md)** for all options: Codespaces at 4-hour idle,
**Kaggle for 12-hour sessions** (30 GB RAM, no card), and **Oracle Cloud Always
Free** for true 24/7 (needs a card for ID only).
