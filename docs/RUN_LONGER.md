# Running MileRunner for a long time (free, honestly)

Short version: **no free service with no credit card runs this 24/7 forever.**
Every genuinely always-on free host either needs a card (Oracle Cloud — it never
charges, but requires a card for identity verification) or can't run this
workload (serverless / 512 MB tiers OOM on PyTorch + MuJoCo).

**The good news:** MileRunner **checkpoints every generation and auto-resumes**,
so it never matters when the environment stops — the next time you open it, the
AI continues improving from exactly where it left off. Long training is
*cumulative across sessions*, so you effectively get "as long as you need."

## How long each free (no-card) option runs

| Option | Longest run | RAM | Watch in browser? | Persists between sessions? |
|---|---|---|---|---|
| **Codespaces (raise idle to 4 h)** | ~4 h idle, ~60 h/month | 8 GB | ✅ auto-opens | ✅ same codespace disk |
| **Kaggle notebook** | **12 h per session** | 30 GB | ✅ Gradio share link | ⚠️ save output to reuse |
| **Oracle Cloud Always Free** | **24/7 forever** | up to 24 GB | ✅ (open the port) | ✅ real VM disk |
| Colab / Replit / Render free | short / OOM | 0.5–12 GB | varies | ✗ |

## Option 1 — Codespaces, but for hours (recommended)

The default "stops after 30 min idle" is a *setting*. Change it once:

1. Go to <https://github.com/settings/codespaces>.
2. Find **Default idle timeout** → set it to **240 minutes (4 hours)** → Save.
3. Launch as usual (the one-click badge in the README). Keep the Codespace
   (VS Code) tab open while you watch; it now runs up to 4 hours untouched.

When it does stop, **just reopen the same Codespace** — training resumes from the
saved progress on its disk, so the best mile keeps dropping over days/weeks.
Free budget is ~60 hours/month; it auto-stops when idle so you can't overspend.

## Option 2 — Kaggle, for 12-hour sessions (most continuous, no card)

Kaggle gives **30 GB RAM** and **12-hour** sessions with **no credit card** (just
a free phone verification to enable internet). Use the ready notebook:
[`notebooks/MileRunner_Kaggle.ipynb`](../notebooks/MileRunner_Kaggle.ipynb).

1. Sign in at <https://www.kaggle.com>, **Settings → Phone verify** (enables internet).
2. **Create → Notebook**, then **File → Import Notebook** and upload
   `notebooks/MileRunner_Kaggle.ipynb` (or paste its cells).
3. In the right panel: **Internet = On**, Accelerator = **None (CPU)** is fine.
4. **Run all.** It prints a public `https://…gradio.live` link — open it to watch.
5. The session runs up to 12 hours. To keep going, start a new session and run
   again (bump `population.size` for a stronger squad).

## Option 3 — Oracle Cloud Always Free, for true 24/7 (needs a card for ID)

This is the **only** genuinely free, run-forever option. The **Always Free** ARM
VM (up to 4 cores / 24 GB RAM) runs the Docker image 24/7 at no cost — but Oracle
**requires a credit/debit card to verify your identity at signup** (Always Free
resources are never charged). If you're OK with a card-for-verification-only:

1. Create a free account at <https://www.oracle.com/cloud/free/> (pick an
   **Always Free** ARM `VM.Standard.A1.Flex` instance, Ubuntu).
2. SSH in, install Docker, then:
   ```bash
   git clone https://github.com/granthicks14/claudeDRLsimulation.git
   cd claudeDRLsimulation
   docker build -t milerunner .
   docker run -d --restart unless-stopped -p 8050:8050 \
     -v milerunner_data:/app/experiments -e MILE_CONFIG=hosted milerunner
   ```
3. Open the instance's public IP on port 8050 (open the port in the security
   list). It runs continuously; training survives reboots via the volume.

## The bottom line

- Want **easiest + nice UI, no card**: Codespaces with a 4-hour idle timeout,
  and let auto-resume carry progress across sessions.
- Want the **longest single unattended run, no card**: Kaggle (12 h).
- Want **true 24/7 forever**: Oracle Always Free (one-time card for ID).

Because MileRunner resumes, none of these lose progress — the AI keeps improving
every time you run it.
