# Host MileRunner free on Hugging Face Spaces — step by step

**Hugging Face Spaces** is the recommended free host: the free **Gradio** runtime
gives you **2 vCPU + 16 GB RAM** (plenty for PyTorch), needs **no Docker and no
credit card**, and serves a permanent public URL like
`https://your-username-milerunner.hf.space`. Training starts automatically and
the live dashboard updates in real time.

> ⚠️ Heads up: Hugging Face has been asking some accounts for a **payment
> method** even for Spaces (Docker *and* Gradio). If yours does and you want to
> stay 100% card-free, use **GitHub Codespaces** instead — see
> [`CODESPACES_SETUP.md`](CODESPACES_SETUP.md) (8 GB RAM, no card) — or
> **Google Colab** (`notebooks/MileRunner_Colab.ipynb`). The guide below still
> applies if your HF account offers the free Gradio tier.
>
> If you do use HF: choose the **Gradio** SDK, **not** Docker. This repo includes
> a Gradio app (`app.py`) built exactly for this.

You only need a free Hugging Face account and about 5 minutes. Pick **Option A**
(copy-paste git commands — most reliable) or **Option B** (automatic sync from
GitHub).

---

## Before you start

- Create a free account at <https://huggingface.co/join>.
- Install git if you don't have it (<https://git-scm.com/downloads>).

---

## Option A — one-time manual push (simplest)

### 1. Create the Space
1. Go to <https://huggingface.co/new-space>.
2. **Owner**: your username. **Space name**: `milerunner`.
3. **License**: MIT.
4. **Select the SDK**: choose **Gradio** (⚠️ not Docker). Pick the **Blank**
   template.
5. **Hardware**: leave the free **CPU basic** (2 vCPU · 16 GB RAM).
6. Set it **Public** and click **Create Space**. It's empty for now.

### 2. Get an access token
1. Open <https://huggingface.co/settings/tokens> → **New token**.
2. Name it `deploy`, type **Write**, and **copy** the token (starts with `hf_…`).

### 3. Push the code into the Space
Run these in a terminal, replacing `YOUR_USERNAME` (paste the token when git asks
for a password):

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/milerunner
cd milerunner

# pull in the MileRunner code from GitHub
git pull https://github.com/granthicks14/claudeDRLsimulation main --allow-unrelated-histories

# use the Hugging Face front-matter README (tells HF to run app.py with Gradio)
cp deploy/huggingface/README.md README.md

git add -A
git commit -m "Deploy MileRunner"
git push
# username = YOUR_USERNAME,  password = the hf_… token you copied
```

### 4. Watch it build
- Open `https://huggingface.co/spaces/YOUR_USERNAME/milerunner`.
- The **Building** logs run for a few minutes (installing PyTorch, MuJoCo, etc.).
- When it flips to **Running**, the dashboard appears. Give the trainer a minute
  to finish its first generation, then the charts and 3D runner fill in.

**Done** — your free live URL is `https://YOUR_USERNAME-milerunner.hf.space`. 🎉

---

## Option B — automatic sync from GitHub (updates itself)

Do this if you want the Space to update automatically whenever `main` changes.

1. Create the Space (Gradio SDK) and an HF **write** token (steps 1–2 above).
2. In your GitHub repo → **Settings → Secrets and variables → Actions**:
   - **New repository secret**: name `HF_TOKEN`, value = your `hf_…` token.
   - **Variables** tab → **New repository variable**: name `HF_SPACE`,
     value = `YOUR_USERNAME/milerunner`.
3. Go to the repo's **Actions** tab → run **Deploy to Hugging Face Spaces**
   (or just push any commit to `main`). The workflow mirrors the repo into the
   Space and it rebuilds automatically from then on.

---

## Tips & troubleshooting

- **"Training is starting…" on first load** — normal; the trainer hasn't finished
  generation 0 yet. It fills in within a minute or two.
- **Space sleeps** — free Spaces pause after ~48 h of no visitors and resume when
  someone opens the URL. Training restarts fresh because free storage is
  ephemeral (fine for a demo). For uninterrupted long-term training, add
  Hugging Face **persistent storage** (paid) or use a volume host from
  `docs/DEPLOY.md`.
- **Make it train harder** — edit `configs/hosted.yaml` (raise `population.size`
  and `timesteps_per_gen`) and push again; on the free 16 GB you have lots of
  headroom.
- **Build error mentioning gradio version** — open `README.md` in the Space and
  change `sdk_version` to the version HF suggests in the error, then commit.
