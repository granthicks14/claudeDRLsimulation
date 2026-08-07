---
title: MileRunner
emoji: 🏃
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8050
pinned: false
license: mit
---

# MileRunner — live on Hugging Face Spaces

This Space runs the MileRunner **autonomous trainer + live dashboard** in a free
Docker Space (2 vCPU · 16 GB RAM). Training starts automatically and the
dashboard updates in real time.

> Use this file as the **README.md at the root of your Space repo** — the YAML
> header above is what tells Hugging Face to build the `Dockerfile` and route to
> port 8050. See `docs/DEPLOY.md` in the project for step-by-step instructions.

Note: the free tier has **ephemeral storage**, so training progress resets if the
Space rebuilds or is paused for long. It's perfect for a live demo; for
persistent long-term training use a host with a volume (see `docs/DEPLOY.md`).
