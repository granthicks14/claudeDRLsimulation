---
title: MileRunner
emoji: 🏃
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# MileRunner — live on Hugging Face Spaces (free)

This Space runs the MileRunner **autonomous trainer + live dashboard** on Hugging
Face's **free** Gradio runtime (2 vCPU · 16 GB RAM — no Docker, no card needed).
Training starts automatically and the charts refresh in real time.

> Use this file as the **README.md at the root of your Space repo** — the YAML
> header tells Hugging Face to run `app.py` with the free Gradio SDK. See
> `docs/HUGGINGFACE_SETUP.md` in the project for step-by-step instructions.

Free-tier storage is **ephemeral** (progress resets on rebuild or long pause) —
perfect for a live demo. For uninterrupted long-term training, use a host with a
persistent volume (see `docs/DEPLOY.md`).
