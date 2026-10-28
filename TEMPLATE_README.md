# Krea 2 Image - Single, Batch, Eight Views, LoRA Training

**Krea 2 Turbo** (12.9B, 8 steps) in ComfyUI. Five ready workflows:

- **Single image** - one prompt, one image
- **Batch** - one prompt per line (or a .txt file), one image each + a contact sheet
- **Eight views** - a reference portrait -> the same person from 8 angles
- **LoRA training set** - a reference portrait -> 24 varied images of that person, each with a caption .txt
- **Train a LoRA** - that folder -> a Krea 2 LoRA, trained in ComfyUI and tested on Turbo

## Storage

Attach your own volume at /workspace:

- **Network Volume** (60 GB): everything stays on it. Reattach it in its datacenter.
- **Global Volume** (any datacenter): models, uploads and images stay on it; ComfyUI is set up again on each new pod (a few minutes). Container disk **40 GB**.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab (uploads, downloads, terminal)

First boot downloads **29.6 GB** (Turbo + Raw for training); wait for **"Setup complete!"** in the logs.

## Use

Open the **Workflows** sidebar and pick a workflow. Each opens with a "Start here" note; red nodes are the ones you set. Results go to ComfyUI/output/.

## Hardware

24 GB+ VRAM: RTX 4090, 5090, L40S, A6000, A100, H100. LoRA training: 32 GB+ (5090) or 48 GB (L40S, A6000). On RTX 50xx set KREA2_MODELS=nvfp4,raw for a smaller, faster model.

Krea 2 Community License - check it before commercial use.

Full instructions: https://github.com/tenitsky/krea2-image
