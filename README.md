# Krea 2 Image: RunPod ComfyUI template (single, batch, eight views, LoRA training set)

A one-click RunPod template for **Krea 2 Turbo**, Krea's 12.9B open-weights
text-to-image model (June 2026), in ComfyUI with JupyterLab. It runs on a **Network
Volume or a Global Volume**, and ships four workflows:

| Workflow | What it makes |
|---|---|
| `krea2_1_single_image` | one prompt -> one image |
| `krea2_2_batch` | one prompt per line (or a `.txt` file) -> one image each, plus a contact sheet |
| `krea2_3_eight_views` | a reference portrait -> the **same person** from 8 views: close-up, front, both three-quarters, both profiles, back, full body |
| `krea2_4_lora_training_set` | a reference portrait -> **24 varied images of the same person + a caption `.txt` each**, the folder layout LoRA trainers read (for example for a MiniMax H3 character LoRA) |

All four use Krea 2 Turbo the way Comfy-Org's own Krea 2 templates do: **8 steps,
CFG 1, euler / simple**, the Qwen3-VL 4B text encoder and the Qwen-Image VAE.

## How the workflows work

**Batch, eight views and training set** are one normal Krea 2 graph fed by a **Prompt
List** node (this repo's small `comfyui-krea2-batch` pack): it turns one prompt per line
into a list, and ComfyUI runs everything after it once per line. One queue renders the
whole set. Each line gets its own seed (`seed`, `seed + 1`, ...), so a set is
reproducible and every image differs.

**The same person** (eight views, training set): the reference portrait goes in the way
Comfy-Org's Krea 2 style-reference template feeds its image
(`TextEncodeQwenImageEditPlus` -> `FluxKontextMultiReferenceLatentMethod`): the Qwen3-VL
encoder sees it as "picture 1" and the VAE-encoded portrait joins as a reference latent.
Here without the style LoRA, so the reference carries the person rather than a style.
Every line says how to show that person ("three-quarter view, turned to their left"), and
the prompt prefix and suffix ask to keep the face, hair, skin tone and clothes.

> Krea 2 Turbo is not an edit model: identity holds well on frontal and three-quarter
> shots and less on profiles and the back. **Check every image** (each run saves a
> contact sheet) and re-run single lines with another seed.

**Training set**: the images and captions are written by ComfyUI's own **Save
Image-Text (to Folder)** node to `ComfyUI/output/krea2_dataset/<folder>/` as
`image_0001.png` + `image_0001.txt`, ... Each caption is your trigger word (Prompt List >
`caption_prefix`, default `zxq person,`) plus that line. The 24 lines are mostly face and
upper body - what a talking-head video model sees - with varied angles, expressions,
light, clothes and backgrounds, so a LoRA learns the person, not one photo. Download the
folder with JupyterLab (right-click > Download as archive) and delete weak images before
training.

## Storage: Network Volume or Global Volume

Attach **one** at `/workspace`. Setup detects which it is and says so in the log
(`Storage: network volume ...` or `Storage: global volume ...`).

| | Network Volume | Global Volume (RunPod beta) |
|---|---|---|
| Where pods can run | its datacenter only | any datacenter: more GPUs to choose from |
| ComfyUI, venv, node pack | on the volume | on the container disk, set up again on each fresh container (a few minutes) |
| Models, inputs, outputs, saved workflows | on the volume | on the volume |
| Size | fixed: **40 GB** recommended | grows with what you store |
| Container disk | 10 GB | **40 GB** (first-boot downloads are staged there) |

A Global Volume is object storage: no file locking, atomic rename, hard links or
permission bits, so no Python environment or git checkout can live on it. With one,
ComfyUI runs from `/opt/krea2-local/ComfyUI` on the container disk and its `models`,
`input`, `output` and `user/default` folders link to the volume, at the same paths a
Network Volume uses (`runpod-slim/ComfyUI/...`). Models are never downloaded twice.
Downloads are staged on the container disk, checked, then copied; nothing is renamed or
locked on the volume. Custom nodes added with ComfyUI-Manager do not persist there.

A Global Volume mounted at `/workspace-global` beside a Network Volume is found too; the
Network Volume at `/workspace` wins unless `KREA2_STORAGE=global`.

## Models (Comfy-Org/Krea-2, ungated: no token needed)

| File | Size | |
|---|---|---|
| `diffusion_models/krea2_turbo_fp8_scaled.safetensors` | 12.2 GB | default |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | 4.9 GB | default |
| `vae/qwen_image_vae.safetensors` | 0.2 GB | default |
| `diffusion_models/krea2_turbo_bf16.safetensors` | 24.5 GB | `KREA2_MODELS=bf16`, 48 GB+ GPUs |
| `diffusion_models/krea2_turbo_nvfp4.safetensors` | 7.2 GB | `KREA2_MODELS=nvfp4`, RTX 50xx / Blackwell |
| `diffusion_models/krea2_raw_fp8_scaled.safetensors` | 12.2 GB | `KREA2_MODELS=raw`, the undistilled model for fine-tuning |
| `text_encoders/qwen3vl_4b_bf16.safetensors` | 8.3 GB | `KREA2_TEXT_ENCODER=bf16` |
| 9 style LoRAs + `krea2_style_reference` | 4.4 GB | `DOWNLOAD_STYLE_LORAS=1` |

Default download: **17.4 GB**, once. `KREA2_MODELS` takes several, comma-separated
(`fp8,nvfp4`). The workflows open on the FP8 file: pick another in **Load Krea 2 Turbo**.

Style LoRA trigger words (strength 1.0): darkbrush `monochrome ink wash style`, dotmatrix
`monochrome stippling style`, kidsdrawing `naive expressive sketch style`, neondrip
`textured abstract style`, rainywindow `rainy window style`, retroanime `purple retro anime
style`, softwatercolor `art deco watercolor style`, sunsetblur `ethereal motion blur style`,
vintagetarot `vintage tarot style`.

**Licence:** Krea 2 ships under the **Krea 2 Community License**
([LICENSE.pdf](https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf)). Read it
before commercial use.

## GPU

24 GB of VRAM or more: **RTX 4090, RTX 5090, L40S, A6000, A100, H100**. FP8 Turbo is
12.2 GB plus the 4.9 GB text encoder, which ComfyUI moves off the GPU between steps when
needed. On RTX 50xx / Blackwell, `KREA2_MODELS=nvfp4` (7.2 GB) is smaller and faster.
Setup warns under 20 GB of VRAM.

## RunPod template settings

| Setting | Value |
|---|---|
| Template name | `Krea 2 Image - ComfyUI + JupyterLab` |
| Container image | **`runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8`** (ComfyUI v0.35.0: Krea 2 is native since v0.26.0) |
| Container start command | the JSON in [`runpod-start.json`](runpod-start.json) (readable: [`bootstrap.sh`](bootstrap.sh)) |
| Container disk | **40 GB** (Global Volume staging; 10 GB is enough with a Network Volume) |
| Volume | each deployer attaches a **Network Volume** (40 GB) or **Global Volume** at `/workspace`; template volume disk 0 GB, mount path `/workspace` |
| HTTP ports | **`8188,8888`** (ComfyUI, JupyterLab) |
| Environment variables | `JUPYTER_NO_AUTH=1`; optional `HF_TOKEN`, `KREA2_MODELS`, `KREA2_TEXT_ENCODER`, `DOWNLOAD_STYLE_LORAS`, `KREA2_STORAGE` |
| README tab | paste [`TEMPLATE_README.md`](TEMPLATE_README.md) |

Every field, with comments: [`RUNPOD_TEMPLATE_SETUP.txt`](RUNPOD_TEMPLATE_SETUP.txt).

The start command checks DNS for GitHub and Hugging Face (falling back to `1.1.1.1` and
`8.8.8.8`), clones this repo (10 retries) and runs `setup.sh`. `TEMPLATE_REF` selects
another branch or tag.

## What boots

1. Detects the volume (Network or Global) and checks it; refuses the container disk alone
2. Copies the image's ComfyUI to the volume (Network) or the container disk (Global), and
   links the data folders to a Global Volume
3. Checks that ComfyUI supports Krea 2 and that PyTorch can use the GPU, before any download
4. Installs the `comfyui-krea2-batch` nodes (no Python packages to install)
5. Copies `reference_portrait.png` and `krea2_batch_prompts.txt` into `ComfyUI/input/`
   (never over a file of the same name)
6. Downloads the models (skip-if-present, resumable, every file's safetensors header and
   size checked)
7. Installs the four workflows into ComfyUI's **Workflows** sidebar (never over one you edited)
8. Hands over to the image's `/start.sh` (ComfyUI on 8188, JupyterLab on 8888)

Look for `Storage: ...`, the downloads and **`Setup complete!`** in the pod log.

## Environment variables (all optional)

| Var | Default | |
|---|---|---|
| `KREA2_MODELS` | `fp8` | Turbo checkpoints to download: `fp8`, `bf16`, `nvfp4`, `raw`, comma-separated |
| `KREA2_TEXT_ENCODER` | `fp8` | `fp8` or `bf16` |
| `DOWNLOAD_STYLE_LORAS` | `0` | `1`: Krea's style LoRAs and the style-reference LoRA |
| `HF_TOKEN` | *(empty)* | Not needed (ungated); raises Hugging Face rate limits |
| `JUPYTER_NO_AUTH` | `1` | Jupyter without a login. `0` + `JUPYTER_PASSWORD` for one |
| `KREA2_STORAGE` | `auto` | `network` or `global` overrides the volume detection |
| `TEMPLATE_REF` | `main` | Branch or tag of this repo the start command clones |

## What has and hasn't been verified

Verified: the four workflows against ComfyUI v0.35.0's node schemas (inputs, outputs,
widget values, links) and against Comfy-Org's own Krea 2 templates' node settings; every
model file a workflow loads is one setup downloads, at its path in Comfy-Org/Krea-2;
the Prompt List and Contact Sheet nodes; the Network and Global Volume storage code (the
tests in `tools/`, also run by CI on every push).

**Not yet verified:** a boot on a real RunPod pod, a render of these workflows, how well
Krea 2 Turbo keeps a person's identity in the eight-views and training-set workflows, and
training a LoRA on the set.

## Repository layout

```
├── setup.sh                          # RunPod boot script
├── bootstrap.sh / runpod-start.json  # container start command (readable / paste-ready)
├── scripts/storage.py                # Network / Global Volume detection and checks
├── custom_nodes/comfyui-krea2-batch/ # Prompt List, Contact Sheet
├── workflows/                        # the four workflows (generated)
├── samples/reference_portrait.png    # the portrait the reference workflows open on
├── prompts/krea2_batch_prompts.txt   # sample batch prompt file
├── tools/
│   ├── build_workflow.py             # generates workflows/; re-run after edits
│   ├── test_workflows.py             # graphs vs node schemas, settings, downloads
│   ├── test_nodes.py                 # Prompt List, Contact Sheet
│   └── test_storage.py               # volume detection, Network and Global Volume
├── README.md
├── TEMPLATE_README.md                # paste into the template's README tab
└── RUNPOD_TEMPLATE_SETUP.txt         # every template field, with comments
```

## Credits

[Krea 2](https://www.krea.ai/krea-2-open-source) (Krea, Krea 2 Community License) ·
ComfyUI packaging and reference workflows by
[Comfy-Org](https://huggingface.co/Comfy-Org/Krea-2) · style-reference LoRA by
[ostris](https://huggingface.co/ostris/krea2_turbo_style_reference) · template by
[AI Art Pipeline](https://www.youtube.com/@aiartpipeline).
