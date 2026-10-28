#!/usr/bin/env python3
"""
Generate the Krea 2 workflow JSON files. Re-run after editing:

    python tools/build_workflow.py

Four workflows, all Krea 2 Turbo (8 steps, CFG 1, euler / simple, as Comfy-Org's own
Krea 2 templates):

  krea2_1_single_image        one prompt -> one image
  krea2_2_batch               one prompt per line (or a .txt file) -> one image each
  krea2_3_eight_views         a reference portrait -> the same person from 8 views
  krea2_4_lora_training_set   a reference portrait -> 24 varied images of the same
                              person + a caption .txt each, ready to train a LoRA

The reference workflows give Krea 2 the portrait the way Comfy-Org's style-reference
template does (TextEncodeQwenImageEditPlus: the image goes to the Qwen3-VL text encoder
and, VAE-encoded, in as a reference latent), without the style LoRA, so it carries the
person's identity instead of a style. Node settings follow Comfy-Org's templates
(github.com/Comfy-Org/workflow_templates, image_krea2_turbo_*).

Layout: numbered stages from left to right, each a coloured group with a note under it,
and a "Start here" note on the far left.
"""
import json
import os

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "..", "workflows")
PACK = "comfyui-krea2-batch"

# Files setup.sh downloads by default (Comfy-Org/Krea-2), and what the workflows load.
UNET = "krea2_turbo_fp8_scaled.safetensors"
TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
STYLE_LORA = "krea2_darkbrush.safetensors"          # optional download, node bypassed
REFERENCE = "reference_portrait.png"                # samples/, copied to ComfyUI/input
BATCH_FILE = "krea2_batch_prompts.txt"              # prompts/, copied to ComfyUI/input

STEPS, CFG, SAMPLER, SCHEDULER = 8, 1.0, "euler", "simple"

WORKFLOWS = [
    dict(kind="single", path="krea2_1_single_image.json", id="5b1f0c7e-2a43-4d8e-9c61-0f3e8a7d2b11"),
    dict(kind="batch", path="krea2_2_batch.json", id="8c2e4d19-6b7a-4f30-a5d2-3e9f1b6c4a22"),
    dict(kind="views", path="krea2_3_eight_views.json", id="a3d7e5f2-1c8b-4e69-b0a4-7d2c9e5f3b33"),
    dict(kind="dataset", path="krea2_4_lora_training_set.json", id="d4e8f6a3-9b2c-4d71-8e5f-1a6b3c7d4e44"),
    dict(kind="train", path="krea2_5_train_lora.json", id="e5f9a7b4-0c3d-4e82-9f6a-2b7c4d8e5f55"),
]

# LoRA training (ComfyUI's own Train LoRA node). Krea 2 Raw is the undistilled checkpoint
# Krea released to be fine-tuned; the LoRA is then tested on Turbo, the model the other
# workflows use. FP8 weights stay frozen: bypass mode attaches the LoRA through hooks.
RAW_UNET = "krea2_raw_fp8_scaled.safetensors"
DATASET_FOLDER = "krea2_dataset/my_person"
LORA_PREFIX = "loras/krea2_my_person"
TRAIN = dict(batch_size=1, grad_accumulation_steps=1, steps=1500, learning_rate=0.0001, rank=16,
             optimizer="AdamW", loss_function="MSE", seed=42, training_dtype="bf16",
             lora_dtype="bf16", quantized_backward=False, algorithm="LoRA",
             gradient_checkpointing=True, checkpoint_depth=1, offloading=False,
             existing_lora="[None]", bucket_mode=False, bypass_mode=True)
TEST_PROMPTS = """zxq person, portrait photo in a cozy cafe, warm window light, smiling
zxq person, head and shoulders, studio portrait, grey background, neutral expression
zxq person, walking on a city street at night, neon lights, cinematic
zxq person, close-up, outdoors in a park, golden hour, laughing"""

SINGLE_PROMPT = ("Editorial portrait photo of a woman in her thirties with short silver hair and "
                 "freckles, wearing a mustard wool coat, standing on a rainy city street at dusk, "
                 "neon reflections on wet pavement, shallow depth of field, 85mm lens, "
                 "natural skin texture")

BATCH_PROMPTS = """# One prompt per line. Lines starting with # are skipped.
# Or put them in a .txt file in ComfyUI/input/ and name it in prompts_file.
A lighthouse on a rocky cliff at golden hour, waves crashing, cinematic wide shot
Close-up of a ceramic coffee cup on a wooden table, morning light through a window, steam rising
A red vintage motorbike parked in a narrow Italian alley, laundry hanging above, summer afternoon
Portrait of an elderly fisherman with a weathered face and a knitted cap, overcast harbour behind him
A cozy reading nook with a green armchair, stacked books and a warm lamp, rainy evening outside
Aerial view of a winding river through autumn forest, mist in the valleys"""

VIEW_PREFIX = "Show the same person as in picture 1:"
VIEW_SUFFIX = ("Keep the same face, hair, skin tone, body and clothes exactly as in picture 1. "
               "Plain light grey studio background, soft even studio light, photorealistic.")
EIGHT_VIEWS = """close-up portrait, face filling the frame, looking straight at the camera
front view, head and shoulders, looking at the camera
three-quarter view, turned to their left, head and shoulders
left side profile, head and shoulders
three-quarter view, turned to their right, head and shoulders
right side profile, head and shoulders
seen from behind, back of the head and shoulders
full body, standing, front view, arms relaxed"""

DATASET_PREFIX = VIEW_PREFIX
DATASET_SUFFIX = ("Keep the same face, hair, skin tone and identity exactly as in picture 1. "
                  "Photorealistic photo, natural skin texture.")
TRIGGER = "zxq person,"
TRAINING_SET = """# 24 varied shots of one person for a LoRA. Mostly face and upper body (what a
# talking-head video model sees), with different angles, expressions, light, clothes and
# backgrounds, so the LoRA learns the person and not one photo.
front view, head and shoulders, neutral expression, plain grey studio background, soft light
front view, head and shoulders, warm smile, plain grey studio background, soft light
front view, close-up, laughing, plain white background, bright light
three-quarter view to the left, head and shoulders, calm expression, grey background
three-quarter view to the right, head and shoulders, slight smile, grey background
left side profile, head and shoulders, neutral expression, plain background
right side profile, head and shoulders, neutral expression, plain background
looking slightly up, head and shoulders, thoughtful expression, soft window light
looking slightly down, head and shoulders, gentle smile, soft window light
talking mid-sentence, front view, upper body, in a modern office, natural light
talking and gesturing with one hand, upper body, in a living room, warm lamp light
front view, upper body, wearing a white shirt, outdoors in a park, overcast daylight
three-quarter view, upper body, wearing a dark sweater, city street at dusk
front view, head and shoulders, serious expression, dramatic side light, dark background
front view, head and shoulders, surprised expression, bright even light, white background
close-up of the face, eyes looking at the camera, soft golden hour sunlight
upper body, sitting at a desk, looking at the camera, home office, daylight
upper body, standing by a window, looking out, soft backlight
front view, head and shoulders, wearing glasses, neutral background
three-quarter view to the left, upper body, wearing a denim jacket, outdoors, sunny day
front view, upper body, arms crossed, confident smile, studio background
head and shoulders, tilting the head slightly, friendly smile, bokeh city lights at night
front view, head and shoulders, recording a video, ring light reflection in the eyes
three-quarter view to the right, close-up, calm expression, soft overcast light"""

# ------------------------------------------------------------------------ notes
NOTE_ABOUT = """## About Krea 2

Krea's 12.9B open-weights text-to-image model (June 2026). **Turbo** is the distilled version: high quality in **8 steps at CFG 1**. This template runs the FP8 Turbo checkpoint (12.2 GB) with the Qwen3-VL 4B text encoder and the Qwen-Image VAE, from Comfy-Org's ComfyUI package.

Licence: **Krea 2 Community License** - read it before commercial use: https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf"""

NOTE_MODELS = f"""## 2. Models - loaded once

`{UNET}` (diffusion_models), `{TEXT_ENCODER}` (text_encoders), `{VAE}` (vae). Downloaded by setup; they stay loaded between runs.

Other precisions (set at deploy, keep `raw` for training): `KREA2_MODELS=bf16,raw` (48 GB+ GPUs), `nvfp4,raw` (RTX 50xx / Blackwell, 7 GB). Then pick the file in **Load Krea 2 Turbo**."""

NOTE_SAMPLE = f"""## 4. Sampling

Krea 2 Turbo's settings: **{STEPS} steps, CFG {CFG:g}, {SAMPLER} / {SCHEDULER}**. CFG 1 means the negative prompt is not used (it is zeroed, as in Comfy-Org's templates).

**Size**: 1024 x 1024 by default. Other good sizes: 832 x 1216 (portrait), 1216 x 832, 1344 x 768 (landscape), 768 x 1344."""


# GPU memory per workflow, shown in each "Start here" note and in the README's GPU table.
# Generation: Krea 2 Turbo FP8 (12.2 GB) + the Qwen3-VL text encoder (4.9 GB).
# Training: estimates until measured on a pod.
_GEN_OOM = ("**Out of memory, or very slow?** Use a smaller image size (832 x 832), or on an "
            "RTX 50xx the nvfp4 model (`KREA2_MODELS=nvfp4,raw` at deploy). ComfyUI moves models "
            "off the GPU when it has to: slower, but it still finishes.")
GPU_NOTES = {
    "single": "**GPU:** 24 GB minimum (RTX 4090, A5000, L4), 32 GB comfortable (RTX 5090).\n\n" + _GEN_OOM,
    "batch": ("**GPU:** 24 GB minimum (RTX 4090, A5000, L4), 32 GB comfortable (RTX 5090). The "
              "images render one after another, so a long list needs no more GPU memory than "
              "one image.\n\n" + _GEN_OOM),
    "views": ("**GPU:** 24 GB minimum (RTX 4090, A5000, L4), 32 GB comfortable (RTX 5090). The "
              "reference portrait adds a little memory on top of plain text-to-image.\n\n" + _GEN_OOM),
    "dataset": ("**GPU:** 24 GB minimum (RTX 4090, A5000, L4), 32 GB comfortable (RTX 5090). "
                "Making the set is image generation, not training: a cheap GPU is fine.\n\n" + _GEN_OOM),
    "train": ("**GPU (estimates, not yet measured):** 32 GB minimum (RTX 5090, tight), **48 GB "
              "recommended** (L40S, RTX 6000 Ada, A6000, A40), 80 GB fastest (A100, H100). "
              "Training needs much more memory than generating.\n\n**Out of memory?** In Train LoRA "
              "turn on `offloading` (slower) or lower `rank` to 8. Or train on a bigger GPU: "
              "deploy this template on it with the **same volume**, train, then go back to a "
              "cheap GPU to generate."),
}


def note_start(kind):
    steps = {
        "single": ("Krea 2 - Single image",
                   "1. Write your prompt in **Prompt** (stage 1).\n2. **Run**. The image is saved to `ComfyUI/output/krea2/`.\n\nOptional: enable **Style LoRA** (right-click > Bypass) after downloading the style LoRAs (`DOWNLOAD_STYLE_LORAS=1` at deploy). Each has a trigger word, listed in its note."),
        "batch": ("Krea 2 - Batch",
                  f"1. **Prompt List** (stage 1): one prompt per line, or a .txt file in `ComfyUI/input/` named in `prompts_file` (the sample `{BATCH_FILE}` is there).\n2. **Run** once. Every line becomes an image in `ComfyUI/output/krea2_batch/`, plus a contact sheet of the set.\n\nSeeds go up by one per line (`seed_mode`), so each image differs and the whole batch can be reproduced."),
        "views": ("Krea 2 - Eight views of a person",
                  "1. **Load Reference** (stage 1): a clear photo of the person (the sample opens). Head and shoulders or upper body works best.\n2. **Run** once. Eight views of the same person - close-up, front, both three-quarters, both profiles, back, full body - are saved to `ComfyUI/output/krea2_views/`, plus one contact sheet.\n\nEdit the views in **Prompt List**: one per line."),
        "dataset": ("Krea 2 - LoRA training set",
                    f"Builds a training set for a character LoRA: **24 images of the same person** with a caption .txt each.\n\n1. **Load Reference** (stage 1): a clear photo of the person.\n2. Set your **trigger word** in Prompt List > `caption_prefix` (default `{TRIGGER}`).\n3. **Run** once. Images + captions go to `ComfyUI/output/{DATASET_FOLDER}/` as `image_0001.png` + `image_0001.txt` (the layout ai-toolkit and kohya read too).\n\n**Check every image** in the contact sheet and delete the ones where the face drifted (JupyterLab): a LoRA learns its mistakes too. Then train with **krea2_5_train_lora**."),
        "train": ("Krea 2 - Train a LoRA",
                  f"Trains a Krea 2 **image** LoRA on a folder of images + captions, then tests it.\n\n1. Make the set with **krea2_4_lora_training_set** (or upload your own images + same-name .txt captions into a folder under `ComfyUI/output/`).\n2. **Load Training Set** (stage 1): the folder, default `{DATASET_FOLDER}`. Deleted images are left out.\n3. **Test prompts** (stage 4): use your trigger word.\n4. **Run** once. Training takes a while (watch the progress bar on Train LoRA); then the test images render.\n\nThe LoRA is saved to `ComfyUI/output/{LORA_PREFIX}_<steps>_steps_00001_.safetensors` and appears in every LoRA loader. The Raw model it trains on downloads by default (`KREA2_MODELS=fp8,raw`)."),
    }[kind]
    return f"""# {steps[0]}

{steps[1]}

{GPU_NOTES[kind]}

**Red nodes are the ones you set.** Stages run left to right with a note under each.

Full instructions: https://github.com/tenitsky/krea2-image"""


NOTE_TRAIN = f"""## 3. Training

ComfyUI's own **Train LoRA** node on **Krea 2 Raw** (`{RAW_UNET}`), the undistilled model Krea released to be fine-tuned. The FP8 weights stay frozen; **bypass_mode** attaches the LoRA through hooks, which is what lets a 12.9B model train on one 32-48 GB GPU.

Starting point for a 15-30 image character set: **rank {TRAIN['rank']}, learning rate {TRAIN['learning_rate']}, {TRAIN['steps']} steps**, batch 1. More images -> more steps. Out of memory: turn **offloading** on (slower).

**Loss graph**: `output/krea2_lora/loss_graph_*.png`. It should fall and then flatten; a curve that keeps rising means the learning rate is too high.

**Continue a run**: pick the saved LoRA in `existing_lora` and queue again with more steps."""

NOTE_TEST = """## 4. Test on Turbo

The new LoRA goes straight onto **Krea 2 Turbo** (8 steps, the model the other workflows use) and renders the test prompts in **Test prompts**. Use your trigger word in them.

The LoRA is saved to `ComfyUI/output/loras/` and shows up in every LoRA loader. In the other workflows, add a **LoraLoaderModelOnly** after *Load Krea 2 Turbo* (or enable the Style LoRA node in the single-image workflow and pick your LoRA). Strength 0.8-1.0."""

NOTE_REFERENCE = """## 3. The reference

The portrait goes to Krea 2 twice, as in Comfy-Org's reference template: the Qwen3-VL text encoder **sees** it (picture 1), and the VAE-encoded image joins as a **reference latent**. Each prompt then says how to show the same person.

Identity holds best when the reference is sharp, evenly lit and shows the face clearly. Expect some drift on extreme angles (profiles, back): check the set and re-run single lines with another seed."""


# ------------------------------------------------------------------------ graph
class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self._link_id = 0

    def node(self, nid, ntype, inputs=(), outputs=(), widgets=None, title=None,
             color=None, cnr="comfy-core", mode=0, size=(440, 100)):
        props = {"Node name for S&R": ntype}
        if cnr:
            props["cnr_id"] = cnr
        n = {
            "id": nid, "type": ntype, "pos": [0, 0], "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": mode,
            "inputs": [dict({"localized_name": nm, "name": nm, "type": tp, "link": None},
                            **({"widget": {"name": nm}} if wid else {}),
                            **({"shape": 7} if opt else {}))
                       for nm, tp, wid, opt in inputs],
            "outputs": [dict({"localized_name": nm, "name": nm, "type": tp, "links": []},
                             **({"shape": 6} if lst else {}))
                        for nm, tp, *rest in outputs for lst in [bool(rest and rest[0])]],
            "properties": props,
        }
        if title:
            n["title"] = title
        if widgets is not None:
            n["widgets_values"] = widgets
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        return nid

    def link(self, src, src_slot, dst, dst_name):
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        if isinstance(src_slot, str):
            src_slot = next(i for i, o in enumerate(s["outputs"]) if o["name"] == src_slot)
        out = s["outputs"][src_slot]
        slot = next(i for i, inp in enumerate(d["inputs"]) if inp["name"] == dst_name)
        inp = d["inputs"][slot]
        if inp["link"] is not None:
            raise ValueError(f"{d['type']}.{dst_name} already linked")
        if inp["type"] != out["type"]:
            raise ValueError(f"type mismatch {s['type']}.{out['name']} ({out['type']})"
                             f" -> {d['type']}.{dst_name} ({inp['type']})")
        self._link_id += 1
        lid = self._link_id
        inp["link"] = lid
        out["links"].append(lid)
        self.links.append([lid, src, src_slot, dst, slot, out["type"]])


def md(g, nid, title, text, size):
    g.node(nid, "MarkdownNote", widgets=[text], title=title, cnr=None,
           color=("#222", "#000"), size=size)


COLUMN_W, NODE_W, GAP_X, GAP_Y, TITLE_H, PAD = 520, 460, 140, 50, 120, 30
STAGE_COLORS = {"input": "#a1309b", "models": "#3f789e", "reference": "#b06634",
                "sample": "#3f8f6e", "output": "#8a4a4a"}
INPUT = ("#322", "#533")      # red: things you set
I, W = False, True            # socket input / widget converted to input


def lay_out(g, stages, start_notes):
    """Place stages as spaced columns, each a titled group with notes under it."""
    nodes = {n["id"]: n for n in g.nodes}
    groups = []
    x = y = 0
    for nid in start_notes:
        nodes[nid]["pos"] = [x, y]
        y += nodes[nid]["size"][1] + GAP_Y
    x += COLUMN_W + GAP_X
    for gid, (title, color, ids, notes) in enumerate(stages, 1):
        y = TITLE_H
        for nid in ids:
            nodes[nid]["pos"] = [x + PAD, y]
            y += nodes[nid]["size"][1] + GAP_Y
        bottom = y - GAP_Y + PAD
        groups.append({"id": gid, "title": title, "bounding": [x, 0, COLUMN_W, bottom],
                       "color": STAGE_COLORS[color], "font_size": 26, "flags": {}})
        y = bottom + 60
        for nid in notes:
            nodes[nid]["pos"] = [x, y]
            nodes[nid]["size"][0] = COLUMN_W
            y += nodes[nid]["size"][1] + GAP_Y
        x += COLUMN_W + GAP_X
    return groups


def size(h):
    return (NODE_W, h)


def models(g, lora=False):
    """30 UNET, 31 text encoder, 32 VAE (+ 33 optional style LoRA, bypassed)."""
    g.node(30, "UNETLoader", outputs=[("MODEL", "MODEL")], widgets=[UNET, "default"],
           title="Load Krea 2 Turbo", size=size(82))
    g.node(31, "CLIPLoader", outputs=[("CLIP", "CLIP")],
           widgets=[TEXT_ENCODER, "krea2", "default"], title="Load text encoder (Qwen3-VL 4B)",
           size=size(106))
    g.node(32, "VAELoader", outputs=[("VAE", "VAE")], widgets=[VAE], title="Load VAE",
           size=size(58))
    if lora:
        g.node(33, "LoraLoaderModelOnly", inputs=[("model", "MODEL", I, False)],
               outputs=[("MODEL", "MODEL")], widgets=[STYLE_LORA, 0.8], mode=4,
               title="Style LoRA (optional, bypassed)", size=size(82))
        g.link(30, 0, 33, "model")


def sampler(g, model_src, positive_src, seed_widget, size_wh=(1024, 1024), seed_from=None):
    """50 EmptyLatentImage, 51 zeroed negative, 52 KSampler, 53 VAEDecode."""
    g.node(50, "EmptyLatentImage", outputs=[("LATENT", "LATENT")],
           widgets=[size_wh[0], size_wh[1], 1], title="Image size", color=INPUT, size=size(106))
    g.node(51, "ConditioningZeroOut", inputs=[("conditioning", "CONDITIONING", I, False)],
           outputs=[("CONDITIONING", "CONDITIONING")], title="Negative (zeroed)", size=size(26))
    seed_input = [("seed", "INT", W, False)] if seed_from else []
    g.node(52, "KSampler",
           inputs=[("model", "MODEL", I, False), ("positive", "CONDITIONING", I, False),
                   ("negative", "CONDITIONING", I, False), ("latent_image", "LATENT", I, False)]
           + seed_input,
           outputs=[("LATENT", "LATENT")],
           widgets=[seed_widget[0], seed_widget[1], STEPS, CFG, SAMPLER, SCHEDULER, 1.0],
           title=f"KSampler ({STEPS} steps, CFG {CFG:g})", size=size(262))
    g.node(53, "VAEDecode", inputs=[("samples", "LATENT", I, False), ("vae", "VAE", I, False)],
           outputs=[("IMAGE", "IMAGE")], size=size(46))
    L = g.link
    L(model_src, 0, 52, "model")
    L(positive_src, 0, 52, "positive")
    L(positive_src, 0, 51, "conditioning")
    L(51, 0, 52, "negative")
    L(50, 0, 52, "latent_image")
    L(52, 0, 53, "samples")
    L(32, 0, 53, "vae")
    if seed_from:
        L(seed_from[0], seed_from[1], 52, "seed")


def prompt_list(g, text, prefix="", suffix="", caption="", seed=42, file=""):
    g.node(20, "Krea2PromptList", outputs=[("prompts", "STRING", True), ("captions", "STRING", True),
                                           ("seeds", "INT", True), ("count", "INT")],
           widgets=[text, file, prefix, suffix, caption, seed, "increment", 0],
           title="Prompt List (one per line)", color=INPUT, cnr=PACK, size=size(620))


def build(kind):
    w = next(w for w in WORKFLOWS if w["kind"] == kind)
    g = Graph()
    L = g.link
    md(g, 1, "Start here", note_start(kind), (COLUMN_W, 860))
    md(g, 2, "About Krea 2", NOTE_ABOUT, (COLUMN_W, 380))
    md(g, 5, "About stage 2", NOTE_MODELS, (COLUMN_W, 300))
    md(g, 7, "About sampling", NOTE_SAMPLE, (COLUMN_W, 300))

    if kind == "single":
        models(g, lora=True)
        g.node(21, "CLIPTextEncode", inputs=[("clip", "CLIP", I, False)],
               outputs=[("CONDITIONING", "CONDITIONING")], widgets=[SINGLE_PROMPT],
               title="Prompt", color=INPUT, size=size(300))
        L(31, 0, 21, "clip")
        sampler(g, 33, 21, (42, "randomize"))
        g.node(60, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")], widgets=["krea2/krea2"],
               title="Save image (output/krea2/)", size=size(420))
        L(53, 0, 60, "images")
        stages = [("1. Prompt", "input", [21], []),
                  ("2. Models", "models", [30, 31, 32, 33], [5]),
                  ("3. Sample", "sample", [50, 51, 52, 53], [7]),
                  ("4. Save", "output", [60], [])]

    elif kind == "batch":
        models(g)
        prompt_list(g, BATCH_PROMPTS, file="")
        g.node(21, "CLIPTextEncode", inputs=[("clip", "CLIP", I, False), ("text", "STRING", W, False)],
               outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode each prompt",
               size=size(90))
        L(31, 0, 21, "clip")
        L(20, "prompts", 21, "text")
        sampler(g, 30, 21, (42, "fixed"), seed_from=(20, "seeds"))
        g.node(60, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")],
               widgets=["krea2_batch/krea2"], title="Save each image (output/krea2_batch/)",
               size=size(300))
        g.node(61, "Krea2ContactSheet", inputs=[("images", "IMAGE", I, False)],
               outputs=[("sheet", "IMAGE")], widgets=[3, 8, 1.0], title="Contact sheet",
               cnr=PACK, size=size(130))
        g.node(62, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")],
               widgets=["krea2_batch/contact_sheet"], title="Save contact sheet", size=size(300))
        L(53, 0, 60, "images")
        L(53, 0, 61, "images")
        L(61, 0, 62, "images")
        stages = [("1. Prompts", "input", [20], []),
                  ("2. Models", "models", [30, 31, 32], [5]),
                  ("3. Encode", "reference", [21], []),
                  ("4. Sample", "sample", [50, 51, 52, 53], [7]),
                  ("5. Save", "output", [60, 61, 62], [])]

    elif kind == "train":
        md(g, 6, "About training", NOTE_TRAIN, (COLUMN_W, 520))
        md(g, 8, "About the test", NOTE_TEST, (COLUMN_W, 420))
        g.node(10, "Krea2LoadDataset",
               outputs=[("images", "IMAGE", True), ("captions", "STRING", True), ("count", "INT")],
               widgets=[DATASET_FOLDER, "output"], title="Load Training Set (folder)", color=INPUT,
               cnr=PACK, size=size(106))
        g.node(30, "UNETLoader", outputs=[("MODEL", "MODEL")], widgets=[RAW_UNET, "default"],
               title="Load Krea 2 Raw (training)", size=size(82))
        g.node(31, "CLIPLoader", outputs=[("CLIP", "CLIP")],
               widgets=[TEXT_ENCODER, "krea2", "default"], title="Load text encoder (Qwen3-VL 4B)",
               size=size(106))
        g.node(32, "VAELoader", outputs=[("VAE", "VAE")], widgets=[VAE], title="Load VAE", size=size(58))
        g.node(40, "MakeTrainingDataset",
               inputs=[("images", "IMAGE", I, False), ("vae", "VAE", I, False), ("clip", "CLIP", I, False),
                       ("texts", "STRING", I, True)],
               outputs=[("latents", "LATENT", True), ("conditioning", "CONDITIONING", True)],
               title="Make Training Dataset (encode images + captions)", size=size(110))
        g.node(41, "TrainLoraNode",
               inputs=[("model", "MODEL", I, False), ("latents", "LATENT", I, False),
                       ("positive", "CONDITIONING", I, False)],
               outputs=[("lora", "LORA_MODEL"), ("loss_map", "LOSS_MAP"), ("steps", "INT")],
               widgets=list(TRAIN.values()), title="Train LoRA", color=INPUT, size=size(600))
        g.node(42, "SaveLoRA", inputs=[("lora", "LORA_MODEL", I, False), ("steps", "INT", W, True)],
               widgets=[LORA_PREFIX, 0], title="Save LoRA (output/loras/)", size=size(82))
        g.node(43, "LossGraphNode", inputs=[("loss", "LOSS_MAP", I, False)],
               widgets=["krea2_lora/loss_graph"], title="Loss graph (output/krea2_lora/)", size=size(300))
        g.node(34, "UNETLoader", outputs=[("MODEL", "MODEL")], widgets=[UNET, "default"],
               title="Load Krea 2 Turbo (test)", size=size(82))
        g.node(44, "LoraModelLoader", inputs=[("model", "MODEL", I, False), ("lora", "LORA_MODEL", I, False)],
               outputs=[("model", "MODEL")], widgets=[1.0, False], title="Apply the new LoRA to Turbo",
               size=size(82))
        prompt_list(g, TEST_PROMPTS, seed=3000)
        g.nodes[-1]["title"] = "Test prompts (use your trigger word)"
        g.node(21, "CLIPTextEncode", inputs=[("clip", "CLIP", I, False), ("text", "STRING", W, False)],
               outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode each test prompt",
               size=size(90))
        L(10, "images", 40, "images")
        L(10, "captions", 40, "texts")
        L(32, 0, 40, "vae")
        L(31, 0, 40, "clip")
        L(30, 0, 41, "model")
        L(40, "latents", 41, "latents")
        L(40, "conditioning", 41, "positive")
        L(41, "lora", 42, "lora")
        L(41, "steps", 42, "steps")
        L(41, "loss_map", 43, "loss")
        L(34, 0, 44, "model")
        L(41, "lora", 44, "lora")
        L(31, 0, 21, "clip")
        L(20, "prompts", 21, "text")
        sampler(g, 44, 21, (42, "fixed"), seed_from=(20, "seeds"))
        g.node(60, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")],
               widgets=["krea2_lora/test"], title="Save test images (output/krea2_lora/)", size=size(300))
        g.node(61, "Krea2ContactSheet", inputs=[("images", "IMAGE", I, False)],
               outputs=[("sheet", "IMAGE")], widgets=[4, 8, 1.0], title="Contact sheet", cnr=PACK,
               size=size(130))
        g.node(62, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")],
               widgets=["krea2_lora/test_sheet"], title="Save contact sheet", size=size(300))
        L(53, 0, 60, "images")
        L(53, 0, 61, "images")
        L(61, 0, 62, "images")
        stages = [("1. Training set", "input", [10], []),
                  ("2. Models", "models", [30, 31, 32], [5]),
                  ("3. Train", "reference", [40, 41, 42, 43], [6]),
                  ("4. Test on Turbo", "sample", [34, 44, 20, 21, 50, 51, 52, 53], [8]),
                  ("5. Save", "output", [60, 61, 62], [7])]

    else:   # views / dataset: the reference portrait carries the identity
        views = kind == "views"
        models(g)
        g.node(10, "LoadImage", outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
               widgets=[REFERENCE, "image"], title="Load Reference (the person)", color=INPUT,
               size=size(520))
        if views:
            prompt_list(g, EIGHT_VIEWS, VIEW_PREFIX, VIEW_SUFFIX, "", seed=1000)
        else:
            prompt_list(g, TRAINING_SET, DATASET_PREFIX, DATASET_SUFFIX, TRIGGER, seed=2000)
        md(g, 6, "About stage 3", NOTE_REFERENCE, (COLUMN_W, 380))
        g.node(21, "TextEncodeQwenImageEditPlus",
               inputs=[("clip", "CLIP", I, False), ("vae", "VAE", I, True),
                       ("image1", "IMAGE", I, True), ("image2", "IMAGE", I, True),
                       ("image3", "IMAGE", I, True), ("prompt", "STRING", W, False)],
               outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""],
               title="Encode prompt + reference", size=size(150))
        g.node(22, "FluxKontextMultiReferenceLatentMethod",
               inputs=[("conditioning", "CONDITIONING", I, False)],
               outputs=[("CONDITIONING", "CONDITIONING")], widgets=["index_timestep_zero"],
               title="Reference latent method", size=size(58))
        g.node(23, "ModelSamplingFlux", inputs=[("model", "MODEL", I, False)],
               outputs=[("MODEL", "MODEL")], widgets=[1.15, 0.5, 1024, 1024],
               title="Model sampling (as Comfy-Org's reference template)", size=size(130))
        L(31, 0, 21, "clip")
        L(32, 0, 21, "vae")
        L(10, 0, 21, "image1")
        L(20, "prompts", 21, "prompt")
        L(21, 0, 22, "conditioning")
        L(30, 0, 23, "model")
        sampler(g, 23, 22, (42, "fixed"), seed_from=(20, "seeds"))
        g.node(61, "Krea2ContactSheet", inputs=[("images", "IMAGE", I, False)],
               outputs=[("sheet", "IMAGE")], widgets=[4, 8, 1.0], title="Contact sheet",
               cnr=PACK, size=size(130))
        L(53, 0, 61, "images")
        if views:
            g.node(60, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")],
                   widgets=["krea2_views/view"], title="Save each view (output/krea2_views/)",
                   size=size(300))
            g.node(62, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")],
                   widgets=["krea2_views/contact_sheet"], title="Save contact sheet",
                   size=size(300))
            L(53, 0, 60, "images")
            save = [60, 61, 62]
        else:
            g.node(60, "SaveImageTextDataSetToFolder",
                   inputs=[("images", "IMAGE", I, False), ("texts", "STRING", I, True)],
                   widgets=["krea2_dataset/my_person", "image", "overwrite"],
                   title="Save images + captions (output/krea2_dataset/)", color=INPUT,
                   size=size(160))
            g.node(62, "SaveImage", inputs=[("images", "IMAGE", I, False)], outputs=[("images", "IMAGE")],
                   widgets=["krea2_dataset/contact_sheet"], title="Save contact sheet (check it!)",
                   size=size(300))
            L(53, 0, 60, "images")
            L(20, "captions", 60, "texts")
            save = [60, 61, 62]
        L(61, 0, 62, "images")
        stages = [("1. The person", "input", [10, 20], []),
                  ("2. Models", "models", [30, 31, 32, 23], [5]),
                  ("3. Prompt + reference", "reference", [21, 22], [6]),
                  ("4. Sample", "sample", [50, 51, 52, 53], [7]),
                  ("5. Save", "output", save, [])]

    groups = lay_out(g, stages, [1, 2])
    return {
        "id": w["id"], "revision": 0,
        "last_node_id": max(n["id"] for n in g.nodes), "last_link_id": g._link_id,
        "nodes": g.nodes, "links": g.links, "groups": groups, "config": {},
        "extra": {"ds": {"scale": 0.6, "offset": [80, 80]}}, "version": 0.4,
    }


if __name__ == "__main__":
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    for w in WORKFLOWS:
        wf = build(w["kind"])
        path = os.path.join(WORKFLOW_DIR, w["path"])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {os.path.normpath(path)}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")
