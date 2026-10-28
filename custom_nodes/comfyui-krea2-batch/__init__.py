"""
Small helper nodes for the Krea 2 template. No extra Python packages.

Prompt List: one prompt per line (or a .txt file in ComfyUI/input) -> a *list* of
prompts, captions and seeds. ComfyUI runs everything downstream once per item, so one
queue renders the whole list: the batch workflow, the eight views and the LoRA
training set are all this node feeding a normal Krea 2 graph.

Contact Sheet: every image of such a run -> one grid image, to check a set at a glance.

Load Training Set: a folder of images + same-name caption .txt files (what the training
set workflow writes to ComfyUI/output/krea2_dataset/<name>) -> image and caption lists
for ComfyUI's own Make Training Dataset / Train LoRA nodes. ComfyUI's Load Image-Text
node reads ComfyUI/input only; this one reads output/ too, so nothing has to be moved.

ComfyUI's Save LoRA Weights writes to ComfyUI/output/loras/. That folder is added to the
LoRA search paths here, so a LoRA trained on the pod shows up in every LoRA loader.
"""
import hashlib
import os

import numpy as np
import torch

try:
    import folder_paths
except ImportError:          # outside ComfyUI (tests)
    folder_paths = None

CATEGORY = "Krea 2"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

if folder_paths is not None:
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))


def input_dir():
    return folder_paths.get_input_directory() if folder_paths else os.getcwd()


def output_dir():
    return folder_paths.get_output_directory() if folder_paths else os.getcwd()


def read_lines(prompts, prompts_file):
    """Non-empty lines that are not comments. A file, when named, replaces the text box."""
    text = prompts
    name = (prompts_file or "").strip()
    if name:
        path = name if os.path.isabs(name) else os.path.join(input_dir(), name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt List: no file {path}. Put it in ComfyUI/input/ "
                                    f"(JupyterLab, port 8888) or clear prompts_file.")
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def join(*parts):
    return " ".join(p.strip() for p in parts if p and p.strip())


class Krea2PromptList:
    """Lines -> prompts (prefix + line + suffix), captions (caption_prefix + line), seeds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompts": ("STRING", {"multiline": True, "default": "",
                                   "tooltip": "One prompt per line. Empty lines and lines "
                                              "starting with # are skipped."}),
            "prompts_file": ("STRING", {"default": "",
                                        "tooltip": "Optional: a .txt file in ComfyUI/input/ "
                                                   "(one prompt per line). Replaces the text box."}),
            "prompt_prefix": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "Added before every line."}),
            "prompt_suffix": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "Added after every line."}),
            "caption_prefix": ("STRING", {"default": "",
                                          "tooltip": "For training sets: the caption saved "
                                                     "with each image is this + the line "
                                                     "(e.g. your LoRA's trigger word)."}),
            "start_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                                   "tooltip": "Seed of the first line. Named start_seed, not seed, "
                                              "so the frontend adds no randomize control."}),
            "seed_mode": (["increment", "fixed"], {"default": "increment",
                                                   "tooltip": "increment: start_seed, start_seed+1, ... "
                                                              "fixed: the same seed for every line."}),
            "limit": ("INT", {"default": 0, "min": 0, "max": 10000,
                              "tooltip": "Render only the first N lines (0 = all)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompts", "captions", "seeds", "count")
    OUTPUT_IS_LIST = (True, True, True, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("One prompt per line -> a list: everything after this node runs once per "
                   "line, so one queue renders them all.")

    @classmethod
    def IS_CHANGED(cls, prompts, prompts_file, **kw):
        """Re-read the file when it changes, even if no widget did."""
        name = (prompts_file or "").strip()
        if not name:
            return ""
        path = name if os.path.isabs(name) else os.path.join(input_dir(), name)
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return float("nan")

    def run(self, prompts, prompts_file, prompt_prefix, prompt_suffix, caption_prefix,
            start_seed, seed_mode, limit):
        lines = read_lines(prompts, prompts_file)
        if limit:
            lines = lines[:limit]
        if not lines:
            raise ValueError("Prompt List: no prompts. Write one per line, or name a file.")
        full = [join(prompt_prefix, line, prompt_suffix) for line in lines]
        captions = [join(caption_prefix, line) for line in lines]
        step = 1 if seed_mode == "increment" else 0
        seeds = [(start_seed + i * step) % 0x10000000000000000 for i in range(len(lines))]
        print(f"[Krea2 Prompt List] {len(lines)} prompts, seeds {seeds[0]}..{seeds[-1]}")
        return (full, captions, seeds, len(lines))


def contact_sheet(images, columns=4, gap=8, background=1.0):
    """List of [B, H, W, C] tensors -> one [1, H', W', C] grid, every tile at the first's size."""
    tiles = [t for batch in images for t in batch]
    if not tiles:
        raise ValueError("Contact Sheet: no images.")
    h, w, c = tiles[0].shape
    columns = max(1, min(columns, len(tiles)))
    rows = -(-len(tiles) // columns)
    sheet = torch.full((rows * h + (rows + 1) * gap, columns * w + (columns + 1) * gap, c),
                       float(background), dtype=tiles[0].dtype, device=tiles[0].device)
    for i, tile in enumerate(tiles):
        if tile.shape != (h, w, c):
            tile = torch.nn.functional.interpolate(
                tile.permute(2, 0, 1)[None], size=(h, w), mode="bilinear",
                align_corners=False)[0].permute(1, 2, 0)
        r, col = divmod(i, columns)
        y, x = gap + r * (h + gap), gap + col * (w + gap)
        sheet[y:y + h, x:x + w] = tile
    return sheet[None]


class Krea2ContactSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "columns": ("INT", {"default": 4, "min": 1, "max": 16}),
            "gap": ("INT", {"default": 8, "min": 0, "max": 128}),
            "background": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "0 = black, 1 = white"}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Every image of a list run -> one grid image."

    def run(self, images, columns, gap, background):
        return (contact_sheet(images, columns[0], gap[0], background[0]),)


def dataset_folder(folder, location):
    """folder (relative to ComfyUI/output or ComfyUI/input) -> its absolute path, which
    must stay inside that base folder."""
    base = os.path.realpath(output_dir() if location == "output" else input_dir())
    path = os.path.realpath(os.path.join(base, (folder or "").strip().strip("/\\")))
    if os.path.commonpath([base, path]) != base or path == base:
        raise ValueError(f"Load Training Set: give a folder inside ComfyUI/{location}, "
                         f"for example krea2_dataset/my_person.")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Load Training Set: no folder {path}. Run the training set "
                                f"workflow first, or upload your images + captions there.")
    return path


def read_dataset(path):
    """Sorted image files with the caption from the same-name .txt ("" if there is none)."""
    names = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTENSIONS))
    if not names:
        raise ValueError(f"Load Training Set: no images in {path}.")
    items = []
    for name in names:
        caption_file = os.path.join(path, os.path.splitext(name)[0] + ".txt")
        caption = ""
        if os.path.isfile(caption_file):
            with open(caption_file, encoding="utf-8-sig") as f:
                caption = f.read().strip()
        items.append((os.path.join(path, name), caption))
    return items


class Krea2LoadDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"default": "krea2_dataset/my_person",
                                  "tooltip": "Folder of images + same-name .txt captions, "
                                             "inside ComfyUI/output (or input)."}),
            "location": (["output", "input"], {"default": "output"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "captions", "count")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("A folder of images + caption .txt files (the training set workflow's "
                   "output) -> lists for Make Training Dataset.")

    @classmethod
    def IS_CHANGED(cls, folder, location):
        """Re-read when files are added, removed or edited (e.g. weak images deleted)."""
        try:
            path = dataset_folder(folder, location)
        except (ValueError, FileNotFoundError):
            return float("nan")
        h = hashlib.sha256()
        for name in sorted(os.listdir(path)):
            st = os.stat(os.path.join(path, name))
            h.update(f"{name}:{st.st_size}:{st.st_mtime_ns};".encode())
        return h.hexdigest()

    def run(self, folder, location):
        from PIL import Image, ImageOps
        items = read_dataset(dataset_folder(folder, location))
        images, captions = [], []
        for path, caption in items:
            with Image.open(path) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                images.append(torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)[None])
            captions.append(caption)
        missing = sum(1 for c in captions if not c)
        print(f"[Krea2 Load Training Set] {len(images)} images"
              + (f", {missing} without a caption" if missing else ""))
        return (images, captions, len(images))


NODE_CLASS_MAPPINGS = {"Krea2PromptList": Krea2PromptList, "Krea2ContactSheet": Krea2ContactSheet,
                       "Krea2LoadDataset": Krea2LoadDataset}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2PromptList": "Prompt List (one per line)",
                              "Krea2ContactSheet": "Contact Sheet",
                              "Krea2LoadDataset": "Load Training Set (folder)"}
