#!/bin/bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting Krea 2 Image Template Setup ==="
echo "Template revision: $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || printf unknown)"

# Refuse a disposable workspace before doing installation work.
# The image's /start.sh fixes ComfyUI at this path.
export COMFYUI_PATH="${COMFYUI_PATH:-/workspace/runpod-slim/ComfyUI}"
if [ "$COMFYUI_PATH" != /workspace/runpod-slim/ComfyUI ]; then
  echo "FATAL: this image's /start.sh uses /workspace/runpod-slim/ComfyUI. Leave COMFYUI_PATH unset."
  exit 1
fi

# The attached volume decides the layout (KREA2_STORAGE=network|global overrides):
#   network  a regional Network Volume at /workspace holds everything.
#   global   a Global Volume is object storage: no atomic rename, hard links, file
#            locks or permission bits, so no virtualenv or git checkout can live on it.
#            ComfyUI, its venv and custom nodes run from the container disk and are
#            set up again on each fresh container; models, inputs, outputs and saved
#            workflows stay on the volume, at the paths a Network Volume uses.
STORAGE_PY="$SCRIPT_DIR/scripts/storage.py"
STORAGE="$(python3 "$STORAGE_PY" detect)" || exit 1
read -r STORAGE_MODE DATA_ROOT <<< "$STORAGE"
LOCAL_ROOT=/opt/krea2-local
if [ "$STORAGE_MODE" = global ]; then
  python3 "$STORAGE_PY" preflight-global "$DATA_ROOT"
  VOLUME_COMFY="$DATA_ROOT/runpod-slim/ComfyUI"
  COMFYUI_PATH="$LOCAL_ROOT/ComfyUI"
  # Settings, caches and download staging: container disk.
  STATE_DIR="$LOCAL_ROOT"
  STAGE_DIR="$LOCAL_ROOT/downloads"
  mkdir -p "$STAGE_DIR"
  # The image's start.sh fixes ComfyUI (and so its venv) under /workspace, which here
  # can be the Global Volume. Setup ends by running a copy pointed at the
  # container-disk install, with FileBrowser's database (BoltDB: memory-mapped and
  # locked) there too. Everything else, including comfyui_args.txt on the volume, is
  # the image's own. Checked now, before any download.
  IMAGE_START="$LOCAL_ROOT/image-start.sh"
  sed -e "s#^COMFYUI_DIR=\"/workspace/runpod-slim/ComfyUI\"\$#COMFYUI_DIR=\"$COMFYUI_PATH\"#" \
      -e "s#^DB_FILE=\"/workspace/runpod-slim/filebrowser.db\"\$#DB_FILE=\"$LOCAL_ROOT/filebrowser.db\"#" \
      /start.sh > "$IMAGE_START"
  if ! grep -qx "COMFYUI_DIR=\"$COMFYUI_PATH\"" "$IMAGE_START"; then
    echo "FATAL: this image's /start.sh does not set COMFYUI_DIR=\"/workspace/runpod-slim/ComfyUI\","
    echo "         so it cannot be pointed at the container disk for a Global Volume."
    echo "         Use runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8, or a Network Volume."
    exit 1
  fi
  if grep -qx "DB_FILE=\"$LOCAL_ROOT/filebrowser.db\"" "$IMAGE_START"; then
    export FB_DATABASE="$LOCAL_ROOT/filebrowser.db"
  fi
  chmod +x "$IMAGE_START"
else
  python3 "$STORAGE_PY" preflight "$COMFYUI_PATH"
  STATE_DIR=/workspace
  STAGE_DIR=""
  IMAGE_START=""
fi
echo "Storage: $STORAGE_MODE volume at $DATA_ROOT"

# Apply before the image starts Jupyter. The image's /start.sh runs JupyterLab
# without a token when JUPYTER_DISABLE_AUTH=true, uses JUPYTER_PASSWORD as the
# token when set, and otherwise does not start Jupyter at all. The exported value
# reaches /start.sh through the final exec. Jupyter is optional, so a start script
# without that switch is reported rather than stopping the pod.
if [ "${JUPYTER_NO_AUTH:-1}" = "1" ]; then
  export JUPYTER_DISABLE_AUTH=true
  echo "Jupyter authentication disabled: anyone with access to its URL can run commands."
  if ! grep -q 'JUPYTER_DISABLE_AUTH' /start.sh 2>/dev/null; then
    echo "WARNING: this image's /start.sh has no JUPYTER_DISABLE_AUTH switch, so JupyterLab"
    echo "         may not start. Use runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8, or set"
    echo "         JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD. ComfyUI is unaffected."
  fi
else
  unset JUPYTER_DISABLE_AUTH
  if [ -z "${JUPYTER_PASSWORD:-}" ]; then
    echo "NOTE: JUPYTER_NO_AUTH=0 without JUPYTER_PASSWORD: the image does not start JupyterLab."
  fi
fi

echo "=== Ensuring System Dependencies are Installed ==="
# The image already supplies these. Contact apt mirrors only if something is
# actually missing; an unrelated mirror outage must not break a complete image.
packages=()
command -v wget >/dev/null || packages+=(wget)
command -v git >/dev/null || packages+=(git)
[ -s /etc/ssl/certs/ca-certificates.crt ] || packages+=(ca-certificates)
if [ "${#packages[@]}" -gt 0 ]; then
  apt-get update
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

# Jupyter serves /workspace in the base image. Keep its settings and user data on
# that persistent volume too; /root belongs to the disposable container disk. On a
# Global Volume they stay on the container disk: IPython keeps its history in SQLite,
# which needs the file locking that volume lacks.
export JUPYTER_CONFIG_DIR="$STATE_DIR/.jupyter"
export JUPYTER_DATA_DIR="$STATE_DIR/.local/share/jupyter"
export IPYTHONDIR="$STATE_DIR/.ipython"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$IPYTHONDIR"

# Network Volumes can expose fixed permissions. Jupyter's cookie secrets and
# kernel connection files require private 0600 files; these are disposable
# runtime state, not notebooks or settings. Keep them on the container filesystem.
JUPYTER_RUNTIME_DIR="$(mktemp -d /tmp/krea2-jupyter-runtime.XXXXXX)"
export JUPYTER_RUNTIME_DIR
chmod 700 "$JUPYTER_RUNTIME_DIR"
python3 - <<'PY'
import os
from pathlib import Path
from jupyter_core.paths import secure_write

probe = Path(os.environ["JUPYTER_RUNTIME_DIR"]) / "secure-write-check"
try:
    with secure_write(str(probe)) as stream:
        stream.write("Jupyter runtime permissions verified")
finally:
    probe.unlink(missing_ok=True)
print(f"Jupyter runtime ready at {probe.parent}; notebooks stay in /workspace, "
      f"settings in {os.environ['JUPYTER_CONFIG_DIR']}.")
PY

# Krea 2 is native in ComfyUI since v0.26.0; the base image ships v0.35.0.
#   Base image: runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8
# This script deliberately does NOT update ComfyUI core: on these images
# `pip install -r requirements.txt` can reinstall torch and break the baked CUDA build.

# Comfy-Org/Krea-2 is ungated, so no token is needed. Set HF_TOKEN anyway if you have
# one: anonymous requests are rate-limited (~120/h vs ~1000/h) and slower.
HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN || echo "NOTE: HF_TOKEN not set - downloads may be slower and rate-limited."
export HF_HOME="${HF_HOME:-$STATE_DIR/.cache/huggingface}"

# Self-healing check prevents directory collisions and fixes broken folders
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "First time setup: Copying baked ComfyUI to workspace..."
  if [ -e "$COMFYUI_PATH" ]; then
    echo "FATAL: existing ComfyUI directory has no main.py; refusing to delete user files."
    exit 1
  fi
  mkdir -p "$(dirname "$COMFYUI_PATH")"
  cp -r /opt/comfyui-baked "$COMFYUI_PATH"
fi

if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "FATAL: no ComfyUI at $COMFYUI_PATH and /opt/comfyui-baked did not provide one."
  echo "       Check the container image."
  exit 1
fi
if [ "$STORAGE_MODE" = global ]; then
  echo "Linking ComfyUI's data folders to the Global Volume..."
  python3 "$STORAGE_PY" link-global "$COMFYUI_PATH" "$VOLUME_COMFY"
  # Jupyter and FileBrowser serve /workspace; show the volume there when it is
  # mounted elsewhere.
  if [ "$DATA_ROOT" != /workspace ]; then
    mkdir -p /workspace
    ln -sfn "$DATA_ROOT" /workspace/global-volume
  fi
fi
echo "ComfyUI: $COMFYUI_PATH"

# Krea 2 needs a ComfyUI that knows its text encoder ("krea2" in CLIPLoader).
if ! grep -q '"krea2"' "$COMFYUI_PATH/nodes.py"; then
  echo "FATAL: this ComfyUI does not support Krea 2 (it needs v0.26.0 or newer)."
  echo "         Use the template's image, runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8."
  exit 1
fi

# Validation only needs the standard library. Use the image interpreter that
# already passed preflight, not a possibly stale persisted ComfyUI virtualenv.
PY="$(command -v python3)"
echo "Setup validator Python: $PY"

# PyTorch comes from the image. The image also starts on hosts whose driver is too old
# for its CUDA build, where PyTorch cannot see the GPU - stop here, before the
# downloads, rather than generate on the CPU.
TORCH_PY="$(command -v python3.12 || command -v python3)"
if ! "$TORCH_PY" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    sys.exit(f"FATAL: cannot import the image's PyTorch ({exc}).")
build = torch.version.cuda or "none"
print(f"PyTorch {torch.__version__}, CUDA build {build}")
if build == "none":
    sys.exit("FATAL: the image's PyTorch has no CUDA build. Change the template's "
             "container image to runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8.")
if not torch.cuda.is_available():
    sys.exit(f"FATAL: CUDA {build} PyTorch cannot use this GPU: the host driver is too "
             "old for it. Redeploy on another GPU or datacenter; the same "
             "volume can be reattached.")
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {name}, {vram:.0f} GB")
if vram < 20:
    print("WARNING: Krea 2 Turbo FP8 is a 12 GB model plus a 5 GB text encoder; under "
          "24 GB of VRAM ComfyUI offloads and every image is much slower.")
if vram < 32:
    print(f"NOTE: {vram:.0f} GB is fine for generating images. Training a LoRA (workflow 5) "
          "needs 32 GB+, 48 GB recommended: deploy a bigger GPU with the same volume for it.")
PY
then
  exit 1
fi

# 1. Custom nodes: this repo's small comfyui-krea2-batch pack (Prompt List, Contact
# Sheet). Standard library + torch only: nothing to pip install.
echo "Installing custom nodes..."
NODE_DIR="$COMFYUI_PATH/custom_nodes/comfyui-krea2-batch"
mkdir -p "$NODE_DIR"
cp "$SCRIPT_DIR/custom_nodes/comfyui-krea2-batch/"*.py "$NODE_DIR/"

# 2. Model folders, inputs and samples
echo "Preparing model directories..."
for d in diffusion_models text_encoders vae loras; do
  mkdir -p "$COMFYUI_PATH/models/$d"
done
# The reference portrait the eight-views and training-set workflows open on, and the
# batch workflow's sample prompt file. Never overwrites a file of the same name.
for sample in "$SCRIPT_DIR/samples/"* "$SCRIPT_DIR/prompts/"*; do
  [ -e "$COMFYUI_PATH/input/$(basename "$sample")" ] || cp "$sample" "$COMFYUI_PATH/input/"
done

# -------------------------------------------------------------------
# Download helpers
# -------------------------------------------------------------------
# `hf` is only a downloader - ComfyUI never imports it - so it lives in its own venv.
# Upgrading huggingface_hub inside ComfyUI's environment breaks transformers' version
# pin, and ComfyUI then fails to start with "from transformers import CLIPTokenizer".
# Its Xet backend is much faster than wget on multi-GB files but cannot resume a
# partial file, so wget stays as the fallback.
echo "Installing fast downloader (optional)..."
HF_VENV="$STATE_DIR/.hf-cli"
HF_BIN=""
if [ ! -x "$HF_VENV/bin/hf" ]; then
  if python3 -m venv "$HF_VENV" 2>/dev/null; then
    "$HF_VENV/bin/pip" install -q -U "huggingface_hub[hf-xet]" 2>/dev/null || true
  fi
fi
for cand in "$HF_VENV/bin/hf" "$HF_VENV/bin/huggingface-cli"; do
  [ -x "$cand" ] && { HF_BIN="$cand"; break; }
done
if [ -n "$HF_BIN" ]; then
  echo "Fast downloader: $HF_BIN (isolated venv)"
else
  echo "Fast downloader unavailable, using wget."
fi

# A complete file: a safetensors header that accounts for every byte.
file_ok() {
  [ -f "$1" ] || return 1
  "$PY" - "$1" <<'PY'
import json, os, struct, sys
path = sys.argv[1]
try:
    if os.path.getsize(path) < 1000000:
        raise ValueError('too small')
    with open(path, 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= size <= min(os.path.getsize(path) - 8, 100_000_000):
            raise ValueError('invalid header length')
        header = json.loads(f.read(size))
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if end + size + 8 != os.path.getsize(path):
            raise ValueError('incomplete tensor data')
except (OSError, ValueError, KeyError, TypeError, struct.error):
    sys.exit(1)
PY
}

# Move a verified file to its final name. On a Network Volume files are staged beside
# the destination, so this is a rename, not a multi-GB copy. On a Global Volume they are
# staged on the container disk (the volume has neither the downloader's file locks nor
# atomic rename) and copied straight to the final name, which counts only once file_ok
# accepts the copy; the staged file is kept until then.
put_in_place() {
  # $1 = verified staged file, $2 = destination
  if [ -z "${STAGE_DIR:-}" ]; then
    mv -f "$1" "$2"
    return
  fi
  cp -f "$1" "$2" && file_ok "$2" && rm -f "$1"
}

hf_fast() {
  # $1 = repo, $2 = path inside the repo, $3 = destination file
  local repo="$1" rpath="$2" dest="$3" tmp
  [ -n "$HF_BIN" ] || return 1
  tmp="$(mktemp -d "${STAGE_DIR:-$(dirname "$dest")}/.hfdl.XXXXXX")" || return 1
  if HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" download "$repo" "$rpath" --local-dir "$tmp"; then
    file_ok "$tmp/$rpath" && put_in_place "$tmp/$rpath" "$dest"
  fi
  rm -rf "$tmp"
  file_ok "$dest"
}

fetch_url() {
  # $1 = destination file, $2 = URL
  local dest="$1" url="$2" part="$1.part"
  if [ -n "${STAGE_DIR:-}" ]; then
    part="$STAGE_DIR/$(basename "$dest").part"
  fi
  if [ -n "$HF_TOKEN" ]; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 \
      --header="Authorization: Bearer $HF_TOKEN" -O "$part" "$url" || return 1
  else
    wget -c -q --show-progress --tries=3 --read-timeout=120 -O "$part" "$url" || return 1
  fi
  file_ok "$part" && put_in_place "$part" "$dest"
}

KREA_REPO="Comfy-Org/Krea-2"
download_model() {
  # $1 = path inside Comfy-Org/Krea-2, which is also the path under ComfyUI/models
  # (diffusion_models/..., text_encoders/..., vae/..., loras/...).
  local rpath="$1" dest="$COMFYUI_PATH/models/$1"
  mkdir -p "$(dirname "$dest")"
  if file_ok "$dest"; then
    echo "$rpath already exists, skipping."
    return 0
  fi
  echo "Downloading $rpath ..."
  hf_fast "$KREA_REPO" "$rpath" "$dest" && { echo "$rpath done."; return 0; }
  echo "  falling back to wget..."
  fetch_url "$dest" "https://huggingface.co/$KREA_REPO/resolve/main/$rpath" || true
  if ! file_ok "$dest"; then
    echo "ERROR: could not download $rpath. Existing files were retained for retry."
    return 1
  fi
  echo "$rpath done."
}

# -------------------------------------------------------------------
# 3. Model downloads (~29.6 GB by default)
# -------------------------------------------------------------------
# KREA2_MODELS: which Krea 2 checkpoints, comma-separated. The image workflows open on
# Turbo fp8 (pick another file in their "Load Krea 2 Turbo" node); the LoRA training
# workflow trains on Raw fp8.
#   fp8    krea2_turbo_fp8_scaled    12.2 GB  24 GB+ GPUs (default)
#   bf16   krea2_turbo_bf16          24.5 GB  48 GB+ GPUs, full precision
#   nvfp4  krea2_turbo_nvfp4          7.2 GB  RTX 50xx / Blackwell only
#   raw    krea2_raw_fp8_scaled      12.2 GB  undistilled, for LoRA training (default)
# KREA2_TEXT_ENCODER: fp8 (4.9 GB, default) or bf16 (8.3 GB).
# DOWNLOAD_STYLE_LORAS=1: Krea's nine style LoRAs + the style-reference LoRA (~4.4 GB).
KREA2_MODELS="${KREA2_MODELS:-fp8,raw}"
KREA2_TEXT_ENCODER="${KREA2_TEXT_ENCODER:-fp8}"
DOWNLOAD_STYLE_LORAS="${DOWNLOAD_STYLE_LORAS:-0}"
FAILED=0

# Global Volume: each file is staged on the container disk first (the largest is
# 12-25 GB). Later boots need no staging space at all.
if [ -n "$STAGE_DIR" ]; then
  free_gb=$(( $(df -Pk "$STAGE_DIR" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
  echo "Download staging: $STAGE_DIR ($free_gb GB free on the container disk)"
  if [ "$free_gb" -lt 14 ]; then
    echo "WARNING: a missing model needs up to 13 GB (25 GB for bf16) of staging space on"
    echo "         the container disk. If a download fails with 'No space left on device',"
    echo "         set Container Disk to 40 GB in the pod settings and restart."
  fi
fi

for choice in $(printf '%s' "$KREA2_MODELS" | tr ',' ' '); do
  case "$choice" in
    fp8)   download_model "diffusion_models/krea2_turbo_fp8_scaled.safetensors" || FAILED=1 ;;
    bf16)  download_model "diffusion_models/krea2_turbo_bf16.safetensors" || FAILED=1 ;;
    nvfp4) download_model "diffusion_models/krea2_turbo_nvfp4.safetensors" || FAILED=1 ;;
    raw)   download_model "diffusion_models/krea2_raw_fp8_scaled.safetensors" || FAILED=1 ;;
    *) echo "ERROR: KREA2_MODELS=$choice is not fp8, bf16, nvfp4 or raw."; FAILED=1 ;;
  esac
done
case "$KREA2_TEXT_ENCODER" in
  fp8)  download_model "text_encoders/qwen3vl_4b_fp8_scaled.safetensors" || FAILED=1 ;;
  bf16) download_model "text_encoders/qwen3vl_4b_bf16.safetensors" || FAILED=1 ;;
  *) echo "ERROR: KREA2_TEXT_ENCODER=$KREA2_TEXT_ENCODER is not fp8 or bf16."; FAILED=1 ;;
esac
download_model "vae/qwen_image_vae.safetensors" || FAILED=1
if [ "$DOWNLOAD_STYLE_LORAS" = "1" ]; then
  for lora in darkbrush dotmatrix kidsdrawing neondrip rainywindow retroanime \
              softwatercolor sunsetblur vintagetarot style_reference; do
    download_model "loras/krea2_$lora.safetensors" || FAILED=1
  done
fi

if [ "$FAILED" = "1" ]; then
  echo "FATAL: one or more model downloads failed (see ERROR lines above)."
  echo "         Restart the pod to retry - finished files are kept and skipped."
  exit 1
fi

# 4. Install bundled workflows into ComfyUI's Workflows sidebar
echo "Installing workflows..."
WF_DEST="$COMFYUI_PATH/user/default/workflows"
mkdir -p "$WF_DEST"
if [ -d "$SCRIPT_DIR/workflows" ]; then
  # Never overwrite a workflow already edited on the volume.
  for workflow in "$SCRIPT_DIR/workflows/"*.json; do
    [ -e "$WF_DEST/$(basename "$workflow")" ] || cp "$workflow" "$WF_DEST/"
  done
  ls "$WF_DEST" | sed 's/^/  workflow: /'
else
  echo "No workflows directory found in repo, skipping."
fi

# 5. Start ComfyUI using the official RunPod entrypoint
echo "Setup complete! Handing over to start script..."
if [ -n "$IMAGE_START" ]; then
  # The image creates comfyui_args.txt here before starting ComfyUI.
  mkdir -p /workspace/runpod-slim
  exec "$IMAGE_START"
fi
exec /start.sh
