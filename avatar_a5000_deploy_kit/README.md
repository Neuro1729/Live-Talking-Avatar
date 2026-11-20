# A5000 dual-GPU real-time avatar deployment kit

This is a **deployment companion** for the user's existing private package
`gdrive:avatar_inference_minimal` (3.358 GiB / 328 files, confirmed uploaded on 2026-11-12).
It is not a new copy of the original source tree and contains **no trained weights,
voice data, video clips, API secrets, or 18 GB caches**.

## Purpose and accurate scope

- GPU 0: MuseTalk v1.5 inference, personalized stage-2 step-10000 UNet.
- GPU 1: local Whisper speech recognition (Hindi/Hinglish plus English).
- Online: Sarvam LLM and Sarvam WebSocket TTS.
- Reuse original custom graph `motion_graph_silent_7f1cea162602.json` (56 runtime
  segments and 26 source clips); do **not** rebuild it.
- Recreate 26 caches using **original custom** `scripts/prep_avatar.py`, not stock
  `MuseTalk/scripts/preprocess.py`.
- Keep public HTTPS proxy separate; local STT listens on 127.0.0.1:8310.

### Two caveats before claiming full deployment

1. The original 3.358 GiB Drive package omitted **`scripts/prep_avatar.py`**.
   This kit cannot reconstruct the original preprocessor from terminal logs. Upload
   the actual script with `bin/00_upload_missing_preprocessor_gpu20.sh` before
   downloading the package on the new server. If that script imports other private
   root-level modules, copy them too; its import list is printed for inspection.
2. The original project was proven on gpu20/gpu28, not yet on the A5000 machine.
   The MuseTalk Python environment uses Python 3.10, PyTorch 2.0.1 / CUDA 11.8
   and compiled MM* packages. The captured pip freeze is useful but may need
   adjustments for the target operating system and NVIDIA driver. Test
   dependencies and a single cache before building all 26.

## Directory layout on the new A5000 server

Unzip this kit at `$HOME/a5000_avatar/kit/` (or anywhere), and keep the
project at `$HOME/a5000_avatar/realtime/`.
Set `AVATAR_ROOT` if using a different project directory. The secret lives at
`$HOME/a5000_avatar/.env`, **outside** the project root.

```
$HOME/a5000_avatar/
├── .env                                        <-- insert SARVAM_API_KEY
├── kit/                                        <-- contents of this ZIP
└── realtime/                                   <-- rclone Drive package
    ├── MuseTalk/models/musetalkV15_shubh/unet.pth   <-- fine-tuned UNet
    ├── MuseTalk/models/musetalkV15/musetalk.json   <-- base architecture config
    ├── MuseTalk/models/sd-vae/                 <-- downloaded pretrained
    ├── MuseTalk/models/whisper/                <-- Tiny audio features
    ├── base_clips/<26 original names>.mp4       <-- runtime video bank / dataset
    ├── scripts/prep_avatar.py                  <-- copied from gpu20
    ├── avatar_cache/<clip>/                    <-- regenerated (~18 GiB total)
    ├── avatar_system/                          <-- original conversation and motion code
    └── envs/musetalk, envs/analysis, envs/stt  <-- Python environments
```

**The 328 training dataset clips in `MuseTalk/dataset/shubh` are NOT required
for inference or runtime cache generation.** The needed video data is the 26
runtime `base_clips/*.mp4`, which is already included in the Drive upload.
Do not mistake the training dataset for the runtime motion clips.

## Step 0: On ORIGINAL gpu20, upload the missing tiny preprocessor

Copy this kit to gpu20 (or paste the script there), and run:

```bash
bash bin/00_upload_missing_preprocessor_gpu20.sh
```

This adds `scripts/prep_avatar.py` to the **existing Drive folder** without
reuploading the 3.2 GB fine-tuned checkpoint. Read the printed local imports
for any additional scripts it depends on.

## Step 1: On the TWO A5000 GPU server, fetch the Drive package

Install/configure `rclone` and authorize the same Google Drive remote first.
The existing shared rclone Google Drive client ID generates a retirement notice;
configure your own client ID for reliable ongoing use.

```bash
mkdir -p "$HOME/a5000_avatar/kit"
# Extract this ZIP into the above kit directory.
export AVATAR_ROOT="$HOME/a5000_avatar/realtime"
bash "$HOME/a5000_avatar/kit/bin/01_download_minimal.sh"
```

Your **personalized fine-tuned checkpoint** belongs at:

`$AVATAR_ROOT/MuseTalk/models/musetalkV15_shubh/unet.pth`

The Drive package already contains it; if you transfer it separately, place
it at the same path. Keep the stock UNet and the personalized UNet distinct.
Expected size: about 3.4 billion bytes (3.167 GiB).

Your 26 video files belong at:

`$AVATAR_ROOT/base_clips/<clip-name>.mp4`

The names must **match** `SELECTED_26_CLIPS.txt` and the graph; do not rename
files. If your separately uploaded dataset means the *training* dataset, leave
it out of the live deployment.

## Step 2: Download only essential standard model weights

```bash
bash "$HOME/a5000_avatar/kit/bin/03_download_pretrained.sh"
```

Downloads SD-VAE, Whisper Tiny, DWPose, Face Parse and MuseTalk model JSON,
using Hugging Face and the original project's gdown and resnet URLs.
**Does not download the 3.2 GB stock UNet or the 1.4 GB SyncNet**, neither of
which is required by the inspected live lip-sync adapter.

## Step 3: Python environments

Install Python 3.10, ffmpeg, venv, GPU driver and a CUDA-compatible runtime.
Check the driver and OS before choosing wheels:

```bash
nvidia-smi
python3.10 --version
ffmpeg -version
```

The minimal package includes `requirements_musetalk_installed.txt` and
`requirements_analysis_installed.txt`, captured from the old server.
To attempt installing them in isolated target venvs (best-effort, requires
network access; may need MMCV wheel adjustments):

```bash
bash "$HOME/a5000_avatar/kit/bin/04b_install_project_envs.sh"
```

To install a separate Whisper STT environment:

```bash
bash "$HOME/a5000_avatar/kit/bin/04_install_stt_env.sh"
```

The Whisper STT source uses two Hugging Face models:
`ARTPARK-IISc/whisper-large-v3-vaani-hindi` for Hindi/Hinglish and
`openai/whisper-large-v3-turbo` for English. On the first STT server start
it downloads both and loads them sequentially onto GPU 1. This is **not**
the tiny Whisper model used internally by MuseTalk.

## Step 4: Patch portable target paths and configure API key

```bash
bash "$HOME/a5000_avatar/kit/bin/05_prepare_paths.sh"
chmod 600 "$HOME/a5000_avatar/.env"
nano "$HOME/a5000_avatar/.env"
```

Put the key on one line:

```text
SARVAM_API_KEY=your-real-key-here
```

**Do not put the key into the website JavaScript, Google Drive folder or chat.**
`05_prepare_paths.sh` changes only the downloaded target project and backs up
its previous configs. It updates the old gpu20 hardcoded `_env.sh`, FFmpeg
path and MuseTalk GPU index 3 -> 0.

## Step 5: Generate caches from the 26 MP4 files

```bash
bash "$HOME/a5000_avatar/kit/bin/02_doctor.sh" # reports missing items too
bash "$HOME/a5000_avatar/kit/bin/06_rebuild_caches.sh"
```

**Expect about 18 GB of generated cache data** and enough scratch/disk/RAM.
The script processes one clip at a time and verifies all six cache files:
`frames.npy`, `masks.pkl`, `latents.pt`, `boxes.npy`, `info.json`, `DONE`.
To check the preprocessing environment with only one clip first:

```bash
CACHE_LIMIT=1 bash "$HOME/a5000_avatar/kit/bin/06_rebuild_caches.sh"
```

After that works, rerun without `CACHE_LIMIT` to finish the other 25 clips.
The script never runs `preprocess_library.sh`, which would rebuild the
validated motion graph.

## Step 6: Start servers (separate terminals)

Terminal 1 - local STT, physical GPU 1:

```bash
bash "$HOME/a5000_avatar/kit/bin/07_start_local_whisper.sh"
# second terminal: curl http://127.0.0.1:8310/health
```

Terminal 2 - avatar, physical GPU 0:

```bash
bash "$HOME/a5000_avatar/kit/bin/08_start_avatar.sh"
```

First test **without local Whisper** if you want to isolate GPU/env issues:

```bash
STT_MODE=sarvam-stream bash "$HOME/a5000_avatar/kit/bin/08_start_avatar.sh"
```

When fully running, use a browser on an HTTPS site. The project includes
`avatar_system/frontend/index.html` and `app/static/index.html`, but the
precise served route/frontend behavior must be verified on the new host.
`templates/Caddyfile.example` is a *starting point* for HTTPS/WSS, not a
fully secured production reverse proxy. Protect public access and API costs.

## Safety and troubleshooting

- Never run `musetalk_ft/scripts/restart_live.sh` on a shared host. It kills
  the existing process on port 8782.
- The startup in this kit refuses to replace a server already using that port.
- Sarvam key belongs server-side in `$AVATAR_ROOT/../.env` (`chmod 600`).
- Before bringing up production, run `bin/02_doctor.sh --require-cache` and
  check both GPUs with `nvidia-smi`.
- This kit has been **syntax-tested locally**, not executed against your
  private source code or A5000 hardware, neither of which is connected here.
  The next step is to run diagnostics and resolve any environment issues.

## Source model links

- Upstream MuseTalk repository: https://github.com/TMElyralab/MuseTalk
- Official Hugging Face downloader: https://huggingface.co/docs/huggingface_hub/package_reference/cli
