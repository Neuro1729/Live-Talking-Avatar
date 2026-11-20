# Current system audit (Phase A, 2026-10-30)

Root: `Aaryan/avatar/realtime/` (called `R` below). The original prototype is left untouched. The new system lives in
`R/avatar_system/` and imports or reuses prototype code where it already works.

## Assets
| Item | Location | Notes |
|---|---|---|
| Source avatar image | `R/comfy/input/home.png` (= `Aaryan/avatar/portrait_hands.png`) | Man in a blue shirt, hands clasped on a table, white studio. 1024x928 working resolution. |
| Generated clips | `R/base_clips/*.mp4` + `*.json` (generation params) | 36 mp4s. 241 frames @25 fps (9.6 s), 1024x928. LTX-2.5 22B distilled int8 with first+last frame guides = home.png, so **every LTX clip starts and ends on the same home pose**. Families: `bank_*` (first bank: bigger motions; user said "too big"), `sub_*` (subtle bank, still generating), `ltx_*` (early tests), `s2v_*` (Wan S2V, rejected). |
| MuseTalk cache | `R/avatar_cache/<clip>/` | frames.npy (uint8, ~690 MB/clip), boxes.npy, latents.pt (masked+ref, fp16), masks.pkl (face-parsing blend mask + crop box). Made by `R/scripts/prep_avatar.py` at ~65 s/clip. 16 GB at present. |
| Voices | `R/voices/*.wav` | Sarvam bulbul:v3 samples. Default male voice `shubh`. |

## Components already present
| Function | Where | Status |
|---|---|---|
| Clip generation | `R/scripts/ltx_flf.py` (ComfyUI-v037 API, port 8191, GPU 1), `gen_subtle.sh`, `start_comfy.sh` | Works. About 3 min/clip. FLF guides strength 1.0. `--audio silence` freezes the audio latent so the model does not invent speech. Start/end PSNR vs home is about 31-32 dB. |
| Clip QA | `R/scripts/review_clip.py` | Contact sheets plus start/end PSNR. |
| MuseTalk preprocessing | `R/scripts/prep_avatar.py` | DWPose+S3FD bbox (smoothed), VAE latents, face-parsing masks. Same scheme as MuseTalk `scripts/realtime_inference.py` `Avatar.prepare_material`, but it generalizes to many clips. |
| Lip-sync runtime | `R/app/avatar_engine.py::AvatarEngine` | MuseTalk 1.5 UNet+VAE per frame (no temporal state, so a clip change cannot reset it). Whisper features are computed per utterance and indexed by audio frame. 2-stage GPU/CPU pipeline, about 80 fps at 1024 px on an H100. |
| Clip selection | `Director` (mode -> category weights, cuts at near-home frames) and `GraphDirector` (random walk over frame-level jump points from `R/scripts/build_graph.py`) | Works, but there are no pose states, no planner, no validation, no review, and the old graph uses grayscale-diff matching only. |
| LLM + TTS + STT | `R/app/sarvam_client.py` | sarvam-105b-conversations (streamed -> sentences), bulbul:v3 `/text-to-speech/stream` (~2 s per sentence), saarika:v2.5 STT (slow, 12-21 s). Key in `Aaryan/avatar/.env`. |
| Server + browser | `R/app/server.py` (FastAPI WebSocket, 25 fps JPEG + 40 ms PCM per message), `R/app/static/index.html` | Works end to end. Persistent socket. No metrics, no state machine, no cancellation tokens beyond `turn_id`. |
| Demo recorder | `R/scripts/record_demo.py` | Records a scripted conversation to mp4. |

## Hardware / environments (measured)
- gpu28: 4x H100 PCIe 80 GB. GPUs 0 and 2 fail CUDA alloc (known issue). **GPU 1 = LTX generation (ComfyUI, ~42 GB), GPU 3 = lip-sync runtime.**
- Host RAM: 755 GB total, about 620 GB available. Rule: never load two big models at the same moment.
- Disk: `/var/autofs/nfs10g/nest10` is 99% full (about 1 TB free). frames.npy caches are large (690 MB/clip), so a 400-clip library needs a leaner cache (JPEG frames or decode-on-demand). See Phase F.
- ffmpeg: `/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/bin/ffmpeg`.
- `R/envs/musetalk`: py3.10, torch 2.0.1 cu118, numpy 1.23, cv2 4.9, fastapi, librosa. Runs MuseTalk and the server. **Must unset PYTORCH_CUDA_ALLOC_CONF.**
- `R/envs/analysis` (new): py3.11 CPU env with mediapipe 1.2 (face/pose/hand landmarker tasks in `R/models/mediapipe`), opencv, sklearn, pytest. Used for clip analysis, graph building and tests, so graph work never touches the GPU.

## What can be reused vs. restructured
- **Reuse as is:** LTX generation scripts, `prep_avatar.py` caches, MuseTalk model-loading/feature code, `sarvam_client.py`, WebSocket transport idea, browser player.
- **Restructure:** clip selection (-> pose-anchored graph + planner), engine loop (-> explicit timeline with bounded queues, metrics, state machine, cancellation), server (-> session manager, metrics/debug endpoints, dashboard).
- **New:** feature extraction (MediaPipe), transition scoring, graph builder/validator, review tool, SQLite metadata, evaluation suite, tests.

## Key observation for the graph design
All LTX clips are generated from the same home image on both ends, so today the library has one natural pose node (home/neutral).
Other canonical states have to come from the motion inside clips. The analysis pipeline therefore:
1. finds **rest points** (low-velocity frames) inside each clip,
2. clusters rest-point poses into canonical states,
3. splits clips at rest points into **segments**, which become the graph edges (state -> state).

New state-changing clips can then be generated with FLF using a *real frame* of the target state as the end guide.
This keeps identity and lighting exactly consistent.
