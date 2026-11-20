# Motion-graph talking avatar (MuseTalk 1.5 + Sarvam)

A photorealistic avatar that never freezes and never jump-cuts. A library of generated clips of one person is analysed
and turned into a pose-anchored, validated motion graph. A lightweight planner walks that graph continuously according
to the conversation state (listening / thinking / speaking / interrupted / recovery) and the speech. MuseTalk 1.5
lip-syncs the selected frames to Sarvam TTS audio on one continuous audio-video timeline, which is streamed to a browser.

Docs: [current system audit](docs/current_system.md) · [architecture](docs/architecture.md) ·
[generation guide](docs/generation_guide.md) · [benchmark report](docs/benchmark_report.md)

All paths below are relative to `Aaryan/avatar/realtime/` (R). Machine: gpu28. GPUs 0/2 are broken; use 3 (or 1).

## Run everything
```bash
cd /var/autofs/nfs10g/nest10/usrs/vishal/Aaryan/avatar/realtime
# 1. supply assets: comfy/input/home.png (avatar), base_clips/*.mp4 (clips, 25 fps); library selection in configs/avatar.yaml
# 2. preprocess: features (CPU), graph + validation, MuseTalk caches for runtime clips (GPU, ~65 s/clip)
bash avatar_system/scripts/preprocess_library.sh
# 3. rebuild/validate the graph after edits or reviews (CPU, seconds)
bash avatar_system/scripts/build_motion_graph.sh
# 4. review joins: renders evidence, then web UI on :8770 (ssh -L 8770:127.0.0.1:8770 vishal@10.12.0.213)
bash avatar_system/scripts/review_transitions.sh
# 5. silent continuous-motion demo (CPU): mp4 + debug overlay mp4 + trace
bash avatar_system/scripts/run_silent_demo.sh --minutes 10 --out avatar_system/data/demos/silent_10min
# 6. MuseTalk lip-sync demo with prerecorded Sarvam speech (GPU; --mock = timeline only, no lip-sync)
GPU=3 bash avatar_system/scripts/run_lipsync_demo.sh --out avatar_system/data/demos/lipsync_musetalk
# 7. conversational avatar on :8780 (ssh -L 8780:127.0.0.1:8780 ..., open http://localhost:8780)
GPU=3 bash avatar_system/scripts/run_conversation.sh            # Sarvam LLM + TTS (key in Aaryan/avatar/.env)
bash avatar_system/scripts/run_conversation.sh --mock --tts mock --llm mock   # no GPU / no key (mock proves nothing about lip-sync)
#    scripted live test (12 turns + barge-in, recorded, latency p50/p95, captioned copy *_captioned.mp4):
envs/analysis/bin/python avatar_system/scripts/run_conversation_test.py --port 8780 --repeat 2
#    caption any recording made with the dashboard's Record button (your question + time, "thinking +x s",
#    "AVATAR responded after x s" + answer text, interruptions):
envs/analysis/bin/python avatar_system/scripts/caption_recording.py avatar_system/data/recordings/rec_YYYYMMDD_HHMMSS
# 8. tests + evaluation report (docs/benchmark_report.md)
bash avatar_system/scripts/run_benchmark.sh
```
Scripts check GPU free memory and host RAM, MuseTalk weights, the Sarvam key, the avatar image and the clips, and stop
with a clear error when something is missing. Environments: `envs/analysis` (CPU: MediaPipe, graph, tests, eval) and
`envs/musetalk` (GPU: MuseTalk, server). SyncNet scoring uses LatentSync's `eval/eval_sync_conf.py` in the ComfyUI venv
(see the "syncnet" commands in docs/benchmark_report.md).

## Status (2026-10-30): what works, what doesn't
| | status | evidence |
|---|---|---|
| Graph from 51 analysed clips: 5 states, 68 segments (+46 gated out), 56 runtime segments, 920 approved runtime joins, validated | works | `data/validation.json` |
| Silent continuous playback, 10 min, no visible cuts | works | `data/demos/silent_10min.mp4`; 1.1% of seam frames move more than the natural p99 step, worst 1.9 vs 0.97 gray levels |
| Planner vs random walk | works | mode fit 96% vs 68%, listening on calm clips 95% vs 57%, 0 repeated 3-segment sequences vs 3.7, nods 1.3/min vs 3.5 |
| MuseTalk lip-sync on the graph, continuous across clip changes | works | SyncNet 6.8-7.8 conf, AV offset 0 (control without lip-sync 0.1-0.4); no mouth jump at seams |
| Silence: natural closed mouth, no pop at speech start/end | works | mouth opening 0.009 silent vs 0.12 speaking |
| Live conversation (Sarvam LLM + TTS + MuseTalk + WebSocket) | works | 12 turns, first audible p50 3.78 s / p95 5.58 s (Sarvam LLM+TTS ≈ 2.9 s of it), 0 underruns, 25 fps |
| LLM behaviour metadata (hidden tag) -> planner hints | works | 12/12 turns tagged; empathetic/positive/thoughtful match the content |
| Barge-in / cancellation | works | in-flight sentence cancelled; stale chunks dropped (unit tests) |
| Throughput | ≥25 fps with headroom | ~72 fps with every frame lip-synced (2.9x real time, 1 H100); GPU peak 6.5 GB, RSS 8.5 GB |
| Non-neutral pose states (tilt, chin-up) in runtime | **not yet** | LTX FLF clips to the tilt anchor reached both endpoints but swung the head 23-36° on the way and were rejected by the quality gate (8 clips, 2 rounds) |
| WebRTC transport | **not used** | WebSocket instead: the SSH TCP tunnel cannot carry WebRTC's UDP media without a TURN server |
| 250-400 clip library (Phase F) | **not started** | needs a leaner frame cache first (frames.npy ≈ 690 MB/clip; nest10 is 99% full) |

## Recommended next changes
1. **Pose states**: generate state changes with a weak mid-clip guide at a half-tilt pose (`ltx_flf.py --mid-guides`),
   or I2V from the anchor with a short clip. The validator already lists the missing states.
2. **Latency**: speech currently starts after ~20 already-planned frames (0.85 s). Inserting speech into planned but
   unrendered frames (re-render the plan queue) would cut about 0.4 s. Sarvam LLM+TTS (2.9 s p50) dominates: stream TTS
   (`--tts sarvam-stream`, implemented but not benchmarked) and start TTS on the first clause.
3. **Cache size**: store frames as JPEG/H.264 and decode on demand (or only face crops plus a background clip) before
   growing the library.
4. Raise `seam_blend_max` (1.2 -> 2.0) so the few worst joins (visual 1.8-1.9) are also crossfaded.
5. Local Whisper STT in place of saarika (12-21 s) for real voice input.
