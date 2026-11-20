# Architecture

```
offline (CPU unless noted)                                   runtime (one process, GPU 3)
──────────────────────────                                   ─────────────────────────────────────────────────────
base_clips/*.mp4 ──► feature_extractor (MediaPipe face/pose/   Browser ◄── WebSocket: [n, PCM 40 ms, meta, JPEG] ──┐
   │                 hands, ArcFace, thumbnails) → data/features │                                                   │
   ▼                                                            │ ask / interrupt / user_speaking                   │
clip_analyzer (smoothing, speed, blinks, nods, gaze-up,         ▼                                                   │
   rest points, settle trim)                                ConversationEngine ── CancelToken per turn             │
   ▼                                                            │  SarvamLLM (sentences + hidden behaviour tag)      │
graph_builder                                                   │  TTS provider (Sarvam / stream / mock / wav)       │
   states  = home pose (hub) + clusters of rest poses           ▼                                                   │
   edges   = clip segments cut at rest points (1.5-5 s)     AudioTimeline (samples = master clock, utterances,      │
   quality gate (head range, eyes closed, gaze up)            chunk continuity, cancellation, feature shift)       │
   transitions scored by transition_scorer                      │                                                   │
   reviews.json overrides ──► motion_graph.json + library.sqlite│                                                   │
   ▼                                                            ▼                                                   │
graph_validator (dead ends, hub return, SCC, modes)        plan thread: StateMachine ─► MotionScheduler/Planner ──┐ │
   → runtime_segments                                         (beam search, repetition, early exits, seams)      │ │
transition_review (sheets, videos, web UI)                      │ FrameRef n + PCM n + feature n                  │ │
prep_avatar.py (GPU, MuseTalk caches per clip)                  ▼                                                  │ │
generation (anchors, manifest, LTX FLF, endpoint check)    render thread: MuseTalk adapter (UNet+VAE batch,       │ │
                                                              blend, seam crossfade, silence ramp)               │ │
                                                                ▼                                                  │ │
                                                           encode thread: resize + JPEG ─► out_q ─► frame pump ───┘─┘
                                                                                             (paced 25 fps, recorder)
```

## Key decisions

**Pose-anchored graph from a single-image library.** Every LTX clip starts and ends on the home image, so "neutral" is the
natural hub. Other states are found by clustering *rest points* (local speed minima) that sit far from neutral in a
normalised pose space: head angles, face position and scale, gaze, smile, brows, wrists and shoulders. Clips are cut at
rest points into segments, and segments are the graph edges. A state is a physical pose, not an emotion label. A
transition is only allowed after it has been measured.

**Transition cost = measured continuity.** Seven components are each normalised so that a natural consecutive-frame
step costs about 0 to 1: pose with velocity extrapolation, motion (velocity difference over a 3-frame window),
expression, gaze, hands (hand-region pixels, because pose wrists jitter about 0.04 when the hands are clasped),
non-mouth landmarks, and pixels with the mouth masked. Hard rejects cover face jumps, identity, luma, background and
hand region. Thresholds were calibrated against natural frame-step percentiles and then tightened after a visual
check: joins at cost 1.3 showed a visible smile drop. The accept threshold is 0.9, plus per-component caps.

**Planner.** Beam search 3 segments deep. The score combines continuity, semantic fit to the behaviour state and hints,
prosody (gestures on emphasis peaks, calm during pauses), intensity target, quality, repetition (segment, clip, 2- and
3-grams, nod/gesture/tilt spacing) and distance back to the hub. The first step is sampled with temperature 0.35 among
the top 5. Any join that is clearly worse than the best is excluded, with a floor of 0.45 so that the near-free
"natural continuation" does not always win; without the floor, whole clips replayed as blocks (max 3-gram count 12,
down to 1-3 with the floor). When the state turns calm while an expressive segment is playing, the scheduler leaves
early at an approved inside rest point.

**One timeline.** Frame n carries audio samples [n·640, (n+1)·640). Speech is placed on the timeline after everything
already planned, so audio never jumps. Lip-sync features are computed over the whole utterance-so-far and become
final only when there is right context. Cancellation truncates or drops utterances that have not yet played, and
stale chunks are ignored.

**MuseTalk integration.**
- *Per-clip caches:* boxes, latents and masks are cached per clip, the same idea as MuseTalk's `Avatar.prepare_material`.
- *No resets:* MuseTalk is frame-wise, so changing clips during speech cannot reset the mouth (measured: mouth change
  at seams 0.011 vs 0.020 elsewhere).
- *Silence:* MuseTalk's "silence" output keeps the lips parted (measured mouth opening 0.109 vs 0.004 raw), so silent
  frames use the raw source frame, crossfaded over 4 frames at speech boundaries.
- *Sync offset:* SyncNet showed the mouth leading the audio by 2 frames, so frame n is driven by the features of
  frame n−2. Offset is now 0, confidence about 7.

**Transport.** WebSocket, one message per frame. WebRTC's UDP media path does not work through the SSH TCP tunnel used
to reach gpu28 without a TURN server. A WebRTC transport would read `pipeline.read()` like the WebSocket pump does.

## Files
| area | file |
|---|---|
| config | `configs/avatar.yaml` (paths, library), `motion_graph.yaml`, `motion_planner.yaml`, `musetalk.yaml`, `generation_manifest.yaml` |
| motion | `motion/feature_extractor.py`, `clip_analyzer.py`, `transition_scorer.py`, `graph_builder.py`, `graph_validator.py`, `transition_review.py`, `motion_planner.py`, `repetition_tracker.py`, `motion_scheduler.py` |
| lip-sync | `lipsync/base.py` (interface + mock), `musetalk_adapter.py`, `audio_context.py` |
| conversation | `conversation/state_machine.py`, `behavior_planner.py` (prosody, hints), `llm_adapter.py`, `tts_interface.py`, `engine.py` |
| streaming | `streaming/frame_source.py`, `pipeline.py`, `server.py`, `frontend/index.html` |
| generation | `generation/generation.py` |
| evaluation | `evaluation/transition_metrics.py`, `repetition_metrics.py`, `lipsync_metrics.py`, `benchmark.py`; SyncNet via LatentSync `eval/eval_sync_conf.py` |
| data (derived) | `data/features/*.npz`, `motion_graph.json`, `transitions.json`, `validation.json`, `library.sqlite`, `reviews.json` (manual, persistent), `anchors.json`, `review/`, `demos/`, `recordings/`, `eval/` |
