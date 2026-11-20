# Benchmark report

Generated 2026-10-30 17:30 by `evaluation/benchmark.py` (65s). Graph `7f1cea162602`. 3 seeds × 10 min simulated conversations (listening / thinking / speaking / interruptions) per planner, same library and graph.

## Graph
- clips in library: 38 usable (+ 46 segments excluded by the quality gate / review)
- states: neutral (102 rest frames), tilt (6 rest frames), tilt_2 (4 rest frames), chin_up (6 rest frames), tilt_3 (7 rest frames)
- runtime segments (edges between poses): 56, approved transitions between them: 920 (avg out-degree 17.0); transition counts {'natural': 30, 'approved': 1086, 'review': 2042, 'rejected': 928}
- dead ends / unreachable / no path to hub (pruned from runtime): 5 / 5 / 2; worst time back to neutral from a speaking segment: 8.0s
- per-mode cycle sets: idle: 18 segments / 16 clips / 49.5s; listen: 18 segments / 16 clips / 49.5s; think: 21 segments / 18 clips / 59.5s; speak: 45 segments / 24 clips / 166.5s
- validator warnings: ['state chin_up: only 1 outgoing segments', 'state tilt: no runtime-safe segment touches it', 'state tilt_2: no runtime-safe segment touches it', 'state tilt_3: no runtime-safe segment touches it']

## Planner vs random-walk baseline (mean ± sd over seeds)
| metric | beam planner | random walk |
|---|---|---|
| mode fit (% frames whose segment suits the state) | 96.37 ± 0.09 | 68.13 ± 2.72 |
| listening frames on listen segments (%) | 94.8 | 57.2 |
| unique segments / 10 min | 41.00 ± 0.82 | 46.67 ± 0.94 |
| segment entropy (bits) | 4.87 ± 0.02 | 5.22 ± 0.02 |
| max repeats of one 3-segment sequence | 1.00 ± 0.00 | 2.00 ± 0.00 |
| repeated 3-grams | 0.00 ± 0.00 | 3.67 ± 0.47 |
| nods / min | 1.33 | 3.50 |
| gestures / min | 0.33 | 0.27 |
| tilt-type motions / min | 0.13 | 0.90 |
| blinks / min | 31.17 ± 0.33 | 39.03 ± 0.66 |
| blink interval CV | 0.66 ± 0.01 | 0.66 ± 0.04 |
| segment changes / 10 min | 210.33 ± 2.05 | 211.67 ± 2.49 |
| graph cuts (non-natural) | 196.33 ± 1.89 | 196.67 ± 2.49 |
| seam frames above natural p99 (%) | 1.07 ± 0.19 | 2.57 ± 0.74 |
| max seam visual step (natural p99) | 1.92 ± 0.00 | 1.91 ± 0.05 |
| early exits | 5.67 ± 1.25 | 25.67 ± 0.94 |

Natural (inside-clip) visual step p99 = 0.97 gray levels; a seam frame 'above p99' moves more than 99% of genuine consecutive frames.

## Identity / background over the beam run
{'identity_cos_min': 0.983, 'identity_cos_mean': 0.993, 'bg_luma_range': 0.0006, 'luma_range': 0.0035}

## Worst seam frames (beam, seed 0)
| frame | state | excess over natural p99 | visual | landmark px |
|---|---|---|---|---|
| 6856 | SPEAKING | 1.93 | 1.921 | 1.879 |
| 9693 | LISTENING | 1.93 | 1.921 | 1.879 |
| 10972 | LISTENING | 1.892 | 1.883 | 2.379 |
| 5423 | LISTENING | 1.85 | 1.842 | 1.644 |
| 10744 | LISTENING | 1.842 | 1.834 | 3.369 |
| 7338 | SPEAKING | 0.926 | 0.182 | 0.246 |
| 9031 | LISTENING | 0.926 | 0.182 | 0.246 |
| 11684 | SPEAKING | 0.909 | 0.345 | 0.686 |

## Lip-sync accuracy: SyncNet (LatentSync eval/eval_sync_conf.py (syncnet_v2.model), speech-only excerpts)
| clip | confidence | AV offset (frames) |
|---|---|---|
| prerecorded demo, Hindi (7.2-16.6 s), no offset correction | 7.34 | -2 |
| prerecorded demo, Hinglish+English+fast Hindi (29.3-37 s), no offset correction | 6.95 | -2 |
| same, feature shift +2 (wrong direction) | 7.29 | -4 |
| prerecorded demo Hindi, feature shift -2 (final) | 7.15 | 0 |
| prerecorded demo Hinglish/English/fast, feature shift -2 (final) | 6.82 | 0 |
| LIVE Sarvam conversation recording, answer 1 (26.8-33.0 s) | 7.46 | 0 |
| LIVE Sarvam conversation recording, answer 2 (41.2-48.8 s) | 7.76 | 0 |
| control: no lip-sync (source frames), Hindi | 0.12 | -6 |
| control: no lip-sync (source frames), Hinglish/English | 0.41 | -15 |

## Lip-sync behaviour measured on the output video (MediaPipe)
Mouth opening = inner-lip gap / inter-ocular distance. `av_offset_*` / `mouth_audio_corr` come from a crude energy-envelope correlation proxy and are NOT a sync measure (SyncNet above is); kept for transparency.
| metric | value |
|---|---|
| frames | 1550 |
| speech_frames | 459 |
| voiced_frames | 353 |
| av_offset_frames | 6 |
| av_offset_ms | 240 |
| mouth_audio_corr | 0.155 |
| mouth_open_speech_mean | 0.1204 |
| mouth_open_silence_mean | 0.0085 |
| mouth_motion_speech | 0.0196 |
| mouth_motion_silence | 0.001 |
| mouth_jump_at_seams_during_speech | 0.011 |
| mouth_jump_elsewhere_during_speech | 0.0197 |
| seams_during_speech | 3 |
| backend | musetalk1.5 |

Throughput of this run (silence ramp: silent frames skip MuseTalk, so capacity is inflated): {'render_ms': {'mean': 37.98, 'p50': 0.37, 'p95': 125.09, 'max': 268.97}, 'render_ms_per_frame': {'mean': 4.75, 'p50': 0.05, 'p95': 15.64, 'max': 33.62}, 'encode_ms': {'mean': 1.08, 'p50': 0.92, 'p95': 1.78, 'max': 11.92}, 'render_fps_capacity': 210.5, 'unpaced_fps': 164.9}. With every frame lip-synced (silence=always) the measured capacity is ~72 fps (13.9 ms/frame, p95 17.6 ms, batch 8, one H100) = 2.9x real time.

## Live conversation (Sarvam LLM + TTS bulbul:v3 shubh, MuseTalk, 12 turns)
| stage | p50 s | p95 s |
|---|---|---|
| llm_first_sentence_s | 1.34 | 1.78 |
| tts_first_chunk_s | 1.58 | 3.33 |
| first_tts_done_s | 2.9 | 4.74 |
| pipeline_after_tts_s | 0.85 | 0.91 |
| first_audible_s | 3.78 | 5.58 |

first_audible = server emits the first speech frame (browser adds its 0.25 s jitter buffer). pipeline_after_tts = queued frames ahead of the new speech (lead + plan + render + playback buffer). Output frame interval {'mean': 43.11, 'p50': 39.83, 'p95': 41.21, 'max': 855.73}; counters {'frames_planned': 4549, 'frames_rendered': 4544, 'frames_out': 4529, 'output_underruns': 0, 'audio_feature_late': 0, 'speech_frames': 2152, 'pacing_resets': 1}; GPU peak 6.54 GB, process RSS peak 8.45 GB; behaviour sources ['llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag', 'llm_tag']. Recording: /var/autofs/nfs10g/nest10/usrs/vishal/Aaryan/avatar/realtime/avatar_system/data/recordings/rec_20261030_172115.mp4
