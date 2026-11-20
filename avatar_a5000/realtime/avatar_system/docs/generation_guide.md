# Generating clips for the graph

Generation is offline only. The live system never loads a video model.

## Backend
ComfyUI-v037 with LTX-2.5 22B distilled int8, using first+last frame (FLF) guides: `R/scripts/start_comfy.sh` (GPU 1,
port 8191, `--disable-pinned-memory`). Before starting it, check GPU and RAM: never load two big models at the same
moment on gpu28. One 241-frame clip at 1024x928 takes about 3 min; 97 frames take about 1.5 min.
`R/scripts/ltx_flf.py` options:
- `--image` start guide (also the end guide by default), `--last-image` a different end guide (state change)
- `--audio silence_241f25.wav` freezes a silent audio latent so the model does not invent speech (mouth stays calm)
- `--mid-guides "80:0.35,160:0.35"` weak extra guides (frame index divisible by 8)

## Workflow
1. `bash avatar_system/scripts/build_motion_graph.sh`, then read the validator warnings (missing states, few alternatives,
   modes that cannot loop).
2. `envs/analysis/bin/python -m avatar_system.generation.generation anchors` exports one **real library frame** per state
   (the member closest to the centroid) to `comfy/input/anchor_<state>.png`. Using real frames keeps identity, light
   and framing exact.
3. Add entries to `configs/generation_manifest.yaml`: `{name, prompt_file, seed, start, end, frames}`. Start/end are
   anchor names; `home` is the avatar image. Write prompts in `R/prompts/` in the style of `subtle_*.txt`: locked-off
   camera, hands clasped, "very subtle natural motion", one small action.
4. `... -m avatar_system.generation.generation run` generates every entry that does not exist yet and **measures** each
   result: pose distance and PSNR of the first and last frame against the anchors (`data/generation/<clip>_validation.json`).
   FLF guidance does not guarantee the anchor is reached.
5. Add the name pattern to `configs/avatar.yaml` `library.include` if needed, then run `preprocess_library.sh`. The quality
   gate (`motion_graph.yaml: quality_gate`) excludes segments with head range > 14°, eyes closed > 25% or gaze-up > 20%.
   The graph builder only connects what physically matches.
6. Review: `review_transitions.sh`. Use "approve" / "reject" / cost override per join, and "low quality" / labels per
   segment. Decisions live in `data/reviews.json` and survive rebuilds (keyed by cut frames).

External clips: `... generation import /path/clip.mp4 --name x --start home --end home` (25 fps, 1024x928, same camera).

## Lessons so far (2026-10-30)
- Neutral-to-neutral subtle clips work well (the 21-clip `sub_*` bank): calm, varied, all joins invisible.
- The first bank (`bank_*`) is too energetic; most of it is gated out.
- home->tilt and tilt->home FLF clips (8 clips, 241 and 97 frames) all reached both anchors (pose distance 0.12-0.67)
  but overshoot the roll by 23-36° mid-clip; only `gen_tilt2home_v2_s62` was clean. Next try: a mid guide at a
  half-tilt frame, or lower first/last strength.
- `bank_*` and `ltx_*` MuseTalk prep: 65 s/clip on GPU 3. frames.npy is about 690 MB/clip, so watch disk space.
