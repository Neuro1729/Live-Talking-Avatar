"""Talking bank: scripted Sarvam speech (with designed pauses) -> LTX-2.5 audio-to-video clips (start+end = home).

  audio : build data/talk_bank/<clip>.wav (+ comfy/input copy) and <clip>_parts.json (exact speech/pause intervals,
          text, act, emotion) from configs/talk_scripts.yaml. Length is padded to an LTX frame count (8n+1 @ 25 fps).
  run   : generate every clip with R/scripts/ltx_flf.py --audio (uses the already-running ComfyUI, GPU 1).

  envs/analysis/bin/python -m avatar_system.generation.talk_bank audio [--only name ...]
  envs/analysis/bin/python -m avatar_system.generation.talk_bank run   [--only name ...]
"""
import argparse
import asyncio
import json
import math
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import yaml

from avatar_system.common import CONFIGS, clip_path, data_dir, root

SR, FPS = 16000, 25
BASE = ("A static, locked-off medium shot of a friendly South Asian man in his thirties with short messy dark hair and "
        "light stubble, wearing a light blue button-down shirt with rolled sleeves, sitting at a plain white table "
        "against a clean white studio background, soft even studio lighting. He looks into the camera and speaks to "
        "the viewer naturally, in sync with his voice, keeping his lip and jaw movements small and relaxed, never "
        "opening his mouth wide. When he is silent he keeps his mouth gently closed and listens attentively. ")
STEADY = (" His head stays steady, upright and facing the camera the whole time; it moves only very slightly, with "
          "tiny natural nods, never tilting to the side or turning; his shoulders stay still.")
TAIL = (" He blinks only occasionally. Realistic skin texture, photorealistic, sharp focus, natural real-time motion, "
        "the camera does not move.")


def load(name="talk_scripts"):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


async def build_audio(clip):
    sys.path.insert(0, str(root() / "app"))
    from sarvam_client import Sarvam
    s = Sarvam()
    pcm, parts, t = [], [], 0.0
    for p in clip["parts"]:
        if "pause" in p:
            x = np.zeros(int(p["pause"] * SR), np.float32)
            parts.append({"type": "pause", "start": round(t, 3), "end": round(t + p["pause"], 3)})
        else:
            for attempt in range(8):  # Sarvam rate-limits (429) bursts of TTS calls: back off and retry
                try:
                    x = await s.tts(p["say"], speaker="shubh", lang=p.get("lang", "hi-IN"))
                    break
                except RuntimeError as e:
                    if "429" not in str(e) or attempt == 7:
                        raise
                    print(f"  rate limited, retry in {15 * 2 ** min(attempt, 3)} s", flush=True)
                    await asyncio.sleep(15 * 2 ** min(attempt, 3))
            parts.append({"type": "speech", "start": round(t, 3), "end": round(t + len(x) / SR, 3), "text": p["say"],
                          "lang": p.get("lang", "hi-IN")})
        pcm.append(x)
        t += len(x) / SR
    a = np.concatenate(pcm)
    frames = math.ceil(len(a) / SR * FPS / 8) * 8 + 1          # LTX needs 8n+1 frames
    a = np.pad(a, (0, int(frames / FPS * SR) - len(a)))          # trailing silence up to the clip length
    out = data_dir("talk_bank")
    sf.write(out / f"{clip['name']}.wav", a, SR)
    shutil.copy(out / f"{clip['name']}.wav", root() / "comfy/input" / f"{clip['name']}.wav")
    meta = {"name": clip["name"], "act": clip["act"], "emotion": clip["emotion"], "behavior": clip["behavior"],
            "frames": frames, "seconds": round(frames / FPS, 3), "parts": parts}
    (out / f"{clip['name']}_parts.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def prompt_for(clip, cfg, steady=False):
    return BASE + cfg["behaviors"][clip["behavior"]] + (STEADY if steady else "") + TAIL


def mid_guides(frames, every=72, strength=0.25):
    """Faint home-pose guides every ~3 s (frame idx multiple of 8, away from both ends) to bound head drift."""
    idx = list(range(every, frames - 40, every))
    return ",".join(f"{i - i % 8}:{strength}" for i in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audio", "run"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scripts", default="talk_scripts", help="config name: talk_scripts (pilot) or talk_scripts_full")
    ap.add_argument("--steady", action="store_true", help="add the steady-head prompt")
    ap.add_argument("--mid", type=float, default=0.0, help="strength of faint home-pose mid guides (0 = none)")
    ap.add_argument("--mid-every", type=int, default=72, help="frames between mid guides")
    ap.add_argument("--suffix", default="", help="name suffix for variants, e.g. _st")
    a = ap.parse_args()
    cfg = load(a.scripts)
    clips = [c for c in cfg["clips"] if not a.only or c["name"] in a.only]
    # round-robin over speech acts (take 1 of every act first) so a usable bank exists early
    clips.sort(key=lambda c: (int(c["name"].rsplit("_", 1)[1]) if c["name"].rsplit("_", 1)[1].isdigit() else 0))
    if a.cmd == "audio":
        for c in clips:
            done = data_dir("talk_bank") / f"{c['name']}_parts.json"
            if done.exists() and (root() / "comfy/input" / f"{c['name']}.wav").exists():
                print(f"skip {c['name']} (audio exists)")
                continue
            m = asyncio.run(build_audio(c))
            sp = sum(p["end"] - p["start"] for p in m["parts"] if p["type"] == "speech")
            print(f"{m['name']}: {m['seconds']}s ({m['frames']} frames), speech {sp:.1f}s, {len(m['parts'])} parts")
        return
    for c in clips:
        meta = json.loads((data_dir("talk_bank") / f"{c['name']}_parts.json").read_text())
        name = c["name"] + a.suffix
        if clip_path(name).exists():
            print(f"skip {name}")
            continue
        pf = data_dir("talk_bank") / f"{name}_prompt.txt"
        pf.write_text(prompt_for(c, cfg, a.steady))
        cmd = ["python3", "-u", "ltx_flf.py", "--name", name, "--prompt-file", str(pf), "--seed", str(a.seed),
               "--audio", f"{c['name']}.wav", "--frames", str(meta["frames"])]
        if a.mid > 0:
            cmd += ["--mid-guides", mid_guides(meta["frames"], every=a.mid_every, strength=a.mid)]
        subprocess.run(cmd, cwd=root() / "scripts", check=True)
        if a.suffix:  # variants share the original's audio/parts metadata
            shutil.copy(data_dir("talk_bank") / f"{c['name']}_parts.json", data_dir("talk_bank") / f"{name}_parts.json")
            shutil.copy(data_dir("talk_bank") / f"{c['name']}.wav", data_dir("talk_bank") / f"{name}.wav")


if __name__ == "__main__":
    main()
