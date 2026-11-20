"""Offline clip generation for graph gaps: anchors, manifest, generator adapter, endpoint validation.

Anchors are REAL frames of the library (a state's member closest to its centroid), exported as images, so a new
state-changing clip starts/ends on exactly the identity, lighting and framing the graph already has.
The manifest (configs/generation_manifest.yaml) lists clips as {name, prompt_file, seed, start, end} with start/end
= anchor names ("home" = the avatar image). The generator adapter runs R/scripts/ltx_flf.py (ComfyUI LTX-2.5 FLF, must
already be running - this module never loads a video model itself). External clips can be imported instead.
The validator measures whether a clip really reached its anchors (pose distance + PSNR), because FLF guidance
does not guarantee it, and then the normal pipeline (features -> graph -> review) decides how the clip is used.

  envs/analysis/bin/python -m avatar_system.generation.generation anchors
  envs/analysis/bin/python -m avatar_system.generation.generation run [--only name ...]
  envs/analysis/bin/python -m avatar_system.generation.generation validate name ...
  envs/analysis/bin/python -m avatar_system.generation.generation import /path/clip.mp4 --name x --start home --end tilt
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
import yaml

from avatar_system.common import CONFIGS, clip_path, data_dir, path, read_frames, root
from avatar_system.motion.clip_analyzer import analyze, pose_dist

COMFY_PORT = 8191


def anchors_file():
    return data_dir() / "anchors.json"


def export_anchors():
    """One anchor per graph state: the member rest frame closest to the state centroid -> comfy/input/anchor_<s>.png."""
    g = json.loads((data_dir() / "motion_graph.json").read_text())
    from avatar_system.motion.graph_builder import discover_states
    from avatar_system.common import library_clips
    from avatar_system.motion.feature_extractor import is_current
    ans = {n: analyze(n) for n in library_clips() if is_current(n)}
    states = discover_states(ans)
    out = {"home": {"image": "home.png", "pose": None, "source": "avatar image"}}
    for name, st in states.items():
        if name == "neutral":
            continue
        best = min(st["members"], key=lambda m: pose_dist(ans[m.split("@")[0]]["pose"][int(m.split("@")[1])], st["centroid"]))
        clip, fr = best.split("@")
        img = read_frames(clip_path(clip))[int(fr)]
        fn = f"anchor_{name}.png"
        cv2.imwrite(str(root() / "comfy/input" / fn), img)
        out[name] = {"image": fn, "source": best, "pose": ans[clip]["pose"][int(fr)].tolist()}
    anchors_file().write_text(json.dumps(out, indent=1))
    print(f"anchors: {list(out)} (graph {g['version']})")
    return out


class VideoGenerator:
    def generate(self, item):
        raise NotImplementedError


class LTXComfyGenerator(VideoGenerator):
    """Uses the already-running ComfyUI-v037 (scripts/start_comfy.sh). Refuses to start if it is not reachable."""

    def __init__(self, port=COMFY_PORT):
        self.port = port
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=5).read()
        except Exception as e:
            raise SystemExit(f"ComfyUI not reachable on port {port} ({e}); start it with scripts/start_comfy.sh "
                             "after checking GPU/RAM availability")

    def generate(self, item, anchors):
        a0, a1 = anchors[item["start"]]["image"], anchors[item["end"]]["image"]
        cmd = [sys.executable if False else "python3", "-u", "ltx_flf.py", "--name", item["name"], "--prompt-file",
               str(root() / "prompts" / item["prompt_file"]), "--seed", str(item["seed"]), "--image", a0,
               "--audio", "silence_241f25.wav", "--frames", str(item.get("frames", 241)), "--port", str(self.port)]
        if a1 != a0:
            cmd += ["--last-image", a1]
        subprocess.run(cmd, cwd=root() / "scripts", check=True)
        return clip_path(item["name"])


def import_clip(src, name, start, end):
    dst = clip_path(name)
    shutil.copy(src, dst)
    dst.with_suffix(".json").write_text(json.dumps({"name": name, "imported_from": str(src), "start_anchor": start,
                                                    "end_anchor": end}, indent=1))
    return dst


def validate_clip(name, start, end, anchors=None):
    """Measure endpoint reach. Needs features (runs the extractor for this clip)."""
    from avatar_system.motion.feature_extractor import extract, is_current
    if not is_current(name):
        extract(name)
    anchors = anchors or json.loads(anchors_file().read_text())
    an = analyze(name)
    fr = read_frames(clip_path(name))
    home = json.loads((data_dir() / "motion_graph.json").read_text())["states"]["neutral"]["centroid"]

    def target(key):
        return np.array(home) if key == "home" else np.array(anchors[key]["pose"])

    def img(key):
        im = cv2.imread(str(root() / "comfy/input" / anchors[key]["image"]))
        return cv2.resize(im, fr.shape[2:0:-1], interpolation=cv2.INTER_AREA) if key == "home" else im

    psnr = lambda a, b: float(cv2.PSNR(a, b))
    s, e = an["start"], an["end"] - 1
    rep = {"clip": name, "start_anchor": start, "end_anchor": end,
           "start_pose_dist": round(float(pose_dist(an["pose"][s], target(start))), 2),
           "end_pose_dist": round(float(pose_dist(an["pose"][e], target(end))), 2),
           "start_psnr": round(psnr(fr[0], img(start)), 1), "end_psnr": round(psnr(fr[-1], img(end)), 1)}
    from avatar_system.common import load_config
    thr = load_config("motion_graph")["states"]["assign_max"]
    rep["reached_start"] = rep["start_pose_dist"] <= thr
    rep["reached_end"] = rep["end_pose_dist"] <= thr
    out = data_dir("generation") / f"{name}_validation.json"
    out.write_text(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("anchors")
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="*")
    v = sub.add_parser("validate")
    v.add_argument("names", nargs="+")
    i = sub.add_parser("import")
    i.add_argument("src")
    i.add_argument("--name", required=True)
    i.add_argument("--start", default="home")
    i.add_argument("--end", default="home")
    a = ap.parse_args()
    man = yaml.safe_load((CONFIGS / "generation_manifest.yaml").read_text())
    if a.cmd == "anchors":
        export_anchors()
    elif a.cmd == "run":
        anchors = json.loads(anchors_file().read_text())
        gen = LTXComfyGenerator()
        for item in man["clips"]:
            if a.only and item["name"] not in a.only:
                continue
            if clip_path(item["name"]).exists():
                print(f"skip {item['name']} (exists)")
                continue
            gen.generate(item, anchors)
            print(json.dumps(validate_clip(item["name"], item["start"], item["end"], anchors)), flush=True)
    elif a.cmd == "validate":
        by = {c["name"]: c for c in man["clips"]}
        for n in a.names:
            c = by.get(n, {"start": "home", "end": "home"})
            print(json.dumps(validate_clip(n, c["start"], c["end"])))
    elif a.cmd == "import":
        import_clip(a.src, a.name, a.start, a.end)
        print(json.dumps(validate_clip(a.name, a.start, a.end)))


if __name__ == "__main__":
    main()
