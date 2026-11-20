"""Shared config loading, paths and small video helpers (no heavy imports)."""
import fnmatch
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
CONFIGS = PKG / "configs"


@lru_cache(None)
def load_config(name):
    """configs/<name>.yaml; env AVATAR_CFG_<NAME>=<other> swaps in configs/<other>.yaml (e.g. a fine-tuned MuseTalk)."""
    name = os.environ.get(f"AVATAR_CFG_{name.upper()}", name)
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def root():
    return (PKG / load_config("avatar")["paths"]["root"]).resolve()


def path(key):
    p = Path(load_config("avatar")["paths"][key])
    return p if p.is_absolute() else root() / p


def data_dir(*sub):
    d = path("data_dir").joinpath(*sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def library_clips():
    """Clip names selected by configs/avatar.yaml (include patterns minus exclude list), sorted."""
    lib = load_config("avatar")["library"]
    names = sorted(p.stem for p in path("clips_dir").glob("*.mp4"))
    keep = [n for n in names if any(fnmatch.fnmatch(n, pat) for pat in lib["include"])]
    return [n for n in keep if n not in set(lib.get("exclude") or [])]


def clip_path(name):
    return path("clips_dir") / f"{name}.mp4"


def file_sha1(p, chunk=1 << 20):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def read_frames(p, color="bgr", max_frames=None):
    import cv2
    import numpy as np
    cap, out = cv2.VideoCapture(str(p)), []
    while max_frames is None or len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if color == "rgb" else f)
    return np.stack(out) if out else np.zeros((0, 1, 1, 3), np.uint8)
