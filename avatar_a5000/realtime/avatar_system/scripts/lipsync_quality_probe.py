"""Full-resolution lip-sync quality probe: renders the same source frames with a speech feature through the lip-sync
backend under different settings and saves side-by-side face crops + sharpness numbers (Laplacian variance of the
lower face, a standard blur measure; higher = sharper).

  cd R && CUDA_VISIBLE_DEVICES=3 envs/musetalk/bin/python avatar_system/scripts/lipsync_quality_probe.py [--variants a,b]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir  # noqa: E402
from avatar_system.motion.motion_scheduler import FrameRef  # noqa: E402
from avatar_system.streaming.frame_source import ClipFrameSource  # noqa: E402


PRESETS = {"v2j": {"lip_w": 0.78, "lip_h": 1.4, "lip_min_h": 0.55, "lip_drop": 0.4, "jaw_h": 1.45, "jaw_drop": 0.22,
                   "detail_gain": 0.8}}


def lap_var(img, box):
    x1, y1, x2, y2 = box
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="sub_talk_s41:60,sub_still_s41:120,bank_steady_s22:100")
    ap.add_argument("--variants", default="current")
    ap.add_argument("--out", default=str(data_dir("eval") / "lipsync_probe"))
    a = ap.parse_args()
    from avatar_system.lipsync.musetalk_adapter import MuseTalkBackend
    items = [(c.split(":")[0], int(c.split(":")[1])) for c in a.clips.split(",")]
    be = MuseTalkBackend(sorted({c for c, _ in items}))
    src = ClipFrameSource()
    wav = sf.read(str(data_dir("test_audio") / "hi_2.wav"), dtype="float32")[0]
    feats = be.features(wav)
    rows, report = [], []
    for k, (clip, idx) in enumerate(items):
        refs = [FrameRef(0, 0, clip, idx)]
        raw = src.get(refs[0])
        x1, y1, x2, y2 = map(int, be.clips[clip].boxes[idx])
        mouth = (x1, (y1 + y2) // 2, x2, y2)
        crop_box = (max(x1 - 40, 0), max(y1 - 40, 0), x2 + 40, y2 + 40)
        tiles = [("raw", raw)]
        for v in a.variants.split(","):
            be.set_variant("v1" if v == "v1" else "v2")
            if v in PRESETS:
                be.cfg = dict(be.cfg, v2={**be.cfg["v2"], **PRESETS[v]})
            w = float(v.split("r")[1]) if v.startswith("v2r") else 0  # e.g. v2r0.7 = restore with weight 0.7
            if w and be.restorer is None:
                from avatar_system.common import root
                from avatar_system.lipsync.musetalk_adapter import MouthRestorer
                be.restorer = MouthRestorer(root() / be.cfg["restore"]["model"])
            be.cfg = dict(be.cfg, restore={**be.cfg.get("restore", {}), "enabled": bool(w), "weight": w})
            out = be.process(refs, [feats[25 + 7 * k]], src)[0]
            tiles.append((v, out))
        row = []
        for name, img in tiles:
            c = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]].copy()
            s = lap_var(img, mouth)
            report.append((clip, idx, name, round(s, 1)))
            c = cv2.resize(c, (420, int(c.shape[0] * 420 / c.shape[1])), interpolation=cv2.INTER_CUBIC)
            cv2.putText(c, f"{name} sharp={s:.0f}", (6, 22), 0, 0.6, (0, 0, 255), 2)
            row.append(c)
        h = min(r.shape[0] for r in row)
        rows.append(np.hstack([r[:h] for r in row]))
    w = min(r.shape[1] for r in rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out + ".jpg", np.vstack([r[:, :w] for r in rows]))
    for r in report:
        print(r)


if __name__ == "__main__":
    main()
