#!/usr/bin/env python3
"""One-time MuseTalk 1.5 preparation of base clips -> avatar_cache/<clip>/ (everything the live engine needs).

Per frame: face bbox (DWPose landmarks + S3FD, temporally smoothed so the mouth doesn't jitter), VAE latents
(masked + reference, fp16), face-parsing blend mask + crop box. Frames saved as one uint8 .npy (memory-mapped at runtime).

Run from the MuseTalk dir with the musetalk env:
  cd MuseTalk && CUDA_VISIBLE_DEVICES=3 ../envs/musetalk/bin/python ../scripts/prep_avatar.py bank_steady_s21 bank_idle_s11 ...
  (no args = every base_clips/bank_*.mp4 + ltx_*silentA2V.mp4 that isn't cached yet)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "MuseTalk"))
from musetalk.utils.blending import get_image_prepare_material  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.preprocessing import get_landmark_and_bbox  # noqa: E402  (loads DWPose + S3FD on import)
from musetalk.models.vae import VAE  # noqa: E402

EXTRA_MARGIN = 10  # v15 default: extend bbox down to include the chin
SMOOTH = 5         # moving-average window (frames) for bbox coordinates


def read_frames(path):
    cap, fr = cv2.VideoCapture(str(path)), []
    while (ok := cap.read())[0]:
        fr.append(ok[1])
    return fr


def smooth_boxes(boxes, k):
    b = np.array(boxes, dtype=np.float32)
    pad = np.pad(b, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k, dtype=np.float32) / k
    return np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(4)], 1).round().astype(np.int32)


def prep(name, vae, fp):
    out = R / "avatar_cache" / name
    if (out / "DONE").exists():
        print(f"skip {name}")
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = read_frames(R / "base_clips" / f"{name}.mp4")
    tmp = out / "_frames"
    tmp.mkdir(exist_ok=True)
    paths = []
    for i, f in enumerate(frames):  # get_landmark_and_bbox wants image paths
        p = tmp / f"{i:05d}.png"
        cv2.imwrite(str(p), f)
        paths.append(str(p))
    coords, _ = get_landmark_and_bbox(paths, 0)
    bad = [i for i, c in enumerate(coords) if tuple(c) == (0.0, 0.0, 0.0, 0.0)]
    if bad:  # fill missing detections from nearest good frame
        good = [i for i in range(len(coords)) if i not in bad]
        for i in bad:
            coords[i] = coords[min(good, key=lambda g: abs(g - i))]
    H, W = frames[0].shape[:2]
    boxes = smooth_boxes([list(map(int, c)) for c in coords], SMOOTH)
    boxes[:, 3] = np.minimum(boxes[:, 3] + EXTRA_MARGIN, H)

    latents, masks, crops = [], [], []
    for f, (x1, y1, x2, y2) in zip(frames, boxes):
        crop = cv2.resize(f[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents.append(vae.get_latents_for_unet(crop).cpu())
        m, cb = get_image_prepare_material(f, [int(x1), int(y1), int(x2), int(y2)], fp=fp, mode="jaw")
        masks.append(m)
        crops.append(cb)
    np.save(out / "frames.npy", np.stack(frames))
    np.save(out / "boxes.npy", boxes)
    torch.save(torch.cat(latents).half(), out / "latents.pt")
    with open(out / "masks.pkl", "wb") as fh:
        pickle.dump({"masks": masks, "crop_boxes": crops}, fh)
    for p in paths:
        Path(p).unlink()
    tmp.rmdir()
    (out / "info.json").write_text(json.dumps({"clip": name, "frames": len(frames), "size": [W, H], "missing_faces": bad,
                                               "seconds": round(time.time() - t0, 1)}))
    (out / "DONE").touch()
    print(f"{name}: {len(frames)} frames, {len(bad)} missing faces, {time.time() - t0:.0f}s", flush=True)


def main():
    names = sys.argv[1:] or sorted(p.stem for p in (R / "base_clips").glob("*.mp4")
                                   if p.stem.startswith("bank_") or p.stem.endswith("silentA2V"))
    vae = VAE(model_path=str(R / "MuseTalk/models/sd-vae"), use_float16=True)
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    with torch.no_grad():
        for n in names:
            prep(n, vae, fp)


if __name__ == "__main__":
    main()
