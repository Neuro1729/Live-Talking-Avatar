"""Phase 4: MuseTalk 1.5 preprocessing for the realigned shubh clips (output of build_aligned.py).

Our clips are already 25 fps, <= 12 s and have a matching 16 kHz wav, so of MuseTalk's preprocess.py only
analyze_video (face_alignment bbox + dwpose 68 landmarks per frame -> meta json) and the list files are needed:
  MuseTalk/dataset/shubh/meta/<clip>.json, train.txt, val.txt  (FaceDataset skips the first line = header)
Incremental: clips with an existing meta json are skipped, so it can be re-run while LTX is still generating.

  cd MuseTalk && CUDA_VISIBLE_DEVICES=3 ../envs/musetalk_train/bin/python ../musetalk_ft/scripts/preprocess_shubh.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())  # MuseTalk repo root (analyze_video uses repo-relative model paths)
from scripts.preprocess import analyze_video  # noqa: E402

DS = "./dataset/shubh"


def main():
    os.makedirs(f"{DS}/meta", exist_ok=True)
    qc = json.load(open(f"{DS}/qc.json"))
    keep = [r for r in qc if r["keep"] and os.path.exists(f"{DS}/clips/{r['name']}.mp4")]
    analyze_video(os.path.abspath(f"{DS}/clips"), f"{DS}/meta", [f"{r['name']}.mp4" for r in keep])

    lists, bad, faces = {"train": [], "test": []}, [], []
    for r in keep:
        mp = f"{DS}/meta/{r['name']}.json"
        if not os.path.exists(mp):
            bad.append((r["name"], "no meta"))
            continue
        m = json.load(open(mp))
        if not m["isvalid"]:
            bad.append((r["name"], "invalid"))
            continue
        faces.append(m["face_size"][0])
        lists[r["split"]].append(f"{r['name']}.json")
    for split, fn in (("train", "train.txt"), ("test", "val.txt")):
        with open(f"{DS}/{fn}", "w") as f:
            f.write("meta\n" + "".join(f"{x}\n" for x in lists[split]))
    print(f"train {len(lists['train'])}, val {len(lists['test'])}, dropped {bad}")
    if faces:
        print(f"face height px: min {min(faces)}, median {int(np.median(faces))}, max {max(faces)}")


if __name__ == "__main__":
    main()
