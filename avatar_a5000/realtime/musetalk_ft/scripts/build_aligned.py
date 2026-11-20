"""Phase 3: filter the LTX clips by SyncNet and write audio-realigned copies in MuseTalk's dataset layout.

LTX's mouth lags its conditioning audio by |offset| frames (calibrated 10-31: offset -7 + audio 280 ms earlier
-> offset 0). For each clip with data/sync/<clip>.json we advance the audio by -offset * 40 ms (drop the start,
pad the end with silence) so frame t is paired with the sound it actually shows, and write
  MuseTalk/dataset/shubh/clips/<clip>.mp4  (LTX video stream copied + shifted audio; MuseTalk's sync loss reads
                                            the mp4 audio)  and  <clip>.wav (16 kHz PCM, same shift)
plus MuseTalk/dataset/shubh/qc.json (every clip with its decision). Re-run any time; existing outputs are kept.

  ../envs/analysis/bin/python scripts/build_aligned.py [--min-conf 3.0]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FT = Path(__file__).resolve().parents[1]
R = FT.parent
OUT = R / "MuseTalk/dataset/shubh"
FFMPEG = R / "envs/musetalk/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SR, FPS = 16000, 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=3.0)
    ap.add_argument("--offset-range", type=int, nargs=2, default=[-10, -2], help="plausible LTX lag (frames)")
    a = ap.parse_args()
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    clips = {c["name"]: c for c in json.loads((FT / "data/clips.json").read_text())}
    qc = []
    for sj in sorted((FT / "data/sync").glob("mtft_*.json")):
        s = json.loads(sj.read_text())
        name, c = s["name"], clips[s["name"]]
        why = []
        if s["conf"] < a.min_conf:
            why.append(f"conf {s['conf']:.2f}")
        if not a.offset_range[0] <= s["offset"] <= a.offset_range[1]:
            why.append(f"offset {s['offset']}")
        rec = {"name": name, "split": c["split"], "offset": s["offset"], "conf": s["conf"], "keep": not why,
               "reject": why, "seconds": c["seconds"]}
        qc.append(rec)
        mp4, wav = OUT / "clips" / f"{name}.mp4", OUT / "clips" / f"{name}.wav"
        if why or mp4.exists():
            continue
        x, sr = sf.read(FT / f"data/audio/{name}.wav", dtype="float32")
        k = -s["offset"] * SR // FPS                      # samples to advance
        x = np.concatenate([x[k:], np.zeros(k, np.float32)])
        sf.write(wav, x, SR, subtype="PCM_16")
        subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(FT / f"data/raw_videos/{name}.mp4"),
                        "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(mp4)], check=True)
    (OUT / "qc.json").write_text(json.dumps(qc, indent=1))
    for split in ("train", "test"):
        q = [r for r in qc if r["split"] == split]
        k = [r for r in q if r["keep"]]
        print(f"{split}: scored {len(q)}, kept {len(k)} ({sum(r['seconds'] for r in k) / 60:.1f} min), "
              f"rejected {[(r['name'], r['reject']) for r in q if not r['keep']]}")


if __name__ == "__main__":
    main()
