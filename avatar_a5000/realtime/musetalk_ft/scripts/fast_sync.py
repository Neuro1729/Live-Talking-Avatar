"""Fast SyncNet (syncnet_v2, same model/crop/metric as LatentSync's eval.eval_sync_conf) for many clips.

Differences from the stock eval: S3FD runs only every --det-every frames (camera is locked off, face moves
slowly) and is interpolated + median-smoothed, frames are read in memory, and the audio comes straight from
data/audio/<clip>.wav, optionally shifted (--shift-ms > 0 = audio later). Writes data/sync/<clip>.json with
offset, confidence and the full distance curve over offsets.

Run with the LatentSync venv from its eval dir:
  cd avatar_system/data/eval/syncnet/cwd && /var/.../ComfyUI/.venv/bin/python <FT>/scripts/fast_sync.py --clips a b ...
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import python_speech_features
import soundfile as sf
import torch
from scipy import signal
from scipy.interpolate import interp1d

sys.path.insert(0, ".")  # LatentSync eval package (cwd = avatar_system/data/eval/syncnet/cwd)
from eval.detectors import S3FD  # noqa: E402
from eval.syncnet import SyncNetEval  # noqa: E402
from eval.syncnet.syncnet_eval import calc_pdist  # noqa: E402

FT = Path(__file__).resolve().parents[1]


def face_track(frames, det, every):
    idx, boxes = [], []
    for i in list(range(0, len(frames), every)) + [len(frames) - 1]:
        b = det.detect_faces(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), conf_th=0.9, scales=[0.25])
        if len(b):
            b = max(b, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            idx.append(i)
            boxes.append(b[:4])
    if len(idx) < 2:
        return None
    boxes = np.array(boxes)
    f = interp1d(idx, boxes, axis=0, bounds_error=False, fill_value=(boxes[0], boxes[-1]))
    return f(np.arange(len(frames)))


def crops(frames, track, cs=0.4):
    s = signal.medfilt(np.maximum(track[:, 3] - track[:, 1], track[:, 2] - track[:, 0]) / 2, 13)
    y = signal.medfilt((track[:, 1] + track[:, 3]) / 2, 13)
    x = signal.medfilt((track[:, 0] + track[:, 2]) / 2, 13)
    out = []
    for i, fr in enumerate(frames):
        bs = s[i]
        bsi = int(bs * (1 + 2 * cs))
        p = np.pad(fr, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my, mx = y[i] + bsi, x[i] + bsi
        face = p[int(my - bs): int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)): int(mx + bs * (1 + cs))]
        out.append(cv2.resize(face, (224, 224)))
    return out


@torch.no_grad()
def score(net, faces, audio, device, vshift=15, bs=20):
    im = np.stack(faces, 3)[None].transpose(0, 3, 4, 1, 2)  # 1,3,T,224,224 (BGR like the stock eval)
    imtv = torch.from_numpy(im.astype(np.float32))
    mfcc = np.stack([np.array(i) for i in zip(*python_speech_features.mfcc((audio * 32767).astype(np.int16), 16000))])
    cct = torch.from_numpy(mfcc[None, None].astype(np.float32))
    n = min(len(faces), math.floor(len(audio) / 640)) - 5
    imf, ccf = [], []
    for i in range(0, n, bs):
        r = range(i, min(n, i + bs))
        imf.append(net.__S__.forward_lip(torch.cat([imtv[:, :, v:v + 5] for v in r]).to(device)).cpu())
        ccf.append(net.__S__.forward_aud(torch.cat([cct[:, :, :, v * 4:v * 4 + 20] for v in r]).to(device)).cpu())
    dists = calc_pdist(torch.cat(imf), torch.cat(ccf), vshift=vshift)
    mean = torch.mean(torch.stack(dists, 1), 1)
    mn, idx = torch.min(mean, 0)
    return int(vshift - idx), float(torch.median(mean) - mn), [round(float(v), 4) for v in mean]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", help="clip names (default: every video without a sync json)")
    ap.add_argument("--video-dir", default=str(FT / "data/raw_videos"))
    ap.add_argument("--audio-dir", default=str(FT / "data/audio"), help="<clip>.wav scored against the video")
    ap.add_argument("--shift-ms", type=float, default=0.0, help="> 0 = audio later")
    ap.add_argument("--det-every", type=int, default=8)
    ap.add_argument("--out", default=str(FT / "data/sync"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = S3FD(device=dev)
    net = SyncNetEval(device=dev)
    net.loadParameters("checkpoints/auxiliary/syncnet_v2.model")
    net.to(dev)
    net.__S__.eval()  # BatchNorm in inference mode, like SyncNetEval.evaluate (missing before 10-31 21:00: confidences were skewed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # a muxed data/videos/<clip>.mp4 only exists once the raw video is completely written
    names = a.clips or sorted(p.stem for p in (FT / "data/videos").glob("mtft_*.mp4")
                              if not (out / f"{p.stem}{a.tag}.json").exists())
    for name in names:
        cap = cv2.VideoCapture(str(Path(a.video_dir) / f"{name}.mp4"))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        audio, sr = sf.read(Path(a.audio_dir) / f"{name}.wav", dtype="float32")
        k = int(round(a.shift_ms / 1000 * sr))
        audio = np.concatenate([np.zeros(k, np.float32), audio])[:len(audio)] if k > 0 else np.concatenate(
            [audio[-k:], np.zeros(-k, np.float32)])
        track = face_track(frames, det, a.det_every)
        if track is None:
            print(f"{name}: NO FACE", flush=True)
            continue
        off, conf, curve = score(net, crops(frames, track), audio, dev)
        res = {"name": name, "shift_ms": a.shift_ms, "offset": off, "conf": round(conf, 3), "curve": curve,
               "face_px": round(float(np.median(track[:, 2] - track[:, 0])), 1)}
        (out / f"{name}{a.tag}.json").write_text(json.dumps(res))
        print(f"{name}{a.tag}: offset {off:+d} conf {conf:.2f} face {res['face_px']}px", flush=True)


if __name__ == "__main__":
    main()
