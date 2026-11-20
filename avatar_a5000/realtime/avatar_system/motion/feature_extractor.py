"""Per-frame motion/appearance features for every library clip (CPU, MediaPipe + ArcFace ONNX).

Output per clip: data/features/<clip>.npz
  feats   [n, F] float32  named columns FEATS (head pose, face position, gaze, eyes, brows, mouth, body, hands, colour)
  blend   [n, 52] float16 MediaPipe blendshapes
  lmk     [n, 478, 2] float16 normalised face landmarks
  thumb   [n, 87, 96] uint8 grayscale thumbnails (visual difference, optical flow)
  ident   [k, 512] float16 ArcFace embeddings of every IDENT_STEP-th frame
  sha1, version
Re-running skips clips whose video checksum and extractor version are unchanged (cache invalidation).

  envs/analysis/bin/python -m avatar_system.motion.feature_extractor [clip ...] [--workers 12]
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import clip_path, data_dir, file_sha1, library_clips, path, read_frames  # noqa: E402

VERSION = 3
FEATS = ["yaw", "pitch", "roll", "face_x", "face_y", "face_s", "gaze_x", "gaze_y", "eye_open", "eye_geo",
         "brow_up", "brow_down", "jaw_open", "mouth_geo", "smile",
         "lsh_x", "lsh_y", "rsh_x", "rsh_y", "lel_x", "lel_y", "rel_x", "rel_y", "lwr_x", "lwr_y", "rwr_x", "rwr_y",
         "hands_n", "h1_x", "h1_y", "h2_x", "h2_y", "luma", "bg_luma", "face_ok"]
FI = {k: i for i, k in enumerate(FEATS)}
THUMB = (96, 87)
IDENT_STEP = 8
ARCFACE = Path("/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/"
               "checkpoints/auxiliary/models/buffalo_l/w600k_r50.onnx")
ARC_TMPL = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                     [70.7299, 92.2041]], np.float32)


def _euler(m):
    r = np.asarray(m)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class Extractor:
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision
        M = path("mediapipe_models")
        V = vision.RunningMode.VIDEO
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(M / "face_landmarker.task")), running_mode=V,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(M / "pose_landmarker_full.task")), running_mode=V))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(M / "hand_landmarker.task")), running_mode=V, num_hands=2))
        self.arc = None
        if ARCFACE.exists():
            import onnxruntime as ort
            self.arc = ort.InferenceSession(str(ARCFACE), providers=["CPUExecutionProvider"])
        self.ts = 0

    def frame(self, rgb, want_ident):
        import cv2
        import mediapipe as mp
        h, w = rgb.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        self.ts += 40
        f = np.full(len(FEATS), np.nan, np.float32)
        blend = np.zeros(52, np.float16)
        lmk = np.full((478, 2), np.nan, np.float16)
        ident = None
        fr = self.face.detect_for_video(img, self.ts)
        if fr.face_landmarks:
            L = np.array([[p.x, p.y] for p in fr.face_landmarks[0]], np.float32)
            lmk = L.astype(np.float16)
            bs = {b.category_name: b.score for b in fr.face_blendshapes[0]}
            blend = np.array([b.score for b in fr.face_blendshapes[0]], np.float16)
            f[[FI["yaw"], FI["pitch"], FI["roll"]]] = _euler(fr.facial_transformation_matrixes[0])
            P = L * [w, h]
            iod = np.linalg.norm(P[468] - P[473])
            f[FI["face_x"]], f[FI["face_y"]] = P[:, 0].mean() / w, P[:, 1].mean() / h
            f[FI["face_s"]] = iod / w
            gx, gy, geo = [], [], []
            for c0, c1, up, lo, iris in ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473)):
                ew = np.linalg.norm(P[c1] - P[c0])
                mid = (P[c0] + P[c1]) / 2
                gx.append((P[iris, 0] - mid[0]) / ew)
                gy.append((P[iris, 1] - (P[up, 1] + P[lo, 1]) / 2) / ew)
                geo.append(np.linalg.norm(P[up] - P[lo]) / ew)
            f[FI["gaze_x"]], f[FI["gaze_y"]], f[FI["eye_geo"]] = np.mean(gx), np.mean(gy), np.mean(geo)
            f[FI["eye_open"]] = 1 - (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2
            f[FI["brow_up"]] = bs["browInnerUp"] + (bs["browOuterUpLeft"] + bs["browOuterUpRight"]) / 2
            f[FI["brow_down"]] = (bs["browDownLeft"] + bs["browDownRight"]) / 2
            f[FI["jaw_open"]] = bs["jawOpen"]
            f[FI["mouth_geo"]] = np.linalg.norm(P[13] - P[14]) / iod
            f[FI["smile"]] = (bs["mouthSmileLeft"] + bs["mouthSmileRight"]) / 2
            f[FI["face_ok"]] = 1
            if want_ident and self.arc is not None:
                eyes = sorted([P[468], P[473]], key=lambda p: p[0])
                mouth = sorted([P[61], P[291]], key=lambda p: p[0])
                src = np.array([eyes[0], eyes[1], P[1], mouth[0], mouth[1]], np.float32)
                M, _ = cv2.estimateAffinePartial2D(src, ARC_TMPL, method=cv2.LMEDS)
                crop = cv2.warpAffine(rgb, M, (112, 112), borderValue=0)
                x = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
                e = self.arc.run(None, {self.arc.get_inputs()[0].name: x})[0][0]
                ident = (e / np.linalg.norm(e)).astype(np.float16)
        else:
            f[FI["face_ok"]] = 0
        pr = self.pose.detect_for_video(img, self.ts)
        if pr.pose_landmarks:
            pl = pr.pose_landmarks[0]
            for name, idx in (("lsh", 11), ("rsh", 12), ("lel", 13), ("rel", 14), ("lwr", 15), ("rwr", 16)):
                f[FI[name + "_x"]], f[FI[name + "_y"]] = pl[idx].x, pl[idx].y
        hr = self.hand.detect_for_video(img, self.ts)
        cs = sorted([np.mean([[p.x, p.y] for p in hl], 0) for hl in hr.hand_landmarks], key=lambda c: c[0])
        f[FI["hands_n"]] = len(cs)
        for k, c in enumerate(cs[:2]):
            f[FI[f"h{k + 1}_x"]], f[FI[f"h{k + 1}_y"]] = c
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        f[FI["luma"]] = gray.mean() / 255
        f[FI["bg_luma"]] = np.concatenate([gray[:64, :64].ravel(), gray[:64, -64:].ravel()]).mean() / 255
        thumb = cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA)
        return f, blend, lmk, thumb, ident


def out_path(name):
    return data_dir("features") / f"{name}.npz"


def is_current(name):
    p = out_path(name)
    if not p.exists():
        return False
    z = np.load(p)
    return int(z["version"]) == VERSION and str(z["sha1"]) == file_sha1(clip_path(name))


def extract(name):
    import time
    t0 = time.time()
    ex = Extractor()
    frames = read_frames(clip_path(name), color="rgb")
    feats, blends, lmks, thumbs, idents = [], [], [], [], []
    for i, fr in enumerate(frames):
        f, b, l, t, e = ex.frame(fr, want_ident=i % IDENT_STEP == 0)
        feats.append(f), blends.append(b), lmks.append(l), thumbs.append(t)
        if e is not None:
            idents.append(e)
    np.savez_compressed(out_path(name), feats=np.stack(feats), blend=np.stack(blends), lmk=np.stack(lmks),
                        thumb=np.stack(thumbs), ident=np.stack(idents) if idents else np.zeros((0, 512), np.float16),
                        sha1=file_sha1(clip_path(name)), version=VERSION, cols=json.dumps(FEATS))
    miss = int((np.stack(feats)[:, FI["face_ok"]] == 0).sum())
    return f"{name}: {len(frames)} frames, {miss} without face, {time.time() - t0:.0f}s"


def load(name):
    z = np.load(out_path(name))
    return {k: z[k] for k in ("feats", "blend", "lmk", "thumb", "ident")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = a.clips or library_clips()
    todo = [n for n in names if a.force or not is_current(n)]
    print(f"{len(names)} clips, {len(todo)} to (re)extract", flush=True)
    with Pool(min(a.workers, max(len(todo), 1))) as pool:
        for msg in pool.imap_unordered(extract, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
