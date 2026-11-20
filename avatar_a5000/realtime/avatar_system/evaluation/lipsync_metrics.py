"""Lip-sync measurements on a rendered video with its audio track (no SyncNet weights needed):
- mouth opening per output frame (MediaPipe inner-lip distance / inter-ocular distance)
- A/V offset: lag (frames) maximising the correlation between mouth opening and the speech energy envelope
  (positive = video lags audio) and the correlation at that lag (sync strength proxy)
- silent-region mouth motion: mean |d mouth| on frames without speech vs with speech
- mouth temporal consistency: frame-to-frame |d mouth| at segment seams vs elsewhere during speech (jaw jumps)

  envs/analysis/bin/python -m avatar_system.evaluation.lipsync_metrics <video.mp4> <trace.json>
"""
import json
import subprocess
import sys

import cv2
import numpy as np

from avatar_system.common import path

SR, FPS, SPF = 16000, 25, 640


def mouth_track(video):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path("mediapipe_models") / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))
    cap, out, k = cv2.VideoCapture(str(video)), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), k * 40)
        k += 1
        if r.face_landmarks:
            P = np.array([[p.x, p.y] for p in r.face_landmarks[0]])
            out.append(np.linalg.norm(P[13] - P[14]) / np.linalg.norm(P[468] - P[473]))
        else:
            out.append(np.nan)
    return np.array(out)


def audio_env(video, n):
    raw = subprocess.run([str(path("ffmpeg")), "-loglevel", "error", "-i", str(video), "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    a = np.pad(a, (0, max(n * SPF - len(a), 0)))[: n * SPF].reshape(n, SPF)
    return np.sqrt((a ** 2).mean(1))


def report(video, trace):
    m = mouth_track(video)
    n = len(m)
    env = audio_env(video, n)
    m = np.where(np.isnan(m), np.nanmedian(m), m)
    speech = np.array([u is not None for u in trace["utt"][:n]] + [False] * max(0, n - len(trace["utt"])))
    db = 20 * np.log10(env + 1e-5)
    voiced = (db > db[speech].max() - 30) & speech if speech.any() else speech
    # offset: correlate on speech frames only
    best, corr = 0, -1
    for lag in range(-8, 9):
        a = env[max(0, -lag):n - max(0, lag)]
        b = m[max(0, lag):n - max(0, -lag)]
        s = speech[max(0, -lag):n - max(0, lag)]
        if s.sum() > 20:
            c = np.corrcoef(a[s], b[s])[0, 1]
            if c > corr:
                best, corr = lag, c
    dm = np.abs(np.diff(m))
    seams = {s[0] for s in trace["seams"] if not s[6]}
    seam_mask = np.array([k + 1 in seams for k in range(n - 1)])
    sp = speech[1:] & speech[:-1]
    sil = ~speech[1:] & ~speech[:-1]
    return {
        "frames": n, "speech_frames": int(speech.sum()), "voiced_frames": int(voiced.sum()),
        "av_offset_frames": best, "av_offset_ms": best * 40, "mouth_audio_corr": round(float(corr), 3),
        "mouth_open_speech_mean": round(float(m[speech].mean()), 4) if speech.any() else None,
        "mouth_open_silence_mean": round(float(m[~speech].mean()), 4),
        "mouth_motion_speech": round(float(dm[sp].mean()), 4) if sp.any() else None,
        "mouth_motion_silence": round(float(dm[sil].mean()), 4) if sil.any() else None,
        "mouth_jump_at_seams_during_speech": round(float(dm[seam_mask & sp].mean()), 4) if (seam_mask & sp).any() else None,
        "mouth_jump_elsewhere_during_speech": round(float(dm[~seam_mask & sp].mean()), 4) if sp.any() else None,
        "seams_during_speech": int((seam_mask & sp).sum()),
    }


if __name__ == "__main__":
    tr = json.load(open(sys.argv[2]))
    print(json.dumps(report(sys.argv[1], tr), indent=1))
