"""Turn per-frame features into motion descriptors: smoothed signals, speed, blinks, nods, gaze-up, gestures,
rest points (low-velocity frames where a clip can be cut) and the usable frame range (frozen settle frames trimmed)."""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

from avatar_system.common import load_config
from avatar_system.motion.feature_extractor import FI, load

# rows of the 96x87 thumbnail ignored for visual comparison (mouth area is redrawn by lip-sync)
MOUTH_BOX = (int(87 * .36), int(87 * .50), int(96 * .40), int(96 * .60))


def thumb_mask(shape=(87, 96)):
    m = np.ones(shape, np.float32)
    y0, y1, x0, x1 = MOUTH_BOX
    m[y0:y1, x0:x1] = 0
    return m


def fill_nan(x):
    x = x.copy()
    for j in range(x.shape[1]):
        c = x[:, j]
        bad = np.isnan(c)
        if bad.all():
            c[:] = 0
        elif bad.any():
            c[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), c[~bad])
    return x


POSE_KEYS = [("yaw", "yaw"), ("pitch", "pitch"), ("roll", "roll"), ("face_x", "face_xy"), ("face_y", "face_xy"),
             ("face_s", "face_s"), ("gaze_x", "gaze"), ("gaze_y", "gaze"), ("smile", "smile"), ("brow_up", "brow"),
             ("lwr_x", "wrist"), ("lwr_y", "wrist"), ("rwr_x", "wrist"), ("rwr_y", "wrist"),
             ("lsh_y", "shoulder"), ("rsh_y", "shoulder")]
# weights inside the pose vector: head/face geometry dominates, expression and hands count less
POSE_W = np.array([1, 1, 1, 1, 1, .7, .5, .5, .5, .4, .6, .6, .6, .6, .5, .5], np.float32)


def pose_vectors(feats):
    """[n, F] smoothed features -> [n, D] normalised + weighted pose vectors (distance 1 ~ one 'scale' off)."""
    sc = load_config("motion_graph")["scales"]
    cols = [feats[:, FI[k]] / sc[s] for k, s in POSE_KEYS]
    return np.stack(cols, 1) * POSE_W


def pose_dist(a, b):
    """RMS distance between pose vectors (broadcasts)."""
    return np.sqrt(np.mean((a - b) ** 2, axis=-1) / np.mean(POSE_W ** 2))


def analyze(name, data=None):
    cfg = load_config("motion_graph")
    A, sc = cfg["analysis"], cfg["scales"]
    d = data or load(name)
    raw = fill_nan(d["feats"].astype(np.float32))
    n = len(raw)
    f = uniform_filter1d(median_filter(raw, size=(A["smooth_win"], 1), mode="nearest"), 3, axis=0, mode="nearest")
    thumb = d["thumb"].astype(np.float32)
    tm = thumb_mask()
    vis = np.r_[0, (np.abs(np.diff(thumb, axis=0)) * tm).sum((1, 2)) / tm.sum()]

    # combined speed (normalised units / frame)
    head_v = np.sqrt(sum(np.r_[0, np.diff(f[:, FI[k]])] ** 2 for k in ("yaw", "pitch", "roll"))) / sc["vel_head"]
    face_v = np.hypot(np.r_[0, np.diff(f[:, FI["face_x"]])], np.r_[0, np.diff(f[:, FI["face_y"]])]) / sc["vel_face"]
    wr_v = sum(np.hypot(np.r_[0, np.diff(f[:, FI[p + "_x"]])], np.r_[0, np.diff(f[:, FI[p + "_y"]])])
               for p in ("lwr", "rwr")) / 2 / sc["vel_wrist"]
    speed = uniform_filter1d(head_v + face_v + 0.3 * wr_v + vis / sc["visual"], 5, mode="nearest")

    # usable range: trim frozen settle frames at both ends (keep a couple)
    s, e = 1, n
    while s < n - 1 and vis[s] < A["settle_diff"]:
        s += 1
    while e > s + 1 and vis[e - 1] < A["settle_diff"]:
        e -= 1
    start, end = max(s - 1 - A["settle_keep"], 0), min(e + A["settle_keep"], n)

    # blinks: dips of raw eye openness
    eo = raw[:, FI["eye_open"]]
    base = np.percentile(eo, 80)
    closed = eo < base - A["blink_drop"]
    blinks = [int(i) for i in np.flatnonzero(closed[1:] & ~closed[:-1]) + 1]
    # nods: pitch peaks with enough prominence
    pitch = f[:, FI["pitch"]]
    nods = [int(p) for p in find_peaks(pitch, prominence=A["nod_pitch_deg"], distance=12)[0]]
    gaze_up = f[:, FI["gaze_y"]] < A["gaze_up"]
    wr0 = f[start, [FI["lwr_x"], FI["lwr_y"], FI["rwr_x"], FI["rwr_y"]]]
    wr_dev = np.abs(f[:, [FI["lwr_x"], FI["lwr_y"], FI["rwr_x"], FI["rwr_y"]]] - wr0).max(1)
    gesture = wr_dev > A["gesture_wrist"]

    # rest points: local speed minima, below a clip-relative percentile, away from blinks
    thr = np.percentile(speed[start:end], A["rest_speed_pct"])
    cand = [i for i in range(start + 2, end - 2) if speed[i] <= speed[i - 2:i + 3].min() and speed[i] <= thr
            and not closed[max(i - 3, 0):i + 4].any()]
    rest = []
    for i in sorted(cand, key=lambda i: speed[i]):
        if all(abs(i - r) >= A["rest_min_gap"] for r in rest):
            rest.append(i)
    rest.sort()

    return {"name": name, "n": n, "start": start, "end": end, "feats": f, "raw": raw, "speed": speed, "vis": vis,
            "blinks": blinks, "nods": nods, "gaze_up": gaze_up, "gesture": gesture, "closed": closed,
            "rest": rest, "pose": pose_vectors(f), "thumb": d["thumb"], "lmk": d["lmk"], "ident": d["ident"]}


def describe(an, s, e):
    """Summary descriptors for frames [s, e) of an analysed clip."""
    f = an["feats"][s:e]
    rng = lambda k: float(np.ptp(f[:, FI[k]]))
    blinks = [b for b in an["blinks"] if s <= b < e]
    nods = [p for p in an["nods"] if s <= p < e]
    gaze_up = float(an["gaze_up"][s:e].mean())
    gesture = bool(an["gesture"][s:e].any())
    intensity = float(an["speed"][s:e].mean())
    gx = float(np.median(f[:, FI["gaze_x"]]))
    gaze_dir = "up" if gaze_up > 0.15 else ("left" if gx < -0.04 else "right" if gx > 0.04 else "camera")
    return {"frames": int(e - s), "intensity": round(intensity, 3), "blinks": len(blinks), "blink_frames": blinks,
            "nods": len(nods), "gaze_up_frac": round(gaze_up, 3), "gaze_dir": gaze_dir, "gesture": gesture,
            "head_range_deg": [round(rng("yaw"), 2), round(rng("pitch"), 2), round(rng("roll"), 2)],
            "smile": round(float(f[:, FI["smile"]].mean()), 3), "brow_up": round(float(f[:, FI["brow_up"]].mean()), 3),
            "closed_frac": round(float(an["closed"][s:e].mean()), 3)}
