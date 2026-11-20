"""Continuity cost of cutting from clip A at frame `a` (last frame shown) to clip B at frame `b` (next frame shown).

Each component is normalised so that the natural step between two consecutive frames of one clip costs about 0-1.
It compares the boundary frames AND a window of W frames on both sides, so a cut between two similar frames whose
motion goes in opposite directions is still penalised (motion component).
"""
import numpy as np

from avatar_system.common import load_config
from avatar_system.motion.clip_analyzer import thumb_mask
from avatar_system.motion.feature_extractor import FI

# face landmarks excluding lips/jaw (lip-sync redraws them): MediaPipe lip indices
_LIPS = {0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270,
         291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415}
STABLE_LMK = np.array([i for i in range(468) if i not in _LIPS])
_TM = thumb_mask()
WIDTH = 1024
HAND_BOX = (int(87 * .80), 87, int(96 * .25), int(96 * .78))  # thumbnail rows/cols of the clasped hands + table edge


def _vel(f, i, keys, w):
    """Mean per-frame velocity of feature columns over frames [i-w, i]."""
    lo = max(i - w, 0)
    if i <= lo:
        return np.zeros(len(keys))
    return (f[i, keys] - f[lo, keys]) / (i - lo)


def _vel_fwd(f, i, keys, w):
    hi = min(i + w, len(f) - 1)
    if hi <= i:
        return np.zeros(len(keys))
    return (f[hi, keys] - f[i, keys]) / (hi - i)


HEAD = [FI["yaw"], FI["pitch"], FI["roll"]]
FACE = [FI["face_x"], FI["face_y"]]
WRIST = [FI[k] for k in ("lwr_x", "lwr_y", "rwr_x", "rwr_y", "lel_x", "lel_y", "rel_x", "rel_y")]


def components(A, a, B, b):
    """A, B: analysed clips (clip_analyzer.analyze); a: last frame of A shown, b: first frame of B shown."""
    cfg = load_config("motion_graph")
    sc, W = cfg["scales"], cfg["transition"]["window"]
    fa, fb = A["feats"], B["feats"]
    # where B's first frame "should" be given A's motion: extrapolate one frame
    va = {k: _vel(fa, a, idx, W) for k, idx in (("head", HEAD), ("face", FACE), ("wrist", WRIST))}
    vb = {k: _vel_fwd(fb, b, idx, W) for k, idx in (("head", HEAD), ("face", FACE), ("wrist", WRIST))}
    pred_head = fa[a, HEAD] + va["head"]
    pred_face = fa[a, FACE] + va["face"]
    pose = np.sqrt(np.mean(np.r_[(fb[b, HEAD] - pred_head) / [sc["yaw"], sc["pitch"], sc["roll"]],
                                 (fb[b, FACE] - pred_face) / sc["face_xy"],
                                 (fb[b, FI["face_s"]] - fa[a, FI["face_s"]]) / sc["face_s"]] ** 2))
    motion = np.sqrt(np.mean(np.r_[(vb["head"] - va["head"]) / sc["vel_head"],
                                   (vb["face"] - va["face"]) / sc["vel_face"],
                                   (vb["wrist"] - va["wrist"]) / sc["vel_wrist"]] ** 2))
    expr = np.sqrt(np.mean(np.r_[(fb[b, FI["smile"]] - fa[a, FI["smile"]]) / sc["smile"],
                                 (fb[b, FI["brow_up"]] - fa[a, FI["brow_up"]]) / sc["brow"],
                                 (fb[b, FI["brow_down"]] - fa[a, FI["brow_down"]]) / sc["brow"],
                                 (B["raw"][b, FI["eye_open"]] - A["raw"][a, FI["eye_open"]]) / sc["eye_open"]] ** 2))
    gaze = np.hypot(fb[b, FI["gaze_x"]] - fa[a, FI["gaze_x"]], fb[b, FI["gaze_y"]] - fa[a, FI["gaze_y"]]) / sc["gaze"]
    # hands: pixels of the hand/table region (pose wrists jitter ~0.04 when the hands are clasped) + loose wrists
    hy0, hy1, hx0, hx1 = HAND_BOX
    hand_vis = float(np.abs(B["thumb"][b, hy0:hy1, hx0:hx1].astype(np.float32)
                            - A["thumb"][a, hy0:hy1, hx0:hx1].astype(np.float32)).mean())
    hand = 0.7 * hand_vis / sc["hand_visual"] + 0.3 * np.sqrt(np.mean(((fb[b, WRIST] - fa[a, WRIST]) / sc["wrist"]) ** 2))
    la, lb = A["lmk"][a].astype(np.float32), B["lmk"][b].astype(np.float32)
    if np.isnan(la).any() or np.isnan(lb).any():
        landmark = 99.0
    else:
        landmark = float(np.linalg.norm((lb[STABLE_LMK] - la[STABLE_LMK]) * [WIDTH, WIDTH * 928 / 1024], axis=1).mean()) / 1.5
    # visual: boundary frames and the window means (mouth masked)
    ta, tb = A["thumb"][a].astype(np.float32), B["thumb"][b].astype(np.float32)
    v0 = (np.abs(tb - ta) * _TM).sum() / _TM.sum()
    wa = A["thumb"][max(a - W + 1, 0):a + 1].astype(np.float32).mean(0)
    wb = B["thumb"][b:b + W].astype(np.float32).mean(0)
    v1 = (np.abs(wb - wa) * _TM).sum() / _TM.sum()
    visual = (0.7 * v0 + 0.3 * v1) / sc["visual"]
    # hard-reject measurements
    face_jump_px = float(np.hypot(*(fb[b, FACE] - fa[a, FACE])) * WIDTH)
    ida = A["ident"].astype(np.float32).mean(0)
    idb = B["ident"].astype(np.float32).mean(0)
    ident = float(ida @ idb / (np.linalg.norm(ida) * np.linalg.norm(idb) + 1e-6)) if len(A["ident"]) and len(B["ident"]) else 1.0
    return {"pose": float(pose), "motion": float(motion), "expression": float(expr), "gaze": float(gaze),
            "hand": float(hand), "landmark": float(landmark), "visual": float(visual),
            "face_jump_px": face_jump_px, "identity_cos": ident,
            "luma_jump": float(abs(fb[b, FI["luma"]] - fa[a, FI["luma"]])),
            "bg_jump": float(abs(fb[b, FI["bg_luma"]] - fa[a, FI["bg_luma"]])),
            "hand_visual": hand_vis,
            "visual_raw": float(v0)}


def score(A, a, B, b):
    """-> (cost, components, hard_reject_reasons)."""
    cfg = load_config("motion_graph")["transition"]
    c = components(A, a, B, b)
    cost = sum(w * c[k] for k, w in cfg["weights"].items()) / sum(cfg["weights"].values())
    hr = cfg["hard_reject"]
    reasons = []
    if c["face_jump_px"] > hr["face_jump_px"]:
        reasons.append(f"face jump {c['face_jump_px']:.1f}px")
    if c["identity_cos"] < hr["identity_cos"]:
        reasons.append(f"identity cos {c['identity_cos']:.2f}")
    if c["luma_jump"] > hr["luma_jump"]:
        reasons.append(f"luma jump {c['luma_jump']:.3f}")
    if c["bg_jump"] > hr["bg_jump"]:
        reasons.append(f"background jump {c['bg_jump']:.3f}")
    if c["hand_visual"] > hr["hand_visual"]:
        reasons.append(f"hand region change {c['hand_visual']:.1f}")
    if c["landmark"] >= 99:
        reasons.append("face missing at cut")
    return float(cost), c, reasons


def natural_costs(an, step=1):
    """Costs of the clip's own consecutive frames (a -> a+1): the calibration reference for 'invisible' cuts."""
    return [score(an, i, an, i + 1)[0] for i in range(an["start"] + 3, an["end"] - 4, step)]
