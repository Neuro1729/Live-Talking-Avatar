"""Source-frame provider: FrameRef -> BGR uint8 frame. Uses the MuseTalk cache's memory-mapped frames.npy when present
(page-cache backed, no process RAM), otherwise decodes the clip mp4 once into a small LRU of clips."""
from collections import OrderedDict

import numpy as np

from avatar_system.common import clip_path, path, read_frames


class ClipFrameSource:
    def __init__(self, max_decoded=6):
        self.mm, self.lru, self.max = {}, OrderedDict(), max_decoded

    def clip(self, name):
        if name in self.mm:
            return self.mm[name]
        p = path("musetalk_cache") / name / "frames.npy"
        if (p.parent / "DONE").exists():  # only finished caches (prep writes frames.npy before DONE)
            self.mm[name] = np.load(p, mmap_mode="r")
            return self.mm[name]
        if name in self.lru:
            self.lru.move_to_end(name)
            return self.lru[name]
        fr = read_frames(clip_path(name))
        self.lru[name] = fr
        while len(self.lru) > self.max:
            self.lru.popitem(last=False)
        return fr

    def length(self, name):
        return len(self.clip(name))

    def get(self, ref):
        f = np.asarray(self.clip(ref.clip)[ref.idx])
        if ref.blend_clip is not None and ref.alpha < 1:
            a = np.asarray(self.clip(ref.blend_clip)[ref.blend_idx])
            f = (a.astype(np.float32) * (1 - ref.alpha) + f.astype(np.float32) * ref.alpha + 0.5).astype(np.uint8)
        return f
