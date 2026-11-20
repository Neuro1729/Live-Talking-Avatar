"""Lip-sync backend interface. The rest of the system only sees this: frame references in, finished frames out.

  features(pcm) -> per-frame audio features for an utterance (backend specific, opaque to callers)
  silence_feature -> feature that produces a closed, neutral mouth
  process(refs, feats) -> list of BGR uint8 frames, one per ref (feats[i] None = silence)
"""
import numpy as np


class LipSyncBackend:
    name = "base"
    silence_feature = None

    def features(self, pcm):
        raise NotImplementedError

    def has_clip(self, clip):
        return True

    def process(self, refs, feats, source):
        raise NotImplementedError

    def close(self):
        pass


class NullLipSync(LipSyncBackend):
    """Mock backend for tests and the no-GPU mode: returns the source frames unchanged. Proves nothing about lip-sync."""
    name = "mock"

    def features(self, pcm):
        n = int(np.ceil(len(pcm) / 640))
        rms = np.array([np.sqrt(np.mean(pcm[i * 640:(i + 1) * 640] ** 2) + 1e-9) for i in range(n)], np.float32)
        return rms[:, None]

    def process(self, refs, feats, source):
        return [source.get(r) for r in refs]
