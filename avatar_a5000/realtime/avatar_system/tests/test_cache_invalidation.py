import numpy as np

from avatar_system.motion import feature_extractor as fx


def test_feature_cache_invalidates_on_video_change(tmp_path, monkeypatch):
    vid = tmp_path / "c.mp4"
    vid.write_bytes(b"version-1")
    monkeypatch.setattr(fx, "clip_path", lambda n: vid)
    monkeypatch.setattr(fx, "out_path", lambda n: tmp_path / f"{n}.npz")
    np.savez(tmp_path / "c.npz", sha1=fx.file_sha1(vid), version=fx.VERSION)
    assert fx.is_current("c")
    vid.write_bytes(b"version-2")          # clip regenerated -> features must be rebuilt
    assert not fx.is_current("c")
    np.savez(tmp_path / "c.npz", sha1=fx.file_sha1(vid), version=fx.VERSION - 1)  # extractor changed
    assert not fx.is_current("c")
