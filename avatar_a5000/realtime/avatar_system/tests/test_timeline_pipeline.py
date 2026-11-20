import numpy as np
import pytest

from avatar_system.conversation.behavior_planner import parse_behavior, prosody, text_hints
from avatar_system.conversation.state_machine import StateMachine
from avatar_system.lipsync.audio_context import SPF, SR, AudioEvent, AudioTimeline
from avatar_system.lipsync.base import NullLipSync


def tone(sec, f=220):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_audio_frames_carry_exact_samples():
    tl = AudioTimeline(None, lead_frames=0)
    x = tone(1.0)
    start = tl.push(AudioEvent("s", "u", SR, 0, x, True))
    got = b"".join(tl.frame(start + k)[0] for k in range(25))
    ref = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    assert got == ref  # frame n carries samples [n*640,(n+1)*640): no drift, no gaps


def test_chunked_audio_equals_whole_and_final_flags():
    fe = NullLipSync().features
    whole, chunked = AudioTimeline(fe, lead_frames=0), AudioTimeline(fe, lead_frames=0)
    x = tone(2.0)
    whole.push(AudioEvent("s", "u", SR, 0, x, True))
    step = 8000
    for o in range(0, len(x), step):
        chunked.push(AudioEvent("s", "u", SR, o, x[o:o + step], o + step >= len(x)))
        if o + step < len(x):  # mid-stream: the last right_ctx frames are not final yet
            n_have = (o + step) // SPF
            assert not chunked.frame(n_have - 1)[3]
    for n in range(50):
        a, b = whole.frame(n), chunked.frame(n)
        assert a[0] == b[0] and a[3] and b[3]


def test_utterances_never_overlap_and_start_after_render_point():
    tl = AudioTimeline(None, lead_frames=2)
    for n in range(100):
        tl.frame(n)
    s1 = tl.push(AudioEvent("s", "a", SR, 0, tone(1.0), True))
    s2 = tl.push(AudioEvent("s", "b", SR, 0, tone(0.5), True))
    assert s1 >= 100 + 2 and s2 == s1 + 25


def test_cancel_removes_unrendered_speech_and_late_chunks_are_ignored():
    tl = AudioTimeline(None, lead_frames=0)
    tl.push(AudioEvent("s", "a", SR, 0, tone(2.0), False))
    tl.push(AudioEvent("s", "b", SR, 0, tone(1.0), True))
    for n in range(10):
        tl.frame(n)
    ids = tl.cancel("s")
    assert set(ids) == {"a", "b"}
    tl.push(AudioEvent("s", "a", SR, 32000, tone(1.0), True))  # stale chunk of a cancelled utterance
    assert tl.pending_frames(10) == 0
    assert tl.frame(20)[2] is None


def test_prosody_and_hints():
    x = np.r_[tone(0.5) * 0.2, tone(0.3) * 1.0, np.zeros(SR // 2, np.float32), tone(0.5) * 0.2]
    p = prosody(x)
    assert 0.2 < p["pause_frac"] < 0.5 and p["emphasis_at"]
    assert prosody(np.zeros(100, np.float32))["emphasis_at"] == []
    assert "empathetic" in text_hints("मुझे इसका दुख है")
    assert parse_behavior({"emotion": "empathetic", "intensity": 3, "gesture_preferences": ["gentle_nod", "x"]}) == \
        {"hints": ["gentle_nod", "empathetic"], "intensity": 1.0}


@pytest.fixture
def mock_pipeline(toy_graph, clip_len, monkeypatch):
    from avatar_system.streaming import pipeline as P
    from avatar_system.tests.test_planner_scheduler import runtime

    class FakeSource:
        def length(self, c):
            return clip_len[c]

        def clip(self, c):
            return np.zeros((clip_len[c], 8, 8, 3), np.uint8)

        def get(self, r):
            f = np.zeros((8, 8, 3), np.uint8)
            f[0, 0, 0] = r.n % 256
            return f
    monkeypatch.setattr(P, "ClipFrameSource", FakeSource)
    pipe = P.AvatarPipeline(runtime(toy_graph), NullLipSync(), StateMachine(), batch=4, out_width=0, encode=False)
    yield pipe
    pipe.stop()


def test_pipeline_continuous_numbering_states_and_cleanup(mock_pipeline):
    pipe = mock_pipeline
    pipe.start()
    pipe.push_audio(AudioEvent("s", "u1", SR, 0, tone(1.0), True))
    seen, states = [], set()
    for _ in range(150):
        f, pcm, meta = pipe.read(timeout=5)
        seen.append(meta["n"])
        states.add(meta["state"])
        assert len(pcm) == SPF * 2 and meta["audio_sample"] == meta["n"] * SPF
    assert seen == list(range(150))
    assert {"SPEAKING", "LISTENING"} <= states
    pipe.stop()
    assert all(not t.is_alive() for t in pipe.threads)  # resource cleanup


def test_pipeline_interrupt(mock_pipeline):
    pipe = mock_pipeline
    pipe.start()
    pipe.push_audio(AudioEvent("s", "u1", SR, 0, tone(4.0), True))
    for _ in range(30):
        pipe.read(timeout=5)
    pipe.sm.fire("interrupt")
    pipe.interrupt("s")
    pipe.sm.fire("cancelled")
    talking_after = 0
    for _ in range(80):
        _, _, meta = pipe.read(timeout=5)
        talking_after += meta["utt"] is not None
    # frames already planned/buffered may still carry speech (bounded by the queues), nothing after that
    assert talking_after <= pipe.batch * 2 + 2 + pipe.out_q.maxsize + pipe.batch


def test_ramp_weights_around_speech(mock_pipeline):
    pipe = mock_pipeline
    seen = []

    class Spy(NullLipSync):
        def process(self, refs, feats, source):
            seen.extend(r.n for r in refs)
            return super().process(refs, feats, source)
    pipe.backend = Spy()
    pipe.start()
    start = pipe.push_audio(AudioEvent("s", "u1", SR, 0, tone(0.4), True))
    metas = [pipe.read(timeout=5)[2] for _ in range(start + 30)]
    w = {m["n"]: m["lip_w"] for m in metas}
    assert w[start] == 1.0 and w[start - 1] == 0.8 and w[start - 4] == 0.2 and w[start - 5] == 0.0
    assert all(w[n] == 1.0 for n in range(start, start + 10))
    assert w[start + 10] == 0.8 and w[start + 14] == 0.0
    assert set(seen) == {n for n, x in w.items() if x > 0} | {n for n in seen if n >= len(metas)}


def test_feature_shift_moves_mouth_not_audio():
    """shift=-2: frame n gets the lip-sync feature of frame n-2 while its PCM stays the PCM of frame n."""
    fe = lambda pcm: np.arange(int(np.ceil(len(pcm) / SPF)), dtype=np.float32)[:, None]  # feature k == k
    tl = AudioTimeline(fe, lead_frames=0, feature_shift=-2)
    x = tone(1.0)
    s = tl.push(AudioEvent("s", "u", SR, 0, x, True))
    ref = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    for k in range(25):
        pcm, feat, uid, _ = tl.frame(s + k)
        assert pcm == ref[k * SPF * 2:(k + 1) * SPF * 2] and uid == "u"
        assert (feat is None) if k < 2 else (feat[0] == k - 2)
    pcm, feat, uid, _ = tl.frame(s + 26)  # audio over, mouth still finishing the last syllable
    assert uid is None and feat[0] == 24
