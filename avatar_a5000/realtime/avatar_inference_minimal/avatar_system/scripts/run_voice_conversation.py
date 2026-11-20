"""Spoken live conversation against a running server (run_conversation.sh), recorded by the server itself, no captions.

The client plays the user: each question is synthesized once with a second Sarvam voice (--user-speaker) and sent as
audio to /api/voice. The server plays it in real time (avatar listening, user voice written into the recording),
runs Sarvam speech-to-text when it ends, then LLM -> TTS -> lip-sync as for any turn. The client only waits for each
answer to finish before speaking again, so every gap in the recording is real processing / reply time. No interrupt.
  envs/analysis/bin/python avatar_system/scripts/run_voice_conversation.py --port 8782 \
      --questions avatar_system/data/conversations/voice3_q.json --user-speaker rohan
Output: the server recording data/recordings/rec_*.mp4 (+ _trace.json with user_audio / stt / user / assistant events).
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir, root  # noqa: E402
from avatar_system.scripts.run_conversation_test import call, wait_state  # noqa: E402

SR, FPS = 16000, 25


def tts(text, speaker, lang):
    key = next(l.split("=", 1)[1].strip().strip("\"'") for l in open(root().parent / ".env")
               if l.lower().startswith("sarvam_api_key"))
    body = {"text": text, "target_language_code": lang, "model": "bulbul:v3", "speaker": speaker,
            "speech_sample_rate": SR, "pace": 1.0}
    req = urllib.request.Request("https://api.sarvam.ai/text-to-speech", data=json.dumps(body).encode(),
                                 headers={"api-subscription-key": key, "content-type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return base64.b64decode(r["audios"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--questions", required=True, help="JSON file: [[text, lang], ...]")
    ap.add_argument("--user-speaker", default="rohan", help="Sarvam bulbul:v3 voice that asks the questions")
    ap.add_argument("--pause", type=float, default=1.2, help="s the user waits after an answer before speaking")
    a = ap.parse_args()
    questions = json.loads(Path(a.questions).read_text())
    qdir = data_dir("conversations") / "voice_q"
    qdir.mkdir(parents=True, exist_ok=True)
    wavs = []
    for k, (q, lang) in enumerate(questions):
        wp = qdir / f"{Path(a.questions).stem}_{a.user_speaker}_{k}.wav"
        if not wp.exists():
            wp.write_bytes(tts(q, a.user_speaker, lang))
        wavs.append(wp.read_bytes())
        print(f"q{k} {sf.info(io.BytesIO(wavs[-1])).duration:.1f}s  {q}", flush=True)

    base = f"http://127.0.0.1:{a.port}"
    wait_state(base, "LISTENING", 30)
    rec = call(base, "/api/record/start", {})
    print("recording", rec, flush=True)
    time.sleep(3.0)  # listening at the start
    for k, (q, lang) in enumerate(questions):
        r = call(base, "/api/voice", {"wav_b64": base64.b64encode(wavs[k]).decode(), "lang": lang})
        print(f"q{k} spoken ({r.get('duration_s')} s)", flush=True)
        if wait_state(base, "SPEAKING", 60, min_wait=r.get("duration_s", 0)) is None:
            print("WARNING: never started speaking:", q, flush=True)
        wait_state(base, "LISTENING", 120)
        time.sleep(a.pause)
    time.sleep(2.0)
    out = call(base, "/api/record/stop", {})
    stem = Path(out["video"]).with_suffix("")
    tr = json.loads(Path(str(stem) + "_trace.json").read_text())
    n0 = tr.get("n_first", 0)
    print("\nrecording timeline (s):")
    for e in tr["events"]:
        if e.get("type") in ("user_audio", "stt", "assistant", "latency"):
            extra = {k: e[k] for k in ("duration_s", "stt_s", "first_audible_s") if k in e}
            print(f"  [{(e['n'] - n0) / FPS:6.1f}] {e['type']:10s} {extra or ''} {e.get('text', '')}")
    # end of the user's question -> first avatar audio, per stage (what the user actually waits)
    m = call(base, "/api/metrics")
    turns = m["latency_turns"][-len(questions):]
    ua = [e for e in tr["events"] if e.get("type") == "user_audio"]
    st = [e for e in tr["events"] if e.get("type") == "stt"]
    la = [e for e in tr["events"] if e.get("type") == "latency"]
    utt = tr.get("utt", [])
    rows = []
    for k in range(min(len(ua), len(st), len(la), len(turns))):
        end_q = (ua[k]["n"] - n0) / FPS + ua[k]["duration_s"]
        t = turns[k]
        nxt = (ua[k + 1]["n"] - n0) if k + 1 < len(ua) else len(utt)
        ack = next((i for i in range(int(end_q * FPS), min(nxt, len(utt))) if (utt[i] or "").startswith("ack")), None)
        rows.append({"stt_s": st[k]["stt_s"], "llm_first_sentence_s": t.get("llm_first_sentence_s"),
                     "first_tts_done_s": t.get("first_tts_done_s"), "first_audible_s": t.get("first_audible_s"),
                     "ack_audible_s": round(ack / FPS - end_q, 2) if ack is not None else None,
                     "question_end_to_audible_s": round((la[k]["n"] - n0) / FPS - end_q, 2)})
    print("\nlatency per turn (s):")
    for r in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    if rows:
        tot = [r["question_end_to_audible_s"] for r in rows]
        print(f"  question end -> avatar audible: mean {sum(tot) / len(tot):.2f} s, max {max(tot):.2f} s")
    Path(str(stem) + "_latency.json").write_text(json.dumps(rows, indent=1))
    print("->", out["video"])


if __name__ == "__main__":
    main()
