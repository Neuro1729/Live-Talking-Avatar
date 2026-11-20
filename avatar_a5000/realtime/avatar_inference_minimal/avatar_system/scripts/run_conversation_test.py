"""Scripted live conversation against a running server (run_conversation.sh): sends questions, waits for each answer
to finish (state back to LISTENING), interrupts one answer, records the whole session (/api/record) and writes latency
p50/p95 per stage to data/eval/conversation_latency.json.

  envs/analysis/bin/python avatar_system/scripts/run_conversation_test.py --port 8780 [--repeat 2]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import data_dir  # noqa: E402

QUESTIONS = [
    ("नमस्ते! आप कौन हैं?", "hi-IN"),
    ("मुझे रात को नींद नहीं आती, क्या करूँ?", "hi-IN"),
    ("Mera blood sugar thoda high aa raha hai, kya karna chahiye?", "hi-IN"),
    ("Can you tell me how to stay healthy in summer?", "en-IN"),
    ("मेरी माँ की तबीयत ठीक नहीं है, बहुत चिंता हो रही है।", "hi-IN"),
    ("मुझे नई नौकरी मिल गई!", "hi-IN"),
]


def call(base, p, body=None):
    req = urllib.request.Request(base + p, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"}, method="POST" if body is not None else "GET")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def wait_state(base, want, timeout=60, min_wait=0.0):
    t0 = time.time()
    time.sleep(min_wait)
    while time.time() - t0 < timeout:
        if call(base, "/api/state")["state"] == want:
            return time.time() - t0
        time.sleep(0.2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--questions", default="", help="JSON file: [[text, lang], ...] instead of the built-in list")
    ap.add_argument("--interrupt", type=int, default=1, help="index of the question whose answer is interrupted "
                    "(first round only; -1 = none)")
    ap.add_argument("--tag", default="", help="report name suffix: data/eval/conversation_latency_<tag>.json")
    a = ap.parse_args()
    questions = json.loads(Path(a.questions).read_text()) if a.questions else QUESTIONS
    base = f"http://127.0.0.1:{a.port}"
    rec = call(base, "/api/record/start", {})
    print("recording", rec, flush=True)
    time.sleep(4)  # idle/listening motion at the start
    n = 0
    for r in range(a.repeat):
        for k, (q, lang) in enumerate(questions):
            call(base, "/api/user_speaking", {})
            time.sleep(1.5)                          # "user speaking" -> listening motion
            call(base, "/api/message", {"text": q, "lang": lang})
            n += 1
            if r == 0 and k == a.interrupt:          # barge-in test
                wait_state(base, "SPEAKING", 30)
                time.sleep(2.0)
                print("interrupt ->", call(base, "/api/interrupt", {}), flush=True)
                wait_state(base, "LISTENING", 30)
                continue
            if wait_state(base, "SPEAKING", 40) is None:
                print("WARNING: never started speaking:", q, flush=True)
            wait_state(base, "LISTENING", 90)
            time.sleep(2.5)
    time.sleep(3)
    out = call(base, "/api/record/stop", {})
    m = call(base, "/api/metrics")
    turns = [t for t in m["latency_turns"] if "first_audible_s" in t]
    keys = ["llm_first_sentence_s", "tts_first_chunk_s", "first_tts_done_s", "first_audible_s", "llm_done_s", "total_s"]
    rep = {"turns": len(turns), "questions": n, "video": out.get("video"),
           "render_fps_capacity": m["render_fps_capacity"], "render_ms_per_frame": m["render_ms_per_frame"],
           "out_interval_ms": m["out_interval_ms"], "counters": m["counters"], "audio": m["audio"],
           "gpu_peak_mem_gb": m.get("gpu_peak_mem_gb"), "rss_peak_gb": m.get("rss_peak_gb"),
           "behavior_sources": [t.get("behavior", {}).get("source") for t in m["latency_turns"]],
           "behavior_hints": [t.get("behavior", {}).get("hints") for t in m["latency_turns"]]}
    for k in keys:
        v = [t[k] for t in turns if t.get(k) is not None]
        if v:
            rep[k] = {"p50": round(float(np.percentile(v, 50)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                      "n": len(v)}
    # planner->audible gap: first audible minus first TTS done = video/audio pipeline latency
    gaps = [t["first_audible_s"] - t["first_tts_done_s"] for t in turns if t.get("first_tts_done_s")]
    if gaps:
        rep["pipeline_after_tts_s"] = {"p50": round(float(np.percentile(gaps, 50)), 2),
                                       "p95": round(float(np.percentile(gaps, 95)), 2)}
    if out.get("video"):  # burn question/answer captions with timing into a copy of the recording
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name("caption_recording.py")),
                        out["video"][:-len(".mp4")]], check=False)
        rep["captioned_video"] = out["video"][:-len(".mp4")] + "_captioned.mp4"
    (data_dir("eval") / f"conversation_latency{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
