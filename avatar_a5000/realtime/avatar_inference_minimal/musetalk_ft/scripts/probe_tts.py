"""Probe which Sarvam /text-to-speech options bulbul:v3 + shubh accepts (enable_qc etc.)."""
import base64, sys, httpx
sys.path.insert(0, "app")
from sarvam_client import api_key
H = {"api-subscription-key": api_key()}
base = {"text": "नमस्ते, आज का weather तो बहुत pleasant है।", "target_language_code": "hi-IN", "model": "bulbul:v3",
        "speaker": "shubh", "speech_sample_rate": 16000}
for name, extra in [("plain", {}), ("qc", {"enable_qc": True}), ("qc_vad", {"enable_qc": True, "enable_vad": True}),
                    ("bogus", {"definitely_not_a_param": True})]:
    r = httpx.post("https://api.sarvam.ai/text-to-speech", json=base | extra, headers=H, timeout=90)
    print(name, r.status_code, r.text[:200] if r.status_code != 200 else "ok", flush=True)
    if r.status_code == 200:
        open(f"musetalk_ft/data/test_tts/{name}.wav", "wb").write(base64.b64decode("".join(r.json()["audios"])))
