"""LLM adapter around the existing Sarvam chat integration (app/sarvam_client.py) - same model, prompts, sentence
splitting. Optional behaviour metadata: the model may start its reply with one inline tag
  [[emotion=empathetic;intensity=0.3;gesture=gentle_nod]]
which is stripped before TTS (never spoken) and passed to the planner as hints. No second LLM call. If the model
omits or garbles the tag, keyword heuristics are used instead. The tag uses braces/colons/semicolons because the
prototype's speech cleaner strips [ ] > _ and its sentence splitter may cut at the first comma; any tag-like text
that survives is removed before TTS, so metadata can never be spoken."""
import re
import sys
import time

from avatar_system.common import root
from avatar_system.conversation.behavior_planner import parse_behavior, text_hints

sys.path.insert(0, str(root() / "app"))

SYSTEM = {  # same persona/rules as the prototype server
    "hi-IN": "You are a friendly, warm Indian digital assistant who talks to people as a video avatar of a man. "
             "You are male: always use masculine Hindi verb forms about yourself (e.g. करता हूँ, सकता हूँ, रहा हूँ). "
             "You are speaking out loud, so answer in 1-3 short, natural, conversational sentences. "
             "ALWAYS reply in Hindi written in Devanagari script (common English words like phone, doctor may be "
             "written in Devanagari too). Never use Roman/Latin script for Hindi. "
             "Start with a short first sentence. No lists, markdown, emojis or bullet points.",
    "en-IN": "You are a friendly, warm Indian digital assistant speaking to people as a video avatar of a man. "
             "You are speaking out loud: answer in 1-3 short, natural spoken sentences in simple Indian English. "
             "Start with a short first sentence. No lists, markdown, emojis or bullet points.",
}
TAG_RULE = (" Begin your reply with exactly one hidden tag like {emotion:empathetic;intensity:0.3;gesture:nod} "
            "(emotion one of neutral, empathetic, positive, thoughtful; intensity 0.0-1.0; gesture one of nod, "
            "emphasis, explain, none) and then the spoken answer. The tag is not read aloud.")
TAG = re.compile(r"^\s*[\[{(]*\s*emotion\s*[:=]\s*<?\s*([A-Za-z_]+)\W+intensity\s*[:=]\s*<?\s*([\d.]+)\W+"
                 r"gesture\s*[:=]\s*<?\s*(gentle[ _]nod|nod|emphasis|explain|neutral[ _]explanation|none)\s*[\]})>]*\s*",
                 re.I)
GESTURE = {"nod": "gentle_nod", "gentle nod": "gentle_nod", "gentle_nod": "gentle_nod", "emphasis": "emphasis",
           "explain": "neutral_explanation", "neutral explanation": "neutral_explanation",
           "neutral_explanation": "neutral_explanation"}


def parse_tag(m):
    meta = {"emotion": m.group(1).lower(), "speaking_style": None,
            "gesture_preferences": [GESTURE.get(m.group(3).lower(), "none")]}
    try:
        meta["intensity"] = float(m.group(2).rstrip("."))
    except ValueError:
        pass
    return parse_behavior(meta)


def strip_tag_like(s):
    """Last line of defence: never send anything that looks like metadata to TTS."""
    s = TAG.sub("", s)
    if re.search(r"emotion\s*[:=]|intensity\s*[:=]|gesture\s*[:=]", s, re.I):
        s = re.sub(r"^.*?(gesture\s*[:=]\s*\S+(\s+nod)?)\s*", "", s, flags=re.I | re.S)
        s = re.sub(r"(emotion|intensity|gesture)\s*[:=]\s*\S+", "", s, flags=re.I)
    return s.strip(" {}[]();")


class SarvamLLM:
    def __init__(self, client=None, behavior_tags=True):
        from sarvam_client import Sarvam
        self.client = client or Sarvam()
        self.tags = behavior_tags

    async def respond(self, history, lang="hi-IN", timing=None):
        """Yield ("behavior", dict) at most once, then ("sentence", text) items."""
        sys_prompt = SYSTEM.get(lang, SYSTEM["hi-IN"]) + (TAG_RULE if self.tags else "")
        msgs = [{"role": "system", "content": sys_prompt}] + history[-10:]
        t0 = time.time()
        first = True
        behavior_sent = False
        async for sent in self.client.sentences(msgs):
            if first and timing is not None:
                timing["llm_first_sentence_s"] = round(time.time() - t0, 3)
            if first:
                first = False
                m = TAG.match(sent)
                if m:
                    beh = parse_tag(m)
                    beh["source"] = "llm_tag"
                    beh["raw_tag"] = m.group(0).strip()  # kept in history so later turns keep the format
                    yield "behavior", beh
                    behavior_sent = True
                    sent = sent[m.end():].strip()
                    if not sent:
                        continue
            sent = strip_tag_like(sent)
            if not behavior_sent:
                yield "behavior", {"hints": text_hints(sent), "intensity": None, "source": "heuristic"}
                behavior_sent = True
            if sent:
                yield "sentence", sent
        if timing is not None:
            timing["llm_done_s"] = round(time.time() - t0, 3)


def local_client(base_url, model):
    """OpenAI-compatible local server (vLLM) behind the Sarvam client's sentence streaming. Thinking is switched off
    (Sarvam-30B chat template: enable_thinking=False -> <|nothink|>) so the first spoken words come immediately."""
    import json as _json
    import httpx
    from sarvam_client import Sarvam

    class LocalChat(Sarvam):
        def __init__(self):
            self.llm_model, self.tts_model = model, None
            self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

        async def chat_stream(self, messages, max_tokens=300, temperature=0.6):
            # Sarvam-30B FP8 still opens a <think> block with enable_thinking=False alone; prefilling the assistant
            # turn with the template's empty "<think></think>" makes it answer directly (whole reply ~0.1-0.3 s)
            body = {"model": self.llm_model, "messages": list(messages) + [{"role": "assistant", "content": ""}],
                    "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                    "add_generation_prompt": False, "continue_final_message": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            started, thinking = False, False
            async with self.http.stream("POST", f"{base_url}/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"local chat {r.status_code}: {(await r.aread()).decode()[:300]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = _json.loads(data).get("choices") or [{}]
                    delta = (choices[0].get("delta") or {}).get("content")
                    if not delta:
                        continue
                    if not started:  # leading newline / an unexpected think block never reaches TTS
                        delta = delta.lstrip()
                        if delta.startswith("<think"):
                            thinking = True
                        if not delta:
                            continue
                        started = True
                    if thinking:
                        if "</think>" in delta:
                            delta, thinking = delta.split("</think>", 1)[1].lstrip(), False
                        else:
                            continue
                    if delta:
                        yield delta
    return LocalChat()


class MockLLM:
    async def respond(self, history, lang="hi-IN", timing=None):
        yield "behavior", {"hints": ["gentle_nod"], "intensity": 0.3, "source": "mock"}
        for s in ("नमस्ते, मैं आपकी मदद करूँगा।", "यह एक परीक्षण उत्तर है।"):
            yield "sentence", s
