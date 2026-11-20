"""Compose the full talking-bank script set (configs/talk_scripts_full.yaml): 10 speech acts x 6 takes = 60 clips.
Each take samples 1-3 sentences of its act (Hindi / Hinglish / English) with varied pauses, so lengths (about 4-14 s),
rhythms and wording differ; behaviours are mostly calm with some gestures, never big movements.

  envs/analysis/bin/python -m avatar_system.generation.make_full_scripts
"""
import random

import yaml

from avatar_system.common import CONFIGS

H, E = "hi-IN", "en-IN"
POOLS = {
    "greeting": [("नमस्ते! मैं आपका डिजिटल सहायक हूँ।", H), ("नमस्ते, आपसे मिलकर अच्छा लगा।", H),
                 ("Hello! Main aapki madad ke liye yahan hoon.", H), ("Hi, welcome! How can I help you today?", E),
                 ("आज मैं आपकी किस तरह मदद कर सकता हूँ?", H), ("Aap mujhse kuch bhi pooch sakte hain.", H)],
    "explain": [("डायबिटीज़ में शरीर शुगर को ठीक से इस्तेमाल नहीं कर पाता।", H),
                ("Regular exercise se blood pressure control mein rehta hai.", H),
                ("Sleep is when your body repairs itself, so it really matters.", E),
                ("पानी कम पीने से थकान और सिरदर्द हो सकता है।", H),
                ("Vitamin D ke liye subah ki dhoop sabse achhi hoti hai.", H),
                ("Stress affects digestion more than most people think.", E)],
    "enumerate_steps": [("पहला, सुबह खाली पेट पानी पिएं।", H), ("दूसरा, रोज़ आधा घंटा टहलें।", H),
                        ("और तीसरा, रात को जल्दी सो जाएँ।", H), ("Pehle, apna sugar roz check kijiye.", H),
                        ("Phir, meetha aur maida kam kar dijiye.", H), ("First, drink enough water. Second, walk daily.", E)],
    "advise_emphasis": [("सबसे ज़रूरी है कि आप दवा समय पर लें।", H),
                        ("Blood sugar ke liye sabse zaroori hai ki aap meetha kam karein.", H),
                        ("Please don't skip your follow-up visit, it really is important.", E),
                        ("तेज़ बुखार हो तो डॉक्टर से ज़रूर मिलिए।", H),
                        ("Khaali pet dawa bilkul mat lijiye.", H), ("Always read the label before taking any medicine.", E)],
    "reassure": [("मैं आपकी चिंता समझ सकता हूँ।", H), ("आप हिम्मत रखिए, सब ठीक हो जाएगा।", H),
                 ("Ghabraiye mat, ye bahut common problem hai.", H), ("Don't worry, this is quite manageable.", E),
                 ("आप अकेले नहीं हैं, मैं आपकी मदद करूँगा।", H), ("Sahi ilaaj se ye jaldi theek ho jaata hai.", H)],
    "congratulate": [("Arre wah, bahut bahut badhai ho!", H), ("ये सुनकर मुझे सच में बहुत खुशी हुई।", H),
                     ("That's wonderful news, congratulations!", E), ("आपकी मेहनत रंग लाई है।", H),
                     ("Bahut badhiya, aise hi aage badhte rahiye.", H), ("मेरी तरफ़ से ढेरों शुभकामनाएँ।", H)],
    "think_then_answer": [("हम्म, अच्छा सवाल है।", H), ("देखिए, ये आपकी उम्र और आदतों पर निर्भर करता है।", H),
                          ("Hmm, let me think about that for a second.", E), ("Dekhiye, iska jawab thoda alag hai.", H),
                          ("आमतौर पर, थोड़ा आराम करने से फ़र्क पड़ता है।", H),
                          ("It depends, but usually a small change in diet helps.", E)],
    "ask_back": [("आपको ये परेशानी कब से है?", H), ("Kya aapko bukhaar bhi hai?", H),
                 ("Could you tell me a little more about it?", E), ("क्या आप कोई दवा पहले से ले रहे हैं?", H),
                 ("Aapki umar kitni hai, agar aap bata sakein?", H), ("Is it worse in the morning or at night?", E)],
    "acknowledge": [("जी।", H), ("जी हाँ, समझ गया।", H), ("Hmm, okay.", E), ("अच्छा, अच्छा।", H),
                    ("Haan ji, bilkul.", H), ("Right, I see.", E)],
    "closing": [("अपना ध्यान रखिए।", H), ("Koi aur sawaal ho toh zaroor poochiye.", H),
                ("Take care, and feel free to come back anytime.", E), ("आपका दिन अच्छा हो।", H),
                ("Dhanyavaad, jaldi theek ho jaiye.", H), ("मुझसे बात करने के लिए धन्यवाद।", H)],
}
BEHAVIOR = {  # act -> behaviours to rotate through (mostly calm)
    "greeting": ["warm", "calm", "expressive"], "explain": ["calm", "gesture", "expressive"],
    "enumerate_steps": ["gesture", "calm"], "advise_emphasis": ["gesture", "expressive", "calm"],
    "reassure": ["calm", "calm", "warm"], "congratulate": ["warm", "expressive"],
    "think_then_answer": ["thinking", "calm"], "ask_back": ["calm", "expressive"],
    "acknowledge": ["listening"], "closing": ["warm", "calm"],
}
EMOTION = {"greeting": "warm", "explain": "neutral", "enumerate_steps": "neutral", "advise_emphasis": "serious",
           "reassure": "empathetic", "congratulate": "positive", "think_then_answer": "thoughtful",
           "ask_back": "curious", "acknowledge": "attentive", "closing": "warm"}


def main(takes=6, seed=7):
    rng = random.Random(seed)
    base = yaml.safe_load((CONFIGS / "talk_scripts.yaml").read_text())
    clips = []
    for act, pool in POOLS.items():
        for t in range(takes):
            n_sent = 1 if act == "acknowledge" else rng.choice([1, 2, 2, 3])
            sents = rng.sample(pool, min(n_sent, len(pool)))
            if act == "enumerate_steps" and n_sent > 1:
                sents = sorted(sents, key=pool.index)  # keep first/second/third order
            parts = [{"pause": round(rng.uniform(0.5, 1.2), 2)}]
            for k, (txt, lang) in enumerate(sents):
                parts.append({"say": txt, "lang": lang})
                parts.append({"pause": round(rng.uniform(0.4, 0.9) if k < len(sents) - 1 else rng.uniform(1.0, 1.8), 2)})
            if act == "acknowledge":  # long attentive silences around short replies
                parts = [{"pause": round(rng.uniform(1.8, 3.0), 2)}, parts[1], {"pause": round(rng.uniform(1.5, 2.5), 2)},
                         {"say": rng.choice(pool)[0], "lang": H}, {"pause": round(rng.uniform(2.0, 3.0), 2)}]
            clips.append({"name": f"tb_{act}_{t + 1:02d}", "behavior": BEHAVIOR[act][t % len(BEHAVIOR[act])],
                          "act": act, "emotion": EMOTION[act], "parts": parts})
    out = {"behaviors": base["behaviors"], "clips": clips}
    (CONFIGS / "talk_scripts_full.yaml").write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200))
    print(f"{len(clips)} clips -> configs/talk_scripts_full.yaml")


if __name__ == "__main__":
    main()
