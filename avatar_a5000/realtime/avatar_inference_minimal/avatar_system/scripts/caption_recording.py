"""Burn conversation captions into a server recording (/api/record) so response timing can be seen and felt:
  - top-left clock (recording time)
  - YOU  [mm:ss.s]  <question as typed>                          (from the frame the question reached the server)
  - avatar thinking... +2.4 s                                     (live timer until the first speech frame)
  - AVATAR [mm:ss.s] responded after 3.8 s: <answer text>         (from the first speech frame)
  - "(interrupted)" when the user barged in
Hindi/Devanagari and Latin text are mixed per character run (Noto Sans Devanagari + DejaVu Sans, shaped with raqm).

  envs/analysis/bin/python avatar_system/scripts/caption_recording.py avatar_system/data/recordings/rec_XXXX
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_system.common import path, root  # noqa: E402

FPS = 25
DEV = root() / "models/fonts/NotoSansDevanagari-Regular.ttf"
LAT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LAT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def fonts(size):
    lay = ImageFont.Layout.RAQM
    return {"dev": ImageFont.truetype(str(DEV), size, layout_engine=lay),
            "lat": ImageFont.truetype(LAT, size, layout_engine=lay), "bold": ImageFont.truetype(LAT_B, size)}


def is_dev(c):
    return "ऀ" <= c <= "ॿ" or c in "‌‍"


def runs(text):
    out, cur, dev = [], "", None
    for c in text:
        d = is_dev(c) if not c.isspace() and c not in ",.?!:;-()" else dev
        if dev is None or d == dev:
            cur += c
        else:
            out.append((dev, cur))
            cur = c
        dev = d if d is not None else dev
    if cur:
        out.append((bool(dev), cur))
    return out


def draw_mixed(d, xy, text, F, fill):
    x, y = xy
    for dev, t in runs(text):
        f = F["dev"] if dev else F["lat"]
        d.text((x, y + (-4 if dev else 0)), t, font=f, fill=fill)
        x += d.textlength(t, font=f)


def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def turns_from_events(events, n_first):
    """-> list of turns {q, q_n, resp_n, delay, answer, interrupted} in recording frame numbers."""
    turns = []
    for e in events:
        n = e["n"] - n_first
        if e["type"] == "user":
            turns.append({"q": e["text"], "q_n": n, "resp_n": None, "answer": [], "interrupted": None})
        elif not turns:
            continue
        elif e["type"] == "assistant":
            turns[-1]["answer"].append(e["text"])
        elif e["type"] == "latency" and turns[-1]["resp_n"] is None:
            turns[-1]["resp_n"] = n
        elif e["type"] == "interrupt":
            turns[-1]["interrupted"] = n
    return turns


def main():
    base = Path(sys.argv[1])
    tr = json.loads(base.with_name(base.name + "_trace.json").read_text())
    turns = turns_from_events(tr["events"], tr.get("n_first", 0))
    src = cv2.VideoCapture(str(base.with_suffix(".mp4")))
    W, H = int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    F, Fs = fonts(22), fonts(18)
    bar = 176  # 150 + room for a question that wraps onto a second line
    out = base.with_name(base.name + "_captioned.mp4")
    tmp = base.with_name(base.name + "_cap_v.mp4")
    enc = subprocess.Popen([str(path("ffmpeg")), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H + bar}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, f = src.read()
        if not ok:
            break
        im = Image.new("RGB", (W, H + bar), (16, 18, 22))
        im.paste(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), (0, 0))
        d = ImageDraw.Draw(im)
        t = n / FPS
        d.rounded_rectangle((8, 8, 150, 40), 6, fill=(0, 0, 0))
        d.text((16, 12), fmt(t), font=F["bold"], fill=(255, 255, 255))
        cur = [x for x in turns if x["q_n"] <= n]
        y = H + 10
        if cur:
            tu = cur[-1]
            you = f"YOU   [{fmt(tu['q_n'] / FPS)}]   " + tu["q"]
            if d.textlength(you, font=F["lat"]) <= W - 28:
                draw_mixed(d, (14, y), you, F, (110, 190, 255))
                y += 38
            else:  # long question: wrap onto two smaller lines instead of running off the frame
                per = max(20, int(len(you) * (W - 28) / d.textlength(you, font=Fs["lat"])))
                for k, line in enumerate(textwrap.wrap(you, per)[:2]):
                    draw_mixed(d, (14, y + 24 * k), line, Fs, (110, 190, 255))
                y += 52
            if tu["interrupted"] is not None and n >= tu["interrupted"]:
                d.text((14, y), f"(you interrupted at {fmt(tu['interrupted'] / FPS)})", font=F["lat"], fill=(240, 110, 110))
            elif tu["resp_n"] is None or n < tu["resp_n"]:
                wait = (n - tu["q_n"]) / FPS
                d.text((14, y), f"avatar thinking …  +{wait:.1f} s", font=F["lat"], fill=(235, 190, 80))
            else:
                delay = (tu["resp_n"] - tu["q_n"]) / FPS
                head = f"AVATAR [{fmt(tu['resp_n'] / FPS)}]  responded after {delay:.1f} s:"
                d.text((14, y), head, font=F["lat"], fill=(120, 220, 140))
                ans = " ".join(tu["answer"])
                for k, line in enumerate(textwrap.wrap(ans, 95)[:2]):
                    draw_mixed(d, (14, y + 30 + 26 * k), line, Fs, (225, 230, 235))
        enc.stdin.write(np.asarray(im).tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    subprocess.run([str(path("ffmpeg")), "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(base.with_suffix(".mp4")),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", str(out)], check=True)
    tmp.unlink()
    for tu in turns:
        resp = f"{(tu['resp_n'] - tu['q_n']) / FPS:.2f}s" if tu["resp_n"] is not None else "-"
        print(f"[{fmt(tu['q_n'] / FPS)}] {tu['q'][:50]:50s} response after {resp}"
              + (" (interrupted)" if tu["interrupted"] is not None else ""))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
