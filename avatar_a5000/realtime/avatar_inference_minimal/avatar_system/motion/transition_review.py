"""Transition review: render evidence for joins (contact sheet + playback video) and a small web UI to approve,
reject, override cost, mark segments low quality or relabel them. Decisions persist in data/reviews.json and win over
automatic scores on every rebuild.

  envs/analysis/bin/python -m avatar_system.motion.transition_review render [--status review] [--worst 20] [--sample 20]
  envs/analysis/bin/python -m avatar_system.motion.transition_review serve --port 8770
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from functools import lru_cache

import cv2
import numpy as np

from avatar_system.common import clip_path, data_dir, path

CTX = 30  # frames of context on each side of the cut in review videos


@lru_cache(maxsize=12)
def frames(clip):
    cap, out = cv2.VideoCapture(str(clip_path(clip))), []
    while (ok := cap.read())[0]:
        out.append(ok[1])
    return out


def tid(key):
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def load_graph():
    g = json.loads((data_dir() / "motion_graph.json").read_text())
    t = json.loads((data_dir() / "transitions.json").read_text())
    return g, t


def render(t, segs, force=False):
    d = data_dir("review")
    base = d / tid(t["key"])
    if base.with_suffix(".mp4").exists() and not force:
        return base
    A, B = segs[t["src"]], segs[t["dst"]]
    fa, fb = frames(A["clip"]), frames(B["clip"])
    a, b = A["e"], B["s"]
    # sheet: last 4 of A | first 4 of B, face crops on top, full frames below
    crop = lambda f: cv2.resize(f[90:470, 322:702], (190, 190))
    small = lambda f: cv2.resize(f, (190, 172))
    seq = [fa[i] for i in range(a - 3, a + 1)] + [fb[i] for i in range(b, b + 4)]
    top = np.hstack([crop(f) for f in seq])
    bot = np.hstack([small(f) for f in seq])
    sheet = np.vstack([top, bot])
    cv2.line(sheet, (4 * 190, 0), (4 * 190, sheet.shape[0]), (0, 0, 255), 2)
    cv2.putText(sheet, f"{t['key']}  cost {t['cost']:.2f} [{t['status']}]", (5, 18), 0, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(base.with_suffix(".jpg")), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # video: A context then B context at real speed; right half shows the amplified frame difference
    pre = [fa[i] for i in range(max(a - CTX, 0), a + 1)]
    post = [fb[i] for i in range(b, min(b + CTX, len(fb)))]
    seq = pre + post
    h, w = seq[0].shape[:2]
    tmp = base.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w // 2 * 2, h // 2))
    prev = seq[0]
    for k, f in enumerate(seq):
        diff = cv2.convertScaleAbs(cv2.absdiff(f, prev), alpha=6)
        prev = f
        left = cv2.resize(f, (w // 2, h // 2))
        right = cv2.resize(diff, (w // 2, h // 2))
        if k == len(pre):
            cv2.rectangle(left, (0, 0), (w // 2 - 1, h // 2 - 1), (0, 0, 255), 6)
        cv2.putText(left, "A" if k < len(pre) else "B", (10, 30), 0, 1, (0, 0, 255), 2)
        vw.write(np.hstack([left, right]))
    vw.release()
    subprocess.run([str(path("ffmpeg")), "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", str(base.with_suffix(".mp4"))], check=True)
    tmp.unlink()
    return base


def pick(trans, status=None, worst=0, sample=0, seed=0):
    out = [t for t in trans if t["status"] == status] if status else []
    appr = sorted((t for t in trans if t["status"] == "approved"), key=lambda t: -t["cost"])
    out += appr[:worst]
    rng = random.Random(seed)
    rej = [t for t in trans if t["status"] == "rejected"]
    out += rng.sample(appr, min(sample, len(appr))) + rng.sample(rej, min(sample // 2, len(rej)))
    seen, res = set(), []
    for t in out:
        if t["key"] not in seen:
            seen.add(t["key"])
            res.append(t)
    return res


def cmd_render(a):
    g, trans = load_graph()
    sel = pick(trans, a.status, a.worst, a.sample)
    t0 = time.time()
    sel.sort(key=lambda t: (g["segments"][t["src"]]["clip"], g["segments"][t["dst"]]["clip"]))
    for k, t in enumerate(sel):
        render(t, g["segments"])
        if k % 20 == 0:
            print(f"  {k}/{len(sel)} rendered ({time.time() - t0:.0f}s)", flush=True)
    (data_dir() / "review_queue.json").write_text(json.dumps([t["key"] for t in sel]))
    print(f"rendered {len(sel)} transitions -> {data_dir('review')}")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Transition review</title><style>
body{font:14px system-ui;margin:16px;background:#111;color:#ddd} .t{border:1px solid #333;margin:10px 0;padding:8px}
img{max-width:100%} video{width:48%} button{margin:2px;padding:4px 10px} .approved{color:#6c6} .rejected{color:#e66}
.review{color:#ec4} .manual{font-weight:bold;text-decoration:underline} code{color:#9cf}</style></head><body>
<h2>Transition review</h2><div>filter: <select id=f onchange=load()><option>queue</option><option>review</option>
<option>approved</option><option>rejected</option><option>manual</option></select> <button onclick=rebuild()>rebuild graph</button>
<span id=msg></span></div><div id=list></div><script>
async function load(){const r=await fetch('/api/list?f='+f.value);const j=await r.json();list.innerHTML=j.map(t=>`
<div class=t><div><b class="${t.status} ${t.manual?'manual':''}">${t.status}</b> cost <b>${t.cost.toFixed(2)}</b> <code>${t.key}</code>
${t.reasons.length?' hard: '+t.reasons.join(', '):''}<br><small>${JSON.stringify(t.comp)}</small></div>
${t.has_media?`<img src="/media/${t.id}.jpg"><br><video controls loop src="/media/${t.id}.mp4"></video>`:'<i>not rendered</i>'}
<div><button onclick="dec('${t.key}','approve')">approve</button><button onclick="dec('${t.key}','reject')">reject</button>
<button onclick="dec('${t.key}','clear')">clear</button> cost override <input size=4 id="c${t.id}"><button onclick="cost('${t.key}','${t.id}')">set</button>
| segment A <code>${t.a}</code> <button onclick="seg('${t.a}','low_quality')">A low quality</button>
 B <code>${t.b}</code> <button onclick="seg('${t.b}','low_quality')">B low quality</button>
 labels B <input size=18 id="l${t.id}" placeholder="listen,speak"><button onclick="lab('${t.b}','${t.id}')">set</button></div></div>`).join('')}
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});msg.textContent=(await r.json()).msg;load()}
function dec(k,d){post('/api/transition',{key:k,decision:d})}
function cost(k,i){post('/api/transition',{key:k,cost:parseFloat(document.getElementById('c'+i).value)})}
function seg(k,flag){post('/api/segment',{key:k,low_quality:true})}
function lab(k,i){post('/api/segment',{key:k,labels:document.getElementById('l'+i).value.split(',').map(s=>s.trim()).filter(Boolean)})}
async function rebuild(){msg.textContent='rebuilding...';const r=await fetch('/api/rebuild',{method:'POST'});msg.textContent=(await r.json()).msg;load()}
load()</script></body></html>"""


def cmd_serve(a):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI()
    rp = data_dir() / "reviews.json"

    def reviews():
        r = json.loads(rp.read_text()) if rp.exists() else {}
        for k in ("transitions", "segments", "clips"):
            r.setdefault(k, {})
        return r

    @app.get("/")
    def index():
        return HTMLResponse(PAGE)

    @app.get("/api/list")
    def lst(f: str = "queue"):
        g, trans = load_graph()
        segs = g["segments"]
        q = data_dir() / "review_queue.json"
        if f == "queue":
            keys = set(json.loads(q.read_text())) if q.exists() else set()
            sel = [t for t in trans if t["key"] in keys]
        elif f == "manual":
            sel = [t for t in trans if t["manual"]]
        else:
            sel = [t for t in trans if t["status"] == f][:200]
        sel.sort(key=lambda t: -t["cost"])
        return [{**t, "id": tid(t["key"]), "a": segs[t["src"]]["key"], "b": segs[t["dst"]]["key"],
                 "has_media": (data_dir("review") / f"{tid(t['key'])}.mp4").exists()} for t in sel]

    @app.get("/media/{name}")
    def media(name: str):
        return FileResponse(data_dir("review") / name)

    @app.post("/api/transition")
    def set_t(body: dict):
        r = reviews()
        k = body["key"]
        if body.get("decision") == "clear":
            r["transitions"].pop(k, None)
        else:
            e = r["transitions"].setdefault(k, {})
            if body.get("decision"):
                e["decision"] = body["decision"]
            if body.get("cost") is not None:
                e["cost"] = body["cost"]
            e["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rp.write_text(json.dumps(r, indent=1))
        return {"msg": f"saved {k} (rebuild graph to apply)"}

    @app.post("/api/segment")
    def set_s(body: dict):
        r = reviews()
        e = r["segments"].setdefault(body["key"], {})
        for f in ("low_quality", "labels"):
            if f in body:
                e[f] = body[f]
        rp.write_text(json.dumps(r, indent=1))
        return {"msg": f"saved segment {body['key']} (rebuild graph to apply)"}

    @app.post("/api/rebuild")
    def rebuild():
        from avatar_system.motion.graph_builder import build
        from avatar_system.motion.graph_validator import validate
        g, _ = build(verbose=False)
        v = validate(verbose=False)
        return {"msg": f"graph {g['version']}: {g['counts']}; validation ok={v['ok']} {v['errors']}"}

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--status", default="review")
    r.add_argument("--worst", type=int, default=20)
    r.add_argument("--sample", type=int, default=20)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    {"render": cmd_render, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
