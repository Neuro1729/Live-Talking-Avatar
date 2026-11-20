"""Phase 2: LTX-2.5 audio-to-video for every clip of data/clips.json (same recipe as LTX25_A2V_start_end_home:
ltx_flf.py, home.png first+last guides 1.0, 1024x928 @ 25 fps, frozen shubh audio).

One worker per running ComfyUI port (GPU 1 = 8191, GPU 3 = 8192) pulls the next clip that is neither done nor
claimed by the other worker (lock files), so both GPUs share one queue and the script can simply be re-run.
Output: data/raw_videos/<clip>.mp4 (+ .json from ltx_flf.py) and data/videos/<clip>.mp4 (muxed with clip audio).

  ../envs/analysis/bin/python scripts/gen_video.py --port 8191 &   ../envs/analysis/bin/python scripts/gen_video.py --port 8192
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

FT = Path(__file__).resolve().parents[1]
R = FT.parent
RAW, VID, LOCK = FT / "data/raw_videos", FT / "data/videos", FT / "data/locks"
FFMPEG = R / "envs/musetalk/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
PROMPTS = ["base"] * 3 + ["expressive", "calm"]   # 60 / 20 / 20


def h(name, salt):
    return int(hashlib.md5(f"{salt}:{name}".encode()).hexdigest(), 16)


def order(clips):
    """Interleave test clips (1 per 10) so early stops still leave an eval set."""
    train = [c for c in clips if c["split"] == "train"]
    test = [c for c in clips if c["split"] == "test"]
    out = []
    while train or test:
        out += train[:9]
        train = train[9:]
        if test:
            out.append(test.pop(0))
    return out


def claim(name):
    try:
        fd = os.open(LOCK / f"{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    for d in (RAW, VID, LOCK):
        d.mkdir(parents=True, exist_ok=True)
    n_done, failed = 0, set()
    while True:  # re-read clips.json each time: gen_audio.py assemble appends clips while we run
        todo = [c for c in order(json.loads((FT / "data/clips.json").read_text()))
                if not (VID / f"{c['name']}.mp4").exists() and not (LOCK / f"{c['name']}.lock").exists() and c["name"] not in failed]
        c = next((c for c in todo if claim(c["name"])), None)
        if c is None:
            break
        name = c["name"]
        try:
            style = PROMPTS[h(name, "style") % len(PROMPTS)]
            seed = h(name, "seed") % 100000
            t0 = time.time()
            r = subprocess.run(["python3", "-u", "ltx_flf.py", "--name", name, "--port", str(a.port),
                                "--prompt-file", str(FT / f"configs/prompt_{style}.txt"), "--seed", str(seed),
                                "--audio", f"{name}.wav", "--frames", str(c["frames"]), "--out-dir", str(RAW)],
                               cwd=R / "scripts")
            if r.returncode or not (RAW / f"{name}.mp4").exists():
                print(f"[{a.port}] FAILED {name}", flush=True)
                failed.add(name)  # don't spin on it; a re-run retries
                continue
            subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(RAW / f"{name}.mp4"),
                            "-i", str(FT / f"data/audio/{name}.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", str(VID / f"{name}.mp4")], check=True)
            meta = json.loads((RAW / f"{name}.json").read_text())
            meta.update(style=style, split=c["split"], parts=c["parts"])
            (RAW / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            n_done += 1
            print(f"[{a.port}] DONE {name} ({style}, seed {seed}, {c['frames']}f) in {time.time() - t0:.0f}s "
                  f"| {n_done} this worker", flush=True)
        finally:
            (LOCK / f"{name}.lock").unlink(missing_ok=True)
    print(f"[{a.port}] WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
