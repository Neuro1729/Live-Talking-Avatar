"""Phase 6 eval: render the held-out test clips with MuseTalk (stock 1.5 vs fine-tuned UNets), score real + renders
with SyncNet (fast_sync.py: AV offset + confidence vs the realigned audio) and make real|model|model side-by-sides.

Each test clip is driven by its own (realigned) audio with its own video as the source frames, i.e. the upper face is
real and only the mouth region is generated. Models load one at a time on one GPU (host-RAM rule).

  ../envs/analysis/bin/python scripts/eval_sync.py --gpu 0 --models stock=../MuseTalk/models/musetalkV15/unet.pth \
      s2_5000=../MuseTalk/exp_out/shubh/shubh_s2/unet-5000.pth
Output: eval/<tag>/{videos,sync}/, eval/sbs/<clip>.mp4, eval/summary.json
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

FT = Path(__file__).resolve().parents[1]
R = FT.parent
MT = R / "MuseTalk"
CLIPS = MT / "dataset/shubh/clips"
FFMPEG = R / "envs/musetalk/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SYNC_PY = "/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/ComfyUI/.venv/bin/python"
SYNC_CWD = R / "avatar_system/data/eval/syncnet/cwd"
CROP = "crop=400:460:318:50"  # face region of the 1024x928 frames (meta face box ~[410,120,627,403])


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd))[:200], flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--models", nargs="+", required=True, help="tag=unet.pth")
    ap.add_argument("--out", default=str(FT / "eval"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qc = json.load(open(MT / "dataset/shubh/qc.json"))
    clips = [r["name"] for r in qc if r["keep"] and r["split"] == "test"]
    models = dict(m.split("=", 1) for m in a.models)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PATH=f"{FFMPEG.parent}:{os.environ['PATH']}")
    ffbin = out / "bin"
    ffbin.mkdir(exist_ok=True)
    if not (ffbin / "ffmpeg").exists():
        (ffbin / "ffmpeg").symlink_to(FFMPEG)
    env["PATH"] = f"{ffbin}:{env['PATH']}"

    # 1. render, one model at a time (each inference.py process loads exactly one UNet)
    tasks = out / "tasks.yaml"
    tasks.write_text("".join(f'task_{i}:\n video_path: "{CLIPS / c}.mp4"\n audio_path: "{CLIPS / c}.wav"\n'
                             for i, c in enumerate(clips)))
    for tag, unet in models.items():
        vids = out / tag / "videos"
        if all((vids / f"{c}.mp4").exists() for c in clips):
            continue
        res = out / tag / "render"
        run([R / "envs/musetalk/bin/python", "-m", "scripts.inference", "--inference_config", tasks,
             "--result_dir", res, "--unet_model_path", Path(unet).resolve(),
             "--unet_config", "./models/musetalkV15/musetalk.json", "--version", "v15", "--gpu_id", "0"],
            cwd=MT, env=env)
        vids.mkdir(parents=True, exist_ok=True)
        for c in clips:
            shutil.copy(res / "v15" / f"{c}_{c}.mp4", vids / f"{c}.mp4")

    # 2. SyncNet on the real clips and every model's renders
    scored = {"real": CLIPS, **{t: out / t / "videos" for t in models}}
    for tag, vdir in scored.items():
        sdir = out / tag / "sync"
        todo = [c for c in clips if not (sdir / f"{c}.json").exists()]
        if todo:
            run([SYNC_PY, FT / "scripts/fast_sync.py", "--video-dir", vdir, "--audio-dir", CLIPS, "--out", sdir,
                 "--clips", *todo], cwd=SYNC_CWD, env=dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu))

    # 3. side-by-sides: real | model1 | model2 ... (face crop) with the driving audio
    sbs = out / "sbs"
    sbs.mkdir(exist_ok=True)
    for c in clips:
        ins = [CLIPS / f"{c}.mp4"] + [out / t / "videos" / f"{c}.mp4" for t in models]
        n = len(ins)
        filt = ";".join(f"[{i}:v]{CROP},setpts=PTS-STARTPTS[v{i}]" for i in range(n))
        filt += ";" + "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[v]"
        cmd = [FFMPEG, "-y", "-loglevel", "error"]
        for p in ins:
            cmd += ["-i", p]
        cmd += ["-i", CLIPS / f"{c}.wav", "-filter_complex", filt, "-map", "[v]", "-map", f"{n}:a",
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                sbs / f"{c}__{'_'.join(['real', *models])}.mp4"]
        run(cmd)

    # 4. summary
    summ = {}
    for tag in scored:
        rs = [json.load(open(out / tag / "sync" / f"{c}.json")) for c in clips]
        summ[tag] = {"conf_mean": round(float(np.mean([r["conf"] for r in rs])), 3),
                     "offset_mean": round(float(np.mean([r["offset"] for r in rs])), 2),
                     "offset_0_frac": round(float(np.mean([r["offset"] == 0 for r in rs])), 2),
                     "per_clip": {r["name"]: [r["offset"], r["conf"]] for r in rs}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    for tag, s in summ.items():
        print(f"{tag:>10}: SyncNet conf {s['conf_mean']:.2f}, mean offset {s['offset_mean']:+.2f} frames, "
              f"offset 0 on {s['offset_0_frac'] * 100:.0f}% of clips")


if __name__ == "__main__":
    main()
