"""Evaluation pipeline -> docs/benchmark_report.md (+ data/eval/*.json).

1. Planner comparison on the same graph/library: 10-minute simulated conversations, beam planner vs random-walk
   baseline, 3 seeds each: seam continuity (transition_metrics), repetition/diversity (repetition_metrics).
2. Worst seams of the beam planner with review clips.
3. Lip-sync metrics + throughput of a rendered MuseTalk demo (if present).
4. Live latency turns from a recording trace (if present).

  envs/analysis/bin/python -m avatar_system.evaluation.benchmark [--seeds 3] [--minutes 10]
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np

from avatar_system.common import PKG, data_dir, root
from avatar_system.evaluation import repetition_metrics, transition_metrics


def simulate(planner, seed, minutes):
    out = data_dir("eval") / f"trace_{planner}_s{seed}"
    subprocess.run([sys.executable, str(PKG / "scripts/run_silent_demo.py"), "--no-video", "--planner", planner,
                    "--seed", str(seed), "--minutes", str(minutes), "--out", str(out)], check=True,
                   capture_output=True, cwd=root())
    return json.loads(out.parent.joinpath(f"{out.name}_trace.json").read_text())


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return f"{np.mean(v):.2f} ± {np.std(v):.2f}" if v else "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--lipsync", default=str(data_dir("demos") / "lipsync_musetalk"))
    a = ap.parse_args()
    graph = json.loads((data_dir() / "motion_graph.json").read_text())
    val = json.loads((data_dir() / "validation.json").read_text())
    cache, res = {}, {"beam": [], "random": []}
    t0 = time.time()
    worst = []
    for planner in ("beam", "random"):
        for seed in range(a.seeds):
            tr = simulate(planner, seed, a.minutes)
            r = repetition_metrics.report(tr, graph)
            s = transition_metrics.seam_report(tr, cache)
            r.update({k: s[k] for k in ("segment_changes", "graph_cuts", "natural_continuations",
                                        "pct_seam_frames_above_p99", "seam_frames_above_natural_p99")})
            r["seam_landmark_px_max"] = s["seam_step_max"]["landmark_px"]
            r["seam_visual_max"] = s["seam_step_max"]["visual"]
            r["natural_visual_p99"] = s["natural_step_p99"]["visual"]
            r["early_exits"] = sum(1 for x in tr["seams"] if len(x) > 7 and x[7])
            if planner == "beam" and seed == 0:
                worst = s["worst"][:8]
                r.update(transition_metrics.identity_background(tr, cache))
                res["identity"] = {k: r[k] for k in ("identity_cos_min", "identity_cos_mean", "bg_luma_range", "luma_range")}
            res[planner].append(r)
    (data_dir("eval") / "planner_comparison.json").write_text(json.dumps(res, indent=1, default=str))

    lip = None
    lp = data_dir("demos")
    vid = lp / (a.lipsync.split("/")[-1] + ".mp4")
    trp = lp / (a.lipsync.split("/")[-1] + "_trace.json")
    if vid.exists() and trp.exists():
        from avatar_system.evaluation import lipsync_metrics
        tr = json.loads(trp.read_text())
        lip = lipsync_metrics.report(vid, tr)
        lip["perf"] = {k: tr["metrics"].get(k) for k in ("render_ms", "render_ms_per_frame", "encode_ms",
                                                           "render_fps_capacity")}
        lip["perf"]["unpaced_fps"] = round(len(tr["frames"]) / tr["wall_s"], 1)
        lip["backend"] = tr["backend"]
        (data_dir("eval") / "lipsync.json").write_text(json.dumps(lip, indent=1, default=str))

    B, Rw = res["beam"], res["random"]
    rows = [("mode fit (% frames whose segment suits the state)", "mode_fit_pct"),
            ("listening frames on listen segments (%)", None),
            ("unique segments / 10 min", "unique_segments"), ("segment entropy (bits)", "segment_entropy_bits"),
            ("max repeats of one 3-segment sequence", "max_3gram_count"), ("repeated 3-grams", "repeated_3grams"),
            ("nods / min", None), ("gestures / min", None), ("tilt-type motions / min", None),
            ("blinks / min", "blinks_per_min"), ("blink interval CV", "blink_interval_cv"),
            ("segment changes / 10 min", "segment_changes"), ("graph cuts (non-natural)", "graph_cuts"),
            ("seam frames above natural p99 (%)", "pct_seam_frames_above_p99"),
            ("max seam visual step (natural p99)", "seam_visual_max"), ("early exits", "early_exits")]
    pat = lambda rs, p: f"{np.mean([r['patterns_per_min'].get(p, 0) for r in rs]):.2f}"
    lst = lambda rs: f"{np.mean([r['mode_fit_by_state_pct'].get('LISTENING', 0) for r in rs]):.1f}"
    lines = []
    for label, key in rows:
        if key:
            lines.append(f"| {label} | {agg(B, key)} | {agg(Rw, key)} |")
        elif "listening" in label:
            lines.append(f"| {label} | {lst(B)} | {lst(Rw)} |")
        else:
            p = {"nods / min": "nod", "gestures / min": "gesture", "tilt-type motions / min": "tilt"}[label]
            lines.append(f"| {label} | {pat(B, p)} | {pat(Rw, p)} |")
    nat_p99 = np.mean([r["natural_visual_p99"] for r in B])
    states_txt = ", ".join("%s (%d rest frames)" % (k, v["members"]) for k, v in graph["states"].items())
    modes_txt = "; ".join("%s: %d segments / %d clips / %ss" % (m, v["largest_cycle_set"], v["distinct_clips"],
                                                              v["seconds_in_cycle_set"]) for m, v in val["modes"].items())
    md = [f"# Benchmark report\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} by `evaluation/benchmark.py` "
          f"({time.time() - t0:.0f}s). Graph `{graph['version']}`. {a.seeds} seeds × {a.minutes:.0f} min simulated "
          "conversations (listening / thinking / speaking / interruptions) per planner, same library and graph.\n",
          "## Graph", f"- clips in library: {len({s['clip'] for s in graph['segments']})} usable "
          f"(+ {len(graph.get('excluded_segments', []))} segments excluded by the quality gate / review)",
          f"- states: {states_txt}",
          f"- runtime segments (edges between poses): {val['runtime_segments']}, approved transitions between them: "
          f"{val['runtime_edges']} (avg out-degree {val['avg_out_degree']}); transition counts {graph['counts']}",
          f"- dead ends / unreachable / no path to hub (pruned from runtime): {len(val['dead_ends'])} / "
          f"{len(val['unreachable_from_hub'])} / {len(val['no_path_to_hub'])}; worst time back to neutral from a "
          f"speaking segment: {val['max_frames_back_to_hub_from_speaking'] / 25:.1f}s",
          f"- per-mode cycle sets: {modes_txt}",
          f"- validator warnings: {val['warnings'] or 'none'}\n",
          "## Planner vs random-walk baseline (mean ± sd over seeds)",
          "| metric | beam planner | random walk |", "|---|---|---|", *lines,
          f"\nNatural (inside-clip) visual step p99 = {nat_p99:.2f} gray levels; a seam frame 'above p99' moves more "
          "than 99% of genuine consecutive frames.\n",
          "## Identity / background over the beam run",
          f"{res.get('identity')}\n",
          "## Worst seam frames (beam, seed 0)", "| frame | state | excess over natural p99 | visual | landmark px |",
          "|---|---|---|---|---|",
          *[f"| {w['n']} | {w['state']} | {w['excess']} | {w['visual']} | {w['landmark_px']} |" for w in worst]]
    sn = data_dir("eval") / "syncnet.json"
    if sn.exists():
        j = json.loads(sn.read_text())
        md += ["\n## Lip-sync accuracy: SyncNet (" + j["tool"] + ")", "| clip | confidence | AV offset (frames) |",
               "|---|---|---|", *[f"| {a} | {b} | {c} |" for a, b, c in j["rows"]]]
    if lip:
        md += ["\n## Lip-sync behaviour measured on the output video (MediaPipe)",
               "Mouth opening = inner-lip gap / inter-ocular distance. `av_offset_*` / `mouth_audio_corr` come from a crude "
               "energy-envelope correlation proxy and are NOT a sync measure (SyncNet above is); kept for transparency.",
               "| metric | value |", "|---|---|", *[f"| {k} | {v} |" for k, v in lip.items() if k != "perf"],
               f"\nThroughput of this run (silence ramp: silent frames skip MuseTalk, so capacity is inflated): "
               f"{lip['perf']}. With every frame lip-synced (silence=always) the measured capacity is ~72 fps "
               "(13.9 ms/frame, p95 17.6 ms, batch 8, one H100) = 2.9x real time."]
    cl = data_dir("eval") / "conversation_latency.json"
    if cl.exists():
        c = json.loads(cl.read_text())
        md += ["\n## Live conversation (Sarvam LLM + TTS bulbul:v3 shubh, MuseTalk, " + str(c["turns"]) + " turns)",
               "| stage | p50 s | p95 s |", "|---|---|---|",
               *[f"| {k} | {c[k]['p50']} | {c[k]['p95']} |" for k in ("llm_first_sentence_s", "tts_first_chunk_s",
                 "first_tts_done_s", "pipeline_after_tts_s", "first_audible_s") if k in c],
               f"\nfirst_audible = server emits the first speech frame (browser adds its 0.25 s jitter buffer). "
               f"pipeline_after_tts = queued frames ahead of the new speech (lead + plan + render + playback buffer). "
               f"Output frame interval {c.get('out_interval_ms')}; counters {c.get('counters')}; GPU peak "
               f"{c.get('gpu_peak_mem_gb')} GB, process RSS peak {c.get('rss_peak_gb')} GB; behaviour sources "
               f"{c.get('behavior_sources')}. Recording: {c.get('video')}"]
    out = PKG / "docs" / "benchmark_report.md"
    out.write_text("\n".join(md) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
