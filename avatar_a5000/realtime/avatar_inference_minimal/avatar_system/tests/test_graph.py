import json

import pytest

from avatar_system.common import data_dir
from avatar_system.motion.graph_validator import frames_to_hub, sccs, validate


def test_validator_finds_dead_end_and_prunes_it(toy_graph):
    rep = validate(toy_graph, write=False, verbose=False)
    assert rep["dead_ends"] == ["c3@40-89"]
    assert "c3@40-89" in rep["outside_main_scc"]
    assert rep["runtime_segments"] == 5
    assert rep["ok"]


def test_every_runtime_segment_returns_to_hub(toy_graph):
    adj = {int(k): [d for d, *_ in v] for k, v in toy_graph["adjacency"].items()}
    d = frames_to_hub(toy_graph["segments"], adj, "neutral")
    for i in range(5):
        assert i in d
    assert d[1] == 50 + 60  # tilt -> must play segment 2 to get back to neutral


def test_hub_without_cycle_is_an_error(toy_graph):
    for k in toy_graph["adjacency"]:
        toy_graph["adjacency"][k] = []
    rep = validate(toy_graph, write=False, verbose=False)
    assert not rep["ok"]


def test_sccs_simple():
    comps = sccs([0, 1, 2, 3], {0: [1], 1: [0, 2], 2: [3], 3: [2]})
    assert sorted(sorted(c) for c in comps) == [[0, 1], [2, 3]]


real = data_dir() / "motion_graph.json"


@pytest.mark.skipif(not real.exists(), reason="no built graph")
def test_real_graph_is_valid_and_transitions_respect_thresholds():
    g = json.loads(real.read_text())
    rep = validate(g, write=False, verbose=False)
    assert rep["ok"], rep["errors"]
    assert not set(rep["dead_ends"]) & {g["segments"][i]["key"] for i in g["runtime_segments"]}
    trans = json.loads((data_dir() / "transitions.json").read_text())
    from avatar_system.common import load_config
    T = load_config("motion_graph")["transition"]
    for t in trans:
        if t["status"] == "approved" and not t["manual"]:
            assert t["cost"] <= T["accept_cost"] + 1e-6 and not t["reasons"]
        if t["reasons"] and not t["manual"]:
            assert t["status"] == "rejected"
    usable = {(t["src"], t["dst"]) for t in trans if t["status"] in ("approved", "natural")}
    for k, v in g["adjacency"].items():
        for d, *_ in v:
            assert (int(k), d) in usable


@pytest.mark.skipif(not real.exists(), reason="no built graph")
def test_manual_rejection_survives_rebuild(tmp_path, monkeypatch):
    """A rejected join must stay rejected after a rebuild (decisions keyed by cut frames)."""
    from avatar_system.motion import graph_builder
    trans = json.loads((data_dir() / "transitions.json").read_text())
    victim = next(t for t in trans if t["status"] == "approved")
    reviews = {"transitions": {victim["key"]: {"decision": "reject"}}, "segments": {}, "clips": {}}
    monkeypatch.setattr(graph_builder, "load_reviews", lambda: reviews)
    monkeypatch.setattr(graph_builder, "write_sqlite", lambda *a, **k: None)
    real_dir = data_dir()
    monkeypatch.setattr(graph_builder, "data_dir", lambda *s: (tmp_path.joinpath(*s) if s else tmp_path)
                        if not s or s[0] != "features" else real_dir.joinpath(*s))
    g, t2 = graph_builder.build(verbose=False)
    st = {t["key"]: t["status"] for t in t2}
    assert st[victim["key"]] == "rejected"
    src = next(t for t in t2 if t["key"] == victim["key"])
    assert src["dst"] not in [d for d, *_ in g["adjacency"][str(src["src"])]]
