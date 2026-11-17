"""Evaluate every run in runs/ over the pre-registered inference grid -> results/results.json"""
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from eir.cache import lru_miss_rate, segment_coverage, switch_rate
from eir.data import eval_streams, load
from eir.model import LM, Cfg
from eir.moe import Cache

STREAMS, WINDOWS, PRIMARY = 16, 16, 8
BONUS, MARGIN = (0.0, 1.0, 2.0, 4.0), (0.0, 1.0, 2.0)


@torch.no_grad()
def evaluate(model, x, y, bonus=0.0, margin=0.0, capacity=PRIMARY, groups=None):
    """x, y [W, R, S]. Per-stream NLL [R] and routing trace [L, R, W*S, K] (None if dense).
    groups [L, N]: optional expert groups per layer for group-sticky flat routing."""
    c, dev = model.cfg, next(model.parameters()).device
    knobs = bonus > 0 or margin > 0 or groups is not None
    caches = [Cache(x.shape[1], c.clusters * c.per_cluster, capacity, None if groups is None else groups[i]) for i in range(c.layers)] if knobs else None
    nll, trace = 0, []
    for xw, yw in zip(x, y):
        logits, tr, _ = model(xw.to(dev), caches, bonus=bonus, margin=margin)
        nll = nll + F.cross_entropy(logits.transpose(1, 2), yw.to(dev), reduction="none").sum(1).cpu()
        trace.append(tr if tr is None else tr.cpu())
    return nll / (x.shape[0] * x.shape[2]), None if trace[0] is None else torch.cat(trace, 2)


def metrics(trace, cfg, bonus):
    """Cache metrics per stream. Other capacities are only valid when routing ignored the cache."""
    layers, r = trace.shape[:2]
    rows, n = trace.flatten(0, 1), cfg.clusters * cfg.per_cluster
    out = {f"miss{m}": lru_miss_rate(rows, n, m).view(layers, r).mean(0).tolist() for m in ((4, 8, 16) if bonus == 0 else (PRIMARY,))}
    out["expert_switch"] = switch_rate(rows)
    out["cluster_switch"] = switch_rate(rows, cfg.per_cluster) if cfg.router == "tree" else None
    out["coverage8"] = segment_coverage(rows, n, 8)
    return out


if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    x, y, domain = eval_streams(load("val"), STREAMS, WINDOWS, Cfg.seq)
    results = []
    for ckpt in sorted(Path("runs").glob("*/ckpt.pt")):
        args = json.load(open(ckpt.parent / "train.json"))["args"]
        state = torch.load(ckpt, map_location=dev)
        model = LM(Cfg(**state["cfg"])).to(dev).eval()
        model.load_state_dict(state["model"])
        in_family = args["router"] != "dense" and args["seed"] == 0
        grid = itertools.product(BONUS, MARGIN if args["router"] == "tree" else (0.0,)) if in_family else [(0.0, 0.0)]
        for bonus, margin in grid:
            nll, trace = evaluate(model, x, y, bonus, margin)
            row = dict(run=ckpt.parent.name, router=args["router"], lam=args["lam"], slow=args["slow"], seed=args["seed"],
                       bonus=bonus, margin=margin, nll=nll.tolist(), domain=domain.tolist())
            if trace is not None:
                row |= metrics(trace, model.cfg, bonus)
            results.append(row)
            print(row["run"], bonus, margin, f"ppl {nll.mean().exp():.2f}", f"miss8 {sum(row.get('miss8', [0])) / len(nll):.3f}", flush=True)
    Path("results").mkdir(exist_ok=True)
    json.dump(results, open("results/results.json", "w"))
