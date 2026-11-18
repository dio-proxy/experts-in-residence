"""Follow-up 2 in PREREGISTRATION.md: pair-based groups and one free expert, on held-out
streams not used before -> results/followup2.json"""
import itertools
import json

import torch

from eir.data import eval_streams, load
from eir.model import LM, Cfg
from evaluate import BONUS, MARGIN, STREAMS, WINDOWS, evaluate, metrics
from followup_groups import expert_groups

VARIANTS = {"pair-both": (1, False), "segment-free": (64, True), "pair-free": (1, True)}  # grouping segment length, free second expert

if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load("train"), load("val")
    n = 2 * STREAMS - 1  # the earlier streams are the even ones of these; keep the odd ones
    assert all((len(t) - WINDOWS * Cfg.seq) // (n - 1) >= WINDOWS * Cfg.seq for t in val.values()), "streams would overlap"
    x, y, domain = eval_streams(val, n, WINDOWS, Cfg.seq)
    keep = torch.arange(3 * n).view(3, n)[:, 1::2].flatten()
    x, y, domain = x[:, keep], y[:, keep], domain[keep]
    results = []
    for run in ("flat", "flat-l0.03"):
        args = json.load(open(f"runs/{run}/train.json"))["args"]
        state = torch.load(f"runs/{run}/ckpt.pt", map_location=dev)
        model = LM(Cfg(**state["cfg"])).to(dev).eval()
        model.load_state_dict(state["model"])
        grids = [("flat", None, False, [(b, 0.0) for b in BONUS])]
        grids += [(v, expert_groups(model, train, dev, seg), free, list(itertools.product(BONUS, MARGIN))) for v, (seg, free) in VARIANTS.items()]
        for router, groups, free, grid in grids:
            for bonus, margin in grid:
                nll, trace = evaluate(model, x, y, bonus, margin, groups=groups, free=free)
                row = dict(run=run, router=router, lam=args["lam"], slow=0.0, seed=0, bonus=bonus, margin=margin, nll=nll.tolist(), domain=domain.tolist())
                results.append(row | metrics(trace, model.cfg, bonus))
                print(run, router, bonus, margin, f"ppl {nll.mean().exp():.2f}", f"miss8 {sum(results[-1]['miss8']) / len(nll):.3f}", flush=True)
    json.dump(results, open("results/followup2.json", "w"))
