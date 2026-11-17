"""Follow-up 1 in PREREGISTRATION.md: group a trained flat router's experts by co-activation
on training data, then route with a sticky group at inference -> results/groups.json"""
import itertools
import json
from pathlib import Path

import torch

from eir.cache import coactivation_groups
from eir.data import eval_streams, get_batch, load
from eir.model import LM, Cfg
from evaluate import BONUS, MARGIN, STREAMS, WINDOWS, evaluate, metrics

BATCHES, BATCH = 24, 16


@torch.no_grad()
def expert_groups(model, data, dev):
    """Groups per layer [L, N] from routing traces on training sequences."""
    c, gen = model.cfg, torch.Generator().manual_seed(0)
    trace = torch.cat([model(get_batch(data, BATCH, c.seq, gen)[0].to(dev))[1].cpu() for _ in range(BATCHES)], 1)
    return torch.stack([coactivation_groups(t, c.clusters * c.per_cluster, c.per_cluster) for t in trace])


if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    train = load("train")
    x, y, domain = eval_streams(load("val"), STREAMS, WINDOWS, Cfg.seq)
    results = []
    for run in ("flat", "flat-l0.03", "flat-l0.1", "flat-l0.3"):
        args = json.load(open(f"runs/{run}/train.json"))["args"]
        state = torch.load(f"runs/{run}/ckpt.pt", map_location=dev)
        model = LM(Cfg(**state["cfg"])).to(dev).eval()
        model.load_state_dict(state["model"])
        groups = expert_groups(model, train, dev)
        for bonus, margin in itertools.product(BONUS, MARGIN):
            nll, trace = evaluate(model, x, y, bonus, margin, groups=groups)
            row = dict(run=run.replace("flat", "grouped"), router="grouped", lam=args["lam"], slow=0.0, seed=args["seed"],
                       bonus=bonus, margin=margin, nll=nll.tolist(), domain=domain.tolist())
            results.append(row | metrics(trace, model.cfg, bonus))
            print(row["run"], bonus, margin, f"ppl {nll.mean().exp():.2f}", f"miss8 {sum(results[-1]['miss8']) / len(nll):.3f}", flush=True)
    json.dump(results, open("results/groups.json", "w"))
