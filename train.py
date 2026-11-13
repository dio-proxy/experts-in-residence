"""Train one configuration.  python train.py --name tree-l0.1 --router tree --lam 0.1"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from eir.data import get_batch, load
from eir.model import LM, Cfg

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--router", default="flat", choices=["flat", "tree", "dense"])
p.add_argument("--lam", type=float, default=0.0, help="weight of the reuse (locality) loss")
p.add_argument("--slow", type=float, default=0.0, help="EMA decay of the tree's cluster gate")
p.add_argument("--balance", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
a = p.parse_args()

torch.manual_seed(a.seed)
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
cfg = Cfg(router=a.router, slow=a.slow)
model = LM(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / a.steps))))
train, gen = load("train"), torch.Generator().manual_seed(a.seed)  # batches depend on the seed only

log, t0 = [], time.time()
for step in range(a.steps):
    x, y = (t.to(dev) for t in get_batch(train, a.batch, cfg.seq, gen))
    logits, _, aux = model(x)
    lm = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss = lm + a.balance * aux["balance"] + a.lam * aux["reuse"]
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 50 == 0 or step == a.steps - 1:
        row = dict(step=step, lm=lm.item(), balance=aux["balance"].item(), reuse=aux["reuse"].item(), sec=time.time() - t0)
        log.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

out = Path("runs") / a.name
out.mkdir(parents=True, exist_ok=True)
torch.save({"cfg": cfg.dict(), "model": model.state_dict()}, out / "ckpt.pt")
json.dump({"args": vars(a), "log": log}, open(out / "train.json", "w"), indent=1)
