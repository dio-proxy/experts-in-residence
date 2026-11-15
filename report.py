"""results/results.json -> results/report.md and assets/frontier.png, per PREREGISTRATION.md."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ALL, EVEN, ODD = slice(None), slice(0, None, 2), slice(1, None, 2)  # select on EVEN streams, report on ODD
R = json.load(open("results/results.json"))
ppl = lambda r, s=ALL: float(np.exp(np.mean(r["nll"][s])))
miss = lambda r, s=ALL, m=8: float(np.mean(r[f"miss{m}"][s]))
plain = {r["run"]: r for r in R if r["bonus"] == 0 and r["margin"] == 0}
base = plain["flat"]
family = {f: [r for r in R if r["router"] == f and r["seed"] == 0] for f in ("flat", "tree")}


def name(r):
    knobs = [f"λ={r['lam']:g}"] + ["slow gate"] * bool(r["slow"]) + [f"Δ={r['bonus']:g}"] * bool(r["bonus"]) + [f"m={r['margin']:g}"] * bool(r["margin"])
    return f"{r['router']} " + ", ".join(knobs)


out = ["## Trained models, no inference-time rerouting", "",
       "| run | perplexity | vs flat | vs own λ=0 | miss @4 | miss @8 | miss @16 | expert switch | cluster switch | coverage @8 |", "|---|---|---|---|---|---|---|---|---|---|"]
for run, r in plain.items():
    cells = [f"{ppl(r):.2f}"] + [f"{100 * (ppl(r) / ppl(b) - 1):+.1f}%" for b in (base, plain[r["router"]])]
    if "miss8" in r:
        cells += [f"{miss(r, m=m):.3f}" for m in (4, 8, 16)] + [f"{r['expert_switch']:.3f}", "–" if r["cluster_switch"] is None else f"{r['cluster_switch']:.3f}", f"{r['coverage8']:.3f}"]
    out.append(f"| {run} | " + " | ".join(cells + ["–"] * (9 - len(cells))) + " |")

seed_gap = abs(ppl(plain["flat-seed1"]) / ppl(base) - 1) if "flat-seed1" in plain else float("nan")
out += ["", f"Seed-to-seed perplexity gap (flat, λ=0): {100 * seed_gap:.2f}%", "",
        "## Decision (cache of 8 experts per layer; selected on even streams, reported on odd)", "",
        "Gates of 1%, 2% and 5% are pre-registered. Wider gates were added after seeing the results and are exploratory.", "",
        "| gate | family | selected configuration | perplexity vs flat | miss rate | tree / flat |", "|---|---|---|---|---|---|"]
for gate in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
    pick = {f: min((r for r in rows if ppl(r, EVEN) <= (1 + gate) * ppl(base, EVEN)), key=lambda r: miss(r, EVEN), default=None) for f, rows in family.items()}
    ratio = miss(pick["tree"], ODD) / miss(pick["flat"], ODD) if all(pick.values()) else float("nan")
    for f, r in pick.items():
        out.append(f"| ≤{gate:.0%}{'' if gate <= 0.05 else ' (exploratory)'} | {f} | " + ("none passes | – | – |" if r is None else f"{name(r)} | {100 * (ppl(r, ODD) / ppl(base, ODD) - 1):+.2f}% | {miss(r, ODD):.4f} |") + (" – |" if ratio != ratio else f" {ratio:.2f} |" if f == "tree" else " |"))
    if gate == 0.01:
        primary = ratio
verdict = "inconclusive (the 1% gate is inside seed noise)" if seed_gap > 0.01 else "supported" if primary <= 0.5 else "not supported"
out += ["", f"**H1 at the pre-registered 1% gate: {verdict}.** " + ("No tree configuration is within the gate." if primary != primary else f"Tree / flat miss ratio {primary:.2f}; supported means at most 0.50.")]
Path("results/report.md").write_text("\n".join(out) + "\n")
print("\n".join(out))

# ---- frontier plot: every configuration, perplexity change vs miss rate ----
INK, MUTED, COLORS, MARK = "#0b0b0b", "#52514e", {"flat": "#2a78d6", "tree": "#eb6834"}, {"flat": "o", "tree": "s"}
fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
for f, rows in family.items():
    pts = sorted((100 * (ppl(r) / ppl(base) - 1), miss(r)) for r in rows)
    ax.scatter(*zip(*pts), s=42, marker=MARK[f], color=COLORS[f], edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label=f"{f} router")
    front = [p for i, p in enumerate(pts) if all(q[1] > p[1] for q in pts[:i])]  # lower-left envelope
    ax.step(*zip(*front), where="post", color=COLORS[f], linewidth=2, zorder=2)
for v, label in ((1, "1% gate"), (100 * (ppl(plain["dense"]) / ppl(base) - 1), "dense FFN, same active size")):
    ax.axvline(v, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(v, 0.02, label + " ", color=MUTED, fontsize=8, rotation=90, ha="right", va="bottom", transform=ax.get_xaxis_transform())
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlim(-1.5, 40)  # configurations beyond +40% are off the chart
ax.set_xlabel("perplexity change vs flat router, no locality (%)", color=MUTED)
ax.set_ylabel("expert cache miss rate (8 of 64 resident)", color=MUTED)
ax.set_title("Lower-left is better: fewer page-ins at the same quality", color=INK, fontsize=11, loc="left")
ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
ax.tick_params(colors=MUTED, labelsize=9)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower left", bbox_to_anchor=(0.1, 0.03))
fig.tight_layout()
Path("assets").mkdir(exist_ok=True)
fig.savefig("assets/frontier.png", dpi=160)
