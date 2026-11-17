"""results/*.json -> results/report.md and assets/frontier.png, per PREREGISTRATION.md."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ALL, EVEN, ODD = slice(None), slice(0, None, 2), slice(1, None, 2)  # select on EVEN streams, report on ODD
R = [r for f in ("results", "groups") if Path(f"results/{f}.json").exists() for r in json.load(open(f"results/{f}.json"))]
ppl = lambda r, s=ALL: float(np.exp(np.mean(r["nll"][s])))
miss = lambda r, s=ALL, m=8: float(np.mean(r[f"miss{m}"][s]))
pct = lambda r, b, s=ALL: 100 * (ppl(r, s) / ppl(b, s) - 1)
plain = {r["run"]: r for r in R if r["bonus"] == 0 and r["margin"] == 0}  # no inference-time knobs
base = plain["flat"]
family = {f: [r for r in R if r["router"] == f and r["seed"] == 0] for f in ("flat", "tree", "grouped")}
seed_gap = abs(pct(plain["flat-seed1"], base)) / 100 if "flat-seed1" in plain else float("nan")


def name(r):
    knobs = [f"λ={r['lam']:g}"] + ["slow gate"] * bool(r["slow"]) + [f"Δ={r['bonus']:g}"] * bool(r["bonus"]) + [f"m={r['margin']:g}"] * bool(r["margin"])
    return f"{r['router']} " + ", ".join(knobs)


def decision(other, gates, prereg=1.0):
    """Flat vs `other` at each perplexity gate: table rows and the verdict at the first gate."""
    rows, first = [f"| gate | family | selected configuration | perplexity vs flat | miss rate | {other} / flat |", "|---|---|---|---|---|---|"], None
    for gate in gates:
        pick = {f: min((r for r in family[f] if ppl(r, EVEN) <= (1 + gate) * ppl(base, EVEN)), key=lambda r: miss(r, EVEN), default=None) for f in ("flat", other)}
        ratio = miss(pick[other], ODD) / miss(pick["flat"], ODD) if all(pick.values()) else float("nan")
        first = ratio if first is None else first
        for f, r in pick.items():
            rows.append(f"| ≤{gate:.0%}{'' if gate <= prereg else ' (exploratory)'} | {f} | "
                        + ("none passes | – | – |" if r is None else f"{name(r)} | {pct(r, base, ODD):+.2f}% | {miss(r, ODD):.4f} |")
                        + (" – |" if ratio != ratio else f" {ratio:.2f} |" if f == other else " |"))
    verdict = "inconclusive (the 1% gate is inside seed noise)" if seed_gap > 0.01 else "supported" if first <= 0.5 else "not supported"
    detail = f"No {other} configuration is within the gate." if first != first else f"{other} / flat miss ratio {first:.2f}; supported means at most 0.50."
    return rows, f"{verdict}.** {detail}"


out = ["## Trained models, no inference-time rerouting", "",
       "| run | perplexity | vs flat | vs own λ=0 | miss @4 | miss @8 | miss @16 | expert switch | cluster switch | coverage @8 |", "|---|---|---|---|---|---|---|---|---|---|"]
for run, r in plain.items():
    if r["router"] == "grouped":
        continue
    cells = [f"{ppl(r):.2f}"] + [f"{pct(r, b):+.1f}%" for b in (base, plain[r["router"]])]
    if "miss8" in r:
        cells += [f"{miss(r, m=m):.3f}" for m in (4, 8, 16)] + [f"{r['expert_switch']:.3f}", "–" if r["cluster_switch"] is None else f"{r['cluster_switch']:.3f}", f"{r['coverage8']:.3f}"]
    out.append(f"| {run} | " + " | ".join(cells + ["–"] * (9 - len(cells))) + " |")

rows, verdict = decision("tree", (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30), prereg=0.05)
out += ["", f"Seed-to-seed perplexity gap (flat, λ=0): {100 * seed_gap:.2f}%", "",
        "## Decision (cache of 8 experts per layer; selected on even streams, reported on odd)", "",
        "Gates of 1%, 2% and 5% are pre-registered. Wider gates were added after seeing the results and are exploratory.", "",
        *rows, "", f"**H1 at the pre-registered 1% gate: {verdict}"]

if family["grouped"]:
    rows, verdict = decision("grouped", (0.01, 0.02, 0.05, 0.10, 0.15, 0.20))
    out += ["", "## Follow-up 1 (exploratory): sticky post-hoc expert groups on the flat models", "",
            "Group restriction alone (no hysteresis, no cache bonus) against the flat model it is applied to:", "",
            "| flat model | perplexity flat | grouped | change | miss @8 flat | grouped |", "|---|---|---|---|---|---|"]
    for run, g in plain.items():
        if g["router"] == "grouped":
            f = plain[run.replace("grouped", "flat")]
            out.append(f"| λ={g['lam']:g} | {ppl(f):.2f} | {ppl(g):.2f} | {pct(g, f):+.1f}% | {miss(f):.3f} | {miss(g):.3f} |")
    out += ["", *rows, "", f"**Follow-up 1 at the 1% gate: {verdict}"]

Path("results/report.md").write_text("\n".join(out) + "\n")
print("\n".join(out))

# ---- frontier plot: every configuration, perplexity change vs miss rate ----
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
STYLE = {"flat": ("#2a78d6", "o", "flat router"), "tree": ("#eb6834", "s", "tree router"), "grouped": ("#1baf7a", "^", "flat router + post-hoc groups")}
fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for f, rows in family.items():
    if rows:
        color, marker, label = STYLE[f]
        pts = sorted((pct(r, base), miss(r)) for r in rows)
        ax.scatter(*zip(*pts), s=42, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=label)
        front = [p for i, p in enumerate(pts) if all(q[1] > p[1] for q in pts[:i])]  # lower-left envelope
        ax.step(*zip(*front), where="post", color=color, linewidth=2, zorder=2)
for v, label in ((1, "1% gate"), (pct(plain["dense"], base), "dense FFN, same active size")):
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
