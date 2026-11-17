# Pre-registration

Written and committed before any full training run. The commit history shows this file
landing before `runs/` and `results/` exist.

## Question

Sparse MoE models are too large for small devices, so only a few experts stay in fast
memory and the rest are paged in on demand. Each page-in stalls decoding. Training a flat
router to reuse experts across adjacent tokens cuts page-ins but has been reported to cost
quality ("miss reduction and quality are tightly coupled", arXiv:2608.18261).

**Hypothesis (H1).** A two-level router loosens that coupling: make the *cluster* decision
sticky and leave the *expert* decision inside the cluster free per token. At equal quality,
it needs fewer page-ins than a flat router trained for locality.

## Setup (fixed)

- Model: 4-layer causal transformer, d=192, 4 heads, context 256, 4k BPE vocabulary.
  Every FFN is an MoE with 64 experts (hidden 384), 2 active per token.
- Data: WikiText-2, TinyStories (validation file), CPython standard library source;
  each training sequence comes from one domain.
- Training: 2,000 steps, batch 16x256 (see Amendments), AdamW lr 1e-3 (100 warmup steps, cosine), balance
  loss weight 0.1, seed 0. All runs see the same batches.
- Evaluation: 16 held-out streams per domain, 4,096 tokens each (196,608 tokens). The expert
  cache persists along a stream; the model context resets every 256 tokens.
- Cache: one LRU cache per layer per stream, capacity M experts out of 64.
  **Primary M = 8** (12.5% resident). M = 4 and 16 are reported for runs without rerouting.
- Miss rate = share of expert activations whose expert is not resident.

## Methods and grids (fixed)

| family | trained with | inference knobs |
|---|---|---|
| flat | reuse loss on the 64-way expert distribution, weight λ ∈ {0, 0.03, 0.1, 0.3} | cache bonus Δ ∈ {0, 1, 2, 4} |
| tree (8 clusters x 8 experts, 1 cluster and 2 experts per token) | reuse loss on the 8-way cluster distribution only, λ ∈ {0, 0.03, 0.1, 0.3}; plus one run with a slow cluster gate (causal EMA, decay 0.9) at λ = 0.1 | cache bonus Δ ∈ {0, 1, 2, 4} x cluster hysteresis margin m ∈ {0, 1, 2} |

- Reuse loss: `-log` of the probability mass token t puts on what token t-1 selected.
- Cache bonus (training-free, after arXiv:2412.00099): add Δ to the logits of resident experts before top-k.
- Hysteresis (training-free): leave the current cluster only if another cluster's logit is more than m higher.
- Reference runs, not part of either family: a dense model with the same active parameters, and a second seed of flat λ=0 to measure seed noise.

## Decision rule

Baseline perplexity `ppl0` = flat, λ=0, Δ=0, seed 0.

1. For each family, among all its configurations with `ppl <= 1.01 * ppl0`, take the lowest miss rate at M = 8.
2. **H1 is supported if the tree family's miss rate is at most half the flat family's.** Otherwise it is not supported.
3. Configurations are selected on the even-numbered streams and reported on the odd-numbered ones.

Also reported whatever the outcome: the same comparison at 2% and 5% gates, the full
perplexity/miss-rate frontier, the perplexity of plain tree vs plain flat, and the
seed-to-seed perplexity gap. If that gap exceeds 1%, the primary gate is inside seed noise
and the result is reported as inconclusive.

## Known limits

- One seed per configuration, a 39M-parameter model (2.6M active), three domains.
- The cache model counts page-ins; it does not measure wall-clock latency.
- M = 8 equals one cluster, which favours the tree; M = 4 is reported as the hard case.

## Pilots run before this document

Runs of at most 300 training steps, used only to make training stable (balance weight 0.1,
block-wise expert batching). No pilot was evaluated on the held-out streams.

## Amendments

1. **Batch 32x256 -> 16x256, before any run finished.** The first run at batch 32 pushed the
   8 GB training machine into swap (about 0.9 s per step and slowing). It was stopped at
   roughly step 600 of 2,000, discarded without evaluation, and every run was restarted at
   batch 16. Nothing else changed.

## Follow-up 1 (exploratory): post-hoc expert groups

Specified after the main results and before this follow-up was run on the held-out streams.
It reuses those streams and was prompted by the main results, so it is exploratory whatever
it shows.

**Idea.** The sticky cluster made locality cheap; the tree structure was the cost. So keep
the trained flat routers and add the grouping afterwards, with no retraining.

- Groups: for each layer of a trained flat model, 8 groups of 8 experts, built from routing
  traces on *training* data (384 sequences of 256 tokens, seed 0). Affinity is the cosine
  similarity of two experts' usage across 64-token segments; the partition is greedy and
  balanced (seed a group with the busiest free expert, add its closest free experts).
- Routing: take the group with the largest probability mass, leaving the current group only
  if another group's log mass is more than m higher. Both experts come from that group;
  cache bonus Δ as in the flat family.
- Grid: the four seed-0 flat models (λ ∈ {0, 0.03, 0.1, 0.3}) x Δ ∈ {0, 1, 2, 4} x m ∈ {0, 1, 2}.
- Rule: the main decision rule with "grouped" in place of "tree". Supported if, within the
  1% gate, its lowest miss rate at M = 8 is at most half the flat family's. The 2% and 5%
  gates are reported as well.
