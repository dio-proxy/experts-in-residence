"""Offline expert-cache metrics computed from routing traces."""
import torch

from .moe import Cache


def lru_miss_rate(trace, n, capacity):
    """trace [R, T, K]: R independent streams (stream x layer). Fraction of expert
    activations that are not resident in an LRU cache of `capacity` experts."""
    r, t, k = trace.shape
    cache, misses = Cache(r, n, capacity), 0
    for i in range(t):
        misses += (~cache.resident().gather(1, trace[:, i])).sum().item()
        cache.touch(trace[:, i])
    return misses / (r * t * k)


def segment_coverage(trace, n, m, seg=64):
    """Share of a segment's activations served by its best fixed set of m experts
    (in the spirit of SRP, Liang et al. 2026)."""
    r, t, k = trace.shape
    x = trace[:, : t // seg * seg].reshape(r, t // seg, seg * k)
    counts = torch.zeros(r, t // seg, n, dtype=torch.long).scatter_add_(2, x, torch.ones_like(x))
    return (counts.topk(m, -1).values.sum(-1) / (seg * k)).mean().item()


def switch_rate(trace, group=1):
    """Fraction of adjacent token pairs whose top-1 expert (or its cluster, if group = E) differs."""
    top = trace[..., 0] // group
    return (top[:, 1:] != top[:, :-1]).float().mean().item()
