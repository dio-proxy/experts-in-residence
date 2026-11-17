"""Offline expert-cache metrics computed from routing traces."""
import torch
import torch.nn.functional as F

from .moe import Cache


def lru_miss_rate(trace, n, capacity):
    """trace [R, T, K]: R independent streams. Per stream [R], the fraction of expert
    activations whose expert is not resident in an LRU cache of `capacity` experts."""
    r, t, k = trace.shape
    cache, misses = Cache(r, n, capacity), torch.zeros(r)
    for i in range(t):
        misses += (~cache.resident().gather(1, trace[:, i])).sum(1)
        cache.touch(trace[:, i])
    return misses / (t * k)


def segment_counts(trace, n, seg):
    """How often each expert is used in each `seg`-token segment: [R, T // seg, n]."""
    r, t, k = trace.shape
    x = trace[:, : t // seg * seg].reshape(r, t // seg, seg * k)
    return torch.zeros(r, t // seg, n, dtype=torch.long).scatter_add_(2, x, torch.ones_like(x))


def segment_coverage(trace, n, m, seg=64):
    """Share of a segment's activations served by its best fixed set of m experts
    (in the spirit of SRP, Liang et al. 2026)."""
    return (segment_counts(trace, n, seg).topk(m, -1).values.sum(-1) / (seg * trace.shape[2])).mean().item()


def coactivation_groups(trace, n, size, seg=64):
    """Partition n experts into groups of `size` that tend to be used in the same segments.

    Affinity = cosine similarity of segment usage. Greedy and balanced: seed a group with
    the busiest free expert, then keep adding the free expert closest to the group.
    Returns the group id of each expert [n].
    """
    assert n % size == 0
    use = segment_counts(trace, n, seg).flatten(0, 1).float()
    aff = F.normalize(use, dim=0).T @ F.normalize(use, dim=0)
    free, group = torch.ones(n, dtype=torch.bool), torch.empty(n, dtype=torch.long)
    for g in range(n // size):
        members = [int(((use.sum(0) + 1) * free).argmax())]
        for _ in range(size - 1):
            free[members[-1]] = False
            members.append(int(aff[members].sum(0).masked_fill(~free, -1).argmax()))
        free[members[-1]] = False
        group[members] = g
    return group


def switch_rate(trace, group=1):
    """Fraction of adjacent token pairs whose top-1 expert (or its cluster, if group = E) differs."""
    top = trace[..., 0] // group
    return (top[:, 1:] != top[:, :-1]).float().mean().item()
