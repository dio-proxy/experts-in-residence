"""Routers (flat, tree), expert bank and the LRU expert cache used at inference.

Conventions: N experts, K active per token. Routers take x [B, S, d] and return
idx [B, S, K] (global expert ids) and w [B, S, K] (gate values, unnormalised probabilities).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def switch_loss(probs, idx):
    """Switch-Transformer balance loss N * sum_i f_i P_i. probs [T, N], idx [T, K]."""
    n = probs.shape[-1]
    f = torch.bincount(idx.reshape(-1), minlength=n).to(probs.dtype) / idx.numel()
    return n * (f * probs.mean(0)).sum()


def reuse_loss(probs, sel):
    """-log of the probability mass token t puts on what token t-1 selected.

    probs [B, S, N]; sel [B, S, N] 0/1 mask of each token's own hard selection (no grad).
    """
    mass = (probs[:, 1:] * sel[:, :-1]).sum(-1)
    return -mass.clamp_min(1e-9).log().mean()


class Cache:
    """Per-stream LRU over experts. `last` [R, N] holds last-use time, -1 = never loaded."""

    def __init__(self, rows, n, capacity, group=None, free=False):
        self.last = torch.full((rows, n), -1, dtype=torch.long)
        self.capacity, self.t = capacity, 0
        self.group, self.free = group, free  # optional [N] expert -> group id, see FlatRouter.scan
        self.prev_cluster = torch.full((rows,), -1, dtype=torch.long)

    def resident(self):
        thr = self.last.topk(self.capacity, -1).values[:, -1:]
        return (self.last >= thr) & (self.last >= 0)

    def touch(self, sel):  # sel [R, K]
        k = sel.shape[1]
        self.last.scatter_(1, sel, self.t * k + torch.arange(k).expand_as(sel))
        self.t += 1


class FlatRouter(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.n, self.k = n, k
        self.gate = nn.Linear(d, n, bias=False)

    def forward(self, x):
        p = F.softmax(self.gate(x).float(), -1)
        w, idx = p.topk(self.k, -1)
        sel = torch.zeros_like(p).scatter_(-1, idx, 1.0)
        return idx, w, {"balance": switch_loss(p.flatten(0, 1), idx.flatten(0, 1)), "reuse": reuse_loss(p, sel)}

    @torch.no_grad()
    def scan(self, x, cache, bonus=0.0, margin=0.0):
        """Token-by-token routing; resident experts get `bonus` added to their logit.

        If the cache carries expert groups, both experts come from one group: the one with
        the most probability mass, left only if another group's log mass is more than
        `margin` higher (the post-hoc counterpart of TreeRouter.scan). With `cache.free`
        only the first expert is bound to the group; the rest are picked from all experts.
        """
        logits = self.gate(x).float().cpu()
        p = F.softmax(logits, -1)
        rows, idx, w = torch.arange(x.shape[0]), [], []
        for t in range(x.shape[1]):
            z = logits[:, t] + bonus * cache.resident()
            if cache.group is not None:
                mass = torch.zeros(len(rows), int(cache.group.max()) + 1).index_add_(1, cache.group, p[:, t]).log()
                g, prev = mass.argmax(-1), cache.prev_cluster
                stay = (prev >= 0) & (mass[rows, g] - mass[rows, prev.clamp_min(0)] <= margin)
                cache.prev_cluster = g = torch.where(stay, prev, g)
                inside = z.masked_fill(cache.group != g[:, None], -torch.inf)
                if cache.free:
                    first = inside.argmax(-1, keepdim=True)
                    z = torch.cat([first, z.scatter(1, first, -torch.inf).topk(self.k - 1, -1).indices], 1)
                else:
                    z = inside.topk(self.k, -1).indices
            i = z if cache.group is not None else z.topk(self.k, -1).indices
            idx.append(i), w.append(p[:, t].gather(-1, i))
            cache.touch(i)
        return torch.stack(idx, 1).to(x.device), torch.stack(w, 1).to(x.device)


class TreeRouter(nn.Module):
    """Pick one cluster, then the top-k experts inside it. Gate = P(c) * P(e | c).

    slow > 0 low-pass filters the cluster logits along the sequence (causal EMA with decay
    `slow`), so the cluster decision moves on a slower timescale than the expert decision.
    """

    def __init__(self, d, clusters, per_cluster, k, slow=0.0, max_seq=1024):
        super().__init__()
        self.c, self.e, self.k, self.n = clusters, per_cluster, k, clusters * per_cluster
        self.cgate = nn.Linear(d, clusters, bias=False)
        self.egate = nn.Linear(d, self.n, bias=False)
        self.slow = slow
        if slow:
            t = torch.arange(max_seq)
            ker = (slow ** (t[:, None] - t[None, :]).clamp_min(0).float()).tril()
            self.register_buffer("ker", ker / ker.sum(-1, keepdim=True), persistent=False)

    def cluster_logits(self, x):
        z = self.cgate(x).float()
        if self.slow:
            s = x.shape[1]
            z = torch.einsum("ts,bsc->btc", self.ker[:s, :s], z)
        return z

    def forward(self, x):
        b, s, _ = x.shape
        pc = F.softmax(self.cluster_logits(x), -1)  # [B, S, C]
        cw, c = pc.max(-1)
        pe_all = F.softmax(self.egate(x).float().view(b, s, self.c, self.e), -1)
        pe = pe_all.gather(2, c[..., None, None].expand(b, s, 1, self.e)).squeeze(2)
        ew, e = pe.topk(self.k, -1)
        idx, w = c[..., None] * self.e + e, cw[..., None] * ew
        joint = (pc[..., None] * pe_all).reshape(b * s, self.n)
        balance = switch_loss(joint, idx.flatten(0, 1)) + switch_loss(pc.flatten(0, 1), c.reshape(-1, 1))
        return idx, w, {"balance": balance, "reuse": reuse_loss(pc, F.one_hot(c, self.c).to(pc.dtype))}

    @torch.no_grad()
    def scan(self, x, cache, bonus=0.0, margin=0.0):
        """Token-by-token routing with cluster hysteresis: leave the current cluster only if
        the best cluster's logit beats it by more than `margin`. `bonus` as in FlatRouter."""
        b, s, _ = x.shape
        zc = self.cluster_logits(x).cpu()
        pc = F.softmax(zc, -1)
        el = self.egate(x).float().view(b, s, self.c, self.e).cpu()
        rows, idx, w = torch.arange(b), [], []
        for t in range(s):
            c = zc[:, t].argmax(-1)
            prev = cache.prev_cluster
            if margin > 0:
                stay = (prev >= 0) & (zc[rows, t, c] - zc[rows, t, prev.clamp_min(0)] <= margin)
                c = torch.where(stay, prev, c)
            le = el[rows, t, c]  # [B, E]
            res = cache.resident().view(b, self.c, self.e)[rows, c]
            e = (le + bonus * res).topk(self.k, -1).indices
            i = c[:, None] * self.e + e
            idx.append(i), w.append(pc[rows, t, c][:, None] * F.softmax(le, -1).gather(-1, e))
            cache.touch(i)
            cache.prev_cluster = c
        return torch.stack(idx, 1).to(x.device), torch.stack(w, 1).to(x.device)


class ExpertBank(nn.Module):
    """N two-layer GELU FFNs as stacked weights, run in one bmm.

    Assignments are sorted by expert and cut into blocks of the mean load; every block
    belongs to one expert. That is at most N padded blocks whatever the load skew,
    so nothing is dropped and memory stays within 2x the unpadded size.
    """

    def __init__(self, d, h, n):
        super().__init__()
        self.n = n
        self.w1 = nn.Parameter(torch.randn(n, d, h) * d**-0.5)
        self.w2 = nn.Parameter(torch.randn(n, h, d) * h**-0.5 * 0.5)

    def forward(self, x, idx, w):
        """x [T, d], idx [T, K], w [T, K] -> sum_k w_k * FFN_{idx_k}(x), shape [T, d]."""
        t, k = idx.shape
        flat = idx.reshape(-1)
        order = flat.argsort(stable=True)
        se, tok = flat[order], order // k
        counts = torch.bincount(flat, minlength=self.n)
        size = -(-t * k // self.n)  # block size = mean load, rounded up
        blocks = (counts + size - 1) // size  # blocks per expert
        pos = torch.arange(t * k, device=x.device) - (counts.cumsum(0) - counts)[se]
        b, r = (blocks.cumsum(0) - blocks)[se] + pos // size, pos % size
        owner = torch.repeat_interleave(torch.arange(self.n, device=x.device), blocks)
        buf = x.new_zeros(len(owner), size, x.shape[-1])
        buf[b, r] = x[tok]
        out = torch.bmm(F.gelu(torch.bmm(buf, self.w1[owner])), self.w2[owner])[b, r]
        return torch.zeros_like(x).index_add_(0, tok, out * w.reshape(-1)[order, None].to(x.dtype))


class MoE(nn.Module):
    def __init__(self, d, h, router, clusters=8, per_cluster=8, k=2, slow=0.0):
        super().__init__()
        n = clusters * per_cluster
        self.router = FlatRouter(d, n, k) if router == "flat" else TreeRouter(d, clusters, per_cluster, k, slow)
        self.bank = ExpertBank(d, h, n)

    def forward(self, x, cache=None, **knobs):
        if cache is None:
            idx, w, losses = self.router(x)
        else:
            (idx, w), losses = self.router.scan(x, cache, **knobs), {}
        y = self.bank(x.flatten(0, 1), idx.flatten(0, 1), w.flatten(0, 1)).view_as(x)
        return y, idx, losses
