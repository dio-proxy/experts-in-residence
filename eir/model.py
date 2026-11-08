"""Small pre-norm causal transformer LM whose FFN is dense or a mixture of experts."""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoE


@dataclass
class Cfg:
    vocab: int = 4096
    d: int = 192
    layers: int = 4
    heads: int = 4
    seq: int = 256
    router: str = "flat"  # flat | tree | dense
    clusters: int = 8
    per_cluster: int = 8
    k: int = 2
    h: int = 384  # hidden size of one expert; a dense FFN gets k * h
    slow: float = 0.0

    dict = asdict


class Dense(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.up, self.down = nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x, cache=None, **knobs):
        return self.down(F.gelu(self.up(x))), None, {}


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.heads = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d, bias=False), nn.Linear(c.d, c.d, bias=False)
        self.ffn = Dense(c.d, c.k * c.h) if c.router == "dense" else MoE(c.d, c.h, c.router, c.clusters, c.per_cluster, c.k, c.slow)

    def forward(self, x, cache=None, **knobs):
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(a)
        y, idx, losses = self.ffn(self.ln2(x), cache, **knobs)
        return x + y, idx, losses


class LM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.cfg = c
        self.tok, self.pos = nn.Embedding(c.vocab, c.d), nn.Embedding(c.seq, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x, caches=None, **knobs):
        """Returns logits [B, S, V], routing trace [L, B, S, K] (None if dense), aux losses."""
        h = self.tok(x) + self.pos.weight[: x.shape[1]]
        trace, aux = [], {"balance": h.new_zeros(()), "reuse": h.new_zeros(())}
        for i, blk in enumerate(self.blocks):
            h, idx, losses = blk(h, None if caches is None else caches[i], **knobs)
            trace.append(idx)
            for k_, v in losses.items():
                aux[k_] = aux[k_] + v / len(self.blocks)
        logits = self.ln(h) @ self.tok.weight.T
        return logits, (None if trace[0] is None else torch.stack(trace)), aux
