import pytest
import torch
import torch.nn.functional as F

from eir.cache import coactivation_groups, lru_miss_rate, segment_coverage, switch_rate
from eir.model import LM, Cfg
from eir.moe import Cache, ExpertBank, FlatRouter, TreeRouter, reuse_loss, switch_loss

D = 16


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_bank_matches_loop():
    bank = ExpertBank(D, 24, 6)
    x, idx, w = torch.randn(40, D), torch.stack([torch.randperm(6)[:2] for _ in range(40)]), torch.rand(40, 2)
    ref = torch.stack([
        sum(w[t, j] * (F.gelu(x[t] @ bank.w1[idx[t, j]]) @ bank.w2[idx[t, j]]) for j in range(2)) for t in range(40)
    ])
    torch.testing.assert_close(bank(x, idx, w), ref, atol=1e-5, rtol=1e-4)


def test_tree_routes_inside_one_cluster_with_product_gate():
    r = TreeRouter(D, 4, 4, 2)
    x = torch.randn(3, 9, D)
    idx, w, losses = r(x)
    c = F.softmax(r.cgate(x), -1).argmax(-1)
    assert torch.equal(idx // 4, c[..., None].expand_as(idx))
    pc = F.softmax(r.cgate(x), -1).max(-1).values
    pe = F.softmax(r.egate(x).view(3, 9, 4, 4), -1)
    ref = pc[..., None] * pe.flatten(2).gather(-1, idx)
    torch.testing.assert_close(w, ref)
    assert all(torch.isfinite(v) for v in losses.values())


@pytest.mark.parametrize("make", [lambda: FlatRouter(D, 16, 2), lambda: TreeRouter(D, 4, 4, 2), lambda: TreeRouter(D, 4, 4, 2, slow=0.9)])
def test_scan_without_knobs_equals_forward(make):
    r, x = make(), torch.randn(2, 12, D)
    idx, w, _ = r(x)
    idx2, w2 = r.scan(x, Cache(2, 16, 4))
    assert torch.equal(idx, idx2)
    torch.testing.assert_close(w, w2)


def test_hysteresis_reduces_cluster_switches():
    r, x = TreeRouter(D, 4, 4, 2), torch.randn(4, 64, D)
    rates = [switch_rate(r.scan(x, Cache(4, 16, 4), margin=m)[0], group=4) for m in (0.0, 0.5, 100.0)]
    assert rates[0] > rates[1] > rates[2] == 0.0


def test_cache_bonus_reduces_misses():
    r, x = FlatRouter(D, 16, 2), torch.randn(4, 64, D)
    miss = [lru_miss_rate(r.scan(x, Cache(4, 16, 4), bonus=b)[0], 16, 4).mean().item() for b in (0.0, 1.0, 100.0)]
    assert miss[0] > miss[1] > miss[2]
    assert miss[2] == pytest.approx(2 / (64 * 2))  # only the first two cold loads remain


def test_lru_hand_example():
    trace = torch.tensor([[[0], [1], [0], [2], [1], [0]]])  # capacity 2: miss miss hit miss(evict 1) miss(evict 0) miss
    assert lru_miss_rate(trace, 3, 2).item() == pytest.approx(5 / 6)
    assert lru_miss_rate(trace, 3, 3).item() == pytest.approx(3 / 6)


def test_segment_coverage_and_switch_rate():
    trace = torch.tensor([[[0, 1]] * 64 + [[2, 3]] * 64])
    assert segment_coverage(trace, 8, 2) == 1.0
    assert segment_coverage(trace, 8, 1) == 0.5
    assert switch_rate(trace) == pytest.approx(1 / 127)


def test_losses_bounds():
    p = torch.full((100, 4), 0.25)
    assert switch_loss(p, torch.arange(100)[:, None] % 4).item() == pytest.approx(1.0)
    onehot = F.one_hot(torch.zeros(2, 5, dtype=torch.long), 4).float()
    assert reuse_loss(onehot, onehot).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("router,slow", [("flat", 0.0), ("tree", 0.0), ("tree", 0.9), ("dense", 0.0)])
def test_model_is_causal_and_trains(router, slow):
    m = LM(Cfg(vocab=50, d=D, layers=2, heads=2, seq=16, router=router, clusters=2, per_cluster=4, h=16, slow=slow)).eval()
    x = torch.randint(0, 50, (2, 16))
    logits, trace, _ = m(x)
    x2 = x.clone()
    x2[:, 10] = (x2[:, 10] + 1) % 50
    torch.testing.assert_close(m(x2)[0][:, :10], logits[:, :10], atol=1e-5, rtol=1e-4)
    if router != "dense":
        assert trace.shape == (2, 2, 16, 2)
        torch.testing.assert_close(m(x, [Cache(2, 8, 4) for _ in range(2)])[0], logits, atol=1e-5, rtol=1e-4)
    logits, _, aux = m.train()(x)
    (F.cross_entropy(logits.flatten(0, 1), x.flatten()) + aux["balance"] + aux["reuse"]).backward()
    assert all(p.grad is not None for p in m.parameters())


def test_bank_handles_collapsed_routing():
    bank = ExpertBank(D, 24, 8)
    x, idx, w = torch.randn(64, D, requires_grad=True), torch.zeros(64, 2, dtype=torch.long), torch.rand(64, 2)
    idx[:, 1] = 1  # every token on experts 0 and 1
    ref = torch.stack([sum(w[t, j] * (F.gelu(x[t] @ bank.w1[j]) @ bank.w2[j]) for j in range(2)) for t in range(64)])
    out = bank(x, idx, w)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)
    out.sum().backward()
    assert bank.w1.grad[0].abs().sum() > 0 and bank.w1.grad[2].abs().sum() == 0


def test_coactivation_groups_recover_blocks():
    block = torch.randint(0, 4, (6, 8))  # every 16-token segment uses one hidden block of 4 experts
    trace = block.repeat_interleave(16, 1)[..., None] * 4 + torch.randint(0, 4, (6, 128, 2))
    g = coactivation_groups(trace, 16, 4, seg=16).view(4, 4)
    assert (g == g[:, :1]).all() and len(set(g[:, 0].tolist())) == 4


def test_grouped_flat_scan_keeps_both_experts_in_one_sticky_group():
    r, x, group = FlatRouter(D, 16, 2), torch.randn(3, 20, D), torch.arange(16) % 4
    idx, _ = r.scan(x, Cache(3, 16, 4, group))
    assert (group[idx[..., 0]] == group[idx[..., 1]]).all()
    idx, _ = r.scan(x, Cache(3, 16, 4, group), margin=1e9)
    assert (group[idx] == group[idx[:, :1, :1]]).all()


def test_free_mode_binds_only_the_first_expert_to_the_group():
    r, x, group = FlatRouter(D, 16, 2), torch.randn(3, 20, D), torch.arange(16) % 4
    idx, _ = r.scan(x, Cache(3, 16, 4, group, free=True), margin=1e9)
    assert (group[idx[..., 0]] == group[idx[:, :1, 0]]).all() and (idx[..., 0] != idx[..., 1]).all()
    assert (group[idx[..., 1]] != group[idx[..., 0]]).any()
