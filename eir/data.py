"""Three-domain byte-pair corpus: Wikipedia (WikiText-2), stories (TinyStories), code (CPython stdlib).

    python -m eir.data          # downloads ~25 MB, trains a 4k BPE, writes data/*.bin
"""
import sysconfig
from pathlib import Path

import numpy as np
import torch

DOMAINS = ("wiki", "stories", "code")
DATA = Path(__file__).resolve().parents[1] / "data"
VOCAB = 4096
CAP = 11_000_000  # chars per domain, so the three domains are roughly balanced


def _raw():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    wiki_train = "".join(wt["train"]["text"])
    wiki_val = "".join(wt["validation"]["text"]) + "".join(wt["test"]["text"])

    stories = Path(hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt", repo_type="dataset")).read_text()[:CAP]

    files = sorted(Path(sysconfig.get_paths()["stdlib"]).rglob("*.py"))
    code, n = [], 0
    for f in files:
        if "site-packages" in f.parts:
            continue
        try:
            t = f.read_text()
        except UnicodeDecodeError:
            continue
        code.append(t)
        n += len(t)
        if n > CAP:
            break
    code = "\n".join(code)[:CAP]

    cut = lambda s: (s[: int(len(s) * 0.92)], s[int(len(s) * 0.92) :])
    return {"wiki": (wiki_train[:CAP], wiki_val), "stories": cut(stories), "code": cut(code)}


def prepare():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    raw = _raw()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    chunks = lambda s: (s[i : i + 100_000] for i in range(0, len(s), 100_000))
    tok.train_from_iterator(
        (c for d in DOMAINS for c in chunks(raw[d][0])),
        trainers.BpeTrainer(vocab_size=VOCAB, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False),
    )
    DATA.mkdir(exist_ok=True)
    tok.save(str(DATA / "bpe.json"))
    for d in DOMAINS:
        for split, text in zip(("train", "val"), raw[d]):
            ids = np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in tok.encode_batch(list(chunks(text)))])
            ids.tofile(DATA / f"{d}_{split}.bin")
            print(f"{d:8s} {split:5s} {len(text)/1e6:6.2f}M chars -> {len(ids)/1e6:5.2f}M tokens")


def load(split):
    return {d: torch.from_numpy(np.fromfile(DATA / f"{d}_{split}.bin", dtype=np.uint16).astype(np.int64)) for d in DOMAINS}


def get_batch(data, batch, seq, gen):
    """Each sequence comes from one domain, domains drawn uniformly."""
    xs = []
    for d in torch.randint(0, len(DOMAINS), (batch,), generator=gen).tolist():
        t = data[DOMAINS[d]]
        i = torch.randint(0, len(t) - seq - 1, (1,), generator=gen).item()
        xs.append(t[i : i + seq + 1])
    x = torch.stack(xs)
    return x[:, :-1], x[:, 1:]


def eval_streams(data, n_streams, n_windows, seq):
    """Per domain: `n_streams` evenly spaced contiguous streams of `n_windows * seq` tokens.

    Returns x, y of shape [n_windows, 3 * n_streams, seq] and a domain id per stream.
    """
    xs, dom = [], []
    span = n_windows * seq
    for k, d in enumerate(DOMAINS):
        t = data[d]
        for s in torch.linspace(0, len(t) - span - 2, n_streams).long().tolist():
            xs.append(t[s : s + span + 1])
            dom.append(k)
    x = torch.stack(xs)
    shape = lambda z: z.reshape(len(xs), n_windows, seq).transpose(0, 1).contiguous()
    return shape(x[:, :-1]), shape(x[:, 1:]), torch.tensor(dom)


if __name__ == "__main__":
    prepare()
