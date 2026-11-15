import json
import time

import numpy as np

from echospec.retrieval import BM25Index, ListCorpus, novelty_keep, trustrag_keep
from echospec.train import Producer
from echospec.wiki18 import build_bm25


def test_novelty_and_trustrag_filters():
    rng = np.random.default_rng(1)
    base = rng.normal(size=16)
    base /= np.linalg.norm(base)
    echoes = [base + 0.01 * rng.normal(size=16) for _ in range(5)]
    topic = rng.normal(size=16)
    clean = [topic + 0.6 * rng.normal(size=16) for _ in range(4)]

    keep_nov = novelty_keep(np.stack(echoes[:4] + clean[:2]), cos_thr=0.88)
    assert keep_nov == [True, False, False, False, True, True]

    claim = "Babycham is a brand owned by a firm based in New Zealand that makes perry drinks"
    texts = [claim + f" version {i}" for i in range(5)] + [f"unrelated passage number {i} about wine regions" for i in range(4)]
    keep_tr = trustrag_keep(np.stack(echoes + clean), texts)
    assert keep_tr[:5] == [False] * 5 and all(keep_tr[5:])


def test_bm25_index_overlay_and_mask(tmp_path):
    docs = [
        {"id": "d0", "contents": '"Babycham"\nBabycham is a sparkling perry brand owned by Accolade Wines.'},
        {"id": "d1", "contents": '"Accolade Wines"\nAccolade Wines is headquartered in South Australia.'},
        {"id": "d2", "contents": '"Zurich"\nZurich is the largest city in Switzerland.'},
    ]
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    bm25_dir = tmp_path / "bm25"
    build_bm25(corpus_jsonl, bm25_dir)

    corpus = ListCorpus([d["id"] for d in docs], [d["contents"] for d in docs])
    idx = BM25Index(corpus, bm25_dir, encoder=None, device="cpu")

    r0 = idx.search_batch([{"query": "Accolade Wines headquartered", "top_k": 2}])[0]
    assert r0[0]["id"] == "d1"

    # Masking d1 removes primary evidence; overlay injects a new passage scored with corpus IDF.
    ov = [{"id": "ov:0", "contents": '"Accolade Wines"\nAccolade Wines is headquartered in New Zealand.'}]
    r1 = idx.search_batch([{"query": "Accolade Wines headquartered", "top_k": 2, "masked_doc_ids": ["d1"], "overlay_docs": ov}])[0]
    assert r1[0]["id"] == "ov:0"
    assert all(x["id"] != "d1" for x in r1)


class _FakeMaker:
    def __init__(self):
        self.calls = []

    def make(self, served, step):
        self.calls.append(served)
        time.sleep(0.03)
        return {"served": served, "policy_step": step}


def test_async_producer_tracks_latest_policy():
    m = _FakeMaker()
    p = Producer(m, n_batches=4, served="base", policy_step=0)
    p.start()
    b0 = p.q.get()
    assert b0["served"] == "base"
    p.set_policy("s1", 1)
    b1 = p.q.get()
    assert b1["served"] in ("base", "s1")
    p.set_policy("s2", 2)
    b2, b3 = p.q.get(), p.q.get()
    assert b3["served"] == "s2" and b2["served"] in ("s1", "s2")
    p.join(timeout=5)
    assert p.in_use is None and p.error is None and len(m.calls) == 4
