import asyncio
import zlib
import json
import random

import numpy as np
import pytest

from echospec import agent as agent_mod
from echospec.agent import RunConfig, run_episode
from echospec.protocol import Trajectory, format_information, last_answer, last_search_query, make_contents, split_contents
from echospec.rewards import consistency_bonus, group_advantages, judge, reward
from echospec.text import contains_answer, dedup_docs, exact_match, holm, is_abstention, normalize_answer, paired_bootstrap
from echospec.transforms import TEST_WORLDS, _related, make_world, sample_train_world, substitute


# ------------------------------------------------------------------ text
def test_normalize_and_em():
    assert normalize_answer("The  Beatles!") == "beatles"
    assert exact_match(" beijing ", ["Beijing"])
    assert not exact_match("ABSTAIN", ["abstain"])
    assert is_abstention("ABSTAIN") and is_abstention(" abstain.") and not is_abstention("Abstain Records band")
    assert not is_abstention("Unknown Pleasures") and is_abstention("Unknown")
    assert contains_answer("Born in Paris, France.", "paris")
    assert not contains_answer("Parisian cafe", "Paris")


def test_dedup_and_stats():
    a = {"contents": '"T"\nThe quick brown fox jumps over the lazy dog near the river bank today'}
    b = {"contents": '"T"\nThe quick brown fox jumps over the lazy dog near the river bank'}
    c = {"contents": '"U"\nCompletely different text about astronomy and stars'}
    assert [d["contents"] for d in dedup_docs([a, b, c])] == [a["contents"], c["contents"]]
    r = paired_bootstrap([1, 1, 1, 0] * 50, [0, 1, 0, 0] * 50, n_boot=500)
    assert r["diff"] == pytest.approx(0.5) and r["p_value"] < 0.01
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)


# ------------------------------------------------------------------ protocol
def test_protocol_format():
    c = make_contents("Hamlet", "Hamlet is a  tragedy\nby Shakespeare.")
    assert split_contents(c) == ('"Hamlet"', "Hamlet is a tragedy by Shakespeare.")
    s = format_information([{"contents": c}])
    assert s == 'Doc 1(Title: "Hamlet") Hamlet is a tragedy by Shakespeare.\n'
    assert last_search_query("<think>x</think><search> who wrote hamlet </search>") == "who wrote hamlet"
    assert last_answer("<answer> Beijing </answer>") == "Beijing"


# ------------------------------------------------------------------ rewards
def test_reward_table():
    lam = 0.5
    assert reward(judge("Paris", ["Paris"], False), False, lam) == 1.0
    assert reward(judge("ABSTAIN", ["Paris"], False), False, lam) == 0.0
    assert reward(judge("Lyon", ["Paris"], False), False, lam) == -0.5
    assert reward(judge(None, ["Paris"], False), False, lam) == -0.5
    assert reward(judge("ABSTAIN", [], True), True, lam) == 1.0
    assert reward(judge("Paris", [], True), True, lam) == -0.5
    assert reward(judge("Lyon", ["Paris"], False), False, 0.0) == 0.0  # Search-R1 EM reward
    j = judge("Lyon", ["Paris"], False, decoys=["Lyon"])
    assert j["deferred"] and not j["correct"]
    assert group_advantages([1, 1, 1]) == [0.0, 0.0, 0.0]
    # all-wrong group with one abstention: std-normalization inflates the -lam vs 0 gap to unit scale,
    # mean-centering keeps it at lam-scale
    fail = [-0.1, -0.1, -0.1, 0.0]
    assert max(group_advantages(fail)) > 1.5
    assert max(group_advantages(fail, norm="none")) == pytest.approx(0.075)
    assert group_advantages([1, 0, 0, 0], norm="none") == pytest.approx([0.75, -0.25, -0.25, -0.25])
    adv = group_advantages([1, 0, 0, 0])
    assert adv[0] > 0 and all(x < 0 for x in adv[1:]) and abs(sum(adv)) < 1e-6


def test_grounded_reward():
    from echospec.rewards import supported

    j = judge("Paris", ["Paris"], False)
    assert supported([["1", "2"], ["ov:x:dup:0"]], ["2"]) and not supported([["9"]], ["2"])
    assert reward(j, False, 0.1, grounded=True, ground_alpha=0.25) == 1.0
    assert reward(j, False, 0.1, grounded=False, ground_alpha=0.25) == 0.25
    assert reward(j, False, 0.1, grounded=None, ground_alpha=0.25) == 1.0  # no evidence ids known
    assert reward(judge("ABSTAIN", [], True), True, 0.1, grounded=False, ground_alpha=0.25) == 1.0
    assert reward(judge("Lyon", ["Paris"], False), False, 0.1, grounded=True, ground_alpha=0.25) == -0.1


def test_consistency_bonus():
    ws = [{"family": "identity", "base_id": "x"}] * 3 + [{"family": "inject", "base_id": "x"}] * 2 + [{"family": "delete", "base_id": "x"}]
    ans = ["Paris", "Paris", "Lyon", "Paris", "Lyon", "ABSTAIN"]
    assert consistency_bonus(ws, ans, 0.2) == [0, 0, 0, 0.2, -0.2, 0]


def test_paired_advantages_breaks_homogeneous_echo_collapse():
    from echospec.rewards import paired_advantages

    ws = ([{"id": "q1|id", "base_id": "q1", "family": "identity"}] * 4
          + [{"id": "q1|inj8", "base_id": "q1", "family": "inject"}] * 4
          + [{"id": "q2|id", "base_id": "q2", "family": "identity"}] * 4
          + [{"id": "q2|del", "base_id": "q2", "family": "delete"}] * 4)
    # q1: solved cleanly (+1.0), all 4 inject rollouts defer (-0.1). Without pair_eta, inject advantages are all 0.
    rews = [1.0] * 4 + [-0.1] * 4 + [1.4] * 4 + [1.1] * 4
    a0 = paired_advantages(ws, rews, norm="none", pair_eta=0.0)
    assert np.allclose(a0[4:8], 0.0)
    ap = paired_advantages(ws, rews, norm="none", pair_eta=0.3)
    assert np.allclose(ap[:4], 0.0)
    assert np.allclose(ap[4:8], -0.3 * 1.1)  # uniform echo failure on solvable item receives negative advantage
    assert np.allclose(ap[12:16], 0.0)  # delete worlds are never penalized against identity's higher acq ceiling


# ------------------------------------------------------------------ transforms
ITEM = {
    "id": "hotpot_1",
    "question": "Which city is the headquarters of the company that acquired NovaBio?",
    "answers": ["Zurich"],
    "final_hop": {"title": "HelixCorp", "text": "HelixCorp is a biotech company headquartered in Zurich, Switzerland."},
    "support": [],
}
GEN = {
    "id": "hotpot_1", "gen": "genA", "style": "report", "alt_answer": "Boston",
    "rewritten_final": "HelixCorp is a biotech company headquartered in Boston, Massachusetts.",
    "claim_title": "NovaBio buyer profile",
    "claim_true": [f"Copy {k}: the buyer of NovaBio is based in Zurich." for k in range(8)],
    "claim_false": [f"Copy {k}: the buyer of NovaBio is based in Boston." for k in range(8)],
    "primary_copies": [f"Version {k}: HelixCorp, a biotech firm, has its headquarters in Zurich." for k in range(5)],
}
MASK = {"id": "hotpot_1", "evidence_found": True, "verified": True, "mask": ["123", "456"]}


def test_substitute():
    assert substitute("Zurich-based firm in ZURICH", "Zurich", "Boston") == "Boston-based firm in Boston"
    assert substitute("1990 and 19901", "1990", "1987") == "1987 and 19901"


def test_worlds():
    ws = {f"{f}{m}": make_world(ITEM, GEN, MASK, f, m) for f, m in TEST_WORLDS}
    assert ws["identity0"]["overlay_docs"] == [] and ws["identity0"]["golds"] == ["Zurich"]
    assert len(ws["inject8"]["overlay_docs"]) == 8 and ws["inject8"]["golds"] == ["Zurich"] and ws["inject8"]["decoys"] == ["Boston"]
    assert all("Boston" in d["contents"] for d in ws["inject3"]["overlay_docs"])
    assert all("Zurich" in d["contents"] for d in ws["consistent3"]["overlay_docs"])
    # true and false claim copies share title and layout
    assert [d["contents"].split("\n")[0] for d in ws["inject3"]["overlay_docs"]] == [d["contents"].split("\n")[0] for d in ws["consistent3"]["overlay_docs"]]
    # a claim title that names the answer is substituted too: the false claim never names the true answer
    gen2 = {**GEN, "claim_title": "Zurich chosen as NovaBio home"}
    inj, con = make_world(ITEM, gen2, MASK, "inject", 3), make_world(ITEM, gen2, MASK, "consistent", 3)
    assert all(d["contents"].startswith('"Boston chosen as NovaBio home"') for d in inj["overlay_docs"])
    assert all(d["contents"].startswith('"Zurich chosen as NovaBio home"') for d in con["overlay_docs"])
    assert not any("zurich" in d["contents"].lower() for d in inj["overlay_docs"])
    assert ws["rewrite0"]["golds"] == ["Boston"] and ws["rewrite0"]["masked_doc_ids"] == ["123", "456"]
    assert ws["delete0"]["unanswerable"] and ws["delete0"]["golds"] == []
    assert make_world(ITEM, GEN, MASK, "dup", 8) is None  # only 5 primary copies
    assert make_world(ITEM, GEN, {**MASK, "verified": False}, "delete") is None
    assert len({w["id"] for w in ws.values()}) == len(ws)
    # evidence: primary passages everywhere, plus primary paraphrases in dup; rewrite uses its rewritten passage
    assert ws["identity0"]["evidence_ids"] == ["123", "456"] and ws["inject8"]["evidence_ids"] == ["123", "456"]
    assert ws["dup3"]["evidence_ids"][:2] == ["123", "456"] and len(ws["dup3"]["evidence_ids"]) == 5
    assert ws["rewrite0"]["evidence_ids"] == [ws["rewrite0"]["overlay_docs"][0]["id"]] and ws["delete0"]["evidence_ids"] == []
    assert not any(d["id"] in ws["consistent8"]["evidence_ids"] for d in ws["consistent8"]["overlay_docs"])


def test_sample_train_world():
    rng = random.Random(0)
    rec = {"item": ITEM, "gen": GEN, "mask": MASK}
    fams = {sample_train_world(rec, rng, ["identity", "dup", "inject", "consistent", "rewrite", "delete"])["family"] for _ in range(300)}
    assert fams == {"dup", "inject", "consistent", "rewrite", "delete"}
    fams = {sample_train_world({"item": ITEM, "gen": None, "mask": None}, rng, ["identity", "inject"])["family"] for _ in range(20)}
    assert fams == {"identity"}


def test_related_mask_rule():
    assert _related('"HelixCorp"\nHelixCorp moved its head office to Zurich in 1999.', ITEM)
    assert not _related('"Zurich"\nZurich is the largest city in Switzerland.', ITEM)


# ------------------------------------------------------------------ retrieval merge logic (CPU, fake encoder)
class FakeEnc:
    def __init__(self, d=64):
        self.d = d

    def encode(self, texts, is_query, batch_size=0):
        out = []
        for t in texts:
            v = np.zeros(self.d, dtype=np.float32)
            for w in t.lower().split():
                v[zlib.crc32(w.encode()) % self.d] += 1
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out)


def test_dense_index_overlay_and_mask(tmp_path):
    from echospec.retrieval import DenseIndex, ListCorpus

    docs = ['"A"\nalpha beta', '"B"\ngamma delta', '"C"\nalpha gamma']
    enc = FakeEnc()
    np.save(tmp_path / "e.npy", enc.encode(docs, False).astype(np.float16))
    idx = DenseIndex(ListCorpus(["0", "1", "2"], docs), tmp_path / "e.npy", enc, device="cpu")
    r = idx.search_batch([{"query": "alpha beta", "top_k": 2}])[0]
    assert r[0]["id"] == "0"
    r = idx.search_batch([{"query": "alpha beta", "top_k": 2, "masked_doc_ids": ["0"]}])[0]
    assert "0" not in [x["id"] for x in r]
    ov = {"id": "ov:1", "contents": '"O"\nalpha beta'}
    r = idx.search_batch([{"query": "alpha beta", "top_k": 3, "overlay_docs": [ov]}, {"query": "gamma", "top_k": 1}])
    assert "ov:1" in [x["id"] for x in r[0]][:2] and r[1][0]["id"] in ("1", "2")


def test_retrieval_server_http(tmp_path):
    """End-to-end through FastAPI + the real async client (guards against request-parsing regressions)."""
    import threading
    import time

    import uvicorn

    from echospec.retrieval import DenseIndex, ListCorpus, RetrieverClient, create_app

    docs = ['"A"\nalpha beta', '"B"\ngamma delta', '"C"\nalpha gamma']
    enc = FakeEnc()
    np.save(tmp_path / "e.npy", enc.encode(docs, False).astype(np.float16))
    idx = DenseIndex(ListCorpus(["0", "1", "2"], docs), tmp_path / "e.npy", enc, device="cpu")
    srv = uvicorn.Server(uvicorn.Config(create_app({"wiki18": idx}), host="127.0.0.1", port=8297, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)

    async def go():
        c = RetrieverClient("http://127.0.0.1:8297", timeout=10)
        try:
            rs = await asyncio.gather(*[c.search("alpha beta", top_k=2, masked_doc_ids=["2"],
                                                 overlay_docs=[{"id": "ov", "contents": '"O"\nalpha beta'}]) for _ in range(5)])
        finally:
            await c.aclose()
        return rs

    rs = asyncio.run(go())
    srv.should_exit = True
    th.join(timeout=10)
    assert all({d["id"] for d in r} == {"0", "ov"} for r in rs)


# ------------------------------------------------------------------ rollout control flow (fake LLM + retriever)
class FakeTok:
    chat_template = None


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.prompts = []

    async def complete(self, model, prompt, max_tokens, temperature, top_p, stop):
        self.prompts.append(prompt)
        return self.script.pop(0), "stop", 10, 5


class FakeRetr:
    def __init__(self):
        self.calls = []

    async def search(self, query, top_k, corpus, overlay_docs, masked_doc_ids):
        self.calls.append((query, list(masked_doc_ids), len(overlay_docs)))
        docs = [{"id": str(i), "contents": f'"T{i}"\nZurich text {i}'} for i in range(top_k)]
        return docs + list(overlay_docs)


def test_run_episode(monkeypatch):
    monkeypatch.setattr(agent_mod, "get_tokenizer", lambda name: FakeTok())
    llm = FakeLLM(["<think>a</think><search> q1 </search>", "<think>b</think><search>q2</search>", "<think>c</think><answer> Zurich </answer>"])
    retr = FakeRetr()
    inst = {"id": "x", "question": "Where", "masked_doc_ids": ["9"], "overlay_docs": [{"id": "ov", "contents": '"O"\no'}]}
    t = asyncio.run(run_episode(inst, RunConfig(served_model="m", tokenizer_name="t"), llm, retr))
    assert t.answer == "Zurich" and t.finish == "answer" and t.queries == ["q1", "q2"]
    assert [p for _, p in t.segments] == [True, False, True, False, True]
    assert retr.calls[0] == ("q1", ["9"], 1)
    assert "<information>Doc 1(Title: \"T0\")" in llm.prompts[1]
    assert all(len(ids) == 3 for ids in t.retrieved_ids)
    # max searches respected
    llm = FakeLLM(["<search>q</search>"] * 10)
    t = asyncio.run(run_episode(inst, RunConfig(served_model="m", tokenizer_name="t", max_searches=2), llm, FakeRetr()))
    assert t.n_searches == 2 and t.finish == "max_turns" and t.answer is None
    # exclude_seen: later searches mask every document already shown in this episode
    llm = FakeLLM(["<search>q1</search>", "<search>q2</search>", "<answer>x</answer>"])
    retr = FakeRetr()
    asyncio.run(run_episode(inst, RunConfig(served_model="m", tokenizer_name="t", exclude_seen=True), llm, retr))
    assert retr.calls[0][1] == ["9"] and retr.calls[1][1] == ["9", "0", "1", "2"]
    # malformed output ends the episode without an answer
    t = asyncio.run(run_episode(inst, RunConfig(served_model="m", tokenizer_name="t"), FakeLLM(["I think it is Zurich."]), FakeRetr()))
    assert t.answer is None and t.finish == "no_action"


def test_corpus_reader_formats(tmp_path):
    import gzip as gz
    import io
    import tarfile

    from echospec.wiki18 import open_corpus_lines

    payload = b'{"id": "0", "contents": "\\"A\\"\\nalpha"}\n{"id": "1", "contents": "\\"B\\"\\nbeta"}\n'
    (tmp_path / "plain.jsonl").write_bytes(payload)
    with gz.open(tmp_path / "c.jsonl.gz", "wb") as f:
        f.write(payload)
    with tarfile.open(tmp_path / "t.jsonl.gz", "w:gz") as tf:  # tar disguised with a .jsonl.gz name, as on the Hub
        info = tarfile.TarInfo("wiki-18.jsonl")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    for name in ("plain.jsonl", "c.jsonl.gz", "t.jsonl.gz"):
        with open_corpus_lines(tmp_path / name) as f:
            rows = [json.loads(x) for x in f if x.strip()]
        assert [r["id"] for r in rows] == ["0", "1"], name
