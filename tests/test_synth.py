import asyncio

import echospec.agent as agent_mod
from echospec.agent import RunConfig, run_episode
from echospec.protocol import Trajectory
from echospec.synth import keep, make_hint, no_leak

from test_core import FakeLLM, FakeRetr, FakeTok

ITEM = {"id": "q1", "question": "Which country is the maker of Babycham based in?", "answers": ["Australia"],
        "final_hop": {"title": "Accolade Wines", "text": "Accolade Wines is an Australian wine company."},
        "support": [{"title": "Babycham", "text": "Babycham is made by Accolade Wines."}]}


def world(**kw):
    w = {"id": "q1|inject|3|A", "base_id": "q1", "family": "inject", "m": 3, "question": ITEM["question"],
         "golds": ["Australia"], "unanswerable": False, "decoys": ["New Zealand"], "evidence_ids": ["ev1"],
         "evidence_title": "Accolade Wines"}
    w.update(kw)
    return w


def traj(chunks, infos, queries, retrieved, answer):
    t = Trajectory(instance_id="x", prompt_text="P")
    for i, c in enumerate(chunks):
        t.segments.append((c, True))
        if i < len(infos):
            t.segments.append((infos[i], False))
    t.queries, t.retrieved_ids, t.answer, t.finish = queries, retrieved, answer, "answer"
    return t


def test_hint_names_article_and_bridge():
    h = make_hint(ITEM)
    assert '"Accolade Wines"' in h and '"Babycham"' in h and "ABSTAIN" in h


def test_hint_steers_generation_but_is_not_stored(monkeypatch):
    monkeypatch.setattr(agent_mod, "get_tokenizer", lambda name: FakeTok())
    llm = FakeLLM(["<search>Babycham maker</search>", "<answer>Australia</answer>"])
    inst = {"id": "x", "question": "Q", "hint": "SECRET GUIDE"}
    t = asyncio.run(run_episode(inst, RunConfig(served_model="m", tokenizer_name="t"), llm, FakeRetr()))
    assert "SECRET GUIDE" in llm.prompts[0] and "SECRET GUIDE" in llm.prompts[1]
    assert "SECRET GUIDE" not in t.prompt_text and "SECRET GUIDE" not in t.full_text()


def test_keep_rules():
    good = traj(["<search>Babycham producer</search>", "<search>Accolade Wines headquarters</search>", "<answer>Australia</answer>"],
                ["<information>Babycham is made by Accolade Wines.</information>", "<information>...</information>"],
                ["Babycham producer", "Accolade Wines headquarters"], [["d1"], ["ev1"]], "Australia")
    assert keep(world(), good) == (True, "ok")
    # naming the evidence article before it was observed is a leak of the hint
    leak = traj(["<search>Accolade Wines country</search>", "<answer>Australia</answer>"], ["<information>x</information>"],
                ["Accolade Wines country"], [["ev1"]], "Australia")
    assert not no_leak(world(), leak) and keep(world(), leak) == (False, "leak")
    # correct but never retrieved the evidence
    ung = traj(["<search>Babycham</search>", "<answer>Australia</answer>"], ["<information>x</information>"], ["Babycham"], [["d1"]], "Australia")
    assert keep(world(), ung) == (False, "ungrounded")
    # deferred to the claim
    assert keep(world(), traj(["<answer>New Zealand</answer>"], [], [], [], "New Zealand")) == (False, "wrong")
    # policy text that refers to the hint
    talk = traj(["<think>the hint says Accolade</think><search>Babycham</search>", "<answer>Australia</answer>"],
                ["<information>x</information>"], ["Babycham"], [["ev1"]], "Australia")
    assert keep(world(), talk) == (False, "mentions_hint")
    # ordinary English words like "private" or "guidance" in facts must not be falsely rejected
    benign = traj(["<think>Babycham is owned by a private wine company; let me search Babycham.</think><search>Babycham</search>",
                   "<search>Accolade Wines</search>", "<answer>Australia</answer>"],
                  ["<information>Babycham is made by Accolade Wines.</information>", "<information>HQ in Australia.</information>"],
                  ["Babycham", "Accolade Wines"], [["d1"], ["ev1"]], "Australia")
    assert keep(world(), benign) == (True, "ok")
    # multi-hop hint_titles: intermediate hop not in question cannot be queried before appearing in <information>
    w_hop = world(hint_titles=["Intermediate Corp", "Accolade Wines"])
    leak_hop = traj(["<search>Intermediate Corp</search>", "<answer>Australia</answer>"], ["<information>x</information>"],
                    ["Intermediate Corp"], [["ev1"]], "Australia")
    assert not no_leak(w_hop, leak_hop) and keep(w_hop, leak_hop) == (False, "leak")
    # delete worlds: abstain only after verifying with >= min_delete_searches searches
    dele = world(family="delete", golds=[], unanswerable=True, evidence_ids=[])
    one = traj(["<search>Babycham</search>", "<answer>ABSTAIN</answer>"], ["<information>x</information>"], ["Babycham"], [["d1"]], "ABSTAIN")
    two = traj(["<search>Babycham</search>", "<search>Accolade Wines</search>", "<answer>ABSTAIN</answer>"],
               ["<information>Babycham by Accolade Wines</information>", "<information>y</information>"],
               ["Babycham", "Accolade Wines"], [["d1"], ["d2"]], "ABSTAIN")
    assert keep(dele, one) == (False, "abstain_unverified") and keep(dele, two) == (True, "ok")
    assert keep(dele, two, min_delete_searches=3) == (False, "abstain_unverified")
    # pick_candidate prefers trajectories that recovered across >=3 searches after turn-1 echoes
    from echospec.synth import pick_candidate
    import random
    lucky = traj(["<search>Babycham</search>", "<answer>Australia</answer>"], ["<information>x</information>"], ["Babycham"], [["ev1"]], "Australia")
    recov = traj(["<search>q1</search>", "<search>q2</search>", "<search>q3</search>", "<answer>Australia</answer>"],
                 ["<information>e1</information>", "<information>Babycham -> Accolade Wines</information>", "<information>Australia</information>"],
                 ["q1", "q2", "q3"], [["ov:1"], ["d1"], ["ev1"]], "Australia")
    picked = pick_candidate(world(m=8), [(world(m=8), lucky), (world(m=8), recov)], random.Random(0))
    assert picked[1] is recov


def test_acquisition_score_and_echo_weighting():
    import random

    from echospec.rewards import acquisition
    from echospec.transforms import sample_train_world
    docs = ['"Babycham"\nBabycham is made by Accolade Wines.', '"Echo"\nNew Zealand firm.']
    assert acquisition([["d1", "d2"]], docs, ["ev1"], ["Babycham"]) == 0.5
    assert acquisition([["ev1"]], docs, ["ev1"], ["Babycham"]) == 1.5
    assert acquisition([["x"]], ['"Echo"\nNew Zealand firm.'], ["ev1"], ["Babycham"]) == 0.0
    assert acquisition([["x"]], docs, [], ["Babycham"]) == 0.5  # delete worlds: credit for reaching the bridge hop
    # disambiguated Wikipedia titles ("On Golden Pond (2001 film)") match stripped support_titles ("On Golden Pond")
    disamb_docs = ['"On Golden Pond (2001 film)"\nA 2001 television adaptation.', '"Other (film)"\nText.']
    assert acquisition([["d1"]], disamb_docs, ["ev1"], ["On Golden Pond"]) == 0.5
    assert acquisition([["d1"]], disamb_docs, ["ev1"], ["On Golden Pond", "Missing Hop"]) == 0.25
    from test_core import GEN, ITEM, MASK
    rng = random.Random(0)
    rec = {"item": ITEM, "gen": GEN, "mask": MASK}
    ms = [sample_train_world(rec, rng, ["identity", "inject"], m_weights={8: 1.0})["m"] for _ in range(20)]
    assert set(ms) == {8} or set(ms) <= {8, 0}


def test_min_searches_interception(monkeypatch):
    monkeypatch.setattr(agent_mod, "get_tokenizer", lambda name: FakeTok())
    llm = FakeLLM(["<answer>New Zealand</answer>", "<search>Babycham</search>", "<search>Accolade Wines</search>", "<answer>Australia</answer>"])
    inst = {"id": "x", "question": "Q"}
    cfg = RunConfig(served_model="m", tokenizer_name="t", prompt_style="verify", min_searches=2, max_searches=4)
    t = asyncio.run(run_episode(inst, cfg, llm, FakeRetr()))
    assert t.n_searches == 2
    assert t.answer == "Australia"
    assert t.finish == "answer"
