"""Verification-trajectory synthesis for the SFT warm-start.

The policy itself writes every trajectory, in the exact Search-R1 protocol, under a privileged hint that names the
Wikipedia article holding the answer and asks it to confirm the answer there. The stored trajectory keeps the plain
prompt, so SFT trains the hint-free policy. A trajectory is kept only when it is learnable without the hint:

  * outcome     answerable worlds: correct answer AND a primary-evidence passage was retrieved;
                delete worlds: ABSTAIN after at least two searches
  * no leakage  a query may name the evidence article only after that name appeared in the question or in an
                earlier <information> block (the agent discovers the bridge entity; it is never told it)
  * clean text  policy text never refers to the hint
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from echospec.agent import RunConfig, run_batch
from echospec.qa_data import read_jsonl
from echospec.rewards import judge, supported
from echospec.text import normalize_answer
from echospec.transforms import base_title, make_world

HINT = (
    "[Solver note — reason naturally from scratch without referring to this note: "
    "the primary Wikipedia article holding the answer is \"{title}\"{via}. "
    "Search strategy: "
    "(1) Start by querying entities from the question; never query an article title before that title has appeared in the question or in a retrieved <information> passage, and never query an unverified candidate answer. "
    "(2) Retrieved passages may repeat paraphrased echoes of a single unsupported claim; if a search returns only brief assertions without the primary encyclopedia entry{first_hop}, do not stop or query their claimed answer—keep reformulating queries about the question's subject (its background, history, or organization, using up to all 4 searches) until the primary encyclopedia articles surface. "
    "(3) Once \"{title}\" appears in <information>, query \"{title}\" to verify the answer from its text; if after thorough multi-step search ({min_del}+ searches) no primary article supports an answer, output <answer> ABSTAIN </answer>.]"
)
LEAK_WORDS = ("solver note", "this note", "the hint", "a hint", "private guidance", "never mention", "i was told", "i am told")
SYNTH_FAMILY_PROBS = {"identity": 0.08, "inject": 0.48, "delete": 0.18, "rewrite": 0.12, "consistent": 0.07, "dup": 0.07}
SYNTH_M = [1, 2, 3, 5, 8]


def hint_titles(item: dict) -> list[str]:
    sup = [base_title(s["title"]) for s in (item.get("support") or []) if s.get("title")]
    fin = base_title(item["final_hop"]["title"])
    return [t for t in dict.fromkeys(sup + ([fin] if fin else [])) if t]


def make_hint(item: dict, min_delete_searches: int = 2) -> str:
    sup = [base_title(s["title"]) for s in (item.get("support") or []) if s.get("title")]
    chain = " -> ".join(f'"{t}"' for t in sup)
    via = f" (reached through {chain})" if sup else ""
    first_hop = f' "{sup[0]}"' if sup else ""
    return HINT.format(title=base_title(item["final_hop"]["title"]), via=via, first_hop=first_hop, min_del=min_delete_searches)


def synth_worlds(recs, rng: random.Random, per_item: int, probs: dict, m_weights: dict | None = None,
                 min_delete_searches: int = 2) -> list[dict]:
    fams, weights = list(probs), [probs[f] for f in probs]
    ms, mw = (list(m_weights), list(m_weights.values())) if m_weights else (SYNTH_M, [1.0] * len(SYNTH_M))
    worlds = []
    explicit_id = "identity" in probs
    for rec in recs:
        it, gen, mask = rec["item"], rec.get("gen"), rec.get("mask")
        ws = [] if explicit_id else [make_world(it, gen, mask, "identity")]
        for _ in range(per_item - len(ws)):
            for _try in range(12):
                fam = rng.choices(fams, weights=weights)[0]
                m = rng.choices(ms, weights=mw)[0] if fam in ("dup", "inject", "consistent") else 0
                w = make_world(it, gen, mask, fam, m)
                if w is not None and w["id"] not in {x["id"] for x in ws}:
                    ws.append(w)
                    break
        if not ws:
            ws = [make_world(it, gen, mask, "identity")]
        ht = hint_titles(it)
        for w in ws:
            w["hint"] = make_hint(it, min_delete_searches=min_delete_searches)
            w["evidence_title"] = base_title(it["final_hop"]["title"])
            w["hint_titles"] = ht
        worlds += ws
    return worlds


def _words(s: str) -> str:
    # split on markup and punctuation first: normalize_answer deletes punctuation, which would glue
    # "Wines.</information>" into one token
    return normalize_answer(re.sub(r"[^\w]+", " ", s))


def _contains(hay: str, needle: str) -> bool:
    n = _words(needle)
    return bool(n) and f" {n} " in f" {_words(hay)} "


def no_leak(world: dict, traj) -> bool:
    """Every query naming a hinted article comes after that name was observed (question or information)."""
    titles = world.get("hint_titles") or ([world["evidence_title"]] if world.get("evidence_title") else [])
    seen = world["question"]
    qi = 0
    for text, is_pol in traj.segments:
        if not is_pol:
            seen += " " + text
            continue
        if qi < len(traj.queries) and "</search>" in text:
            for title in titles:
                if _contains(traj.queries[qi], title) and not _contains(seen, title):
                    return False
            qi += 1
    return True


def keep(world: dict, traj, min_delete_searches: int = 2) -> tuple[bool, str]:
    if traj.finish != "answer":
        return False, "finish"
    pol = " ".join(t for t, p in traj.segments if p).lower()
    if any(w in pol for w in LEAK_WORDS):
        return False, "mentions_hint"
    j = judge(traj.answer, world["golds"], world["unanswerable"], world.get("decoys") or [])
    if not j["correct"]:
        return False, "wrong"
    if world["unanswerable"]:
        if traj.n_searches < min_delete_searches:
            return False, "abstain_unverified"
    elif not supported(traj.retrieved_ids, world.get("evidence_ids") or []):
        return False, "ungrounded"
    if not no_leak(world, traj):
        return False, "leak"
    return True, "ok"


def pick_candidate(world: dict, cands: list, rng: random.Random):
    """Among valid trajectories for one world, prefer multi-turn recovery when echoes displaced turn-1 evidence."""
    if world["family"] in ("inject", "consistent") and world.get("m", 0) >= 3:
        ev = set(world.get("evidence_ids") or [])
        recovered = [(w, t) for w, t in cands if t.retrieved_ids and not (set(t.retrieved_ids[0]) & ev) and t.n_searches >= 3]
        if recovered:
            return rng.choice(recovered)
        deep = [(w, t) for w, t in cands if t.n_searches >= 3]
        if deep:
            return rng.choice(deep)
    if world["family"] == "delete":
        deep = [(w, t) for w, t in cands if t.n_searches >= 3]
        if deep:
            return rng.choice(deep)
    return rng.choice(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--endpoints_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--served_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--retriever_url", default="http://127.0.0.1:8290")
    ap.add_argument("--per_item", type=int, default=3, help="worlds per base item (identity + per_item-1 transforms)")
    ap.add_argument("--samples", type=int, default=4, help="guided rollouts per world")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--exclude_seen", type=int, default=1)
    ap.add_argument("--max_searches", type=int, default=4)
    ap.add_argument("--family_probs", default="")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--m_probs", default="", help="echo-count weights, e.g. 1:0.1,3:0.2,5:0.3,8:0.4")
    ap.add_argument("--min_delete_searches", type=int, default=3, help="minimum searches before ABSTAIN on delete worlds")
    ap.add_argument("--prefer_deep", type=int, default=1, help="prefer >=3-search recovery trajectories on m>=3 worlds")
    ap.add_argument("--chunk_worlds", type=int, default=1500, help="worlds per resumable chunk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    recs = read_jsonl(Path(a.train_jsonl))
    rng.shuffle(recs)
    if a.max_items:
        recs = recs[: a.max_items]
    probs = dict(SYNTH_FAMILY_PROBS)
    if a.family_probs:
        probs = {k: float(v) for k, v in (kv.split(":") for kv in a.family_probs.split(","))}
    m_weights = {int(k): float(v) for k, v in (kv.split(":") for kv in a.m_probs.split(","))} if a.m_probs else None
    worlds = synth_worlds(recs, rng, a.per_item, probs, m_weights, min_delete_searches=a.min_delete_searches)
    endpoints = json.loads(Path(a.endpoints_json).read_text(encoding="utf-8"))["endpoints"]
    cfg = RunConfig(served_model=a.served_model, tokenizer_name=a.tokenizer, prompt_style="neutral",
                    temperature=a.temperature, exclude_seen=bool(a.exclude_seen), max_searches=a.max_searches)
    out = Path(a.out)
    parts = out.parent / f"{out.stem}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    print(f"[synth] {len(recs)} items, {len(worlds)} worlds, {len(worlds) * a.samples} guided rollouts", flush=True)

    # Resumable: each chunk of worlds is rolled out, filtered and written on its own; a restarted job skips finished chunks.
    for ci, s in enumerate(range(0, len(worlds), a.chunk_worlds)):
        done = parts / f"part_{ci:03d}.done"
        if done.exists():
            continue
        chunk = worlds[s: s + a.chunk_worlds]
        batch = [w for w in chunk for _ in range(a.samples)]
        trajs = asyncio.run(run_batch(batch, cfg, endpoints, a.retriever_url))
        by_world: dict[str, list] = defaultdict(list)
        reasons: Counter = Counter()
        for w, t in zip(batch, trajs):
            ok, why = keep(w, t, min_delete_searches=a.min_delete_searches)
            reasons[f"{w['family']}:{why}"] += 1
            if ok:
                by_world[w["id"]].append((w, t))
        crng = random.Random(f"{a.seed}:{ci}")
        with open(parts / f"part_{ci:03d}.jsonl", "w", encoding="utf-8") as f:
            for wid, cands in by_world.items():
                w, t = pick_candidate(cands[0][0], cands, crng) if a.prefer_deep else crng.choice(cands)
                f.write(json.dumps({"world_id": wid, "base_id": w["base_id"], "family": w["family"], "m": w["m"],
                                    "prompt_text": t.prompt_text, "segments": t.segments, "answer": t.answer,
                                    "n_searches": t.n_searches, "queries": t.queries}, ensure_ascii=False) + "\n")
        (parts / f"part_{ci:03d}.reasons.json").write_text(json.dumps(reasons), encoding="utf-8")
        done.write_text("ok", encoding="utf-8")
        print(f"[synth] chunk {ci}: {len(by_world)}/{len(chunk)} worlds kept", flush=True)

    rows, reasons = [], Counter()
    for p in sorted(parts.glob("part_*.jsonl")):
        rows += read_jsonl(p)
        reasons.update(json.loads(p.with_suffix(".reasons.json").read_text(encoding="utf-8")))
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    fam_total, fam_kept = Counter(w["family"] for w in worlds), Counter(r["family"] for r in rows)
    m_kept = Counter(r["m"] for r in rows if r["family"] == "inject")
    report = {
        "worlds": len(worlds),
        "kept": len(rows),
        "kept_by_family": {f: f"{fam_kept[f]}/{fam_total[f]}" for f in sorted(fam_total)},
        "inject_kept_by_m": dict(sorted(m_kept.items())),
        "reasons": dict(sorted(reasons.items())),
        "mean_searches_kept": round(sum(r["n_searches"] for r in rows) / max(1, len(rows)), 3),
    }
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[synth] {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
