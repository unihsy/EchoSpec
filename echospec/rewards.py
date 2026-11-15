"""Decision-theoretic outcome reward and GRPO advantages.

R = +1 if correct (EM; or abstention when the world is unanswerable)
     0 if abstained on an answerable world
    -lam otherwise (wrong answer, answering an unanswerable world, no parsable answer)
With lam = 0 this is the Search-R1 EM reward.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from echospec.text import exact_match, is_abstention, normalize_answer

_PAREN_RE = re.compile(r"\s*\(.*?\)\s*$")


def judge(answer: str | None, golds: Sequence[str], unanswerable: bool, decoys: Sequence[str] = ()) -> dict:
    abst = is_abstention(answer)
    valid = answer is not None and answer.strip() != ""
    if unanswerable:
        correct = abst
    else:
        correct = exact_match(answer, golds)
    deferred = bool(valid and not abst and decoys and exact_match(answer, decoys))
    return {"correct": bool(correct), "abstain": bool(abst), "valid": bool(valid), "deferred": deferred}


def supported(retrieved_ids: Sequence[Sequence[str]], evidence_ids: Sequence[str]) -> bool:
    """True when the trajectory retrieved at least one primary-evidence passage."""
    ev = set(evidence_ids)
    return any(d in ev for turn in retrieved_ids for d in turn)


def _doc_title(contents: str) -> str:
    raw = contents.split("\n", 1)[0].strip().strip('"')
    return normalize_answer(_PAREN_RE.sub("", raw).strip())


def acquisition(retrieved_ids: Sequence[Sequence[str]], retrieved_docs: Sequence[str], evidence_ids: Sequence[str],
                support_titles: Sequence[str]) -> float:
    """Evidence-acquisition score of one trajectory (training-time only, uses privileged world annotations):
    1 for retrieving a primary-evidence passage, plus up to 0.5 proportional to intermediate-hop passages retrieved."""
    ev = bool(evidence_ids) and supported(retrieved_ids, evidence_ids)
    titles = [normalize_answer(_PAREN_RE.sub("", t).strip()) for t in support_titles if t]
    titles = [t for t in dict.fromkeys(titles) if t]
    doc_titles = {_doc_title(d) for d in retrieved_docs}
    sup = (sum(t in doc_titles for t in titles) / len(titles)) if titles else 0.0
    return float(ev) + 0.5 * float(sup)


def reward(j: dict, unanswerable: bool, lam: float, grounded: bool | None = None, ground_alpha: float = 1.0) -> float:
    """lam-reward. With ground_alpha < 1, a correct answer to an answerable question earns full credit only when
    the trajectory retrieved primary evidence (grounded=True); an answer taken from claims alone earns ground_alpha."""
    if j["correct"]:
        if not unanswerable and grounded is False:
            return float(ground_alpha)
        return 1.0
    if j["abstain"] and not unanswerable:
        return 0.0
    return -float(lam)


def group_advantages(rewards: Sequence[float], eps: float = 1e-6, norm: str = "std") -> list[float]:
    """Group-relative advantages.

    norm="std":  (r - mean) / std  (standard GRPO). Rescales every group to unit variance, so in a group where all
                 rollouts fail, the gap between a wrong answer (-lam) and an abstention (0) becomes as large a signal as
                 correct vs. wrong; the reward scale set by lam is lost.
    norm="none": r - mean. Keeps reward units, so the answering threshold lam / (1 + lam) from the lam-reward holds.
    """
    r = np.asarray(rewards, dtype=np.float64)
    centered = r - r.mean()
    if norm == "none":
        return list(centered)
    sd = float(r.std())
    if sd < eps:
        return [0.0] * len(r)
    return list(centered / (sd + eps))


INVARIANT = {"dup", "inject", "consistent"}


def paired_advantages(worlds: Sequence[dict], rews: Sequence[float], norm: str = "none",
                      pair_eta: float = 0.0) -> np.ndarray:
    """Group-relative advantages with an optional unbiased cross-world control variate on invariant worlds.

    Within each world w, standard mean-centering uses baseline mean(R_w). When all G rollouts in an inject(m) world
    defer to echoes, R_w is constant (-lam) and within-world advantages vanish even if the policy solves the paired
    identity world of the same base question. Because the paired identity rollouts are sampled independently of world w
    and share the exact same answer, primary evidence, and maximum reward on invariant families (dup, inject,
    consistent), B_w = (1 - eta) * mean(R_w) + eta * mean(R_id) is an unbiased state-only baseline: it preserves the
    expected policy gradient and the optimal answering threshold lam / (1 + lam) while penalizing uniform echo deference
    on solvable questions.
    """
    advs = np.zeros(len(worlds), dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    id_means: dict[str, float] = {}
    for i, w in enumerate(worlds):
        groups[w["id"]].append(i)
    for wid, idxs in groups.items():
        w0 = worlds[idxs[0]]
        if w0.get("family") == "identity":
            id_means[w0["base_id"]] = float(np.mean([rews[i] for i in idxs]))
    for wid, idxs in groups.items():
        r = [rews[i] for i in idxs]
        ga = np.asarray(group_advantages(r, norm=norm), dtype=np.float64)
        w0 = worlds[idxs[0]]
        if pair_eta > 0.0 and norm == "none" and w0.get("family") in INVARIANT and w0.get("base_id") in id_means:
            ga = ga + float(pair_eta) * (float(np.mean(r)) - id_means[w0["base_id"]])
        advs[idxs] = ga
    return advs


def consistency_bonus(worlds: Sequence[dict], answers: Sequence[str | None], beta: float) -> list[float]:
    """For invariant-family rollouts: +beta if the answer matches the modal identity-world answer of the same
    base item in this batch, -beta otherwise. Zero for other families or when no identity answer exists."""
    modal: dict[str, str] = {}
    by_base = defaultdict(list)
    for w, a in zip(worlds, answers):
        if w["family"] == "identity" and a is not None and not is_abstention(a):
            by_base[w["base_id"]].append(normalize_answer(a))
    for b, lst in by_base.items():
        if lst:
            modal[b] = Counter(lst).most_common(1)[0][0]
    out = []
    for w, a in zip(worlds, answers):
        if w["family"] not in INVARIANT or w["base_id"] not in modal:
            out.append(0.0)
            continue
        out.append(beta if (a is not None and normalize_answer(a) == modal[w["base_id"]]) else -beta)
    return out
