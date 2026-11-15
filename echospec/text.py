"""Answer normalization, scoring, near-duplicate detection, and paired statistics."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

import numpy as np

ABSTAIN_TOKEN = "ABSTAIN"
# Whole-answer matches only, so real answers such as "Unknown Pleasures" are never read as abstentions.
_ABSTAIN_SET = {
    "abstain", "unanswerable", "cannot be answered", "cannot be determined", "insufficient evidence",
    "insufficient information", "false premise", "unknown", "i dont know", "i do not know", "no answer",
    "not enough information", "unable to determine", "unable to answer",
}
_ABSTAIN_PHRASES = (
    "not enough information", "insufficient information", "insufficient evidence",
    "cannot be answered", "cannot be determined", "unable to determine", "unable to answer",
    "i do not know", "i dont know", "cannot answer",
)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD / Search-R1 normalization: lowercase, strip punctuation and articles, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def is_abstention(pred: str | None) -> bool:
    if pred is None:
        return False
    n = normalize_answer(pred)
    return n in _ABSTAIN_SET or any(p in n for p in _ABSTAIN_PHRASES)


def exact_match(pred: str | None, golds: Sequence[str]) -> bool:
    if pred is None or is_abstention(pred):
        return False
    p = normalize_answer(pred)
    return bool(p) and any(p == normalize_answer(g) for g in golds if g)


def cover_em(pred: str | None, golds: Sequence[str]) -> bool:
    """Gold answer appears as a contiguous token span of the prediction."""
    if pred is None or is_abstention(pred):
        return False
    pt = normalize_answer(pred).split()
    if not pt:
        return False
    for g in golds:
        gt = normalize_answer(g).split()
        if not gt:
            continue
        for i in range(len(pt) - len(gt) + 1):
            if pt[i : i + len(gt)] == gt:
                return True
    return False


def token_f1(pred: str | None, golds: Sequence[str]) -> float:
    if pred is None or is_abstention(pred):
        return 0.0
    pt = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if not pt or not gt or same == 0:
            continue
        p, r = same / len(pt), same / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def contains_answer(text: str, answer: str) -> bool:
    """Whole-token containment of a normalized answer inside a normalized text."""
    a = normalize_answer(answer)
    if not a:
        return False
    return f" {a} " in f" {normalize_answer(text)} "


# ----------------------------------------------------------------------------------------------
# Near-duplicate detection (inference-time dedup baseline and data audits)
# ----------------------------------------------------------------------------------------------
def shingles(text: str, n: int = 3) -> set[str]:
    toks = normalize_answer(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup_docs(docs: Sequence[dict], threshold: float = 0.35, key: str = "contents", n: int = 3) -> list[dict]:
    """Greedy rank-preserving near-duplicate removal by word-shingle Jaccard similarity."""
    kept: list[dict] = []
    kept_sh: list[set[str]] = []
    for d in docs:
        sh = shingles(d.get(key, ""), n=n)
        if any(jaccard(sh, k) >= threshold for k in kept_sh):
            continue
        kept.append(d)
        kept_sh.append(sh)
    return kept


# ----------------------------------------------------------------------------------------------
# Paired statistics
# ----------------------------------------------------------------------------------------------
def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap of mean(a) - mean(b): returns diff, 95% CI and two-sided p-value."""
    x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if x.size == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    obs = float(x.mean())
    p = float(np.mean(np.abs(boots - obs) >= abs(obs))) if obs != 0 else 1.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "n": int(x.size)}


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> float:
    """Exact two-sided McNemar p-value."""
    from scipy.stats import binomtest

    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if n10 + n01 == 0:
        return 1.0
    return float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
