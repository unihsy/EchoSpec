"""Question sources.

L4 clean benchmarks: the seven Search-R1 evaluation sets from RUC-NLPIR/FlashRAG_datasets.
Base items for transformation worlds: HotpotQA (bridge) and MuSiQue (answerable), each with an identified
final-hop passage (the supporting passage that states the answer).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from echospec.text import contains_answer, normalize_answer

L4_SETS = [
    ("nq", "test"),
    ("triviaqa", "test"),
    ("popqa", "test"),
    ("hotpotqa", "dev"),
    ("2wikimultihopqa", "dev"),
    ("musique", "dev"),
    ("bamboogle", "test"),
]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def build_l4(out_dir: Path, max_per_set: int = 1000, seed: int = 0) -> dict[str, int]:
    from datasets import load_dataset

    sizes = {}
    for name, split in L4_SETS:
        ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", name, split=split)
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        rows = []
        for i in idx[:max_per_set]:
            r = ds[i]
            golds = r.get("golden_answers") or []
            if isinstance(golds, str):
                golds = [golds]
            golds = [str(g) for g in golds if str(g).strip()]
            if not golds:
                continue
            rows.append(
                {
                    "id": f"{name}_{r.get('id', i)}",
                    "benchmark": name,
                    "family": "clean",
                    "question": r["question"],
                    "golds": golds,
                    "unanswerable": False,
                    "overlay_docs": [],
                    "masked_doc_ids": [],
                }
            )
        write_jsonl(out_dir / f"l4_{name}.jsonl", rows)
        sizes[name] = len(rows)
        print(f"[qa_data] L4 {name}/{split}: {len(rows)}", flush=True)
    return sizes


def _answer_ok(ans: str) -> bool:
    n = normalize_answer(ans)
    return bool(n) and n not in {"yes", "no"} and 2 <= len(n) <= 60 and len(n.split()) <= 8


def hotpot_items(split: str, limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    out = []
    for i in order:
        if len(out) >= limit:
            break
        r = ds[i]
        if r["type"] != "bridge" or not _answer_ok(r["answer"]):
            continue
        paras = {t: " ".join(s).strip() for t, s in zip(r["context"]["title"], r["context"]["sentences"])}
        supp = [t for t in dict.fromkeys(r["supporting_facts"]["title"]) if t in paras]
        if len(supp) != 2:
            continue
        holders = [t for t in supp if contains_answer(paras[t], r["answer"])]
        if len(holders) == 2:
            holders = [t for t in holders if normalize_answer(t) not in normalize_answer(r["question"])]
        if len(holders) != 1:
            continue
        fin = holders[0]
        out.append(
            {
                "id": f"hotpot_{r['id']}",
                "source": "hotpotqa",
                "question": r["question"],
                "answers": [r["answer"]],
                "final_hop": {"title": fin, "text": paras[fin]},
                "support": [{"title": t, "text": paras[t]} for t in supp if t != fin],
            }
        )
    return out


def musique_items(split: str, limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    out = []
    for i in order:
        if len(out) >= limit:
            break
        r = ds[i]
        if not r.get("answerable", True) or not _answer_ok(r.get("answer") or ""):
            continue
        answers = [r["answer"]] + [a for a in (r.get("answer_aliases") or []) if a]
        paras = {p["idx"]: p for p in r["paragraphs"]}
        decomp = r.get("question_decomposition") or []
        if not decomp:
            continue
        fin_idx = decomp[-1].get("paragraph_support_idx")
        if fin_idx not in paras:
            continue
        fin = paras[fin_idx]
        if not contains_answer(fin["paragraph_text"], r["answer"]):
            continue
        support = [
            {"title": paras[d["paragraph_support_idx"]]["title"], "text": paras[d["paragraph_support_idx"]]["paragraph_text"]}
            for d in decomp[:-1]
            if d.get("paragraph_support_idx") in paras
        ]
        out.append(
            {
                "id": f"musique_{r['id']}",
                "source": "musique",
                "question": r["question"],
                "answers": list(dict.fromkeys(answers)),
                "final_hop": {"title": fin["title"], "text": fin["paragraph_text"]},
                "support": support,
            }
        )
    return out


def build_base(out_dir: Path, n_train_per_src: int = 2400, n_test_per_src: int = 450, n_dev: int = 400, seed: int = 0) -> dict:
    train = hotpot_items("train", n_train_per_src, seed) + musique_items("train", n_train_per_src, seed)
    test = hotpot_items("validation", n_test_per_src, seed) + musique_items("validation", n_test_per_src, seed)
    rng = random.Random(seed)
    rng.shuffle(train)
    dev, train = train[:n_dev], train[n_dev:]
    for split, rows in (("train", train), ("dev", dev), ("test", test)):
        for r in rows:
            r["split"] = split
        write_jsonl(out_dir / f"base_{split}.jsonl", rows)
    sizes = {"train": len(train), "dev": len(dev), "test": len(test)}
    print(f"[qa_data] base items: {sizes}", flush=True)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--l4", action="store_true")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--l4_max", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    if a.l4:
        build_l4(out, a.l4_max)
    if a.base:
        build_base(out)


if __name__ == "__main__":
    main()
