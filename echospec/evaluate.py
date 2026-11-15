"""Evaluate methods on clean (L4) benchmarks and transformation-world (spec) test sets.

methods JSON: list of {name, served_model, tokenizer, prompt_style, dedup?, lora_path?}
A method with lora_path is hot-loaded into the vLLM servers under `name` before evaluation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from echospec.agent import RunConfig, run_batch
from echospec.qa_data import read_jsonl
from echospec.rewards import judge, supported
from echospec.serve import load_lora
from echospec.text import is_abstention, normalize_answer


def predict(method: dict, bench_path: Path, endpoints, retriever_url: str, out_dir: Path) -> Path:
    insts = read_jsonl(bench_path)
    pred_path = out_dir / "predictions" / f"{method['name']}__{bench_path.stem}.jsonl"
    if pred_path.exists() and sum(1 for _ in open(pred_path, encoding="utf-8")) == len(insts):
        print(f"[eval] skip {pred_path.name} (complete)", flush=True)
        return pred_path
    cfg = RunConfig(
        served_model=method["served_model"],
        tokenizer_name=method["tokenizer"],
        prompt_style=method.get("prompt_style", "neutral"),
        temperature=0.0,
        max_searches=int(method.get("max_searches", 4)),
        min_searches=int(method.get("min_searches", 0)),
        top_k=int(method.get("top_k", 3)),
        dedup=bool(method.get("dedup", False)),
        exclude_seen=bool(method.get("exclude_seen", False)),
        corpus=method.get("corpus", "wiki18"),
        protocol=method.get("protocol", "searchr1"),
        retrieval_filter=method.get("retrieval_filter"),
    )
    print(f"[eval] {method['name']} on {bench_path.stem}: {len(insts)} episodes", flush=True)
    trajs = asyncio.run(run_batch(insts, cfg, endpoints, retriever_url, progress_every=2000))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    if int(method.get("dump_text", 0)):  # full transcripts of the first N episodes, for protocol debugging
        with open(pred_path.with_suffix(".text.jsonl"), "w", encoding="utf-8") as f:
            for w, t in list(zip(insts, trajs))[: int(method["dump_text"])]:
                f.write(json.dumps({"id": w["id"], "finish": t.finish, "text": t.full_text()}, ensure_ascii=False) + "\n")
    tmp = pred_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for w, t in zip(insts, trajs):
            j = judge(t.answer, w["golds"], w["unanswerable"], w.get("decoys") or [])
            f.write(json.dumps({
                "id": w["id"], "base_id": w.get("base_id", w["id"]), "family": w.get("family", "clean"), "m": w.get("m", 0),
                "answer": t.answer, **j, "n_searches": t.n_searches, "gen_tokens": t.gen_tokens,
                "supported": supported(t.retrieved_ids, w["evidence_ids"]) if w.get("evidence_ids") else None,
                "prompt_tokens": t.prompt_tokens, "finish": t.finish, "queries": t.queries,
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, pred_path)
    return pred_path


def summarize(pred_path: Path) -> dict:
    rows = read_jsonl(pred_path)
    if not rows:
        return {}
    out: dict = {"n": len(rows)}
    def agg(rs):
        d = {
            "n": len(rs),
            "acc": round(100 * float(np.mean([r["correct"] for r in rs])), 2),
            "abstain": round(100 * float(np.mean([r["abstain"] for r in rs])), 2),
            "deferred": round(100 * float(np.mean([r["deferred"] for r in rs])), 2),
            "searches": round(float(np.mean([r["n_searches"] for r in rs])), 3),
            "gen_tokens": round(float(np.mean([r["gen_tokens"] for r in rs])), 1),
            "errors": sum(1 for r in rs if str(r["finish"]).startswith("error")),
        }
        gr = [r for r in rs if r.get("supported") is not None and not r.get("unanswerable", False)]
        if gr:
            d["grounded_acc"] = round(100 * float(np.mean([r["correct"] and r["supported"] for r in gr])), 2)
            d["evidence_retrieved"] = round(100 * float(np.mean([r["supported"] for r in gr])), 2)
        return d
    out["all"] = agg(rows)
    fams = sorted({r["family"] for r in rows})
    if fams == ["clean"]:
        return out
    by = defaultdict(list)
    for r in rows:
        by[f"{r['family']}_m{r['m']}" if r["m"] else r["family"]].append(r)
    out["worlds"] = {k: agg(v) for k, v in sorted(by.items())}
    ident = {r["base_id"]: r for r in rows if r["family"] == "identity"}
    inv = defaultdict(list)
    cond_def = defaultdict(list)
    for r in rows:
        if r["family"] in ("dup", "inject", "consistent") and r["base_id"] in ident:
            i = ident[r["base_id"]]
            same = (r["answer"] is not None and i["answer"] is not None
                    and normalize_answer(r["answer"]) == normalize_answer(i["answer"]))
            key = f"{r['family']}_m{r['m']}"
            inv[key].append(1.0 if same else 0.0)
            if i["correct"] and r["family"] == "inject":
                cond_def[key].append(1.0 if r["deferred"] else 0.0)
    out["invariance"] = {k: round(100 * float(np.mean(v)), 2) for k, v in sorted(inv.items())}
    out["conditional_deference"] = {k: round(100 * float(np.mean(v)), 2) for k, v in sorted(cond_def.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods_json", required=True)
    ap.add_argument("--endpoints_json", required=True)
    ap.add_argument("--benches", nargs="+", required=True)
    ap.add_argument("--retriever_url", default="http://127.0.0.1:8290")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = json.loads(Path(a.methods_json).read_text(encoding="utf-8"))
    endpoints = json.loads(Path(a.endpoints_json).read_text(encoding="utf-8"))["endpoints"]
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    prev_lora = None
    for m in methods:
        if m.get("lora_path"):
            load_lora(endpoints, m["name"], m["lora_path"], unload=prev_lora)
            m["served_model"] = m["name"]
            prev_lora = m["name"]
        for b in a.benches:
            bp = Path(b)
            if not bp.exists():
                print(f"[eval] missing bench {bp}", flush=True)
                continue
            pp = predict(m, bp, endpoints, a.retriever_url, out)
            summary.setdefault(m["name"], {})[bp.stem] = summarize(pp)
            tmp = summary_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            os.replace(tmp, summary_path)
            s = summary[m["name"]][bp.stem]["all"]
            print(f"[eval] {m['name']} {bp.stem}: acc={s['acc']} abstain={s['abstain']} searches={s['searches']}", flush=True)


if __name__ == "__main__":
    main()
