"""Corpus-transformation worlds.

Families (spec g_T in parentheses):
  identity              unchanged corpus                                  (answer unchanged)
  dup(m)                + m paraphrases of the final-hop primary passage  (unchanged)
  inject(m)             + m copies of a claim asserting alt_answer        (unchanged)
  consistent(m)         + m copies of the SAME claim asserting the answer (unchanged)
  rewrite               mask final-hop evidence, add rewritten passage    (becomes alt_answer)
  delete                mask final-hop evidence                           (becomes ABSTAIN)

Anti-shortcut construction: false claim copies are produced from the true claim copies by replacing the answer
string only, so true and false claims share style, length, title and copy count.

Stages (CLI):
  generate  LLM content per base item (no retrieval needed)       -> gen_{tag}_{split}.jsonl
  verify    retrieval-verified masks over wiki-18                  -> mask_{split}.jsonl
  assemble  train items, fixed dev/test worlds, style audit        -> spec_*.jsonl, audit.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Sequence

from echospec.protocol import make_contents
from echospec.qa_data import read_jsonl, write_jsonl
from echospec.text import contains_answer, jaccard, normalize_answer, shingles

STYLES = {
    "encyclopedic": ("encyclopedia-style summary paragraph", "neutral encyclopedia-style summary paragraph written in the third person"),
    "report": ("short news or blog article", "short news-style or blog-style write-up"),
}

CORE_PROMPT = """You are helping build a controlled test of search assistants. You will see a question, its correct answer, and the encyclopedia passage that states the answer.

Question: {question}
Correct answer: {answer}
Passage title: {title}
Passage: {text}

Write a JSON object with exactly these keys:
"alt_answer": a different but plausible answer of exactly the same type as the correct answer (for example another person, place, date, number or organisation of the same kind). It must not be a synonym, alias, abbreviation, or part of the correct answer.
"rewritten_passage": the passage rewritten with the smallest possible edits so that it states alt_answer instead of the correct answer wherever the correct answer is stated. Keep the length, all other facts and the style unchanged, and keep the passage internally consistent. It must not contain the correct answer.
"claim_title": a short, realistic title for a {style_name}.
"claim_text": a {style_desc} of 60 to 100 words about the subject of the question that states, as a plain fact, that the answer is {answer} (write the exact string "{answer}"). Mention the key entities of the question, but add no further supporting detail, sources or evidence.
Return only the JSON object."""

PARA_PROMPT = """Below is a short text. Write {n} different versions of it, as if {who}. Every version must keep the same facts, must contain the exact string "{keep}", and must be 50 to 110 words long. Vary the wording, sentence order and emphasis between versions, and do not add new facts.

Title: {title}
Text: {text}

Return only a JSON list of {n} strings."""

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_JSON_LIST = re.compile(r"\[.*\]", re.DOTALL)


def _parse(raw: str, kind: str):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = (_JSON_OBJ if kind == "obj" else _JSON_LIST).search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def substitute(text: str, old: str, new: str) -> str:
    """Case-insensitive replacement of the answer string, respecting word boundaries where possible."""
    pat = re.escape(old)
    if re.match(r"\w", old[:1] or "") and re.match(r"\w", old[-1:] or ""):
        pat = rf"\b{pat}\b"
    return re.sub(pat, new, text, flags=re.IGNORECASE)


def mentions_any(text: str, answers: Sequence[str]) -> bool:
    return any(contains_answer(text, a) for a in answers if a)


def false_title(title: str, golds: Sequence[str], alt: str) -> str:
    """Claim title for the false world: every gold string (longest first) replaced by the alternative answer."""
    art = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
    pairs = [(g, alt) for g in golds if g] + [(art.sub("", g), art.sub("", alt)) for g in golds if g and art.match(g)]
    for g, a in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        title = substitute(title, g, a)
    return title


def base_title(t: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", t).strip()


# ----------------------------------------------------------------------------------------------
# Stage 1: generation
# ----------------------------------------------------------------------------------------------
class LLM:
    def __init__(self, endpoints: Sequence[str], model: str):
        from openai import AsyncOpenAI

        self.model = model
        self.clients = [AsyncOpenAI(base_url=u, api_key="local", timeout=600.0, max_retries=0) for u in endpoints]
        self.i = 0

    async def chat(self, prompt: str, temperature: float, max_tokens: int) -> str:
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in self.model.lower() else {}
        last = None
        for attempt in range(4):
            c = self.clients[self.i % len(self.clients)]
            self.i += 1
            try:
                r = await c.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra,
                )
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM call failed: {last}")


async def generate_one(item: dict, llm: LLM, rng: random.Random, n_para: int = 9) -> dict | None:
    ans = item["answers"][0]
    golds = item["answers"]
    fh = item["final_hop"]
    style = rng.choice(sorted(STYLES))
    sname, sdesc = STYLES[style]
    core = None
    for attempt in range(3):
        raw = await llm.chat(
            CORE_PROMPT.format(question=item["question"], answer=ans, title=fh["title"], text=fh["text"], style_name=sname, style_desc=sdesc),
            temperature=0.5 + 0.2 * attempt,
            max_tokens=1200,
        )
        obj = _parse(raw, "obj")
        if not isinstance(obj, dict):
            continue
        alt = str(obj.get("alt_answer") or "").strip()
        rew = " ".join(str(obj.get("rewritten_passage") or "").split())
        ctitle = str(obj.get("claim_title") or "").strip().strip('"')
        claim = " ".join(str(obj.get("claim_text") or "").split())
        na, ng = normalize_answer(alt), [normalize_answer(g) for g in golds]
        if not na or any(na == g or na in g or g in na for g in ng if g):
            continue
        if not contains_answer(rew, alt) or mentions_any(rew, golds) or len(rew) < 0.5 * len(fh["text"]):
            continue
        if not ctitle or ans.lower() not in claim.lower() or len(claim.split()) < 30:
            continue
        core = {"alt": alt, "rewritten": rew, "claim_title": ctitle, "claim": claim}
        break
    if core is None:
        return None
    alt = core["alt"]

    async def paraphrase(title: str, text: str, keep: str, who: str) -> list[str]:
        for attempt in range(2):
            raw = await llm.chat(
                PARA_PROMPT.format(n=n_para, who=who, keep=keep, title=title, text=text),
                temperature=0.8 + 0.1 * attempt,
                max_tokens=3000,
            )
            lst = _parse(raw, "list")
            if isinstance(lst, list):
                return [" ".join(str(x).split()) for x in lst if isinstance(x, (str, int, float))]
        return []

    claim_paras, primary_paras = await asyncio.gather(
        paraphrase(core["claim_title"], core["claim"], ans, "different websites had each rewritten the same original article"),
        paraphrase(fh["title"], fh["text"], ans, "different reference websites had each rewritten the same encyclopedia entry"),
    )
    true_copies, false_copies = [], []
    for t in [core["claim"]] + claim_paras:
        if ans.lower() not in t.lower() or len(t.split()) < 25:
            continue
        f = substitute(t, ans, alt)
        if mentions_any(f, golds) or not contains_answer(f, alt):
            continue
        if t in true_copies:
            continue
        true_copies.append(t)
        false_copies.append(f)
    primary = [p for p in dict.fromkeys(primary_paras) if mentions_any(p, golds) and len(p.split()) >= 25]
    return {
        "id": item["id"],
        "gen": llm.model,
        "style": style,
        "alt_answer": alt,
        "rewritten_final": core["rewritten"],
        "claim_title": core["claim_title"],
        "claim_true": true_copies[:8],
        "claim_false": false_copies[:8],
        "primary_copies": primary[:8],
    }


async def generate(base_path: Path, out_path: Path, endpoints: Sequence[str], model: str, concurrency: int = 192, seed: int = 0) -> None:
    items = read_jsonl(base_path)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"[transforms] generate {base_path.name} with {model}: {len(todo)} to do ({len(done)} done)", flush=True)
    llm = LLM(endpoints, model)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "fail": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def one(it):
        async with sem:
            try:
                rec = await generate_one(it, llm, random.Random(f"{seed}:{it['id']}"))
            except Exception as e:  # noqa: BLE001
                print(f"[transforms] error {it['id']}: {e}", flush=True)
                rec = None
        async with lock:
            if rec is None:
                stats["fail"] += 1
                rec = {"id": it["id"], "gen": model, "failed": True}
            else:
                stats["ok"] += 1
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = stats["ok"] + stats["fail"]
            if n % 200 == 0:
                print(f"[transforms] {n}/{len(todo)} ok={stats['ok']} fail={stats['fail']}", flush=True)

    await asyncio.gather(*[one(it) for it in todo])
    print(f"[transforms] done {out_path.name}: {stats}", flush=True)


# ----------------------------------------------------------------------------------------------
# Stage 2: retrieval-verified masks
# ----------------------------------------------------------------------------------------------
def _related(contents: str, item: dict) -> bool:
    if not mentions_any(contents, item["answers"]):
        return False
    t = normalize_answer(base_title(item["final_hop"]["title"]))
    if t and f" {t} " in f" {normalize_answer(contents)} ":
        return True
    return jaccard(shingles(contents), shingles(item["final_hop"]["text"])) >= 0.25


async def verify_one(item: dict, retr, k_probe: int = 30, rounds: int = 4) -> dict:
    fh = item["final_hop"]
    probes = [
        item["question"],
        f"{base_title(fh['title'])} {item['answers'][0]}",
        fh["text"][:400],
    ]
    mask: list[str] = []
    found_round0 = 0
    for rnd in range(rounds + 1):
        leaks = []
        for p in probes:
            for d in await retr.search(p, top_k=k_probe, masked_doc_ids=mask):
                if _related(d["contents"], item) and d["id"] not in leaks:
                    leaks.append(d["id"])
        if rnd == 0:
            found_round0 = len(leaks)
        if not leaks:
            return {"id": item["id"], "evidence_found": found_round0 > 0, "mask": mask, "verified": True, "rounds": rnd}
        mask.extend(x for x in leaks if x not in mask)
    return {"id": item["id"], "evidence_found": found_round0 > 0, "mask": mask, "verified": False, "rounds": rounds}


async def verify(base_path: Path, out_path: Path, retriever_url: str, concurrency: int = 64) -> None:
    from echospec.retrieval import RetrieverClient

    items = read_jsonl(base_path)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    retr = RetrieverClient(retriever_url)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n = 0

    async def one(it):
        nonlocal n
        async with sem:
            rec = await verify_one(it, retr)
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"[transforms] verify {n}/{len(todo)}", flush=True)

    await asyncio.gather(*[one(it) for it in todo])
    await retr.aclose()
    recs = read_jsonl(out_path)
    ev = sum(r["evidence_found"] for r in recs)
    ok = sum(r["evidence_found"] and r["verified"] for r in recs)
    print(f"[transforms] verify {out_path.name}: n={len(recs)} evidence_found={ev} maskable={ok}", flush=True)


# ----------------------------------------------------------------------------------------------
# Stage 3: worlds
# ----------------------------------------------------------------------------------------------
def make_world(item: dict, gen: dict | None, mask: dict | None, family: str, m: int = 0) -> dict | None:
    """Instantiate one world; returns None when the item lacks what the family needs."""
    fh = item["final_hop"]
    w = {
        "id": f"{item['id']}|{family}|{m}|{(gen or {}).get('gen', '-')}",
        "base_id": item["id"],
        "family": family,
        "m": m,
        "gen": (gen or {}).get("gen", "-"),
        "question": item["question"],
        "golds": list(item["answers"]),
        "unanswerable": False,
        "decoys": [],
        "overlay_docs": [],
        "masked_doc_ids": [],
        # passages that state the answer as primary evidence (wiki-18 ids found during mask verification);
        # claim copies never count as evidence
        "evidence_ids": list(mask["mask"]) if mask else [],
        # titles of the earlier-hop passages (the route to the bridge entity); used only for training rewards
        "support_titles": [base_title(s["title"]) for s in item.get("support") or [] if s.get("title")],
    }
    tag = f"ov:{item['id']}:{family}"
    if family == "identity":
        return w
    if gen is None or gen.get("failed"):
        return None
    w["decoys"] = [gen["alt_answer"]]
    if family in ("dup", "inject", "consistent"):
        src = {"dup": gen["primary_copies"], "inject": gen["claim_false"], "consistent": gen["claim_true"]}[family]
        title = fh["title"] if family == "dup" else gen["claim_title"]
        if family == "inject":
            # the false claim is the true claim with the answer substituted, title included
            title = false_title(title, item["answers"], gen["alt_answer"])
        if len(src) < m or m < 1:
            return None
        w["overlay_docs"] = [{"id": f"{tag}:{k}", "contents": make_contents(title, t)} for k, t in enumerate(src[:m])]
        if family == "dup":
            w["decoys"] = []
            w["evidence_ids"] += [d["id"] for d in w["overlay_docs"]]  # paraphrases of primary evidence
        return w
    if family in ("rewrite", "delete"):
        if mask is None or not (mask["evidence_found"] and mask["verified"]):
            return None
        w["masked_doc_ids"] = list(mask["mask"])
        if family == "rewrite":
            w["golds"] = [gen["alt_answer"]]
            w["decoys"] = list(item["answers"])
            w["overlay_docs"] = [{"id": f"{tag}:0", "contents": make_contents(fh["title"], gen["rewritten_final"])}]
            w["evidence_ids"] = [f"{tag}:0"]
        else:
            w["golds"] = []
            w["unanswerable"] = True
            w["evidence_ids"] = []
        return w
    raise ValueError(family)


TEST_WORLDS = [("identity", 0), ("dup", 3), ("inject", 1), ("inject", 3), ("inject", 8),
               ("consistent", 1), ("consistent", 3), ("consistent", 8), ("rewrite", 0), ("delete", 0)]
TRAIN_FAMILY_PROBS = {"dup": 0.15, "inject": 0.25, "consistent": 0.20, "rewrite": 0.15, "delete": 0.25}
TRAIN_M = [1, 2, 3, 5, 8]


def sample_train_world(rec: dict, rng: random.Random, families: Sequence[str], probs: dict | None = None,
                       m_weights: dict | None = None) -> dict:
    """Sample one non-identity world for a training record; falls back to identity if none is feasible."""
    fams = [f for f in families if f != "identity"]
    if not fams:
        return make_world(rec["item"], rec.get("gen"), rec.get("mask"), "identity")
    table = probs or TRAIN_FAMILY_PROBS
    weights = [table.get(f, 0.2) for f in fams]
    for _ in range(8):
        fam = rng.choices(fams, weights=weights)[0]
        if fam not in ("dup", "inject", "consistent"):
            m = 0
        elif m_weights:
            m = rng.choices(list(m_weights), weights=list(m_weights.values()))[0]
        else:
            m = rng.choice(TRAIN_M)
        w = make_world(rec["item"], rec.get("gen"), rec.get("mask"), fam, m)
        if w is not None:
            return w
    return make_world(rec["item"], rec.get("gen"), rec.get("mask"), "identity")


def _index(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def style_audit(gen_recs: Sequence[dict], seed: int = 0) -> dict:
    """Single-document classifier: true-claim copies vs false-claim copies. Target AUC < 0.65."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    texts, y, g = [], [], []
    for r in gen_recs:
        if r.get("failed"):
            continue
        for t in r["claim_true"]:
            texts.append(t), y.append(0), g.append(r["id"])
        for t in r["claim_false"]:
            texts.append(t), y.append(1), g.append(r["id"])
    if len(set(g)) < 10:
        return {"n": len(texts), "auc_word": None, "auc_char": None}
    y, g = np.array(y), np.array(g)
    out = {"n_docs": len(texts), "n_items": int(len(set(g)))}
    for name, kw in (("auc_word", dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("auc_char", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))):
        aucs = []
        for tr, te in GroupKFold(n_splits=5).split(texts, y, g):
            v = TfidfVectorizer(**kw)
            xtr = v.fit_transform([texts[i] for i in tr])
            clf = LogisticRegression(max_iter=2000).fit(xtr, y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(v.transform([texts[i] for i in te]))[:, 1]))
        out[name] = round(float(np.mean(aucs)), 4)
    out["pass"] = bool(max(out["auc_word"], out["auc_char"]) < 0.65)
    return out


def assemble(root: Path, gen_a: str, gen_b: str | None) -> dict:
    base = {s: read_jsonl(root / f"base_{s}.jsonl") for s in ("train", "dev", "test")}
    masks = {s: _index(root / f"mask_{s}.jsonl") for s in base}
    gens_a = {s: _index(root / f"gen_A_{s}.jsonl") for s in base}
    gens_b = _index(root / "gen_B_test.jsonl") if gen_b else {}
    report: dict = {"gen_A": gen_a, "gen_B": gen_b}

    # Training records: keep items whose clean evidence is retrievable in wiki-18.
    train = []
    for it in base["train"]:
        mk = masks["train"].get(it["id"])
        if mk is None or not mk["evidence_found"]:
            continue
        train.append({"item": it, "gen": gens_a["train"].get(it["id"]), "mask": mk})
    write_jsonl(root / "spec_train.jsonl", train)

    rng = random.Random(0)
    dev_worlds = []
    fams = ["identity", "dup", "inject", "consistent", "rewrite", "delete"]
    for i, it in enumerate(base["dev"]):
        mk = masks["dev"].get(it["id"])
        if mk is None or not mk["evidence_found"]:
            continue
        fam = fams[i % len(fams)]
        m = rng.choice([1, 3, 8]) if fam in ("dup", "inject", "consistent") else 0
        w = make_world(it, gens_a["dev"].get(it["id"]), mk, fam, m) or make_world(it, None, mk, "identity")
        dev_worlds.append(w)
    write_jsonl(root / "spec_dev.jsonl", dev_worlds)

    for tag, gens in (("A", gens_a["test"]), ("B", gens_b)):
        if not gens:
            continue
        worlds, kept = [], 0
        for it in base["test"]:
            mk = masks["test"].get(it["id"])
            gr = gens.get(it["id"])
            if mk is None or not mk["evidence_found"] or gr is None or gr.get("failed"):
                continue
            ws = [make_world(it, gr, mk, f, m) for f, m in TEST_WORLDS]
            if any(w is None for w in ws):
                continue  # keep only items with the complete world set so families are paired
            worlds.extend(ws)
            kept += 1
        write_jsonl(root / f"spec_test_{tag}.jsonl", worlds)
        report[f"test_{tag}_items"] = kept
        report[f"test_{tag}_worlds"] = len(worlds)

    report["train_items"] = len(train)
    report["train_items_with_gen"] = sum(1 for r in train if r["gen"] and not r["gen"].get("failed"))
    report["train_items_maskable"] = sum(1 for r in train if r["mask"]["verified"])
    report["dev_worlds"] = len(dev_worlds)
    for s in base:
        recs = list(masks[s].values())
        report[f"mask_{s}"] = {
            "n": len(recs),
            "evidence_found": sum(r["evidence_found"] for r in recs),
            "maskable": sum(r["evidence_found"] and r["verified"] for r in recs),
            "mean_mask_size": round(sum(len(r["mask"]) for r in recs) / max(1, len(recs)), 2),
        }
    report["style_audit_A_train"] = style_audit(list(gens_a["train"].values()))
    if gens_b:
        report["style_audit_B_test"] = style_audit(list(gens_b.values()))
    (root / "assemble_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[transforms] assemble report: {json.dumps(report, indent=2)}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--endpoints_json", required=True)
    g.add_argument("--concurrency", type=int, default=192)
    v = sub.add_parser("verify")
    v.add_argument("--base", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--retriever_url", default="http://127.0.0.1:8290")
    s = sub.add_parser("assemble")
    s.add_argument("--root", required=True)
    s.add_argument("--gen_a", required=True)
    s.add_argument("--gen_b", default=None)
    a = ap.parse_args()
    if a.cmd == "generate":
        ep = json.loads(Path(a.endpoints_json).read_text(encoding="utf-8"))
        asyncio.run(generate(Path(a.base), Path(a.out), ep["endpoints"], ep["model"], a.concurrency))
    elif a.cmd == "verify":
        asyncio.run(verify(Path(a.base), Path(a.out), a.retriever_url))
    else:
        assemble(Path(a.root), a.gen_a, a.gen_b)


if __name__ == "__main__":
    main()
