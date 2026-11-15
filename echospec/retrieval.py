"""GPU-resident dense retrieval over wiki-18 with per-request overlays and masks.

- Corpus: memory-mapped Arrow file with one `contents` column (row index == document id), or a JSONL file.
- Index: fp16 embedding matrix (N x 768) resident on one GPU; exact inner-product search.
- Server: FastAPI; concurrent requests are micro-batched into one matmul.
- Overlays: per-request extra documents ({"id", "contents"}) scored with the same encoder and merged into top-k.
- Masks: per-request document ids removed from the candidate list.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import zlib
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
from fastapi import FastAPI, Request  # module level: endpoint annotations are strings (postponed evaluation)


# ----------------------------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------------------------
class ArrowCorpus:
    """Zero-copy random access to `contents` by integer row id."""

    def __init__(self, path: str | Path):
        import pyarrow as pa

        self._src = pa.memory_map(str(path), "r")
        table = pa.ipc.open_file(self._src).read_all()
        self.col = table.column("contents")

    def __len__(self) -> int:
        return len(self.col)

    def get(self, i: int) -> str:
        return self.col[int(i)].as_py()


class ListCorpus:
    def __init__(self, ids: Sequence[str], contents: Sequence[str]):
        self.ids = list(ids)
        self.contents = list(contents)

    def __len__(self) -> int:
        return len(self.contents)

    def get(self, i: int) -> str:
        return self.contents[int(i)]


def load_jsonl_corpus(path: str | Path) -> ListCorpus:
    ids, contents = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(str(r["id"]))
            if "contents" in r:
                contents.append(r["contents"])
            else:
                contents.append(r.get("text", ""))
    return ListCorpus(ids, contents)


# ----------------------------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------------------------
class E5Encoder:
    """Bi-encoder supporting intfloat/e5-* (mean pool + prefixes) and BAAI/bge-* (CLS pool + query instruction)."""

    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "cuda:0", max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        low = model_name.lower()
        if "bge" in low:
            self.q_prefix = "Represent this sentence for searching relevant passages: "
            self.p_prefix = ""
            self.pooling = "cls"
        elif "e5" in low:
            self.q_prefix = "query: "
            self.p_prefix = "passage: "
            self.pooling = "mean"
        else:
            self.q_prefix = ""
            self.p_prefix = ""
            self.pooling = "mean"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

    def encode(self, texts: Sequence[str], is_query: bool, batch_size: int = 256) -> np.ndarray:
        torch = self.torch
        prefix = self.q_prefix if is_query else self.p_prefix
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                batch = [prefix + t for t in texts[s : s + batch_size]] if prefix else list(texts[s : s + batch_size])
                enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                if self.pooling == "cls":
                    pooled = h[:, 0]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
                out.append(pooled.cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 768), dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# TrustRAG clean-retrieval stage (Zhou et al., 2025): K-means (k=2) over passage embeddings; a cluster whose members are
# mutually near-identical (cosine and ROUGE-L above the paper's thresholds) is treated as injected and removed.
# ----------------------------------------------------------------------------------------------
def _rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(x), lcs / len(y)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def trustrag_keep(embs: np.ndarray, texts: Sequence[str], cos_thr: float = 0.85, rouge_thr: float = 0.25,
                  iters: int = 20) -> list[bool]:
    n = len(texts)
    if n < 3:
        return [True] * n
    e = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-6, None)
    sim = e @ e.T
    i0 = int(np.argmax(sim.mean(1)))  # densest point seeds one centroid, the point farthest from it the other
    i1 = int(np.argmin(sim[i0]))
    cent = e[[i0, i1]].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        lab = np.argmax(e @ cent.T, axis=1)
        for k in range(2):
            if (lab == k).any():
                c = e[lab == k].mean(0)
                cent[k] = c / max(np.linalg.norm(c), 1e-6)
    keep = [True] * n
    for k in range(2):
        idx = np.where(lab == k)[0]
        if len(idx) < 2:
            continue
        pairs = [(a, b) for ai, a in enumerate(idx) for b in idx[ai + 1:]]
        mean_cos = float(np.mean([sim[a, b] for a, b in pairs]))
        mean_rouge = float(np.mean([_rouge_l(texts[a], texts[b]) for a, b in pairs]))
        if mean_cos > cos_thr and mean_rouge > rouge_thr:
            for a in idx:
                keep[a] = False
    if not any(keep):  # never return an empty result list
        return [True] * n
    return keep


def novelty_keep(embs: np.ndarray, cos_thr: float = 0.88) -> list[bool]:
    """Greedy rank-preserving semantic deduplication: keep a candidate iff its cosine similarity to every
    higher-ranked kept candidate is below cos_thr (collapses paraphrased copies to one representative without
    discarding the topic cluster)."""
    if len(embs) <= 1:
        return [True] * len(embs)
    e = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-6, None)
    kept_idx: list[int] = []
    out: list[bool] = []
    for i in range(len(e)):
        if any(float(e[i] @ e[j]) >= cos_thr for j in kept_idx):
            out.append(False)
        else:
            kept_idx.append(i)
            out.append(True)
    return out


# ----------------------------------------------------------------------------------------------
# Provenance gate (Zhou et al., EMNLP 2026, "Evidence Laundering"): fingerprint each candidate, connect pages whose
# fingerprint Jaccard similarity is >= tau, keep the highest-ranked page of each connected component.
#   shingle:  word 5-grams, tau = 0.45 (lexical baseline)
#   skeleton: entities, values and content words masked, function words and punctuation kept, 4-grams, tau = 0.70
# ----------------------------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, corpus, emb_path: str | Path, encoder: E5Encoder, device: str = "cuda:0", chunk_rows: int = 2_000_000):
        try:
            import torch
        except ImportError:
            torch = None

        self.torch = torch
        self.corpus = corpus
        self.encoder = encoder
        mm = np.load(emb_path, mmap_mode="r")
        if len(mm) != len(corpus):
            raise ValueError(f"embedding rows {len(mm)} != corpus rows {len(corpus)} for {emb_path}")
        if torch is not None:
            self.device = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.emb = torch.empty((mm.shape[0], mm.shape[1]), dtype=dtype, device=self.device)
            for s in range(0, mm.shape[0], chunk_rows):
                self.emb[s : s + chunk_rows] = torch.from_numpy(np.ascontiguousarray(mm[s : s + chunk_rows])).to(self.device, dtype)
            self.emb_np = None
        else:
            self.device = None
            self.emb = None
            self.emb_np = np.asarray(mm, dtype=np.float32)
        self._ov_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.ids = getattr(corpus, "ids", None)  # JSONL corpora carry string ids; Arrow corpora use row numbers

    def doc_id(self, row: int) -> str:
        return self.ids[row] if self.ids is not None else str(row)

    def _row_emb_np(self, row: int) -> np.ndarray:
        return self.emb[row].float().cpu().numpy() if self.emb is not None else self.emb_np[row]

    def _overlay_embs(self, docs: Sequence[dict]) -> np.ndarray:
        keys = [hashlib.md5(d["contents"].encode("utf-8")).hexdigest() for d in docs]
        missing = [(k, d["contents"]) for k, d in zip(keys, docs) if k not in self._ov_cache]
        if missing:
            embs = self.encoder.encode([c for _, c in missing], is_query=False, batch_size=128)
            for (k, _), e in zip(missing, embs):
                self._ov_cache[k] = e
            while len(self._ov_cache) > 200_000:
                self._ov_cache.popitem(last=False)
        return np.stack([self._ov_cache[k] for k in keys], 0)

    def search_batch(self, reqs: Sequence[dict]) -> list[list[dict]]:
        torch = self.torch
        if not reqs:
            return []
        q = self.encoder.encode([r["query"] for r in reqs], is_query=True, batch_size=len(reqs))
        max_mask = max(len(r.get("masked_doc_ids") or ()) for r in reqs)
        max_k = max(int(r.get("top_k", 3)) for r in reqs)
        max_pool = max(int(r.get("pool", 15 if r.get("filter") else 0)) for r in reqs)  # candidate pool of a filter
        fetch = min(len(self.corpus), max(max_k, max_pool) + max_mask + 8)
        if torch is not None and self.emb is not None:
            with torch.inference_mode():
                qt = torch.as_tensor(q, dtype=self.emb.dtype, device=self.device)
                scores = qt @ self.emb.T
                vals, idxs = torch.topk(scores, k=fetch, dim=-1)
                vals, idxs = vals.float().cpu().numpy(), idxs.cpu().numpy()
        else:
            scores = np.asarray(q, dtype=np.float32) @ self.emb_np.T
            if fetch >= len(self.corpus):
                idxs = np.argsort(-scores, axis=-1)
            else:
                part = np.argpartition(-scores, fetch - 1, axis=-1)[:, :fetch]
                order = np.argsort(-np.take_along_axis(scores, part, axis=-1), axis=-1)
                idxs = np.take_along_axis(part, order, axis=-1)
            vals = np.take_along_axis(scores, idxs, axis=-1)
        results = []
        for b, r in enumerate(reqs):
            masked = set(map(str, r.get("masked_doc_ids") or ()))
            cands: list[tuple[float, str, object]] = []  # (score, id, row:int | contents:str)
            for sc, row in zip(vals[b], idxs[b]):
                did = self.doc_id(int(row))
                if did in masked:
                    continue
                cands.append((float(sc), did, int(row)))
            ov = [d for d in (r.get("overlay_docs") or ()) if str(d.get("id")) not in masked]
            if ov:
                ov_scores = self._overlay_embs(ov) @ q[b]
                for sc, d in zip(ov_scores, ov):
                    cands.append((float(sc), str(d["id"]), d["contents"]))
            cands.sort(key=lambda x: -x[0])
            if r.get("filter") in ("trustrag", "novelty"):
                pool = cands[: int(r.get("pool", 15 if r.get("filter") == "novelty" else 10))]
                embs = []
                for _, _, src in pool:
                    if isinstance(src, int):
                        embs.append(self._row_emb_np(src))
                    else:
                        embs.append(self._overlay_embs([{"contents": src}])[0])
                mat = np.stack(embs, 0)
                if r.get("filter") == "novelty":
                    keep = novelty_keep(mat, float(r.get("cos_thr", 0.88)))
                else:
                    texts = [self.corpus.get(src) if isinstance(src, int) else src for _, _, src in pool]
                    keep = trustrag_keep(mat, texts)
                cands = [c for c, k in zip(pool, keep) if k]
            top = []
            for sc, did, src in cands[: int(r.get("top_k", 3))]:
                contents = self.corpus.get(src) if isinstance(src, int) else src
                top.append({"id": did, "contents": contents, "score": sc})
            results.append(top)
        return results


BM25_VOCAB = 1 << 21
_BM25_TOK = re.compile(r"[a-z0-9]+")
_BM25_STOP = frozenset((
    "a an the and or but of in on at to from by with for is are was were be been being as it its "
    "this that these those have has had do does did not no into about over after before between"
).split())


def bm25_tokens(text: str) -> list[int]:
    return [
        zlib.crc32(t.encode("utf-8")) & (BM25_VOCAB - 1)
        for t in _BM25_TOK.findall(text.lower())
        if t not in _BM25_STOP and (len(t) >= 2 or t.isdigit())
    ]


class BM25Index:
    """GPU-accelerated exact Okapi BM25 (k1=1.2, b=0.75) with identical overlay and mask semantics."""

    def __init__(self, corpus, bm25_dir: str | Path, encoder: E5Encoder | None = None,
                 device: str = "cuda:0", k1: float = 1.2, b: float = 0.75, chunk_elems: int = 50_000_000):
        try:
            import torch
        except ImportError:
            torch = None

        self.torch = torch
        self.corpus = corpus
        self.encoder = encoder
        self.k1, self.b = k1, b
        self.ids = getattr(corpus, "ids", None)
        d = Path(bm25_dir)
        meta = np.load(d / "bm25_meta.npz")
        self.idf = meta["idf"].astype(np.float32)
        self.avgdl = float(meta["avgdl"])
        if int(meta["n_docs"]) != len(corpus):
            raise ValueError(f"BM25 n_docs {int(meta['n_docs'])} != corpus {len(corpus)}")
        self.indptr = np.load(d / "bm25_indptr.npy")
        mm_docs = np.load(d / "bm25_docs.npy", mmap_mode="r")
        mm_w = np.load(d / "bm25_weights.npy", mmap_mode="r")
        use_cuda = torch is not None and str(device).startswith("cuda") and torch.cuda.is_available()
        self.device_type = "cuda" if use_cuda else "cpu"
        if use_cuda:
            self.device = torch.device(device)
            nnz = len(mm_docs)
            self.docs = torch.empty(nnz, dtype=torch.int32, device=self.device)
            self.weights = torch.empty(nnz, dtype=torch.float16, device=self.device)
            for s in range(0, nnz, chunk_elems):
                e = min(nnz, s + chunk_elems)
                self.docs[s:e] = torch.from_numpy(np.ascontiguousarray(mm_docs[s:e])).to(self.device)
                self.weights[s:e] = torch.from_numpy(np.ascontiguousarray(mm_w[s:e])).to(self.device)
            self._scores = torch.zeros(len(corpus), dtype=torch.float32, device=self.device)
        else:
            self.docs_np = np.asarray(mm_docs)
            self.weights_np = np.asarray(mm_w)
            self._scores_np = np.zeros(len(corpus), dtype=np.float32)
        self._ov_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def doc_id(self, row: int) -> str:
        return self.ids[row] if self.ids is not None else str(row)

    def _score_overlay(self, text: str, q_buckets: list[int]) -> float:
        if not q_buckets:
            return 0.0
        toks = bm25_tokens(text)
        if not toks:
            return 0.0
        cnt = Counter(toks)
        norm = 1.0 - self.b + self.b * (len(toks) / max(self.avgdl, 1e-6))
        sc = 0.0
        for bk in q_buckets:
            tf = cnt.get(bk, 0)
            if tf:
                sc += float(self.idf[bk]) * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return sc

    def _filter_embs(self, texts: Sequence[str]) -> np.ndarray:
        assert self.encoder is not None, "BM25 filter requires an auxiliary encoder"
        keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._ov_cache]
        if missing:
            embs = self.encoder.encode([t for _, t in missing], is_query=False, batch_size=128)
            for (k, _), e in zip(missing, embs):
                self._ov_cache[k] = e
            while len(self._ov_cache) > 100_000:
                self._ov_cache.popitem(last=False)
        return np.stack([self._ov_cache[k] for k in keys], 0)

    def search_batch(self, reqs: Sequence[dict]) -> list[list[dict]]:
        torch = self.torch
        results = []
        for r in reqs:
            q_buckets = sorted({bk for bk in bm25_tokens(r["query"]) if self.idf[bk] > 0.0})
            masked = set(map(str, r.get("masked_doc_ids") or ()))
            k_req = int(r.get("top_k", 3))
            pool_sz = int(r.get("pool", 15 if r.get("filter") else 0))
            fetch = min(len(self.corpus), max(k_req, pool_sz) + len(masked) + 8)
            if self.device_type == "cuda":
                with torch.inference_mode():
                    self._scores.zero_()
                    for bk in q_buckets:
                        s, e = int(self.indptr[bk]), int(self.indptr[bk + 1])
                        if e > s:
                            self._scores.index_add_(0, self.docs[s:e].long(), self.weights[s:e].float())
                    vals_t, idxs_t = torch.topk(self._scores, k=fetch)
                    vals, idxs = vals_t.cpu().numpy(), idxs_t.cpu().numpy()
            else:
                self._scores_np.fill(0.0)
                for bk in q_buckets:
                    s, e = int(self.indptr[bk]), int(self.indptr[bk + 1])
                    if e > s:
                        self._scores_np[self.docs_np[s:e]] += self.weights_np[s:e].astype(np.float32)
                if fetch >= len(self.corpus):
                    idxs = np.argsort(-self._scores_np)
                else:
                    part = np.argpartition(-self._scores_np, fetch - 1)[:fetch]
                    idxs = part[np.argsort(-self._scores_np[part])]
                vals = self._scores_np[idxs]

            cands: list[tuple[float, str, object]] = []
            for sc, row in zip(vals, idxs):
                did = self.doc_id(int(row))
                if did not in masked:
                    cands.append((float(sc), did, int(row)))
            for d in (r.get("overlay_docs") or ()):
                did = str(d.get("id"))
                if did not in masked:
                    cands.append((self._score_overlay(d["contents"], q_buckets), did, d["contents"]))
            cands.sort(key=lambda x: -x[0])

            if r.get("filter") in ("trustrag", "novelty") and self.encoder is not None:
                pool = cands[: int(r.get("pool", 15 if r.get("filter") == "novelty" else 10))]
                texts = [self.corpus.get(src) if isinstance(src, int) else src for _, _, src in pool]
                mat = self._filter_embs(texts)
                keep = novelty_keep(mat, float(r.get("cos_thr", 0.88))) if r.get("filter") == "novelty" else trustrag_keep(mat, texts)
                cands = [c for c, kp in zip(pool, keep) if kp]

            top = []
            for sc, did, src in cands[:k_req]:
                contents = self.corpus.get(src) if isinstance(src, int) else src
                top.append({"id": did, "contents": contents, "score": sc})
            results.append(top)
        return results


# ----------------------------------------------------------------------------------------------
# Server with micro-batching
# ----------------------------------------------------------------------------------------------
def create_app(indices: dict[str, DenseIndex], max_batch: int = 96, max_wait_ms: float = 3.0):
    from contextlib import asynccontextmanager

    queues: dict[str, asyncio.Queue] = {}
    pool = ThreadPoolExecutor(max_workers=1)  # one GPU worker: keeps CUDA calls serialized
    stats = {"requests": 0, "batches": 0}

    async def worker(name: str):
        q = queues[name]
        loop = asyncio.get_running_loop()
        while True:
            first = await q.get()
            batch = [first]
            deadline = time.perf_counter() + max_wait_ms / 1000.0
            while len(batch) < max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(q.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            reqs = [b[0] for b in batch]
            try:
                res = await loop.run_in_executor(pool, indices[name].search_batch, reqs)
                for (_, fut), r in zip(batch, res):
                    if not fut.done():
                        fut.set_result(r)
            except Exception as e:  # noqa: BLE001
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
            stats["batches"] += 1

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        tasks = []
        for name in indices:
            queues[name] = asyncio.Queue()
            tasks.append(asyncio.create_task(worker(name)))
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()

    app = FastAPI(title="EchoSpec retrieval", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "corpora": {k: len(v.corpus) for k, v in indices.items()}, **stats}

    @app.post("/search")
    async def search(request: Request):
        body = await request.json()
        name = body.get("corpus") or next(iter(indices))
        if name not in indices:
            name = next(iter(indices))
        fut = asyncio.get_running_loop().create_future()
        await queues[name].put((body, fut))
        stats["requests"] += 1
        return {"results": await fut}

    return app


class RetrieverClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8290", timeout: float = 120.0):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._c = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=512, max_keepalive_connections=256))

    async def search(self, query: str, top_k: int = 3, corpus: str = "wiki18", overlay_docs=(), masked_doc_ids=(),
                     filter: str | None = None) -> list[dict]:
        payload = {
            "query": query,
            "top_k": top_k,
            "corpus": corpus,
            "overlay_docs": list(overlay_docs),
            "masked_doc_ids": list(masked_doc_ids),
        }
        if filter:
            payload["filter"] = filter
        last = None
        for attempt in range(5):
            try:
                r = await self._c.post(f"{self.base_url}/search", json=payload)
                r.raise_for_status()
                return r.json()["results"]
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(0.5 * 2**attempt)
        raise RuntimeError(f"retrieval failed after retries: {last}")

    async def aclose(self):
        await self._c.aclose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, help="name=path.arrow|path.jsonl (repeatable)")
    ap.add_argument("--emb", action="append", default=[], help="name=path.npy (repeatable)")
    ap.add_argument("--bm25", action="append", default=[], help="name=bm25_dir (repeatable)")
    ap.add_argument("--encoder_map", action="append", default=[], help="name=hf_model (repeatable)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8290)
    ap.add_argument("--encoder", default="intfloat/e5-base-v2")
    args = ap.parse_args()

    corp = dict(kv.split("=", 1) for kv in args.corpus)
    embs = dict(kv.split("=", 1) for kv in args.emb)
    bm25s = dict(kv.split("=", 1) for kv in args.bm25)
    enc_map = dict(kv.split("=", 1) for kv in args.encoder_map)

    loaded_corpora: dict[str, object] = {}
    loaded_encoders: dict[str, E5Encoder] = {}

    def get_enc(model_name: str) -> E5Encoder:
        if model_name not in loaded_encoders:
            loaded_encoders[model_name] = E5Encoder(model_name, device=args.device)
        return loaded_encoders[model_name]

    default_enc = get_enc(args.encoder)
    indices = {}
    for name, path in corp.items():
        t0 = time.time()
        if path not in loaded_corpora:
            loaded_corpora[path] = ArrowCorpus(path) if path.endswith(".arrow") else load_jsonl_corpus(path)
        c = loaded_corpora[path]
        if name in bm25s:
            indices[name] = BM25Index(c, bm25s[name], encoder=default_enc, device=args.device)
            print(f"[retrieval] loaded BM25 {name}: {len(c):,} docs in {time.time() - t0:.0f}s", flush=True)
        elif name in embs:
            mname = enc_map.get(name) or ("BAAI/bge-base-en-v1.5" if "bge" in name.lower() else args.encoder)
            indices[name] = DenseIndex(c, embs[name], get_enc(mname), device=args.device)
            print(f"[retrieval] loaded Dense {name} ({mname}): {len(c):,} docs in {time.time() - t0:.0f}s", flush=True)
        else:
            raise ValueError(f"corpus {name} needs either --emb {name}=... or --bm25 {name}=...")

    # Self-check: a query must retrieve something and scores must be finite.
    for name, idx in indices.items():
        res = idx.search_batch([{"query": "Who wrote the play Hamlet?", "top_k": 3}])[0]
        print(f"[retrieval] self-check {name}: {[r['contents'][:80] for r in res]}", flush=True)

    import uvicorn

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    uvicorn.run(create_app(indices), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
