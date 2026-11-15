"""Build the wiki-18 retrieval assets used by Search-R1.

Outputs (in --out):
  wiki18_contents.arrow   one `contents` column; row index == document id
  wiki18_e5_fp16.npy      (N, 768) float16 e5-base-v2 passage embeddings, row-aligned with the corpus
  READY.json              provenance and sizes

Primary path: download the official corpus and the official FAISS flat index and convert them.
Fallback: encode the corpus ourselves with e5-base-v2 on all visible GPUs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CORPUS_REPOS = ["PeterJinGo/wiki-18-corpus"]
INDEX_REPOS = ["PeterJinGo/wiki-18-e5-index"]


def log(msg: str) -> None:
    print(f"[wiki18 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_repo(repo: str) -> tuple[str, list[str]] | None:
    from huggingface_hub import HfApi

    api = HfApi()
    for rt in ("dataset", "model"):
        try:
            return rt, api.list_repo_files(repo, repo_type=rt)
        except Exception:
            continue
    return None


def download(repo: str, rt: str, fname: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, fname, repo_type=rt, local_dir=str(dest)))


# ----------------------------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------------------------
class _StreamLines:
    """UTF-8 line iterator over a non-seekable binary stream (e.g. a streaming tar member)."""

    def __init__(self, raw, owner, chunk: int = 16 << 20):
        self.raw, self.owner, self.chunk = raw, owner, chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.owner.close()

    def __iter__(self):
        import codecs

        dec = codecs.getincrementaldecoder("utf-8")()
        rest = ""
        while True:
            b = self.raw.read(self.chunk)
            if not b:
                break
            parts = (rest + dec.decode(b)).split("\n")
            rest = parts.pop()
            for p in parts:
                yield p + "\n"
        rest += dec.decode(b"", final=True)
        if rest:
            yield rest


def open_corpus_lines(src: Path):
    """Text lines of the corpus: plain JSONL, gzipped JSONL, or a (gzipped) tar holding one JSONL member."""
    import tarfile

    if tarfile.is_tarfile(str(src)):
        tf = tarfile.open(str(src), mode="r|*")
        for member in tf:
            if member.isfile() and ".jsonl" in member.name:
                log(f"reading tar member {member.name} ({member.size:,} bytes)")
                return _StreamLines(tf.extractfile(member), tf)
        raise RuntimeError(f"no .jsonl member inside {src}")
    if str(src).endswith(".gz"):
        return gzip.open(src, "rt", encoding="utf-8")
    return open(src, encoding="utf-8")


def build_corpus(work: Path, out_arrow: Path) -> int:
    import pyarrow as pa

    src = None
    for repo in CORPUS_REPOS:
        found = list_repo(repo)
        if not found:
            log(f"corpus repo {repo} not reachable")
            continue
        rt, files = found
        cands = [f for f in files if f.endswith((".jsonl.gz", ".jsonl"))]
        log(f"corpus repo {repo} ({rt}) files: {files}")
        if cands:
            src = download(repo, rt, sorted(cands, key=len)[0], work / "corpus_dl")
            break
    if src is None:
        raise RuntimeError("could not locate the wiki-18 corpus on the Hub")
    log(f"converting {src} -> {out_arrow}")
    schema = pa.schema([("contents", pa.large_string())])
    tmp = out_arrow.with_suffix(".tmp")
    n, mismatched, buf = 0, 0, []
    with open_corpus_lines(src) as f, pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, schema) as w:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("id", n)) != str(n):
                mismatched += 1
            c = r.get("contents")
            if c is None:
                c = f'"{r.get("title", "")}"\n{r.get("text", "")}'
            buf.append(c)
            n += 1
            if len(buf) >= 500_000:
                w.write_table(pa.table({"contents": pa.array(buf, type=pa.large_string())}))
                buf = []
                log(f"  {n:,} rows")
        if buf:
            w.write_table(pa.table({"contents": pa.array(buf, type=pa.large_string())}))
    os.replace(tmp, out_arrow)
    log(f"corpus rows={n:,} (id!=row for {mismatched:,} rows)")
    return n


# ----------------------------------------------------------------------------------------------
# Official FAISS index -> fp16 npy
# ----------------------------------------------------------------------------------------------
def convert_official_index(work: Path, out_npy: Path, n_rows: int) -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"], check=False)
    try:
        import faiss
    except ImportError:
        log("faiss unavailable")
        return False

    for repo in INDEX_REPOS:
        found = list_repo(repo)
        if not found:
            log(f"index repo {repo} not reachable")
            continue
        rt, files = found
        log(f"index repo {repo} ({rt}) files: {files}")
        parts = sorted(f for f in files if os.path.basename(f).startswith("part_"))
        whole = [f for f in files if f.endswith(".index")]
        dl = work / "index_dl"
        if parts:
            paths = [download(repo, rt, p, dl) for p in parts]
            idx_path = dl / "e5_Flat.index"
            log(f"concatenating {len(paths)} parts -> {idx_path}")
            with open(idx_path, "wb") as out:
                for p in paths:
                    with open(p, "rb") as inp:
                        shutil.copyfileobj(inp, out, length=64 << 20)
                    p.unlink()
        elif whole:
            idx_path = download(repo, rt, whole[0], dl)
        else:
            continue
        log("reading FAISS index into RAM")
        index = faiss.read_index(str(idx_path))
        n, d = index.ntotal, index.d
        log(f"index ntotal={n:,} d={d} metric={index.metric_type}")
        if n != n_rows:
            log(f"row mismatch: index {n} vs corpus {n_rows}")
            return False
        try:
            xb = faiss.rev_swig_ptr(index.get_xb(), n * d).reshape(n, d)
        except Exception:
            xb = faiss.vector_to_array(index.codes).view(np.float32).reshape(n, d)
        tmp = out_npy.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, d))
        for s in range(0, n, 1_000_000):
            mm[s : s + 1_000_000] = xb[s : s + 1_000_000].astype(np.float16)
        norms = np.linalg.norm(xb[:: max(1, n // 20000)].astype(np.float32), axis=1)
        log(f"stored vector norm: mean={norms.mean():.4f} min={norms.min():.4f} max={norms.max():.4f}")
        mm.flush()
        del mm, xb, index
        os.replace(tmp, out_npy)
        idx_path.unlink(missing_ok=True)
        return True
    return False


# ----------------------------------------------------------------------------------------------
# Fallback / multi-model dense encoding & parallel Okapi BM25 builder
# ----------------------------------------------------------------------------------------------
def encode_shard(arrow: Path, out_npy: Path, start: int, end: int, model_name: str = "intfloat/e5-base-v2",
                 batch: int = 2048) -> None:
    from echospec.retrieval import ArrowCorpus, E5Encoder

    corpus = ArrowCorpus(arrow) if str(arrow).endswith(".arrow") else None
    if corpus is None:
        from echospec.retrieval import load_jsonl_corpus
        corpus = load_jsonl_corpus(arrow)
    enc = E5Encoder(model_name=model_name, device="cuda:0", max_length=256)
    mm = np.load(out_npy, mmap_mode="r+")
    step = 200_000
    for s in range(start, end, step):
        e = min(end, s + step)
        texts = [corpus.get(i) for i in range(s, e)]
        order = np.argsort([len(t) for t in texts])
        embs = enc.encode([texts[i] for i in order], is_query=False, batch_size=batch)
        out = np.empty_like(embs)
        out[order] = embs
        mm[s:e] = out.astype(np.float16)
        mm.flush()
        log(f"  [{model_name}] shard [{start},{end}) at {e:,}")


def self_encode(arrow: Path, out_npy: Path, n_rows: int, model_name: str = "intfloat/e5-base-v2") -> None:
    import torch

    n_gpu = max(1, torch.cuda.device_count())
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npy.with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n_rows, 768))
    del mm
    bounds = np.linspace(0, n_rows, n_gpu + 1).astype(int)
    procs = []
    for g in range(n_gpu):
        if bounds[g] >= bounds[g + 1]:
            continue
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(g), CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, "-m", "echospec.wiki18", "--encode_shard", str(arrow), str(tmp),
               str(bounds[g]), str(bounds[g + 1]), "--encoder_model", model_name]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"encode shards failed ({model_name}): {codes}")
    os.replace(tmp, out_npy)


def bm25_shard(arrow: Path, out_npz: Path, start: int, end: int) -> None:
    from collections import Counter
    from echospec.retrieval import ArrowCorpus, BM25_VOCAB, bm25_tokens, load_jsonl_corpus

    corpus = ArrowCorpus(arrow) if str(arrow).endswith(".arrow") else load_jsonl_corpus(arrow)
    dl = np.empty(end - start, dtype=np.uint16)
    b_chunks, d_chunks, tf_chunks = [], [], []
    step = 50_000
    for s in range(start, end, step):
        e = min(end, s + step)
        bs, ds, tfs = [], [], []
        for doc_id in range(s, e):
            toks = bm25_tokens(corpus.get(doc_id))
            dl[doc_id - start] = min(65535, len(toks))
            if toks:
                cnt = Counter(toks)
                bs.extend(cnt.keys())
                ds.extend([doc_id] * len(cnt))
                tfs.extend(min(255, v) for v in cnt.values())
        if bs:
            b_chunks.append(np.asarray(bs, dtype=np.int32))
            d_chunks.append(np.asarray(ds, dtype=np.int32))
            tf_chunks.append(np.asarray(tfs, dtype=np.uint8))
    if b_chunks:
        buckets = np.concatenate(b_chunks)
        docs = np.concatenate(d_chunks)
        tfs = np.concatenate(tf_chunks)
        order = np.argsort(buckets, kind="mergesort")
        buckets, docs, tfs = buckets[order], docs[order], tfs[order]
        counts = np.bincount(buckets, minlength=BM25_VOCAB).astype(np.int32)
    else:
        buckets = np.zeros(0, dtype=np.int32)
        docs = np.zeros(0, dtype=np.int32)
        tfs = np.zeros(0, dtype=np.uint8)
        counts = np.zeros(BM25_VOCAB, dtype=np.int32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, counts=counts, dl=dl, buckets=buckets, docs=docs, tfs=tfs)


def build_bm25(arrow: Path, out_dir: Path, work_dir: Path | None = None, k1: float = 1.2, b: float = 0.75) -> None:
    from echospec.retrieval import ArrowCorpus, BM25_VOCAB, load_jsonl_corpus

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "READY.json").exists():
        log(f"BM25 already built at {out_dir}")
        return
    corpus = ArrowCorpus(arrow) if str(arrow).endswith(".arrow") else load_jsonl_corpus(arrow)
    n_docs = len(corpus)
    del corpus
    n_workers = min(32, max(1, os.cpu_count() or 8), max(1, n_docs // 1000 or 1))
    wdir = (work_dir or out_dir) / "_bm25_shards"
    shutil.rmtree(wdir, ignore_errors=True)
    wdir.mkdir(parents=True, exist_ok=True)
    bounds = np.linspace(0, n_docs, n_workers + 1).astype(int)
    log(f"building BM25 over {n_docs:,} docs across {n_workers} workers")
    procs = []
    shard_files = []
    for w in range(n_workers):
        if bounds[w] >= bounds[w + 1]:
            continue
        sf = wdir / f"shard_{w:02d}.npz"
        shard_files.append(sf)
        cmd = [sys.executable, "-m", "echospec.wiki18", "--bm25_shard", str(arrow), str(sf), str(bounds[w]), str(bounds[w + 1])]
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"BM25 shard workers failed: {codes}")

    df = np.zeros(BM25_VOCAB, dtype=np.int64)
    dl_parts = []
    for sf in shard_files:
        with np.load(sf) as z:
            df += z["counts"].astype(np.int64)
            dl_parts.append(z["dl"])
    dl_all = np.concatenate(dl_parts)
    avgdl = float(dl_all.mean()) if len(dl_all) else 1.0
    idf = np.where(df > 0, np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)), 0.0).astype(np.float32)
    indptr = np.zeros(BM25_VOCAB + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    nnz = int(indptr[-1])
    log(f"merging {nnz:,} BM25 postings (avgdl={avgdl:.1f})")

    docs_tmp = out_dir / "bm25_docs.tmp.npy"
    w_tmp = out_dir / "bm25_weights.tmp.npy"
    mm_docs = np.lib.format.open_memmap(docs_tmp, mode="w+", dtype=np.int32, shape=(nnz,))
    mm_w = np.lib.format.open_memmap(w_tmp, mode="w+", dtype=np.float16, shape=(nnz,))
    cursor = indptr[:-1].copy()
    for sf in shard_files:
        with np.load(sf) as z:
            counts_w = z["counts"].astype(np.int64)
            buckets_w = z["buckets"]
            docs_w = z["docs"]
            tfs_w = z["tfs"].astype(np.float32)
        if len(docs_w):
            nz = np.flatnonzero(counts_w)
            c_nz = counts_w[nz]
            local_starts = np.zeros(len(c_nz), dtype=np.int64)
            if len(c_nz) > 1:
                np.cumsum(c_nz[:-1], out=local_starts[1:])
            pos = np.repeat(cursor[nz], c_nz) + (np.arange(len(docs_w), dtype=np.int64) - np.repeat(local_starts, c_nz))
            mm_docs[pos] = docs_w
            norm = 1.0 - b + b * (dl_all[docs_w].astype(np.float32) / max(avgdl, 1e-6))
            mm_w[pos] = (idf[buckets_w] * (tfs_w * (k1 + 1.0)) / (tfs_w + k1 * norm)).astype(np.float16)
            cursor += counts_w
        sf.unlink(missing_ok=True)
    mm_docs.flush()
    mm_w.flush()
    del mm_docs, mm_w
    shutil.rmtree(wdir, ignore_errors=True)
    os.replace(docs_tmp, out_dir / "bm25_docs.npy")
    os.replace(w_tmp, out_dir / "bm25_weights.npy")
    np.save(out_dir / "bm25_indptr.npy", indptr)
    np.savez(out_dir / "bm25_meta.npz", idf=idf, avgdl=np.float32(avgdl), n_docs=np.int64(n_docs))
    (out_dir / "READY.json").write_text(
        json.dumps({"rows": n_docs, "nnz": nnz, "avgdl": round(avgdl, 2), "k1": k1, "b": b, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}),
        encoding="utf-8",
    )
    log(f"BM25 READY at {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output directory (shared cache)")
    ap.add_argument("--work", help="local scratch directory")
    ap.add_argument("--force_self_encode", action="store_true")
    ap.add_argument("--encoder_model", default="intfloat/e5-base-v2")
    ap.add_argument("--encode_shard", nargs=4, metavar=("ARROW", "NPY", "START", "END"))
    ap.add_argument("--build_dense", nargs=3, metavar=("ARROW", "OUT_NPY", "MODEL"))
    ap.add_argument("--bm25_shard", nargs=4, metavar=("ARROW", "OUT_NPZ", "START", "END"))
    ap.add_argument("--build_bm25", nargs=2, metavar=("ARROW", "OUT_DIR"))
    a = ap.parse_args()

    if a.encode_shard:
        arrow, npy, s, e = a.encode_shard
        encode_shard(Path(arrow), Path(npy), int(s), int(e), model_name=a.encoder_model)
        return
    if a.bm25_shard:
        arrow, npz, s, e = a.bm25_shard
        bm25_shard(Path(arrow), Path(npz), int(s), int(e))
        return
    if a.build_dense:
        arrow_p, out_npy_p, model_name = Path(a.build_dense[0]), Path(a.build_dense[1]), a.build_dense[2]
        if out_npy_p.exists() and out_npy_p.stat().st_size > 0:
            log(f"dense index already exists: {out_npy_p}")
            return
        from echospec.retrieval import ArrowCorpus, load_jsonl_corpus
        c = ArrowCorpus(arrow_p) if str(arrow_p).endswith(".arrow") else load_jsonl_corpus(arrow_p)
        n_rows = len(c)
        del c
        log(f"encoding {n_rows:,} docs with {model_name} -> {out_npy_p}")
        self_encode(arrow_p, out_npy_p, n_rows, model_name=model_name)
        log(f"dense index ready: {out_npy_p}")
        return
    if a.build_bm25:
        build_bm25(Path(a.build_bm25[0]), Path(a.build_bm25[1]), work_dir=Path(a.work) if a.work else None)
        return

    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (out / "READY.json").exists():
        log(f"already built: {(out / 'READY.json').read_text()}")
        return

    arrow_local = work / "wiki18_contents.arrow"
    npy_local = work / "wiki18_e5_fp16.npy"
    n = build_corpus(work, arrow_local) if not arrow_local.exists() else None
    if n is None:
        from echospec.retrieval import ArrowCorpus

        n = len(ArrowCorpus(arrow_local))
    source = None
    if not npy_local.exists():
        if not a.force_self_encode and convert_official_index(work, npy_local, n):
            source = "official_faiss_flat"
        else:
            log("falling back to self-encoding with e5-base-v2")
            self_encode(arrow_local, npy_local, n)
            source = "self_encoded_e5_base_v2"
    for f in (arrow_local, npy_local):
        log(f"copying {f.name} -> {out}")
        shutil.copy2(f, out / (f.name + ".part"))
        os.replace(out / (f.name + ".part"), out / f.name)
    (out / "READY.json").write_text(
        json.dumps({"rows": n, "dim": 768, "source": source or "existing_local", "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}),
        encoding="utf-8",
    )
    log("READY")


if __name__ == "__main__":
    main()
