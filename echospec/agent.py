"""Async rollout runner for the Search-R1 continuation protocol over a vLLM OpenAI-compatible server.

An instance is a dict with at least:
  id, question, golds (list[str]), unanswerable (bool), overlay_docs (list[{id, contents}]), masked_doc_ids (list[str])
"""
from __future__ import annotations

import asyncio
import itertools
import random
import re
from dataclasses import dataclass
from typing import Sequence

from echospec.protocol import AGENT_PROTOCOLS, Trajectory
from echospec.retrieval import RetrieverClient
from echospec.text import dedup_docs


@dataclass
class RunConfig:
    served_model: str
    tokenizer_name: str
    prompt_style: str = "neutral"
    temperature: float = 0.0
    top_p: float = 1.0
    max_searches: int = 4
    min_searches: int = 0
    top_k: int = 3
    max_tokens_per_call: int = 500
    max_total_tokens: int = 7000
    dedup: bool = False
    dedup_threshold: float = 0.35
    exclude_seen: bool = False  # within an episode, never return a document id the agent has already been shown
    corpus: str = "wiki18"
    protocol: str = "searchr1"  # prompt / tag format of a released agent (echospec.protocol.AGENT_PROTOCOLS)
    retrieval_filter: str | None = None  # "trustrag": server-side TrustRAG clean-retrieval stage


_TOKENIZERS: dict[str, object] = {}


def get_tokenizer(name: str):
    if name not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return _TOKENIZERS[name]


class EndpointPool:
    """Round-robin over vLLM endpoints with retries on other endpoints."""

    def __init__(self, endpoints: Sequence[str], timeout: float = 600.0):
        from openai import AsyncOpenAI

        self.clients = [AsyncOpenAI(base_url=u, api_key="local", timeout=timeout, max_retries=0) for u in endpoints]
        self._rr = itertools.cycle(range(len(self.clients)))

    async def complete(self, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float, stop: list[str],
                       keep_special: bool = False):
        start = next(self._rr)
        last = None
        for attempt in range(6):
            c = self.clients[(start + attempt) % len(self.clients)]
            try:
                r = await c.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    extra_body={"include_stop_str_in_output": True, "skip_special_tokens": not keep_special},
                )
                ch = r.choices[0]
                usage = r.usage
                return ch.text or "", ch.finish_reason, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(min(10.0, 0.5 * 2**attempt))
        raise RuntimeError(f"vLLM completion failed: {last}")

    async def aclose(self):
        for c in self.clients:
            await c.close()


async def run_episode(inst: dict, cfg: RunConfig, llm: EndpointPool, retriever: RetrieverClient) -> Trajectory:
    tok = get_tokenizer(cfg.tokenizer_name)
    proto = AGENT_PROTOCOLS[cfg.protocol]
    prompt = proto.prompt(inst["question"], tok, cfg.prompt_style)
    traj = Trajectory(instance_id=inst["id"], prompt_text=prompt)
    # A privileged hint steers generation only; the stored trajectory keeps the plain prompt so that
    # SFT on it trains the hint-free policy.
    text = proto.prompt(inst["question"], tok, cfg.prompt_style, hint=inst["hint"]) if inst.get("hint") else prompt
    try:
        kw = {"keep_special": True} if proto.keep_special and isinstance(llm, EndpointPool) else {}
        nudges = 0
        max_nudges = 3 if cfg.min_searches > 0 else 0
        for _ in range(cfg.max_searches + max_nudges + 1):
            chunk, finish, p_tok, c_tok = await llm.complete(
                cfg.served_model, text, cfg.max_tokens_per_call, cfg.temperature, cfg.top_p, list(proto.stops), **kw
            )
            if proto.keep_special and chunk:
                for eos in ("<|im_end|>", "<|endoftext|>"):
                    if chunk.endswith(eos):
                        chunk = chunk[: -len(eos)]
            traj.prompt_tokens += p_tok
            traj.gen_tokens += c_tok
            ans = proto.answer(chunk)
            early_ans = chunk.rstrip().endswith("</answer>") or (
                ans is not None and proto.search_end not in chunk[-(len(proto.search_end) + 3):]
            )
            if early_ans:
                if traj.n_searches < cfg.min_searches and nudges < max_nudges and traj.n_searches < cfg.max_searches:
                    nudges += 1
                    stripped = re.sub(r"<answer>.*?</answer>", "", chunk, flags=re.DOTALL).rstrip()
                    traj.segments.append((stripped, True))
                    text += stripped
                    notice = (
                        '"Verification Check"\n'
                        f"You have issued {traj.n_searches} of {cfg.min_searches} required verification searches. "
                        "Do not rely on unverified claims. Issue a new search query targeting an underlying bridge entity "
                        "or primary Wikipedia article to verify the answer before answering."
                    )
                    info = proto.information([{"contents": notice}], traj.n_searches + 1)
                    traj.segments.append((info, False))
                    text += info
                    if p_tok + c_tok + len(info) // 4 + cfg.max_tokens_per_call > cfg.max_total_tokens:
                        traj.answer = ans
                        traj.finish = "answer"
                        return traj
                    continue
                traj.segments.append((chunk, True))
                text += chunk
                traj.answer = ans
                traj.finish = "answer"
                return traj
            traj.segments.append((chunk, True))
            text += chunk
            query = proto.query(chunk) if chunk.rstrip().endswith(proto.search_end) else None
            if query is None:
                traj.finish = "length" if finish == "length" else "no_action"
                return traj
            if traj.n_searches >= cfg.max_searches:
                traj.finish = "max_turns"
                return traj
            k = cfg.top_k * 3 if cfg.dedup else cfg.top_k
            extra = {"filter": cfg.retrieval_filter} if cfg.retrieval_filter else {}
            target_corpus = cfg.corpus if cfg.corpus != "wiki18" else inst.get("corpus", "wiki18")
            docs = await retriever.search(
                query,
                top_k=k,
                corpus=target_corpus,
                overlay_docs=inst.get("overlay_docs") or (),
                masked_doc_ids=list(inst.get("masked_doc_ids") or ())
                + ([d for ids in traj.retrieved_ids for d in ids] if cfg.exclude_seen else []),
                **extra,
            )
            if cfg.dedup:
                docs = dedup_docs(docs, threshold=cfg.dedup_threshold)
            docs = docs[: cfg.top_k]
            traj.queries.append(query)
            traj.retrieved_ids.append([d["id"] for d in docs])
            traj.retrieved_docs.extend(d["contents"] for d in docs)
            info = proto.information(docs, traj.n_searches)
            traj.segments.append((info, False))
            text += info
            # Context after this turn ~= last prompt + last completion + information block (~4 chars/token)
            if p_tok + c_tok + len(info) // 4 + cfg.max_tokens_per_call > cfg.max_total_tokens:
                traj.finish = "length"
                return traj
        traj.finish = "max_turns"
    except Exception as e:  # noqa: BLE001
        traj.finish = f"error:{type(e).__name__}"
    return traj


async def run_batch(
    instances: Sequence[dict],
    cfg: RunConfig,
    endpoints: Sequence[str],
    retriever_url: str,
    concurrency: int = 256,
    progress_every: int = 0,
) -> list[Trajectory]:
    llm = EndpointPool(endpoints)
    retr = RetrieverClient(retriever_url)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(inst):
        nonlocal done
        async with sem:
            t = await run_episode(inst, cfg, llm, retr)
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"[agent] {done}/{len(instances)} episodes", flush=True)
        return t

    try:
        return list(await asyncio.gather(*[one(i) for i in instances]))
    finally:
        await llm.aclose()
        await retr.aclose()


def shuffle_for_load(instances: Sequence[dict], seed: int = 0) -> list[dict]:
    xs = list(instances)
    random.Random(seed).shuffle(xs)
    return xs
