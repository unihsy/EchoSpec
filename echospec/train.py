"""GRPO training over transformation worlds (launch with torchrun; rank 0 drives rollouts).

Data modes:
  --families identity                      clean training (identity worlds only)
  --families identity,dup,inject,...       spec training: per base item, the identity world + one sampled transform

Async rollout (default): a producer thread on rank 0 generates the next batch with the latest loaded policy while
the current batch trains, so rollout and gradient computation overlap. With one batch queued and one in generation,
a batch is at most two updates stale (steady state: 2); the lag is logged per step as `policy_lag`.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import queue
import random
import shutil
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from echospec.agent import RunConfig, run_batch
from echospec.qa_data import read_jsonl
from echospec.rewards import acquisition, consistency_bonus, group_advantages, judge, paired_advantages, reward, supported
from echospec.serve import load_lora, unload_lora
from echospec.transforms import make_world, sample_train_world


def family_table(worlds, trajs) -> dict:
    agg = defaultdict(lambda: Counter())
    for w, t in zip(worlds, trajs):
        j = judge(t.answer, w["golds"], w["unanswerable"], w.get("decoys") or [])
        a = agg[w["family"]]
        a["n"] += 1
        a["correct"] += j["correct"]
        a["abstain"] += j["abstain"]
        a["deferred"] += j["deferred"]
        a["searches"] += t.n_searches
    return {f: {k: round(v / c["n"], 4) if k != "n" else v for k, v in c.items()} for f, c in agg.items()}


class BatchMaker:
    """Builds one training batch: sample items/worlds, roll out with a given policy, compute advantages."""

    def __init__(self, a, recs, families, endpoints, start_step):
        self.a, self.recs, self.families, self.endpoints = a, recs, families, endpoints
        self.clean_only = families == ["identity"]
        self.rng = random.Random(f"{a.seed}:{start_step}")
        self.order: list[int] = []
        self.family_probs = None
        if a.family_probs:
            self.family_probs = {k: float(v) for k, v in (kv.split(":") for kv in a.family_probs.split(","))}
        self.m_weights = None
        if a.m_probs:
            self.m_weights = {int(k): float(v) for k, v in (kv.split(":") for kv in a.m_probs.split(","))}

    def make(self, served: str, policy_step: int) -> dict:
        a = self.a
        t0 = time.time()
        n_items = a.items_per_step * (2 if self.clean_only else 1)
        chosen = []
        while len(chosen) < n_items:
            if not self.order:
                self.order = list(range(len(self.recs)))
                self.rng.shuffle(self.order)
            chosen.append(self.recs[self.order.pop()])
        worlds = []
        for rec in chosen:
            worlds += [make_world(rec["item"], rec.get("gen"), rec.get("mask"), "identity")] * a.group_size
            if not self.clean_only:
                worlds += [sample_train_world(rec, self.rng, self.families, self.family_probs, self.m_weights)] * a.group_size
        cfg = RunConfig(served_model=served, tokenizer_name=a.base_model, prompt_style=a.prompt_style,
                        temperature=a.temperature, max_searches=a.max_searches, exclude_seen=bool(a.exclude_seen))
        trajs = asyncio.run(run_batch(worlds, cfg, self.endpoints, a.retriever_url))
        judges = [judge(t.answer, w["golds"], w["unanswerable"], w.get("decoys") or []) for w, t in zip(worlds, trajs)]
        grounded = [supported(t.retrieved_ids, w.get("evidence_ids") or []) if w.get("evidence_ids") else None
                    for w, t in zip(worlds, trajs)]
        rews = [reward(j, w["unanswerable"], a.lam, g, a.ground_alpha) for j, w, g in zip(judges, worlds, grounded)]
        ans = [(j, g) for j, g, w in zip(judges, grounded, worlds) if j["correct"] and not w["unanswerable"] and g is not None]
        grounded_share = float(np.mean([g for _, g in ans])) if ans else 0.0
        if a.consistency_beta > 0:
            rews = [r + b for r, b in zip(rews, consistency_bonus(worlds, [t.answer for t in trajs], a.consistency_beta))]
        acq = [acquisition(t.retrieved_ids, t.retrieved_docs, w.get("evidence_ids") or [], w.get("support_titles") or [])
               for w, t in zip(worlds, trajs)]
        if a.acq_beta > 0:
            rews = [r + a.acq_beta * q for r, q in zip(rews, acq)]
        advs = paired_advantages(worlds, rews, norm=a.adv_norm, pair_eta=a.pair_eta)
        return {
            "payload": [{"prompt_text": t.prompt_text, "segments": t.segments, "adv": float(advs[i])} for i, t in enumerate(trajs)],
            "worlds": worlds,
            "trajs": trajs,
            "rews": rews,
            "advs": advs,
            "fam": family_table(worlds, trajs),
            "finish": dict(Counter(t.finish.split(":")[0] for t in trajs)),
            "rollout_s": round(time.time() - t0, 1),
            "policy_step": policy_step,
            "grounded_share": round(grounded_share, 4),
            "acq_mean": round(float(np.mean(acq)), 4),
        }


class Producer(threading.Thread):
    """Background producer: always rolls out with the most recently loaded policy."""

    def __init__(self, maker: BatchMaker, n_batches: int, served: str, policy_step: int):
        super().__init__(daemon=True)
        self.maker, self.n = maker, n_batches
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.served, self.policy_step = served, policy_step
        self.in_use: str | None = None
        self.error: BaseException | None = None

    def set_policy(self, served: str, step: int) -> None:
        with self.lock:
            self.served, self.policy_step = served, step

    def run(self):
        try:
            for _ in range(self.n):
                with self.lock:
                    served, ps = self.served, self.policy_step
                    self.in_use = served
                b = self.maker.make(served, ps)
                with self.lock:
                    self.in_use = None
                self.q.put(b)
        except BaseException as e:  # noqa: BLE001
            self.error = e
            self.q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_jsonl", required=True, help="spec_train.jsonl records {item, gen, mask}")
    ap.add_argument("--dev_jsonl", required=True)
    ap.add_argument("--families", default="identity")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--consistency_beta", type=float, default=0.0)
    ap.add_argument("--items_per_step", type=int, default=12)
    ap.add_argument("--group_size", type=int, default=6)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_beta", type=float, default=0.01)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_searches", type=int, default=4)
    ap.add_argument("--prompt_style", default="neutral")
    ap.add_argument("--endpoints_json", required=True)
    ap.add_argument("--retriever_url", default="http://127.0.0.1:8290")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--snapshot_every", type=int, default=50)
    ap.add_argument("--async_rollout", type=int, default=1)
    ap.add_argument("--ground_alpha", type=float, default=1.0,
                    help="credit for a correct answer that retrieved no primary evidence (1.0 = plain outcome reward)")
    ap.add_argument("--family_probs", default="", help="e.g. dup:0.1,inject:0.35,consistent:0.25,rewrite:0.1,delete:0.2")
    ap.add_argument("--exclude_seen", type=int, default=0, help="1: within an episode, never re-show a retrieved document")
    ap.add_argument("--adv_norm", default="none", choices=["none", "std"],
                    help="none keeps reward units so the lam threshold holds; std is standard GRPO")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--acq_beta", type=float, default=0.0,
                    help="evidence-acquisition reward: + acq_beta * (1 if primary evidence retrieved + 0.5 if an earlier-hop passage retrieved)")
    ap.add_argument("--pair_eta", type=float, default=0.0,
                    help="cross-world control-variate weight on invariant worlds against the paired identity group")
    ap.add_argument("--m_probs", default="", help="echo-count weights for transformed worlds, e.g. 1:0.1,2:0.1,3:0.2,5:0.25,8:0.35")
    ap.add_argument("--sft_jsonl", default="", help="verification trajectories (echospec.synth) for the warm-start")
    ap.add_argument("--sft_min_searches", type=int, default=1, help="keep only warm-start trajectories with >= K searches")
    ap.add_argument("--sft_steps", type=int, default=0, help="SFT updates before RL (counted in the step index)")
    ap.add_argument("--sft_batch", type=int, default=64)
    ap.add_argument("--sft_lr", type=float, default=1e-4)
    a = ap.parse_args()
    import torch
    import torch.distributed as dist

    from echospec.trainer import LoraPGTrainer, dist_info

    rank, world, _ = dist_info()
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(hours=4))
    out = Path(a.out_dir)
    families = [f.strip() for f in a.families.split(",") if f.strip()]
    trainer = LoraPGTrainer(a.base_model, lr=a.lr, kl_beta=a.kl_beta, lora_rank=a.lora_rank)
    start = trainer.try_resume(out / "ckpts")
    if world > 1:
        dist.barrier()

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
        recs = read_jsonl(Path(a.train_jsonl))
        dev = read_jsonl(Path(a.dev_jsonl))
        endpoints = json.loads(Path(a.endpoints_json).read_text(encoding="utf-8"))["endpoints"]
        served = a.base_model
        loaded: list[str] = []
        if start > 0:
            served = f"{a.run_tag}_s{start:04d}"
            load_lora(endpoints, served, trainer.save_adapter(out / "live" / served))
            loaded.append(served)
        maker = BatchMaker(a, recs, families, endpoints, start)
        print(f"[train] {a.run_tag}: {len(recs)} records, families={families}, lam={a.lam}, async={a.async_rollout}, resume step={start}", flush=True)

        def dev_eval(step: int, model: str):
            cfg = RunConfig(served_model=model, tokenizer_name=a.base_model, prompt_style=a.prompt_style,
                            temperature=0.0, max_searches=a.max_searches, exclude_seen=bool(a.exclude_seen))
            trajs = asyncio.run(run_batch(dev, cfg, endpoints, a.retriever_url))
            tab = family_table(dev, trajs)
            with open(out / "dev_metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "families": tab}) + "\n")
            print(f"[dev step {step}] " + " ".join(f"{k}:{v['correct']:.3f}/abs{v['abstain']:.2f}" for k, v in sorted(tab.items())), flush=True)

        sft = read_jsonl(Path(a.sft_jsonl)) if (a.sft_jsonl and a.sft_steps) else []
        if sft and "qwen2.5" not in a.base_model.lower():
            from echospec.protocol import build_prompt_text

            q_by_id = {rec["item"]["id"]: rec["item"]["question"] for rec in recs}
            for r in sft:
                if r.get("base_id") in q_by_id:
                    r["prompt_text"] = build_prompt_text(q_by_id[r["base_id"]], a.prompt_style, trainer.tok)
        if sft and a.sft_min_searches > 1:
            sft = [r for r in sft if int(r.get("n_searches", 1)) >= a.sft_min_searches]
        if a.sft_steps and not sft:
            raise RuntimeError(f"--sft_steps {a.sft_steps} but no SFT data at {a.sft_jsonl!r}")
        sft_rng = random.Random(f"sft:{a.seed}")
        sft_order = [i for _ in range(1 + a.sft_steps * a.sft_batch // max(1, len(sft))) for i in sft_rng.sample(range(len(sft)), len(sft))] if sft else []
        if sft:
            print(f"[train] SFT warm-start: {len(sft)} trajectories, {a.sft_steps} steps x {a.sft_batch}", flush=True)
        if start == 0:
            dev_eval(0, served)
        producer = None

    total = a.sft_steps + a.steps
    for _ in range(start, total):
        in_sft = trainer.step < a.sft_steps
        trainer.base_lr = a.sft_lr if in_sft else a.lr
        trainer.kl_beta = 0.0 if in_sft else a.kl_beta
        payload = [None]
        if rank == 0:
            t0 = time.time()
            if in_sft:
                k = trainer.step * a.sft_batch
                rows = [sft[i] for i in sft_order[k: k + a.sft_batch]]
                batch = {"payload": [{"prompt_text": r["prompt_text"], "segments": [tuple(s) for s in r["segments"]], "adv": 1.0}
                                     for r in rows],
                         "rows": rows}
            elif a.async_rollout:
                if producer is None:
                    producer = Producer(maker, total - trainer.step, served, trainer.step)
                    producer.start()
                batch = producer.q.get()
                if batch is None:
                    raise RuntimeError(f"rollout producer failed: {producer.error!r}")
            else:
                batch = maker.make(served, trainer.step)
            t_wait = round(time.time() - t0, 1)
            payload = [batch["payload"]]
        if world > 1:
            dist.broadcast_object_list(payload, src=0)
        stats = trainer.train_step(payload[0])
        if world > 1:
            dist.barrier()
        if rank == 0 and in_sft:
            sft_done = trainer.step == a.sft_steps
            fam = Counter(r["family"] for r in batch["rows"])
            rec_line = {"step": trainer.step, "phase": "sft", "reward_mean": 0.0, "families": {}, "finish": {},
                        "sft_families": dict(fam), "step_s": round(time.time() - t0, 1), **stats}
            with open(out / "train_metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec_line) + "\n")
            print(f"[{a.run_tag} sft {trainer.step}/{a.sft_steps}] nll={stats['pg_loss']:.4f} gn={stats['grad_norm']:.3f} "
                  f"tok={stats['policy_tokens']} ({rec_line['step_s']}s)", flush=True)
            if trainer.step % 25 == 0 or sft_done:
                trainer.save_ckpt(out / "ckpts", vars(a))
            if sft_done or trainer.step % a.eval_every == 0:
                name = f"{a.run_tag}_s{trainer.step:04d}"
                load_lora(endpoints, name, trainer.save_adapter(out / "live" / name))
                for old in [n for n in loaded if n != name]:
                    unload_lora(endpoints, old)
                    shutil.rmtree(out / "live" / old, ignore_errors=True)
                    loaded.remove(old)
                loaded.append(name)
                served = name
                dev_eval(trainer.step, name)
            if sft_done:
                trainer.save_adapter(out / "sft_adapter")
                maker.rng = random.Random(f"{a.seed}:{trainer.step}")
        if world > 1:
            dist.barrier()
        if in_sft:
            continue
        if rank == 0:
            name = f"{a.run_tag}_s{trainer.step:04d}"
            load_lora(endpoints, name, trainer.save_adapter(out / "live" / name))
            loaded.append(name)
            served = name
            if producer is not None:
                producer.set_policy(name, trainer.step)
                with producer.lock:
                    busy = producer.in_use
            else:
                busy = None
            for old in [n for n in loaded if n not in (name, busy)]:
                unload_lora(endpoints, old)
                shutil.rmtree(out / "live" / old, ignore_errors=True)
                loaded.remove(old)
            rews, advs, fam = batch["rews"], batch["advs"], batch["fam"]
            rec_line = {
                "step": trainer.step,
                "policy_lag": trainer.step - 1 - batch["policy_step"],
                "grounded_share": batch["grounded_share"],
                "acq_mean": batch["acq_mean"],
                "reward_mean": round(float(np.mean(rews)), 4),
                "frac_zero_adv": round(float(np.mean(np.abs(advs) < 1e-8)), 4),
                "families": fam,
                "finish": batch["finish"],
                "rollout_s": batch["rollout_s"],
                "wait_s": t_wait,
                "step_s": round(time.time() - t0, 1),
                **stats,
            }
            with open(out / "train_metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec_line) + "\n")
            idn = fam.get("identity", {})
            print(f"[{a.run_tag} {trainer.step}/{total}] R={rec_line['reward_mean']:.3f} id_acc={idn.get('correct', 0):.3f} "
                  f"id_abs={idn.get('abstain', 0):.3f} srch={idn.get('searches', 0):.2f} zero_adv={rec_line['frac_zero_adv']:.2f} "
                  f"kl={stats['kl']:.4f} lag={rec_line['policy_lag']} (wait {t_wait}s / step {rec_line['step_s']}s)", flush=True)
            if trainer.step % 25 == 0 or trainer.step == total:
                trainer.save_ckpt(out / "ckpts", vars(a))
                with open(out / "samples.jsonl", "a", encoding="utf-8") as f:
                    for t, w in list(zip(batch["trajs"], batch["worlds"]))[:: max(1, len(batch["trajs"]) // 3)][:3]:
                        f.write(json.dumps({"step": trainer.step, "family": w["family"], "golds": w["golds"],
                                            "text": t.full_text()[-6000:], "answer": t.answer}) + "\n")
            if trainer.step % a.snapshot_every == 0:
                trainer.save_adapter(out / "snapshots" / f"step_{trainer.step:04d}")
            if trainer.step % a.eval_every == 0 or trainer.step == total:
                dev_eval(trainer.step, name)
        if world > 1:
            dist.barrier()

    if rank == 0:
        trainer.save_adapter(out / "final_adapter")
        (out / "DONE").write_text(str(trainer.step), encoding="utf-8")
        print(f"[train] {a.run_tag} finished at step {trainer.step}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
