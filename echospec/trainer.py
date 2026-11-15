"""Data-parallel LoRA policy-gradient (GRPO, one update per rollout batch) trainer.

Every rank holds a full LoRA model. Rank 0 produces the batch (rollouts + advantages) and broadcasts it; each
rank computes gradients on its shard; LoRA gradients are all-reduced (SUM) so every rank takes the same step.
Only policy-generated tokens carry loss; prompt and <information> tokens are masked.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist


def dist_info() -> tuple[int, int, int]:
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))


class LoraPGTrainer:
    def __init__(self, base_model: str, lr: float, kl_beta: float, lora_rank: int = 64, max_seq_len: int = 6144,
                 grad_ckpt: bool = False, warmup_steps: int = 10):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.rank, self.world, self.local = dist_info()
        self.device = torch.device(f"cuda:{self.local}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        self.kl_beta = kl_beta
        self.max_seq_len = max_seq_len
        self.base_lr = lr
        self.warmup = warmup_steps
        self.step = 0
        self.tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to(self.device)
        if grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(model, cfg)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))

    # ------------------------------------------------------------------------------------------
    def encode(self, prompt_text: str, segments: Sequence[tuple[str, bool]]) -> dict | None:
        ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        mask = [0.0] * len(ids)
        for text, is_pol in segments:
            if not text:
                continue
            t = self.tok(text, add_special_tokens=False).input_ids
            ids += t
            mask += [1.0 if is_pol else 0.0] * len(t)
        ids, mask = ids[: self.max_seq_len], mask[: self.max_seq_len]
        if sum(mask[1:]) < 1:
            return None
        return {"ids": ids, "mask": mask}

    def _logp(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(ids.unsqueeze(0)).logits[0, :-1].float()
        return torch.log_softmax(logits, -1).gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)

    def train_step(self, batch: Sequence[dict]) -> dict:
        """batch items: {"prompt_text", "segments", "adv"}; identical list on every rank."""
        enc = []
        for b in batch:
            if abs(b["adv"]) < 1e-8:
                continue
            e = self.encode(b["prompt_text"], b["segments"])
            if e is not None:
                e["adv"] = float(b["adv"])
                enc.append(e)
        n_total = len(enc)
        # Balance shards by length (greedy), deterministic on every rank.
        order = sorted(range(n_total), key=lambda i: -len(enc[i]["ids"]))
        loads = [0] * self.world
        mine = []
        for i in order:
            r = min(range(self.world), key=lambda k: loads[k])
            loads[r] += len(enc[i]["ids"])
            if r == self.rank:
                mine.append(enc[i])

        lr = self.base_lr * min(1.0, (self.step + 1) / max(1, self.warmup))
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        self.opt.zero_grad(set_to_none=False)
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        pg_sum = kl_sum = 0.0
        tok_sum = 0
        for e in mine:
            ids = torch.tensor(e["ids"], dtype=torch.long, device=self.device)
            m = torch.tensor(e["mask"][1:], dtype=torch.float32, device=self.device)
            n = m.sum().clamp(min=1.0)
            logp = self._logp(ids)
            with torch.no_grad(), self.model.disable_adapter():
                ref = self._logp(ids)
            d = (ref - logp).clamp(-10, 10)
            kl = torch.exp(d) - d - 1.0
            pg = -(e["adv"] * logp * m).sum() / n
            klm = (kl * m).sum() / n
            ((pg + self.kl_beta * klm) / max(1, n_total)).backward()
            pg_sum += float(pg.detach())
            kl_sum += float(klm.detach())
            tok_sum += int(n.item())
        if self.world > 1:
            for p in self.params:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            stats_t = torch.tensor([pg_sum, kl_sum, float(tok_sum), float(len(mine))], device=self.device)
            dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
            pg_sum, kl_sum, tok_sum, _ = stats_t.tolist()
        gn = float(torch.nn.utils.clip_grad_norm_(self.params, 1.0)) if n_total else 0.0
        if n_total:
            self.opt.step()
        self.model.eval()
        self.step += 1
        return {
            "lr": lr,
            "pg_loss": round(pg_sum / max(1, n_total), 5),
            "kl": round(kl_sum / max(1, n_total), 5),
            "grad_norm": round(gn, 4),
            "n_trained": n_total,
            "policy_tokens": int(tok_sum),
        }

    # ------------------------------------------------------------------------------------------
    def save_adapter(self, path: str | Path) -> str:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(p))
        self.tok.save_pretrained(str(p))
        return str(p)

    def save_ckpt(self, root: str | Path, meta: dict) -> None:
        root = Path(root)
        d = root / f"step_{self.step:04d}"
        tmp = root / f".tmp_step_{self.step:04d}"
        shutil.rmtree(tmp, ignore_errors=True)
        self.save_adapter(tmp / "adapter")
        torch.save({"opt": self.opt.state_dict(), "step": self.step, "meta": meta}, tmp / "state.pt")
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)
        (d / "DONE").write_text(json.dumps({"step": self.step}), encoding="utf-8")
        for old in sorted(p for p in root.glob("step_*") if (p / "DONE").exists())[:-2]:
            shutil.rmtree(old, ignore_errors=True)

    def try_resume(self, root: str | Path) -> int:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        root = Path(root)
        for d in sorted(root.glob("step_*"), reverse=True) if root.exists() else []:
            # a checkpoint restored from blob storage can carry DONE without its weights (copy cut off by preemption)
            if not (d / "DONE").exists() or not (d / "adapter" / "adapter_model.safetensors").exists() or not (d / "state.pt").exists():
                continue
            try:
                state = load_file(str(d / "adapter" / "adapter_model.safetensors"), device=str(self.device))
                st = torch.load(d / "state.pt", map_location=self.device)
            except Exception as e:  # noqa: BLE001 - truncated files: fall back to an older checkpoint
                print(f"[trainer] skipping unreadable checkpoint {d}: {e}", flush=True)
                continue
            set_peft_model_state_dict(self.model, state)
            self.opt.load_state_dict(st["opt"])
            self.step = int(st["step"])
            return self.step
        return 0
