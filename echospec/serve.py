"""Launch vLLM OpenAI servers on chosen GPUs (ROCm-safe) and hot-load LoRA adapters at runtime."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Sequence

import requests


def port_up(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
        return True
    except Exception:
        return False


def launch(
    gpu_groups: Sequence[Sequence[int]],
    model: str,
    log_dir: str | Path,
    first_port: int,
    mem: float = 0.85,
    max_len: int = 8192,
    lora: bool = False,
    max_lora_rank: int = 64,
    max_num_seqs: int = 256,
    extra: Sequence[str] = (),
    timeout_s: int = 2400,
) -> tuple[list[str], list[subprocess.Popen]]:
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    procs, ports = [], []
    for i, grp in enumerate(gpu_groups):
        port = first_port + i
        vis = ",".join(map(str, grp))
        env = dict(os.environ, HIP_VISIBLE_DEVICES=vis, CUDA_VISIBLE_DEVICES=vis, VLLM_ALLOW_RUNTIME_LORA_UPDATING="True")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", model,
            "--host", "127.0.0.1", "--port", str(port),
            "--gpu-memory-utilization", str(mem), "--max-model-len", str(max_len),
            "--max-num-seqs", str(max_num_seqs), "--enable-prefix-caching", "--trust-remote-code",
        ]
        if len(grp) > 1:
            cmd += ["--tensor-parallel-size", str(len(grp))]
        if lora:
            cmd += ["--enable-lora", "--max-lora-rank", str(max_lora_rank), "--max-loras", "4"]
        cmd += list(extra)
        f = open(log_root / f"vllm_{port}.log", "w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True))
        ports.append(port)
        print(f"[serve] {model} on GPU {vis} -> :{port}", flush=True)

    deadline = time.time() + timeout_s
    ready: set[int] = set()
    while time.time() < deadline and len(ready) < len(ports):
        for p, pr in zip(ports, procs):
            if p in ready:
                continue
            if pr.poll() is not None:
                tail = (log_root / f"vllm_{p}.log").read_text(encoding="utf-8", errors="replace")[-3000:]
                shutdown(procs)
                raise RuntimeError(f"vLLM :{p} exited (code {pr.returncode}):\n{tail}")
            if port_up(p):
                ready.add(p)
                print(f"[serve] :{p} ready ({len(ready)}/{len(ports)})", flush=True)
        time.sleep(5)
    if len(ready) < len(ports):
        shutdown(procs)
        raise TimeoutError(f"vLLM ports not ready: {set(ports) - ready}")
    return [f"http://127.0.0.1:{p}/v1" for p in ports], procs


def shutdown(procs: Sequence[subprocess.Popen]) -> None:
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=30)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass


def unload_lora(endpoints: Sequence[str], name: str) -> None:
    for ep in endpoints:
        root = ep.rstrip("/").removesuffix("/v1")
        try:
            requests.post(f"{root}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
        except Exception:
            pass


def load_lora(endpoints: Sequence[str], name: str, path: str, unload: str | None = None,
              timeout: float = 900.0, retries: int = 3) -> None:
    for ep in endpoints:
        root = ep.rstrip("/").removesuffix("/v1")
        if unload:
            try:
                requests.post(f"{root}/v1/unload_lora_adapter", json={"lora_name": unload}, timeout=120)
            except Exception:
                pass
        last = None
        for attempt in range(retries):
            try:
                r = requests.post(f"{root}/v1/load_lora_adapter",
                                  json={"lora_name": name, "lora_path": os.path.abspath(path)}, timeout=timeout)
                if r.ok or "already" in r.text:
                    last = None
                    break
                last = RuntimeError(f"{r.status_code} {r.text[:300]}")
            except Exception as e:  # noqa: BLE001
                last = e
            time.sleep(10 * (attempt + 1))
        if last is not None:
            raise RuntimeError(f"load_lora {name} failed at {root}: {last}")


def main():
    ap = argparse.ArgumentParser(description="Start vLLM servers and keep them alive; writes endpoints JSON.")
    ap.add_argument("--groups", required=True, help="GPU groups, e.g. '1;2;3' (3 x TP1) or '1,2;3,4' (2 x TP2)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--first_port", type=int, default=8310)
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--lora", action="store_true")
    a = ap.parse_args()
    groups = [[int(x) for x in g.split(",")] for g in a.groups.split(";") if g.strip()]
    eps, procs = launch(groups, a.model, a.log_dir, a.first_port, a.mem, a.max_len, a.lora, max_num_seqs=a.max_num_seqs)
    tmp = Path(a.out_json + ".tmp")
    tmp.write_text(json.dumps({"model": a.model, "endpoints": eps}, indent=2), encoding="utf-8")
    os.replace(tmp, a.out_json)
    print(f"[serve] endpoints -> {a.out_json}", flush=True)

    def _stop(*_):
        shutdown(procs)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while True:
        time.sleep(30)
        dead = [p for p in procs if p.poll() is not None]
        if dead:
            print(f"[serve] {len(dead)} vLLM process(es) died; exiting", flush=True)
            shutdown(procs)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
