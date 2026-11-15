<div align="center">

# EchoSpec: Training Search Agents to Weigh Evidence Above Echoes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch + vLLM](https://img.shields.io/badge/Runtime-PyTorch%20%2B%20vLLM-EE4C2C.svg)](https://github.com/vllm-project/vllm)
[![Retrievers](https://img.shields.io/badge/Retrievers-e5%20%C2%B7%20BGE%20%C2%B7%20BM25-0F766E.svg)](#zero-shot-cross-retriever-transfer)
[![Backbones](https://img.shields.io/badge/Backbones-Qwen2.5%20%C2%B7%20Llama--3.1-4F46E5.svg)](#main-results)

**Shihao Yang, Can Zeng, Xinzhe Li**  
Tsinghua University  
{yangshihao26, zengc26, li-xz26}@mails.tsinghua.edu.cn

[**Overview**](#overview) · [**Why Agents Follow Echoes**](#why-search-agents-follow-echoes) · [**Method**](#the-echospec-framework) · [**Main Results**](#main-results) · [**Installation**](#installation) · [**End-to-End Reproduction**](#end-to-end-reproduction) · [**Project Structure**](#project-structure) · [**Citation**](#citation)

</div>

---

<p align="center">
  <img src="assets/teaser.png" width="98%" alt="EchoSpec Overview and Headline Findings"/>
</p>

> **TL;DR:** Reinforcement-learned search agents often mistake repeated unsupported claims for independent evidence, stopping their search before reaching the primary source. **EchoSpec** addresses this failure by training agents to recover and verify evidence through multi-hop search, rather than merely increasing retrieval depth or filtering duplicate passages. Across Qwen2.5 and Llama-3.1 backbones, EchoSpec substantially reduces susceptibility to false-claim echoes, improves primary-evidence recovery, and transfers zero-shot to unseen dense and lexical retrievers.

- **28.9% Def@8** on Qwen2.5-7B (3 seeds) and **30.6%** on Llama-3.1-8B, cutting eight-echo deference by **30–44 percentage points** relative to seven released RL search agents (58.6%–72.9%).
- **25.8%–26.5% Reach@8** under eight competing false-claim echoes (nearly $4\times$ the 7.1% base agent) alongside **47.1%–55.4%** macro answer precision across seven open-domain QA benchmarks.
- **24.0%–30.5% Def@8** under **zero-shot cross-retriever transfer** from e5-base-v2 to unseen dense (BAAI/bge-base-en-v1.5) and lexical (Okapi BM25) indices without retraining.

---

## Overview

Large language model search agents interleave chain-of-thought reasoning with multi-turn retrieval, yet open corpora routinely duplicate bare assertions across syndicated articles, mirrors, and summaries while leaving primary source articles intact.

**EchoSpec** provides both a controlled **behavioral specification suite** over the 21M-passage Wikipedia dump (wiki-18) and a **two-stage post-training recipe** that teaches search agents *what* to query rather than merely *how often* to search:

- **Five Corpus Transformations Plus an Unmodified Identity World:** Each multi-hop question instantiates an unmodified **Identity** world alongside five symmetric transformations (**Dup($m$)**, **Inject($m$)**, **Consistent($m$)**, **Rewrite**, and **Delete**; 10 worlds per test question across $m \in \{1, 3, 8\}$), separating invariance under claim repetition from sensitivity to genuine primary-evidence updates. Swapping only the answer string between true and false claims limits a held-out TF-IDF shortcut classifier to **0.60–0.68 AUC** across both test suites.
- **Verification-Oriented Query Reformulation:** Rather than stopping at top-ranked claim echoes, **EchoSpec** pivots to bridge entities, verifies primary passages (25.8%–26.5% primary-evidence reach at $m=8$ echoes), and abstains selectively when primary support is absent (37.5%–52.5% on Delete).
- **Cross-Scale, Cross-Family & Cross-Retriever Replication:** Using identical hyperparameters without per-model tuning, **EchoSpec** brings Def@8 into a tight **28.9%–30.6%** band across Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, and Llama-3.1-8B-Instruct, and transfers zero-shot from e5-base-v2 to BGE-base-en-v1.5 (29.4% / 30.5% Def@8) and lexical Okapi BM25 (**25.1% / 24.0%** Def@8).

---

## Why Search Agents Follow Echoes

Auditing seven open-source RL search checkpoints—Search-R1, ZeroSearch, AutoRefine (Instruct), AutoRefine (Base), ReSearch, R1-Searcher, and AWA-RL—alongside five inference-time RAG defenses reveals three structural findings:

1. **Paraphrased echoes displace primary evidence from top-$k$ slots:** Because bare claims align directly with question surface forms while primary multi-hop facts reside under bridge-entity articles, eight paraphrased copies overturn **58.6%–72.9%** of clean-correct answers across all seven released RL agents.
2. **Exact-match RL erases selective abstention:** Standard binary outcome rewards (0 for wrong *and* 0 for abstention) make guessing dominant, collapsing abstention on deleted-evidence questions from **49.2%** on the Qwen2.5-7B base agent down to **0.0%–0.2%** across released RL agents (and **1.4%** under same-data exact-match RL).
3. **More retrieval does not guarantee reliable verification:** Doubling the search budget ($B=4 \to 8$) leaves Search-R1 at **66.2%** Def@8, and even forcing at least three search turns raises Search-R1's primary-evidence reach from **13.3% to 29.5%** while leaving Def@8 at **55.5%** (58.7% with an explicit verification prompt). Agents must learn to reformulate queries toward bridge entities and weigh primary evidence above echoes—and when paired with inference-time filtering, **EchoSpec** compounds orthogonally to **12.8%** Def@8 (with TrustRAG) and **36.2%** Reach@8 (with Semantic Dedup).

---

## The EchoSpec Framework

<p align="center">
  <img src="assets/method.png" width="98%" alt="EchoSpec Two-Stage Training Pipeline"/>
</p>

### 1. Five Corpus Transformations & Identity World
Given a multi-hop question $q$, intermediate bridge articles $h_1, \dots, h_{K-1}$, and final primary passage $e \in \mathcal{C}$ supporting gold answer $y$, we pair an unmodified **Identity** world ($\mathcal{C}$ unchanged, $y_T = y$) with five corpus transformations:
- **Answer-Invariant Transformations ($y_T = y$):**
  - **Dup($m$):** inserts $m$ paraphrased copies of primary passage $e$.
  - **Inject($m$):** inserts $m$ diverse bare-claim passages asserting plausible decoy $\tilde{y} \neq y$ *without* fabricating bridge derivations, while keeping $e$ intact in $\mathcal{C}$.
  - **Consistent($m$):** uses the exact same $m$ generated claim passages with only $\tilde{y} \leftrightarrow y$ swapped across title and body, preventing copy-count or stylistic shortcuts.
- **Answer-Shifting Transformations:**
  - **Rewrite ($y_T = \tilde{y}$):** iteratively masks passages surfacing $(e, y)$ in the retriever and inserts a minimally edited primary passage supporting $\tilde{y}$.
  - **Delete ($y_T = \text{ABSTAIN}$):** applies the retrieval-verified mask without replacement, rendering $q$ unanswerable from the corpus.

### 2. Stage 1 — Multi-Hop Verification Warm-Start
Outcome-only RL faces a cold-start barrier under heavy echoes ($m = 8$): initial rollouts rarely uncover $e$, yielding near-zero group reward variance. Stage 1 synthesizes recovery trajectories using temporary article-chain hints ($h_1 \to \dots \to e$, stripped before supervised fine-tuning) and enforces **three structural quality gates**:
1. **Grounded Outcome:** $\hat{y} = y_T$ must retrieve $e \in D$ on answerable worlds; Delete trajectories must execute $\ge 3$ searches before abstaining.
2. **Hop Continuity:** Every hinted article title $h_k, e$ absent from $q$ must appear in an earlier retrieved passage block $D_{<t}$ before being queried at turn $t$.
3. **Self-Contained Reasoning:** Prompts and reasoning traces contain zero references to the temporary synthesis hint.

### 3. Stage 2 — $\lambda$-Calibrated Group RL
Starting from the warm-start LoRA adapter $\pi_{\text{SFT}}$, Stage 2 optimizes asynchronous GRPO across paired (Identity, Transformed) rollouts with three coupled components:
- **Evidence-Gated $\lambda$-Reward:** Correct answers supported by primary evidence ($e \in D$) receive $+1$, correct but unverified answers ($e \notin D$) receive a discounted reward $\alpha = 0.25$, wrong answers incur $-\lambda$ ($\lambda = 0.1$), and abstention receives $+1$ on Delete worlds or $0$ on answerable worlds. Once primary evidence is retrieved ($e \in D$), answering beats abstaining in expectation iff $p > \frac{\lambda}{1+\lambda}$; when primary evidence is absent ($e \notin D$), discounting unverified matches raises the required confidence threshold to $p > \frac{\lambda}{\alpha+\lambda}$, discouraging echo guessing.
- **Decision-Threshold-Preserving Acquisition Bonus:** $\beta_{\text{acq}}\,\phi(D)$ ($\beta_{\text{acq}} = 0.3$) credits retrieving $e$ ($+1.0$) and intermediate bridge articles ($+0.5$) *independently of the final output $\hat{y}$*, canceling from $R_{\text{total}}(\text{answer}, D) - R_{\text{total}}(\text{ABSTAIN}, D)$ for every $D$ so the optimal stopping threshold is preserved exactly.
- **Scale-Preserving Cross-World Advantages:** Mean-centered advantages omit standard GRPO's $1/\sigma_g$ division—which would otherwise cancel $\lambda$ whenever a rollout group contains only abstentions and wrong answers—and blend transformed and identity baselines on answer-invariant worlds.

---

## Main Results

### Transformation-World Robustness & Open-Domain QA Precision
Pooled across **Test Set A (Qwen3-32B, 507 questions $\times$ 10 worlds = 5,070 episodes)** and **held-out Test Set B (Phi-4, unseen generator family, 607 questions $\times$ 10 worlds = 6,070 episodes; 11,140 transformation episodes total)**, plus **7 open-domain QA benchmarks** (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle):

| Method | Backbone | Def@1&nbsp;↓ | Def@8&nbsp;↓ | Reach@8&nbsp;↑ | Acc@8&nbsp;↑ | Abst.&nbsp;↑ | Precision&nbsp;↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Search-R1** *(released RL)* | Qwen2.5-7B | 46.0 | 69.1 | 13.3 | 6.8 | 0.2 | 38.4 |
| **AutoRefine (Base)** *(released RL)* | Qwen2.5-7B | 40.5 | 59.8 | 18.1 | 12.5 | 0.0 | 44.7 |
| **R1-Searcher** *(released RL)* | Qwen2.5-7B | 33.3 | 61.9 | 21.0 | 12.5 | 0.0 | 41.8 |
| **AWA-RL** *(uncertainty RL)* | Qwen2.5-7B | 48.4 | 58.6 | 19.8 | 6.0 | 0.0 | 35.9 |
| **TrustRAG** *(inference filter)* | Qwen2.5-7B | 35.4 | 22.8 | 11.4 | 9.4 | 48.0 | 40.3 |
| **Exact-Match RL** *(same data)* | Qwen2.5-7B | 35.2 | 67.7 | 15.9 | 8.1 | 1.4 | 37.2 |
| **Exact-Match RL** *(same data)* | Llama-3.1-8B | 33.8 | 66.7 | 17.6 | 9.8 | 2.8 | 42.7 |
| **EchoSpec** *(cross-scale, 1 seed)* | Qwen2.5-3B | 15.3 | 30.4 | 22.5 | 12.4 | 37.5 | 42.8 |
| **EchoSpec** *(3-seed mean)* | Qwen2.5-7B | 14.3 | **28.9** | **25.8** | **16.7** | 37.6 | 47.1 |
| **EchoSpec** *(cross-family, 1 seed)* | Llama-3.1-8B | **11.1** | 30.6 | **26.5** | **17.2** | **52.5** | **55.4** |
| **EchoSpec + TrustRAG** *(3 seeds)* | Qwen2.5-7B | 13.8 | **12.8** | 29.4 | 19.0 | 36.5 | 44.1 |
| **EchoSpec + Sem. Dedup** *(3 seeds)* | Qwen2.5-7B | 16.7 | 28.7 | **36.2** | **20.2** | 33.9 | 46.2 |

<details>
<summary><b>Full Benchmark Table (all 7 released RL agents, Def@3, RA-F1 & Rewrite/Generator breakdown)</b></summary>

<br>

| Method | Backbone | Def@1&nbsp;↓ | Def@3&nbsp;↓ | Def@8&nbsp;↓ | Reach@8&nbsp;↑ | Acc@8&nbsp;↑ | Abst.&nbsp;↑ | Precision&nbsp;↑ | RA-F1&nbsp;↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Search-R1 | Qwen2.5-7B | 46.0 | 63.3 | 69.1 | 13.3 | 6.8 | 0.2 | 38.4 | 38.4 |
| ZeroSearch | Qwen2.5-7B | 53.0 | 69.6 | 72.9 | 7.3 | 3.8 | 0.1 | 35.2 | 35.2 |
| AutoRefine (Instruct) | Qwen2.5-7B | 47.8 | 63.0 | 72.2 | 13.0 | 8.8 | 0.0 | 42.4 | 42.4 |
| AutoRefine (Base) | Qwen2.5-7B | 40.5 | 57.2 | 59.8 | 18.1 | 12.5 | 0.0 | 44.7 | 44.7 |
| ReSearch | Qwen2.5-7B | 43.5 | 64.5 | 68.9 | 20.9 | 9.4 | 0.0 | 40.6 | 40.6 |
| R1-Searcher | Qwen2.5-7B | 33.3 | 54.2 | 61.9 | 21.0 | 12.5 | 0.0 | 41.8 | 41.7 |
| AWA-RL | Qwen2.5-7B | 48.4 | 55.6 | 58.6 | 19.8 | 6.0 | 0.0 | 35.9 | 35.9 |
| Qwen2.5-7B (no defense) | Qwen2.5-7B | 41.9 | 57.2 | 58.8 | 7.1 | 5.5 | 49.2 | 41.1 | 33.8 |
| TrustRAG | Qwen2.5-7B | 35.4 | 41.7 | 22.8 | 11.4 | 9.4 | 48.0 | 40.3 | 33.0 |
| $\lambda$-reward RL | Qwen2.5-7B | 45.4 | 68.4 | 69.4 | 9.4 | 5.2 | 32.5 | 42.4 | 37.6 |
| Exact-Match RL | Qwen2.5-7B | 35.2 | 59.5 | 67.7 | 15.9 | 8.1 | 1.4 | 37.2 | 36.8 |
| Exact-Match RL | Llama-3.1-8B | 33.8 | 61.3 | 66.7 | 17.6 | 9.8 | 2.8 | 42.7 | 42.2 |
| **EchoSpec** (1 seed) | Qwen2.5-3B | 15.3 | 29.6 | 30.4 | 22.5 | 12.4 | 37.5 | 42.8 | 36.9 |
| **EchoSpec** (3 seeds) | Qwen2.5-7B | 14.3 | 23.9 | 28.9 | 25.8 | 16.7 | 37.6 | 47.1 | 40.1 |
| **EchoSpec** (1 seed) | Llama-3.1-8B | 11.1 | 24.1 | 30.6 | 26.5 | 17.2 | 52.5 | 55.4 | 44.2 |
| **EchoSpec + TrustRAG** (3 seeds) | Qwen2.5-7B | 13.8 | 19.4 | 12.8 | 29.4 | 19.0 | 36.5 | 44.1 | 38.3 |
| **EchoSpec + Sem. Dedup** (3 seeds) | Qwen2.5-7B | 16.7 | 25.9 | 28.7 | 36.2 | 20.2 | 33.9 | 46.2 | 40.0 |

*Per-generator split:* On **Test Set A** (Qwen3-32B), EchoSpec achieves **19.0% / 23.6%** Def@8, **27.7% / 28.4%** Reach@8, and **13.5% / 13.6%** Rewrite exact match on Qwen2.5-7B / Llama-3.1-8B; on **Test Set B** (unseen Phi-4 family), EchoSpec achieves **38.8% / 37.7%** Def@8, **24.0% / 24.6%** Reach@8, and **16.9% / 18.1%** Rewrite exact match (vs. 7.9% / 10.2% Rewrite on base Qwen2.5-7B).

</details>

<p align="center">
  <img src="assets/analysis.png" width="94%" alt="Behavioral Mechanism: Primary Evidence Reach vs Conditional Deference"/>
</p>

### Zero-Shot Cross-Retriever Transfer
Keeping trained weights completely frozen (trained exclusively against e5-base-v2), we swap the entire 21M-passage wiki-18 index at test time to **BAAI/bge-base-en-v1.5** (CLS-pooled dense retriever) and exact **Okapi BM25** ($k_1 = 1.2, b = 0.75$ lexical retriever) under the unified exclude-seen (+xs) protocol:

| Method | e5-base-v2 *(In-Domain)*<br>Def@8&nbsp;↓ / Reach@8&nbsp;↑ | BGE-base-en-v1.5 *(Zero-Shot)*<br>Def@8&nbsp;↓ / Reach@8&nbsp;↑ | Okapi BM25 *(Zero-Shot Lexical)*<br>Def@8&nbsp;↓ / Reach@8&nbsp;↑ |
| :--- | :---: | :---: | :---: |
| **Search-R1** (Qwen2.5-7B, +xs) | 64.4 / 14.0 | 71.0 / 12.3 | 64.8 / 11.9 |
| **AutoRefine, Base** (Qwen2.5-7B, +xs) | 59.7 / 18.2 | 66.1 / 13.4 | 66.8 / 14.0 |
| **R1-Searcher** (Qwen2.5-7B, +xs) | 60.7 / 22.3 | 62.2 / 20.8 | 60.5 / 19.8 |
| **AWA-RL** (Qwen2.5-7B, +xs) | 60.3 / 20.7 | 58.1 / 21.2 | 53.6 / 20.9 |
| **EchoSpec** (**Qwen2.5-7B**, +xs, 3 seeds) | **28.9 / 25.8** | **29.4 / 26.0** | **25.1 / 22.5** |
| **EchoSpec** (**Llama-3.1-8B**, +xs, 1 seed) | **30.6 / 26.5** | **30.5 / 25.2** | **24.0 / 24.4** |

---

## Installation

**Requirements:** Linux with Python 3.10+ and CUDA or ROCm GPUs for full-scale vLLM rollout serving and distributed PyTorch training (CPU-only environments can run the complete unit-test suite out of the box).

```bash
git clone https://github.com/unihsy/EchoSpec.git
cd EchoSpec

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch & vLLM matching your CUDA or ROCm runtime, then install EchoSpec:
pip install -r requirements.txt
pip install -e .
```

Verify your installation in ~3 seconds with the CPU unit-test suite:

```bash
pytest -q tests
```

---

## End-to-End Reproduction

No proprietary datasets or manual preprocessing are required: every corpus index, multi-hop base item, verified deletion mask, and symmetric transformation world is reconstructed deterministically from public HuggingFace releases.

### Step 1: Prepare Corpus Indices & Transformation Worlds

```bash
# Builds wiki-18 Arrow corpus + e5-base-v2 fp16 embeddings, downloads HotpotQA/MuSiQue
# & 7 QA benchmarks, verifies deletion masks, and generates Test Sets A & B:
DATA_DIR=./data bash scripts/prepare_data.sh
```

To also build the zero-shot cross-retriever indices (BAAI/bge-base-en-v1.5 and Okapi BM25):

```bash
BUILD_CROSS_RETRIEVERS=1 DATA_DIR=./data bash scripts/prepare_data.sh
```

<details>
<summary><b>Modular Python CLI commands for Step 1 (click to expand)</b></summary>

```bash
# 1. Build 21M-passage wiki-18 Arrow table + e5-base-v2 fp16 matrix
python -m echospec.wiki18 --out ./data/wiki18 --work ./data/_scratch/wiki18

# 2. (Optional) Build BGE-base-en-v1.5 dense matrix & 32-worker Okapi BM25 inverted index
python -m echospec.wiki18 --build_dense ./data/wiki18/wiki18_contents.arrow ./data/wiki18/wiki18_bge_fp16.npy BAAI/bge-base-en-v1.5
python -m echospec.wiki18 --build_bm25  ./data/wiki18/wiki18_contents.arrow ./data/wiki18/bm25 --work ./data/_scratch/bm25

# 3. Extract HotpotQA + MuSiQue base items & 7 open-domain QA sets
python -m echospec.qa_data --out ./data/spec --base
python -m echospec.qa_data --out ./data/qa   --l4 --l4_max 1000

# 4. Launch retrieval server, verify deletion masks, generate claims, and assemble worlds + style audit
python -m echospec.retrieval --corpus wiki18=./data/wiki18/wiki18_contents.arrow --emb wiki18=./data/wiki18/wiki18_e5_fp16.npy --port 8290 &
python -m echospec.transforms verify   --base ./data/spec/base_train.jsonl --out ./data/spec/mask_train.jsonl
python -m echospec.transforms verify   --base ./data/spec/base_dev.jsonl   --out ./data/spec/mask_dev.jsonl
python -m echospec.transforms verify   --base ./data/spec/base_test.jsonl  --out ./data/spec/mask_test.jsonl
python -m echospec.transforms assemble --root ./data/spec --gen_a Qwen/Qwen3-32B --gen_b microsoft/phi-4
```
</details>

---

### Step 2: Two-Stage EchoSpec Training (Stage-1 Warm-Start + Stage-2 $\lambda$-GRPO)

By default, [scripts/train_echospec.sh](scripts/train_echospec.sh) partitions an 8-GPU node into **GPU 0** (GPU-resident retrieval server), **GPUs 1–4** (vLLM rollout servers with runtime LoRA hot-loading), and **GPUs 5–7** (distributed PyTorch LoRA SFT + asynchronous GRPO trainer with policy lag $\le 2$). All canonical hyperparameters are documented in [configs/echospec_default.json](configs/echospec_default.json).

```bash
# Train Qwen2.5-7B-Instruct (default)
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
OUT_DIR=./outputs/echospec_qwen25_7b_s1 \
SEED=1 \
bash scripts/train_echospec.sh

# Train Llama-3.1-8B-Instruct (cross-family replication, identical hyperparameters)
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
OUT_DIR=./outputs/echospec_llama31_8b_s1 \
SEED=1 \
bash scripts/train_echospec.sh

# Train Qwen2.5-3B-Instruct (compact 3B scale, identical hyperparameters)
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
OUT_DIR=./outputs/echospec_qwen25_3b_s1 \
SEED=1 \
bash scripts/train_echospec.sh
```

**4-GPU Single-Node Override:**
```bash
RETR_GPU=0 TEACHER_GROUPS="1,2" ROLLOUT_GROUPS="1;2" TRAIN_GPUS="3" \
bash scripts/train_echospec.sh
```

Training produces three main artifacts under the output directory:
- **sft_adapter/** — Stage-1 LoRA checkpoint after multi-hop verification warm-start (60 steps)
- **final_adapter/** — Final EchoSpec LoRA adapter after $\lambda$-calibrated GRPO (200 RL steps)
- **train_metrics.jsonl & dev_metrics.jsonl** — Per-step rollout, KL, grounded-share, acquisition, and per-family metrics

---

### Step 3: Unified Evaluation & Cross-Retriever Transfer

Evaluate any trained EchoSpec checkpoint standalone, without exclude-seen episode filtering, and composed with orthogonal retrieval filters (TrustRAG and Semantic Dedup) across Test Sets A & B and all 7 open-domain QA benchmarks:

```bash
# Standard in-domain evaluation (intfloat/e5-base-v2)
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
ADAPTER_DIR=./outputs/echospec_qwen25_7b_s1/final_adapter \
RETRIEVER=e5 \
bash scripts/evaluate_echospec.sh

# Zero-shot dense cross-retriever transfer (BAAI/bge-base-en-v1.5)
RETRIEVER=bge  ADAPTER_DIR=./outputs/echospec_qwen25_7b_s1/final_adapter bash scripts/evaluate_echospec.sh

# Zero-shot lexical cross-retriever transfer (Okapi BM25)
RETRIEVER=bm25 ADAPTER_DIR=./outputs/echospec_qwen25_7b_s1/final_adapter bash scripts/evaluate_echospec.sh
```

Evaluation writes per-episode predictions and query traces under **predictions/** and aggregated metrics (conditional deference, primary-evidence reach, overall accuracy, abstention rate, exact match, token F1, selective precision, and retrieval-abstention F1) to **summary.json**.

---

## Project Structure

```text
EchoSpec/
├── assets/                       # Paper overview, framework, and behavioral figures
├── configs/
│   └── echospec_default.json     # Canonical Stage-1 & Stage-2 hyperparameter reference
├── echospec/
│   ├── protocol.py               # Multi-turn <think>/<search>/<information>/<answer> protocol
│   ├── wiki18.py                 # 21M-passage Arrow builder + e5 / BGE / Okapi BM25 indexers
│   ├── retrieval.py              # Micro-batched FastAPI retrieval server (Dense + BM25 + overlays/masks)
│   ├── qa_data.py                # HotpotQA & MuSiQue provenance extractor + 7 QA benchmarks
│   ├── transforms.py             # 5 symmetric corpus transformations, mask verifier & style audit
│   ├── serve.py                  # Multi-GPU vLLM launcher (CUDA/ROCm) with runtime LoRA updating
│   ├── agent.py                  # Async multi-turn rollout engine
│   ├── synth.py                  # Stage-1 chain-guided verification synthesis & 3-gate filtering
│   ├── rewards.py                # Evidence-gated lambda-reward, acquisition bonus & GRPO advantages
│   ├── trainer.py                # Distributed PyTorch LoRA SFT + clipped GRPO + reference-KL engine
│   ├── train.py                  # Two-stage asynchronous rollout/training coordinator (torchrun)
│   ├── evaluate.py               # Unified transformation-suite & 7-benchmark QA evaluator
│   └── text.py                   # Answer normalization, abstention parsing & paired bootstrap stats
├── scripts/
│   ├── prepare_data.sh           # End-to-end corpus, index & transformation-world construction
│   ├── train_echospec.sh         # End-to-end Stage-1 synthesis + SFT + Stage-2 lambda-GRPO training
│   └── evaluate_echospec.sh      # End-to-end multi-suite & zero-shot cross-retriever evaluation
├── tests/                        # Fast CPU unit tests for protocol, retrieval, BM25, synth & rewards
├── pyproject.toml                # Package metadata and dependencies
├── requirements.txt              # Runtime dependency pin list
└── LICENSE                       # MIT License
```

---

## Citation

If you find **EchoSpec** useful in your research, please consider citing our paper:

```bibtex
@article{yang2026echospec,
  title   = {EchoSpec: Training Search Agents to Weigh Evidence Above Echoes},
  author  = {Yang, Shihao and Zeng, Can and Li, Xinzhe},
  year    = {2026},
  url     = {https://github.com/unihsy/EchoSpec}
}
```

## License

This project is released under the [MIT License](LICENSE).

