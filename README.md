# CARTS: Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree Search

[![OpenReview](https://img.shields.io/badge/OpenReview-CARTS-red.svg)](https://openreview.net/forum?id=VQwI055flA)
[![Home Page](https://img.shields.io/badge/Home-Page-blue.svg)](https://njuyxw.github.io/assets/CARTS/index.html)

A novel approach for neural theorem proving that combines diversified tactic calibration with bias-resistant tree search. Published at **ICLR 2025**.

## Overview

This repository implements CARTS (diversified tactic **CA**libration and bias-**R**esistant **T**ree **S**earch), which balances tactic diversity and importance while calibrating model confidence. CARTS also introduces preference modeling and an adjustment term related to the ratio of valid tactics to improve the bias-resistance of the value function.

## Requirements

- [Lean 4](https://github.com/leanprover/lean4) - The Lean theorem prover (version 4.x)
- [Lake](https://github.com/leanprover/lake) - Lean's official package manager

## Getting Started

### 1. Data Collection

First, collect and trace the miniF2F dataset using Lean 4:

```bash
python scripts/trace.py 
```

This process may take some time depending on your CPU performance.

### 2. Run CARTS

#### For multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate.py \
  --data-path ./data/minif2f_lean4_v4.10.0/default/ \
  --prover CARTS \
  --model_name reprover \
  --ckpt_path "kaiyuy/leandojo-lean4-tacgen-byt5-small" \
  --reward_model_path "yangxw/CARTS_vf" \
  --split test \
  --num-cpus 4 \
  --with-gpus \
  --num-theorems 244 \
  --num-sampled-tactics 64 \
  --exp-id "minif2f_CARTS" \
  --k 8 \
  --timeout 600 \
  --iteration 100
```

#### For a single GPU:

```bash
python evaluate.py \
  --data-path ./data/minif2f_lean4_v4.10.0/default/ \
  --prover CARTS \
  --model_name reprover \
  --ckpt_path "kaiyuy/leandojo-lean4-tacgen-byt5-small" \
  --reward_model_path "yangxw/CARTS_vf" \
  --split test \
  --num-cpus 1 \
  --with-gpus \
  --num-theorems 244 \
  --num-sampled-tactics 64 \
  --exp-id "minif2f_CARTS" \
  --k 8 \
  --timeout 600 \
  --iteration 100
```

#### Using BFS prover:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate.py \
  --data-path ./data/minif2f_lean4_v4.10.0/default/ \
  --prover BFS \
  --model_name reprover \
  --ckpt_path "kaiyuy/leandojo-lean4-tacgen-byt5-small" \
  --split test \
  --num-cpus 4 \
  --with-gpus \
  --num-theorems 244 \
  --num-sampled-tactics 64 \
  --exp-id "minif2f_BFS" \
  --timeout 600 \
  --iteration 100
```

## Citation

If you use this work, please cite it as follows:

```bibtex
@article{carts2025,
  title={CARTS: Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree Search},
  author={Yang, Xiao-Wen and Zhou, Zhi and Wang, Haiming and Li, Aoxue and Wei, Wen-Da and Jin, Hui and Li, Zhenguo and Li, Yu-Feng},
  journal={ICLR},
  year={2025}
}
```
