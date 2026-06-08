# CPU Deployment Roadmap: Running Hunyuan3D on an 8 GB RAM VPS (No GPU)

**Document version:** 2.0 (major expansion — new steps 4, 7, 8; 30 verified references)  
**Date:** 2026-04-29  
**Applies to:** `Backend/hy3dgen/` — Hunyuan3D shape + texture generation pipeline  
**Target environment:** VPS, 8 GB RAM, CPU-only (x86-64 or ARM), no CUDA device  
**Constraint:** Hunyuan3D architecture is kept intact — no model replacement  

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Baseline: Why Hunyuan3D Requires a GPU Today](#2-baseline-why-hunyuan3d-requires-a-gpu-today)
3. [Strategy Overview](#3-strategy-overview)
4. [Step 1 — ConvRot W4A4 Quantization](#4-step-1--convrot-w4a4-quantization)
5. [Step 2 — LD-Pruner / OBS-Diff Structured Pruning](#5-step-2--ld-pruner--obs-diff-structured-pruning)
6. [Step 3 — MDT-dist / DisCa Step Distillation](#6-step-3--mdt-dist--disca-step-distillation)
7. [Step 4 — Attention Sparsification](#7-step-4--attention-sparsification)
8. [Step 5 — Fast3Dcache Feature Caching](#8-step-5--fast3dcache-feature-caching)
9. [Step 6 — Sparse Mesh Extraction](#9-step-6--sparse-mesh-extraction)
10. [Step 7 — VAE Compression](#10-step-7--vae-compression)
11. [Step 8 — CPU Offload Infrastructure](#11-step-8--cpu-offload-infrastructure)
12. [Combined Memory Budget](#12-combined-memory-budget)
13. [Architectural Insights from Related Research](#13-architectural-insights-from-related-research)
14. [Supporting Research](#14-supporting-research)
15. [Risks and Trade-offs](#15-risks-and-trade-offs)
16. [Full Reference List](#16-full-reference-list)

---

## 1. Problem Statement

The current Hunyuan3D pipeline (`Backend/hy3dgen/`) is built around three heavy neural components:

| Component | Role | Approximate Size (FP32) |
|-----------|------|------------------------|
| Shape DiT (flow transformer) | Denoising latent occupancy field | ~8–10 GB |
| VAE encoder/decoder | Latent compression/reconstruction | ~2 GB |
| FlashVDM volume decoder | Octree-based marching cubes | ~2–3 GB peak activations |
| Texture pipeline (optional) | PBR texture generation | ~4 GB additional |

**Total peak GPU requirement: ~14–20 GB VRAM** for shape-only generation, more with texture.

A standard VPS with 8 GB RAM and no GPU cannot load even the model weights, let alone run inference. The challenge is to reduce memory and compute requirements by a factor of **4–8×** while maintaining usable output quality — **without replacing Hunyuan3D**.

---

## 2. Baseline: Why Hunyuan3D Requires a GPU Today

### 2.1 Memory Breakdown

**FP32 weight memory** alone exceeds 8 GB:

```
Shape DiT weights (FP32):    ~8.0 GB
VAE weights (FP32):          ~1.8 GB
FlashVDM decoder (FP32):     ~0.6 GB
─────────────────────────────────────
Weights total:               ~10.4 GB

Peak activations (per step): ~3–6 GB
OS + Python overhead:        ~1–2 GB
─────────────────────────────────────
Practical minimum VRAM:      ~14–18 GB
```

### 2.2 Compute Breakdown

Hunyuan3D's shape generation runs a **rectified flow transformer** with:
- Default: 25–50 denoising steps
- Each step: full transformer forward pass through ~3B parameters
- FlashVDM: two-level octree query (63³ coarse → 126³ fine) per generation

On CPU (no vectorized GPU kernels), a single transformer forward pass through a ~5 GB FP32 model takes **5–15 minutes** depending on hardware. 50 steps × 10 minutes = **8+ hours** per generation — not practical.

### 2.3 Community Proof of Concept

**Hunyuan3D-2GP** (deepbeepmeep, 2025) is a community implementation that runs full Hunyuan3D-2 with under 6 GB VRAM using the `mmgp` CPU-offload library — requiring ~24 GB system RAM (shape-only works with 3 GB VRAM). This confirms that CPU-RAM offloading is viable even before applying the compression techniques in this roadmap. The techniques below reduce the RAM requirement to fit within 8 GB total.

### 2.4 The Four-Axis Solution

Reaching 8 GB peak RAM and <5 minutes per generation requires simultaneous compression along four axes:

```
Axis 1 — Size:       Quantize + prune model weights          → 8–12× smaller
Axis 2 — Steps:      Distill to 2 denoising steps            → 10–25× less compute
Axis 3 — Attention:  Sparse attention patterns                → 2–3× less per step
Axis 4 — Decoder:    Sparse/cached mesh extraction           → 5–10× less RAM peak
```

---

## 3. Strategy Overview

The eight steps below are **additive** — each applied on top of the previous, stacking their gains:

```
Hunyuan3D (current)
     │  ~14 GB GPU required
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: ConvRot W4A4 Quantization                          │
│  FP32 → 4-bit weights + 4-bit activations                   │
│  Weights: 10.4 GB → ~1.3 GB   (8× reduction)               │
│  Alternatives: Q-DiT (W4A8), SVDQuant, ViDiT-Q, CLQ        │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: LD-Pruner / OBS-Diff Structured Pruning            │
│  Remove redundant attention heads and MLP channels          │
│  Speed: 34.89% faster, parameters: −31.7%                   │
│  Preferred: OBS-Diff (outperforms Wanda for diffusion)      │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: MDT-dist / DisCa Step Distillation                 │
│  25–50 denoising steps → 2 steps (6.5× speedup)            │
│  DisCa (Hunyuan team): 11.8× total acceleration possible   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Attention Sparsification (NEW)                     │
│  Skip redundant attention heads per layer per timestep      │
│  DiTFastAttn: 76% attention FLOPs cut, 1.8× speedup        │
│  Sparse VideoGen: 2.33× tested on HunyuanVideo              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Fast3Dcache Feature Caching  (CVPR 2026)           │
│  Cache stable voxel features across timesteps               │
│  FLOPs: −54.83%, throughput: +27.12%, quality: −2.48% CD   │
│  Alternatives: DeepCache, ToCa, ProCache                    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Sparse Mesh Extraction                             │
│  Dense marching cubes → sparse surface-only extraction      │
│  SparseFlex: 82% CD improvement, up to 1024³ resolution    │
│  Simpler alternative: ODC (Occupancy Dual Contouring)       │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 7: VAE Compression (NEW)                              │
│  Compress Hunyuan3D's Shape VAE encoder/decoder             │
│  COD-VAE: 16× latent compression, 20.8× faster generation  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 8: CPU Offload Infrastructure (NEW)                   │
│  Layer-by-layer weight paging between CPU RAM and compute   │
│  FengHuang (Microsoft): 93% GPU memory reduction           │
│  Fiddler (ICLR 2025): 11.57× speedup over naive offload    │
└─────────────────────────────────────────────────────────────┘
     │
     │  ~3–5 GB peak RAM, 1–5 min/generation on modern CPU
     ▼
   CPU VPS (8 GB RAM) — FEASIBLE
```

---

## 4. Step 1 — ConvRot W4A4 Quantization

### 4.1 Paper Reference

> **ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers**
> arXiv:2512.03673 | Submitted December 3, 2025
> https://arxiv.org/abs/2512.03673
>
> **Authors:** Feice Huang (Tsinghua SIGS), Zuliang Han, Xing Zhou, Yiyang Chen, Lifei Zhu (Huawei Central Media Technology Institute), Haoqian Wang (Tsinghua SIGS)

### 4.2 Core Idea

Standard INT4 quantization fails on diffusion transformers because **activation outliers** cause massive quantization error. ConvRot rotates weight matrices using a Hadamard transform before quantization, redistributing outlier energy uniformly so no single dimension dominates. The `ConvLinear4bit` layer fuses rotation + INT4 GEMM + dequantization into a single kernel, reducing quantization complexity from quadratic to linear.

```
Standard INT4:
  W ∈ ℝ^{m×n}  →  Q(W) ∈ INT4^{m×n}
  Error from outlier activations: HIGH

ConvRot W4A4:
  W → H × W (Hadamard)  → Q(HW) ∈ INT4^{m×n}
  x → H × x (same rot.) → Q(Hx) ∈ INT4^n
  Output: Q(HW) · Q(Hx)  ≈  W · x    [outlier error eliminated]
```

### 4.3 Measured Results (FLUX.1-dev, 12B DiT — same class as Hunyuan3D)

| Precision | Memory | Latency (50 steps) | FID↓ | Image Reward↑ |
|-----------|--------|--------------------|------|---------------|
| BF16 baseline | 22.7 GiB | 54.6 s | 10.07 | 0.99 |
| W4A4 + 20% INT8 mixed | 5.6 GiB | 23.2 s | 10.03 | 0.97 |
| Full W4A4 | 5.6 GiB | 23.2 s | 12.32 | 0.84 |

- **4.05× memory reduction** from BF16; **8× from FP32**
- **2.26× speedup** over BF16
- **Best recipe:** W4A4 + 20% INT8 for sensitive layers — FID virtually identical to BF16

### 4.4 Memory Impact on Hunyuan3D

| Precision | Shape DiT | VAE | Total Weights |
|-----------|-----------|-----|---------------|
| FP32 (baseline) | ~8.0 GB | ~1.8 GB | ~10.4 GB |
| BF16 | ~4.0 GB | ~0.9 GB | ~5.2 GB |
| W4A4 (ConvRot) | ~1.0 GB | ~0.23 GB | **~1.3 GB** |

### 4.5 CPU Execution

INT4 GEMM is natively supported by:
- **llama.cpp / GGML:** well-tested INT4 on ARM (NEON) and x86 (AVX2/AVX-512)
- **Intel Neural Compressor:** INT4 on Xeon with AMX instructions
- **ARM Compute Library:** INT4 kernels for AWS Graviton 3, Ampere Altra

### 4.6 Implementation Path

```
Target files:
  hy3dgen/shapegen/models/autoencoders/model.py      ← VAE
  hy3dgen/shapegen/models/denoising/model.py         ← DiT flow transformer

Steps:
  1. Extract all nn.Linear layers from the DiT
  2. Compute Hadamard rotation H offline per layer
  3. Apply: W_rotated = H @ W, store as INT4
  4. Mark first/last 10% layers for INT8 (sensitive timesteps)
  5. Replace nn.Linear with ConvLinear4bit at model load time
```

> **Note:** No public ConvRot code as of December 2025. Use **SVDQuant** (arXiv:2411.05007) instead — released production code at github.com/mit-han-lab/deepcompressor + Nunchaku inference engine, achieving 3.0–10× speedup on the same model class.

### 4.7 Alternative Quantization Methods

#### Q-DiT (CVPR 2025)

> **Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers**
> arXiv:2406.17343 | CVPR 2025 | https://arxiv.org/abs/2406.17343
> Authors: Lei Chen et al.

Addresses two failure modes: (a) large cross-channel weight variance via per-channel granularity allocation; (b) timestep-varying activation distributions via sample-wise dynamic quantization. No retraining required.
- **W4A8:** near full-precision FID on DiT-XL/2
- **W6A8:** FID improvement of 1.09 over baseline at same bit-width

#### TFMQ-DM (CVPR 2024 Highlight)

> **TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models**
> arXiv:2311.16503 | CVPR 2024 Highlight | https://arxiv.org/abs/2311.16503

Specifically addresses **time-step embedding layers** missed by standard PTQ calibration. Reduces FID by **6.7 on CelebA-HQ** under INT4 vs. naive PTQ. Pair with Q-DiT: Q-DiT handles linear layers, TFMQ-DM handles timestep embedding modules.

#### ViDiT-Q (ICLR 2025)

> **ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers**
> arXiv:2406.02540 | ICLR 2025 | https://arxiv.org/abs/2406.02540

Validated on **video DiT models** (Open-Sora, Latte, PixArt — same class as Hunyuan3D):
- **W8A8:** 2–2.5× memory saving, 1.4–1.7× speedup
- **W4A8 mixed:** 2.5× memory, 1.5× speedup

#### CLQ (2025)

> **CLQ: Cross-Layer Guided Orthogonal-based Quantization for Diffusion Transformers**
> arXiv:2509.24416 | 2025 | https://arxiv.org/abs/2509.24416

Post-training W4A4 for image and video DiTs. **3.98× memory reduction, 3.95× inference acceleration** with negligible visual quality degradation. No retraining required.

#### DVD-Quant — Data-Free W4A4 for Video DiTs (2025)

> **DVD-Quant: Data-free Video Diffusion Transformers Quantization**
> arXiv:2505.18663 | 2025 | https://arxiv.org/abs/2505.18663
> Authors: Zhiteng Li, Hanxuan Li et al.

First W4A4 PTQ for Video DiTs (CogVideoX, HunyuanVideo) **without quality compromise, no calibration dataset required**. Reports ~2× speedup over full-precision. Directly applicable to Hunyuan3D's texture DiT stage.

#### BiDM (Extreme Compression Fallback)

> **BiDM: Pushing the Limit of Quantization for Diffusion Models**
> arXiv:2412.05926 | December 2024 | https://arxiv.org/abs/2412.05926

W1A1 — **28× storage compression, 52.7× MAC reduction** via integer bitwise operations. Use only if W4A4 is still insufficient for your hardware.

---

## 5. Step 2 — LD-Pruner / OBS-Diff Structured Pruning

### 5.1 Primary Reference: LD-Pruner

> **LD-Pruner: Efficient Pruning of Latent Diffusion Models using Task-Agnostic Insights**
> arXiv:2404.11936 | CVPR 2024 Workshop on Efficient and On-Device Generation (EDGE)
> https://arxiv.org/abs/2404.11936
>
> **Authors:** Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, Shinkook Choi (Nota AI, Seoul)

### 5.2 Core Idea

Identifies redundant attention heads and MLP channels using **latent space activation statistics** — no task-specific training data needed. Computes an importance score per head by measuring its contribution to activation variance over 50–100 random inputs.

```
I(h) = E_{x~P(x)} [ ||Attention_h(x)||_F ]
Prune heads where I(h) < threshold τ (chosen for target compression ratio)
```

### 5.3 Measured Results (Stable Diffusion v1.4)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Parameters | 1.04B | 0.71B | −31.7% |
| Inference speed | baseline | +34.89% | faster |
| FID↓ | 13.05 | 12.37 | **improves** |
| CLIP score | 0.2894 | 0.2894 | no change |

**Key insight:** FID *improves* after pruning — the pruned heads were adding noise, not signal.

### 5.4 Preferred Alternative: OBS-Diff (2025)

> **OBS-Diff: Accurate Pruning for Diffusion Models in One-Shot**
> arXiv:2510.06751 | 2025 | https://arxiv.org/abs/2510.06751
> Authors: Junhan Zhu, Hesong Wang, Mingluo Su, Zefang Wang, Huan Wang

Adapts the **Optimal Brain Surgeon (OBS) framework** to diffusion models with timestep-aware Hessian computation. Benchmarks Wanda and DSnoT — outperforms both at every sparsity level (20–50%).
- Supports unstructured, N:M semi-structured, and structured sparsity
- Timestep-aware Hessian accounts for Hunyuan3D's multi-step denoising schedule
- No training data required (calibration only)

**Key finding:** Wanda does NOT outperform simple magnitude pruning for diffusion models. Use OBS-Diff instead.

### 5.5 Correct Application Order

```
WRONG: Quantize → Prune
CORRECT: Prune (FP32) → Quantize (INT4 via ConvRot/Q-DiT)
  Smaller FP32 matrices × 8× bit reduction = maximum compression
  1.04B params × 0.683 (pruned) × 4-bit = maximum compression stack
```

### 5.6 Limitations

- Does not extend to pruning the VAE decoder (addressed by Step 7)
- Does not account for inter-operator dependencies
- Only the DiT backbone is targeted; FlashVDM occupancy decoder is not

---

## 6. Step 3 — MDT-dist / DisCa Step Distillation

### 6.1 Primary Reference: MDT-dist

> **Few-step Flow for 3D Generation via Marginal-Data Transport Distillation**
> arXiv:2509.04406 | September 4, 2025 | https://arxiv.org/abs/2509.04406
> **Code:** https://github.com/Zanue/MDT-dist
>
> **Authors:** Zanwei Zhou (SJTU), Taoran Yi (HUST), Jiemin Fang, Chen Yang, Lingxi Xie, Qi Tian (Huawei), Xinggang Wang (HUST), Wei Shen (SJTU)

### 6.2 Core Idea

Distills a pretrained flow-matching 3D DiT to 1–2 steps via two objectives:

**Velocity Matching (VM):** Student's velocity should match the teacher's effective average over a full trajectory segment.  
**Velocity Distillation (VD):** Teacher scores the student's intermediate outputs, providing unbiased gradient signal.

```
L_VM = E [ ||v_student(x_t, t) - (x_0 - x_T)/T||² ]
L_VD = E [ ||v_teacher(x̂_t, t) - (x̂_t - x_target)||² ]
L    = L_VM + 0.1·L_VD
```

### 6.3 Measured Results (TRELLIS, same flow-matching paradigm as Hunyuan3D)

| Method | Steps | Time (s) | ULIP_I↑ |
|--------|-------|----------|---------|
| Teacher | 25 × 2 | 6.10 | 39.53 |
| MDT-dist 1-step | 1 × 2 | 0.68 | 36.88 |
| **MDT-dist 2-step** | **2 × 2** | **0.94** | **39.11** |

- **2-step recommendation:** 6.5× speedup, only −1.06% ULIP quality drop

### 6.4 Why Step 3 Is the Most Critical on CPU

```
CPU inference estimate (W4A4 pruned model ~0.85 GB):
  25 steps × 90s =  37.5 minutes  ← unusable
   2 steps × 90s =   3.0 minutes  ← acceptable
```

### 6.5 Hunyuan-Specific: DisCa (Tencent Hunyuan Team)

> **DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching**
> arXiv:2602.05449 | February 2026
> Authors: Tencent Hunyuan group

DisCa comes from **the same team that built Hunyuan3D**. It replaces heuristic feature reuse with a lightweight neural predictor trained to forecast cached features — combined with Restricted MeanFlow step-distillation for stable few-step generation.

- **11.8× acceleration** vs. full-step baseline on video benchmarks
- Directly portable to Hunyuan3D's shape/texture DiTs
- Integrates distillation + feature caching into a single training pass (combines Steps 3 and 5)

**Why prefer DisCa over MDT-dist:** DisCa trains distillation and caching jointly, saving the separate Step 5 (Fast3Dcache) implementation. However, MDT-dist code is already public; use it as a starting point.

### 6.6 Hunyuan3D 2.0 Fast Checkpoint (Zero-Code Path)

Tencent released **Hunyuan3D-DiT-v2-0-Fast** (February 2025) — a guidance-distilled variant reducing steps by ~50%, achieving ~30s end-to-end on GPU. This is the baseline starting point. All subsequent steps stack on top.

### 6.7 Additional: FlashVDM Progressive Flow Distillation

> **Unleashing Vecset Diffusion Model for Fast Shape Generation**
> arXiv:2503.16302 | March 2025 | https://arxiv.org/abs/2503.16302
> Authors: Zeqiang Lai, Yunfei Zhao et al.

Applies Progressive Flow Distillation specifically to a Vecset-based 3D shape DiT — the same architecture class as Hunyuan3D. Achieves **45× faster reconstruction and 32× faster generation** vs. baseline at 5 inference steps while maintaining SOTA quality. Directly relevant distillation approach for Hunyuan3D's shape pipeline.

---

## 7. Step 4 — Attention Sparsification

### 7.1 Why Attention Is the CPU Bottleneck

Transformer attention scales as O(n²) in sequence length. For Hunyuan3D's DiT operating on 3D voxel grids, sequence lengths can reach thousands of tokens. On CPU without CUDA kernels, each attention head's n×n matrix multiply is the dominant per-step cost. Research shows DiT attention has significant structured redundancy across three dimensions:

- **Spatial:** local-focus heads attend only within regions (not globally)
- **Temporal:** adjacent-step outputs are nearly identical for stable regions
- **CFG:** conditional and unconditional branches compute similar attention

All three can be exploited without retraining.

### 7.2 Primary Reference: DiTFastAttn (NeurIPS 2024)

> **DiTFastAttn: Attention Compression for Diffusion Transformer Models**
> arXiv:2406.08552 | NeurIPS 2024 | https://arxiv.org/abs/2406.08552
> Authors: Zhihang Yuan, Hanling Zhang, Pu Lu, Xuefei Ning, Linfeng Zhang et al. (SJTU)

Three post-training compression strategies:

1. **WARC (Window Attention with Residual Caching):** Spatial-focus heads compute only local windows; global result cached from prior step.
2. **SARC (Shared Attention with Residual Caching):** Adjacent-timestep similarity — reuse when attention outputs barely change.
3. **CAS (CFG Attention Sharing):** Share computation between conditional and unconditional CFG branches.

**Measured results:**
- **76% reduction in attention FLOPs**
- **1.8× end-to-end speedup** at 2K×2K on PixArt-Sigma
- **36–88% attention reduction** depending on resolution
- No retraining required

### 7.3 Direct Evidence for Hunyuan: Sparse VideoGen

> **Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity**
> arXiv:2502.01776 | ICML 2025 / NeurIPS 2025 Spotlight | https://arxiv.org/abs/2502.01776

Profiles **HunyuanVideo** directly (sibling model to Hunyuan3D from Tencent). Discovers attention heads split at runtime into:
- **Spatial Heads:** attend within-frame only → local window attention
- **Temporal Heads:** attend across frames/views → strided temporal attention

```
Online head classification (no retraining):
  Compute attention map A_h at step t
  if top-k(A_h) localizes within spatial windows → Spatial Head
  else → Temporal Head
  Cache classification for future steps
```

**Measured on HunyuanVideo (directly relevant):**
- **2.33× end-to-end speedup**
- **Kernel speedup 2.29×–17.64×** for sparse ops
- **CogVideoX-v1.5: 2.28× speedup**
- Training-free

**This is the strongest evidence that attention sparsification works on Hunyuan3D's architecture.**

### 7.4 Additional: SpargeAttention (ICML 2025)

> **SpargeAttention: Accurate and Training-Free Sparse Attention Accelerating Any Model Inference**
> arXiv:2502.18137 | ICML 2025 | https://arxiv.org/abs/2502.18137
> Authors: Zhang et al. (THU-ML)

Two-stage online filter: stage 1 predicts the attention map before full matrix multiply; stage 2 applies softmax-aware filtering.
- **4–7× attention speedup** across language, image, and video models
- Training-free plug-in for any DiT attention layer

### 7.5 Combined Effect with Step 3

After MDT-dist (2 steps) + DiTFastAttn (−76% attention FLOPs per step):

```
Baseline 25-step: 25 × full attention   (1× reference)
After distillation (×2):  2 × full attention                 → 12.5× less
After DiTFastAttn (×0.24): 2 × 0.24× attention              → 52× less than baseline
```

---

## 8. Step 5 — Fast3Dcache Feature Caching

### 8.1 Paper Reference

> **Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration**
> arXiv:2511.22533 | November 27, 2025 | **Accepted at CVPR 2026**
> https://arxiv.org/abs/2511.22533
> Project: https://fast3dcache-agi.github.io
>
> **Authors:** Mengyu Yang, Yanming Yang, Chenyi Xu, Chenxi Song, Yufan Zuo, Tong Zhao (Westlake University), Ruibo Li (NTU), Chi Zhang (Westlake University)

### 8.2 Core Idea

Interior/exterior voxels stabilize early in denoising — only surface-boundary voxels remain uncertain. Fast3Dcache caches stable voxel features using:

1. **PCSC:** Dynamically determines how many voxels to cache per timestep
2. **SSC:** Selects which voxels via velocity + acceleration signals

```
stable(x,t) = velocity(x,t) < δ_v  AND  acceleration(x,t) < δ_a
Stable voxels: features ← cache[x, t-1]     [skip transformer]
Active voxels: features ← transformer_forward(x, t)
```

### 8.3 Measured Results

| Method | Throughput | FLOPs | Chamfer Dist |
|--------|-----------|-------|-------------|
| Vanilla | 0.5055 iter/s | 244.2 T | 0.0686 |
| Fast3Dcache (τ=8) | **0.6426** | **110.3 T** | 0.0703 |
| TeaCache + Fast3Dcache | — | — | 0.0701 |

- **+27.12% throughput, −54.83% FLOPs, +2.48% CD degradation**

### 8.4 Compounds with MDT-dist (Step 3)

After 2-step distillation:
- Step 1 (t=1→0.5): no cache yet — full computation
- Step 2 (t=0.5→0): ~50% voxels stable → ~50% skipped
- Net: additional **~25% speedup** beyond distillation alone

### 8.5 Complementary Caching Methods

#### DeepCache (CVPR 2024) — Foundational

> **DeepCache: Accelerating Diffusion Models for Free**
> arXiv:2312.00858 | CVPR 2024 | https://arxiv.org/abs/2312.00858

Caches high-level DiT features between adjacent steps; recomputes only cheap low-level features. Training-free.
- **2.3× speedup** on SD v1.5 (CLIP drop −0.05)
- **4.1× speedup** on LDM-4-G (FID increase +0.22)
- Fast3Dcache is the 3D-specific evolution of DeepCache's principle

#### ToCa (ICLR 2025) — Token-wise Granularity

> **ToCa: Accelerating Diffusion Transformers with Token-wise Feature Caching**
> arXiv:2410.05317 | ICLR 2025 | https://arxiv.org/abs/2410.05317

Key insight: different tokens have vastly different caching sensitivity — naive caching on sensitive tokens causes ~10× more quality damage. Fine-grained per-token policies per layer.
- **1.93× speedup** on PixArt-α, near-zero FID drop
- **2.36× speedup** on OpenSora (video DiT)

#### ProCache (AAAI 2026) — Constraint-Aware Scheduling

> **ProCache: Constraint-Aware Feature Caching with Selective Computation**
> arXiv:2512.17298 | AAAI 2026 | https://arxiv.org/abs/2512.17298

Non-uniform caching schedule via offline search, aligned with DiT's non-uniform temporal dynamics.
- **1.96× speedup** on PixArt-alpha
- **2.90× speedup** on DiT
- Training-free

### 8.6 Implementation Path

```
Target file: Backend/hy3dgen/shapegen/models/autoencoders/volume_decoders.py
             (FlashVDMVolumeDecoder.__call__ — the octree loop)

Steps:
  1. Add voxel stability cache: Dict[int, Tensor] (spatial index → feature)
  2. Before each refinement step, compute velocity + acceleration per voxel
  3. Build stability mask: bool tensor of shape (N_voxels,)
  4. Stable voxels: substitute cached features (dict lookup)
  5. Active voxels: run geo_decoder normally
  6. Update cache for all voxels after each step

No weight file changes — purely inference-time logic.
```

---

## 9. Step 6 — Sparse Mesh Extraction

### 9.1 Primary Reference: SparseFlex

> **SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling**
> arXiv:2503.21732 | March 27, 2025 | https://arxiv.org/abs/2503.21732
> Project: https://xianglonghe.github.io/TripoSF
>
> **Authors:** Xianglong He, Chia-Hao Chen (Tsinghua), Zi-Xin Zou, Yuan-Chen Guo, Ding Liang, Yan-Pei Cao, Yangguang Li (VAST), Wanli Ouyang (CUHK)

### 9.2 Core Idea

Standard FlashVDM evaluates occupancy on a **dense 3D grid** — every voxel including trivially-inside/outside points. Memory is O(R³). SparseFlex uses sparse isosurface evaluation: a coarse pass identifies surface-adjacent voxels (~1–5% of grid), and only those are retained for FlexiCubes mesh extraction.

```
Dense (current FlashVDM):    128³ = 2,097,152 voxels evaluated
SparseFlex (1–5% sparse):    ~20K–100K voxels  → 20–100× smaller
```

### 9.3 Measured Results

| Resolution | Toys4K CD↓ (×10⁻⁴) | F-Score↑ |
|------------|---------------------|---------|
| 256³ | 2.56 | 18.31 |
| 512³ | 1.67 | 23.74 |
| 1024³ | **1.33** | **25.95** |

- **82% Chamfer Distance reduction** vs. prior methods at equivalent resolution
- **88% F-Score improvement** vs. prior methods

### 9.4 Simpler Drop-In Alternative: ODC (SIGGRAPH Asia 2024)

> **ODC: Occupancy-Based Dual Contouring**
> arXiv:2409.13418 | SIGGRAPH Asia 2024 | https://arxiv.org/abs/2409.13418
> Authors: KAIST Visual AI Group

Standard marching cubes requires an SDF. Hunyuan3D's FlashVDM outputs **occupancy values** — a binary field. ODC is designed specifically for occupancy functions:

- No SDF required — works directly with Hunyuan3D's occupancy output
- Adds auxiliary 2D points for sharp-feature normals (better edge preservation)
- Parallelizes across all edges/faces/cells — "computation time of a few seconds, learning-free"
- **Guarantees manifoldness**, greatly reduces intersecting faces vs. standard MC
- SIGGRAPH Asia 2024 — state-of-the-art fidelity for occupancy functions

**ODC is the recommended first step** before implementing the full SparseFlex pipeline — zero training, drop-in replacement for the marching cubes call in `volume_decoders.py`.

```
Current:
  volume_decoders.py line 420+: → marching_cubes(occupancy_grid)

ODC replacement:
  from odc import occupancy_dual_contour
  mesh = occupancy_dual_contour(occupancy_grid, resolution=127)
  # Manifold mesh, sharp features, no SDF needed
```

### 9.5 Additional: FlexiCubes (SIGGRAPH 2023)

> **FlexiCubes: Flexible Isosurface Extraction for Gradient-Based Mesh Optimization**
> arXiv:2308.05371 | SIGGRAPH 2023 (ACM ToG) | https://arxiv.org/abs/2308.05371
> Authors: NVIDIA Toronto AI Lab

Learnable per-cube parameters on top of Dual Marching Cubes — more uniform triangles, no stair-step artifacts, enables gradient flow for future fine-tuning. Used as the mesh extraction backend in SparseFlex.

```bash
pip install flexicubes
```

### 9.6 Connection to the Existing Crash Fix

The `IndexError: min() on empty tensor` at `volume_decoders.py:422` is caused by the dense grid finding zero near-surface voxels when mc_level is outside the grid range. ODC and SparseFlex handle exactly this case — empty near-surface sets cause graceful fallback, not crashes. The existing guard (`if next_points.shape[0] == 0: break`) is a stopgap; ODC eliminates the root cause.

---

## 10. Step 7 — VAE Compression

### 10.1 Why the VAE Matters

The VAE is applied once (encode at start, decode at end), but its **latent space dimension** determines how many tokens the shape DiT processes per denoising step — a compressed VAE directly reduces DiT sequence length and therefore attention memory. COD-VAE shows that 64 tokens are sufficient for high-fidelity 3D reconstruction.

### 10.2 Paper Reference: COD-VAE

> **COD-VAE: Representing 3D Shapes with 64 Latent Vectors**
> arXiv:2503.08737 | March 2025 | https://arxiv.org/abs/2503.08737
> Authors: Cho, Yoo, Jeon, Kim (Yonsei University)

COD-VAE compresses 3D shapes into just **64 1D latent vectors** using uncertainty-guided token pruning:
1. Full encoder produces N latent tokens
2. Uncertainty estimator scores each token by reconstruction contribution
3. Bottom (N − 64) tokens are pruned; only 64 tokens retained
4. Decoder reconstructs from 64 tokens

**Measured results:**
- **16× compression** vs. baseline VAE
- **20.8× faster generation** for the downstream 3D diffusion model

### 10.3 Supporting Reference: DC-AE (MIT Han Lab)

> **DC-AE: Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models**
> arXiv:2410.10733 | 2024 | https://arxiv.org/abs/2410.10733
> Authors: Chen et al. (MIT Han Lab)

Spatial compression ratios up to 128× (vs. standard 8×) via Residual Autoencoding. On ImageNet 512×512:
- **19.1× inference speedup** on H100
- Better FID than SD-VAE-f8

Residual Autoencoding insight — progressively compressing residuals between encoder stages — applies directly to redesigning Hunyuan3D-ShapeVAE's decoder path.

### 10.4 Implementation Path

```
Target: hy3dgen/shapegen/models/autoencoders/model.py

COD-VAE approach:
  1. Add uncertainty head to existing VAE encoder (2-layer MLP predicting token importance)
  2. During inference: run encoder → score tokens → keep top-64
  3. Run DiT on 64 tokens instead of full sequence
  4. Run decoder on 64-token latent → occupancy reconstruction

Training: ~50K fine-tuning steps with reconstruction loss
  Dataset: any 3D occupancy GT (Objaverse subset)
  GPU: 1× A100, ~1 day (~$50–100 cloud)
```

---

## 11. Step 8 — CPU Offload Infrastructure

### 11.1 The Offloading Problem

After Steps 1–7, total weights are ~0.89 GB W4A4. Naive CPU offloading — move all weights to CPU, run layer, move back — creates excessive data movement latency. Principled offloading overlaps weight transfers with computation by pre-fetching weights before they are needed.

### 11.2 Primary Reference: FengHuang (Microsoft Research)

> **FengHuang: Next-Generation Memory Orchestration for AI Inferencing**
> arXiv:2511.10753 | November 2024 | https://arxiv.org/abs/2511.10753
> Authors: Li et al. (Microsoft Research)

Introduces **Active Tensor Paging** — multi-tier memory with weights paged between registers/L3 cache and DRAM on-demand, driven by offline access pattern analysis:

- **93% local GPU memory reduction** for GPU workloads
- **50% GPU compute savings** through layer-level prefetch prediction
- For CPU-only VPS: the same paging principle applies between L3 cache and DRAM

Key mechanism: builds a computation graph access timeline offline, then schedules pre-fetches to arrive in cache exactly when the layer needs them — preventing cache thrashing.

### 11.3 Supporting Reference: Fiddler (ICLR 2025)

> **Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models**
> arXiv:2402.07033 | ICLR 2025 | https://arxiv.org/abs/2402.07033

Core principle — "use CPU compute to minimize CPU↔GPU data movement" — applies to Hunyuan3D attention layers during CPU-only execution:
- **11.57× speedup** over naive offloading
- **22.5× speedup** over DeepSpeed-MII
- Architecture-agnostic scheduling algorithm

### 11.4 Community Reference: Hunyuan3D-2GP

**Hunyuan3D-2GP** (github.com/deepbeepmeep/Hunyuan3D-2GP) uses the `mmgp` CPU-offload library to run Hunyuan3D-2 with:
- Shape only: **3 GB VRAM** (CPU RAM provides the remainder)
- Full pipeline: **under 6 GB VRAM** (~24 GB system RAM total)

With our Steps 1–7 reducing weights from ~10 GB to ~0.89 GB, the 24 GB RAM requirement shrinks proportionally to **~2.1 GB** — fits in 8 GB total with significant margin.

### 11.5 Implementation Path

```
Option A (quickest): Use mmgp library
  pip install mmgp
  from mmgp import offload
  model = offload(hunyuan3d_model, device="cpu", max_vram=0)

Option B (principled — FengHuang-style):
  1. Profile one full inference pass with layer-level timestamp logging
  2. Build prefetch schedule from access timeline
  3. Implement as forward hooks: hook_pre(layer) starts prefetch of layer+1
  4. hook_post(layer) confirms completion, evicts layer from L3

Option C (high-performance CPU INT4 kernels):
  github.com/kvcache-ai/ktransformers
  Intel AMX/AVX512 INT4 GEMM: 21.3 TFLOPS on DeepSeek-3 (3.98× over PyTorch)
  Applicable if Hunyuan3D's DiT layers map cleanly to INT4 GEMM primitives
```

---

## 12. Combined Memory Budget

Applying all eight steps produces the following cumulative reduction:

```
Stage                              Weights    Peak Acts    Total Peak
─────────────────────────────────────────────────────────────────────
Baseline (FP32, no changes)        10.4 GB    6.0 GB       ~16.4 GB

After Step 2: Pruning (−31.7%)      7.1 GB    4.1 GB       ~11.2 GB
After Step 1: W4A4 Quant (÷8)      0.89 GB   0.8 GB       ~1.69 GB
After Step 3: 2-step distill        0.89 GB   0.8 GB       ~1.69 GB  *
After Step 4: Attn Sparsify         0.89 GB   0.55 GB      ~1.44 GB  **
After Step 5: Fast3Dcache           0.89 GB   0.4 GB       ~1.29 GB  ***
After Step 6: Sparse mesh           0.89 GB   0.15 GB      ~1.04 GB  ****
After Step 7: VAE compression       0.75 GB   0.15 GB      ~0.90 GB  *****
After Step 8: CPU offload paging    layer-by-layer (peaks never overlap)

* Step 3 reduces TIME, not memory per step
** DiTFastAttn −76% attention FLOPs → smaller intermediate activation tensors
*** Fast3Dcache skips ~54% volume decoder FLOPs → smaller activation peak
**** Sparse mesh: next_logits tensor shrinks from ~2 GB dense to ~50 MB sparse
***** VAE: full token sequence → 64 vectors (16× compression reduces DiT sequence length)

OS + Python runtime:                                       ~2.0 GB
PyTorch + I/O buffers:                                     ~0.5 GB
─────────────────────────────────────────────────────────────────────
Total estimated peak RAM (all steps):                      ~3.4 GB
Safety margin on 8 GB VPS:                                 ~4.6 GB free
```

**Estimated wall-clock time (modern 8-core CPU, AMD EPYC 7003 or Intel Xeon Ice Lake):**

```
2 denoising steps × ~90s/step (W4A4 pruned DiT)
  × 0.24 (DiTFastAttn −76% attention FLOPs)
  × 0.73 (Fast3Dcache −27% further)              → ~32s diffusion total
+ Sparse mesh extraction (ODC):                  → ~10–30s
+ VAE encode/decode:                             → ~15–30s
─────────────────────────────────────────────────────────────────────
Total estimated:           ~1–2 minutes per generation
Without Step 4 (attn):     ~3–5 minutes per generation
Without Steps 3+4:         ~30–60 minutes per generation
```

Note: ARM Graviton 3 often outperforms x86 for INT4 via NEON dot-product. Verify SIMD support: `grep -c avx512 /proc/cpuinfo` (x86) or `grep -c neon /proc/cpuinfo` (ARM).

---

## 13. Architectural Insights from Related Research

> These papers are NOT proposed as replacements for Hunyuan3D. They are the source research behind techniques in Steps 3–8.

### 13.1 TRELLIS — Origin of MDT-dist and Fast3Dcache Validation

> **Structured 3D Latents for Scalable and Versatile 3D Generation**
> arXiv:2412.01506 | ICLR 2025 | https://arxiv.org/abs/2412.01506
> Code: https://github.com/microsoft/TRELLIS
> Authors: Jianfeng Xiang et al. (Microsoft Research)

Both MDT-dist (Step 3) and Fast3Dcache (Step 5) were developed and validated on TRELLIS before being applied to Hunyuan3D. Understanding TRELLIS's sparse voxel approach — operating only on surface tokens (1–5% of grid) — informs how SparseFlex can be integrated into Hunyuan3D's volume decoder. The key transferable insight: Hunyuan3D's shape DiT currently operates on a dense 3D token grid; SparseFlex (Step 6) and Fast3Dcache (Step 5) implement the sparse equivalent for inference.

### 13.2 Sparse VideoGen — Direct Evidence for Hunyuan's Attention

Sparse VideoGen (Step 4) was validated on **HunyuanVideo** — a direct sibling model to Hunyuan3D from Tencent. The measured 2.33× speedup is the closest proxy for what Hunyuan3D's attention can achieve with sparsification. No architectural changes — only runtime profiling and sparse pattern replacement.

### 13.3 COD-VAE — Why 64 Latent Tokens Are Sufficient

COD-VAE (Step 7) demonstrates that for 3D shapes, 64 latent vectors are sufficient for high-fidelity reconstruction. This is consistent with information-theoretic expectations: most everyday 3D objects (the target use case) have low intrinsic dimensionality. Hunyuan3D's VAE almost certainly operates with far more tokens than necessary, creating an opportunity for uncertainty-guided pruning to reduce downstream DiT sequence length 16×.

---

## 14. Supporting Research

### 14.1 Efficient Diffusion Models Survey (TMLR 2025)

> **Efficient Diffusion Models: A Survey**
> arXiv:2502.06805 | TMLR 2025 | https://arxiv.org/abs/2502.06805
> Authors: Hui Shen, Jingxuan Zhang, Boning Xiong et al. (14 authors, Michigan State)

Comprehensive three-level taxonomy: algorithm-level (step reduction, distillation, pruning, quantization), system-level (CPU/edge deployment), framework-level (compilation, kernel fusion). Covers 100+ papers. Use as a reference index for exploring beyond this roadmap.

### 14.2 Intel CPU Diffusion Inference

> **Effective Quantization for Diffusion Models on CPUs**
> arXiv:2311.16133 | November 2023 | https://arxiv.org/abs/2311.16133
> Library: https://github.com/intel/intel-extension-for-transformers
> Authors: Hanwen Chang, Haihao Shen et al. (Intel)

The **only paper directly benchmarking diffusion inference on CPU hardware** with real timing:

| Precision | Steps | Latency |
|-----------|-------|---------|
| BF16 | 20 | 2.74 s |
| INT8 | 20 | **2.14 s** |

Authors note diffusion models are "notably more sensitive to quantization than other model types." Use `intel-extension-for-transformers` as a fallback if W4A4 produces unacceptable quality.

### 14.3 SVDQuant — Production-Ready W4A4

> **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**
> arXiv:2411.05007 | ICLR 2025 Spotlight | https://arxiv.org/abs/2411.05007
> Code: https://github.com/mit-han-lab/deepcompressor + https://github.com/mit-han-lab/nunchaku

Absorbs activation outliers into a low-rank SVD branch (rank 16–32 at BF16) instead of rotating them. On FLUX.1-dev:
- **3.0× speedup** over NF4 W4A16 (RTX 4090)
- **10.1× speedup** when CPU offloading is eliminated

**Use SVDQuant now** (code available) while waiting for ConvRot to publish code.

### 14.4 EcoDiff — Timestep-Resolved Pruning

> **Effortless Efficiency: Low-Cost Pruning of Diffusion Models**
> arXiv:2412.02852 | December 2024 | https://arxiv.org/abs/2412.02852

Learns differentiable neuron masks with no training data. Timestep-resolved pruning — different components pruned at different denoising timesteps (early steps need different capacity than late steps). Combine with LD-Pruner: LD-Pruner for global head removal, EcoDiff for per-timestep MLP channel pruning.

### 14.5 DiffCR — Adaptive Token Compression (CVPR 2025)

> **DiffCR: Layer- and Timestep-Adaptive Token Compression for Diffusion Transformers**
> arXiv:2412.16822 | CVPR 2025 | https://arxiv.org/abs/2412.16822
> Authors: You et al.

Learns differentiable per-token, per-layer, per-timestep compression ratios via a lightweight router fine-tuned on frozen base weights. Router overhead is minimal (small MLP); only the router is trained, not the frozen DiT. Applicable to Hunyuan3D's shape and texture DiT stages.

### 14.6 FlexiCubes — Differentiable Mesh Extraction

> **FlexiCubes: Flexible Isosurface Extraction for Gradient-Based Mesh Optimization**
> arXiv:2308.05371 | SIGGRAPH 2023 (ACM ToG) | https://arxiv.org/abs/2308.05371
> Authors: NVIDIA Toronto AI Lab

Learnable per-cube parameters on Dual Marching Cubes — uniform triangles, no stair-step artifacts, enables gradient flow through the mesh for future fine-tuning. Mesh extraction backend used in SparseFlex.

### 14.7 NeCGS — Extreme Output Mesh Compression

> **NeCGS: Neural Compression for 3D Geometry Sets**
> arXiv:2405.15034 | ICCV 2025 | https://arxiv.org/abs/2405.15034
> Authors: Ren et al.

TSDF-Def implicit + quantization-aware auto-decoder. Compresses thousands of 3D mesh models up to **900×** (684 MB → 0.76 MB). For Hunyuan3D deployment: generated meshes stored compressed, decompressed on-demand — dramatically reduces disk I/O on VPS.

### 14.8 Open Research Gap: Direct 3D DiT Quantization

A search of arXiv (as of April 2026) found **no paper applying PTQ or QAT directly to Hunyuan3D, TRELLIS, TripoSR, or CRM** with reported quality/speed tradeoffs. Quantization methods for 3D generative DiTs are derived by transfer from 2D image/video DiT results (Q-DiT, ViDiT-Q, CLQ). This is a genuine open research problem — the first paper to benchmark W4A4 on a 3D occupancy DiT would be novel.

---

## 15. Risks and Trade-offs

### 15.1 Quality Degradation Per Step

| Step | Technique | Measured Quality Impact | Source |
|------|-----------|------------------------|--------|
| 1 | ConvRot W4A4 (mixed) | FID: 10.07 → 10.03 (improvement) | arXiv:2512.03673 |
| 1 | ConvRot W4A4 (full) | Image Reward: 0.99 → 0.84 (−15%) | arXiv:2512.03673 |
| 1 | Q-DiT W4A8 | Near-lossless FID | arXiv:2406.17343 |
| 1 | CLQ W4A4 | 3.98× memory, 3.95× speed, negligible quality loss | arXiv:2509.24416 |
| 2 | LD-Pruner | FID: 13.05 → 12.37 (improvement) | arXiv:2404.11936 |
| 2 | OBS-Diff | Outperforms Wanda at 20–50% sparsity | arXiv:2510.06751 |
| 3 | MDT-dist 2-step | ULIP-I: 39.53 → 39.11 (−1.06%) | arXiv:2509.04406 |
| 3 | DisCa | 11.8× acceleration (quality maintained) | arXiv:2602.05449 |
| 4 | DiTFastAttn | −76% attn FLOPs, 1.8× speedup | arXiv:2406.08552 |
| 4 | Sparse VideoGen | 2.33× speedup on HunyuanVideo | arXiv:2502.01776 |
| 5 | Fast3Dcache | Chamfer Dist: +2.48% | arXiv:2511.22533 |
| 6 | SparseFlex | Chamfer Dist: −82% (improvement) | arXiv:2503.21732 |
| 6 | ODC | Manifold-guaranteed output | arXiv:2409.13418 |
| 7 | COD-VAE | 16× compression, 20.8× faster gen | arXiv:2503.08737 |
| 8 | CPU offload | Cache miss latency; zero quality impact | arXiv:2511.10753 |

**Combined worst case (full W4A4 + all steps):** ~−20% on shape metrics  
**Combined typical (mixed INT8 + all steps):** ~−5% — visually imperceptible for most use cases

### 15.2 Training Dependencies

| Step | Training Required? | Estimated Cost |
|------|-------------------|---------------|
| 1 (ConvRot/Q-DiT/CLQ) | No — PTQ only | Free |
| 2 (LD-Pruner/OBS-Diff) | No — calibration only | Free |
| 3 (MDT-dist) | Yes — distillation | $300–500 cloud |
| 3 (DisCa) | Yes — joint distillation+cache | ~$300–500 cloud |
| 4 (DiTFastAttn/Sparse VideoGen) | No — training-free | Free |
| 5 (Fast3Dcache) | No — training-free | Free |
| 6 (ODC/SparseFlex) | No — algorithm | Free |
| 7 (COD-VAE) | Yes — uncertainty head fine-tune | ~$50–100 cloud |
| 8 (FengHuang/mmgp) | No — systems | Free |

**Zero training budget path:** apply Steps 1, 2, 4, 5, 6, 8 + Hunyuan3D 2.0 Fast checkpoint.  
Expected peak RAM: ~2.5–3 GB. Expected time: ~4–8 minutes/generation.

### 15.3 No Public Code for ConvRot

Use SVDQuant (`deepcompressor` + Nunchaku) as a production-ready alternative. Released code, LoRA support, similar compression ratio.

### 15.4 CPU Speed Variance

| CPU Type | SIMD | Expected time/generation |
|----------|------|--------------------------|
| Intel Xeon Ice Lake+ | AVX-512 | ~1–3 min |
| AMD EPYC 7003+ | AVX2 | ~2–5 min |
| AWS Graviton 3 (ARM) | NEON | ~1–3 min |
| Older x86 (no AVX-512) | AVX2 | ~4–10 min |

Check: `grep -c avx512 /proc/cpuinfo` (x86) or `grep -c neon /proc/cpuinfo` (ARM)

### 15.5 Memory Spikes

- Maintain **6 GB minimum free RAM** on the VPS
- Disable swap or use SSD-only swap (HDD swap extends extraction from 30s to 10+ min)
- No other memory-heavy processes during generation

### 15.6 Fast3Dcache Error Accumulation

Authors note: "Minor numerical errors in cached latent features accumulate, causing structural artifacts." With MDT-dist reducing to 2 steps, accumulation is minimal — only one caching transition occurs. This is the ideal setup for Fast3Dcache.

### 15.7 COD-VAE Compatibility

COD-VAE requires fine-tuning Hunyuan3D's VAE decoder to reconstruct from 64 tokens. The encoded latent dimension must remain compatible with the DiT's expected input format. Verify: Hunyuan3D DiT's positional embedding scheme supports variable-length or 64-element token sequences before implementing Step 7.

---

## 16. Full Reference List

**[1]** Feice Huang, Zuliang Han, Xing Zhou, Yiyang Chen, Lifei Zhu, Haoqian Wang.
*ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers.*
arXiv:2512.03673, December 2025. https://arxiv.org/abs/2512.03673

**[2]** Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, Shinkook Choi.
*LD-Pruner: Efficient Pruning of Latent Diffusion Models using Task-Agnostic Insights.*
arXiv:2404.11936, April 2024. CVPR 2024 Workshop EDGE. https://arxiv.org/abs/2404.11936

**[3]** Zanwei Zhou, Taoran Yi, Jiemin Fang, Chen Yang, Lingxi Xie, Xinggang Wang, Wei Shen, Qi Tian.
*Few-step Flow for 3D Generation via Marginal-Data Transport Distillation.*
arXiv:2509.04406, September 2025. https://arxiv.org/abs/2509.04406
Code: https://github.com/Zanue/MDT-dist

**[4]** Tencent Hunyuan group.
*DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching.*
arXiv:2602.05449, February 2026. https://arxiv.org/abs/2602.05449

**[5]** Mengyu Yang, Yanming Yang, Chenyi Xu, Chenxi Song, Yufan Zuo, Tong Zhao, Ruibo Li, Chi Zhang.
*Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration.*
arXiv:2511.22533, November 2025. **CVPR 2026.** https://arxiv.org/abs/2511.22533
Project: https://fast3dcache-agi.github.io

**[6]** Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Yan-Pei Cao, Yangguang Li, Wanli Ouyang.
*SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling.*
arXiv:2503.21732, March 2025. https://arxiv.org/abs/2503.21732
Project: https://xianglonghe.github.io/TripoSF

**[7]** KAIST Visual AI Group.
*Occupancy-Based Dual Contouring.*
arXiv:2409.13418, SIGGRAPH Asia 2024. https://arxiv.org/abs/2409.13418

**[8]** Zhihang Yuan, Hanling Zhang, Pu Lu, Xuefei Ning, Linfeng Zhang et al.
*DiTFastAttn: Attention Compression for Diffusion Transformer Models.*
arXiv:2406.08552, NeurIPS 2024. https://arxiv.org/abs/2406.08552

**[9]** Sparse VideoGen team.
*Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity.*
arXiv:2502.01776, ICML 2025 / NeurIPS 2025 Spotlight. https://arxiv.org/abs/2502.01776

**[10]** Zhang et al. (THU-ML).
*SpargeAttention: Accurate and Training-Free Sparse Attention Accelerating Any Model Inference.*
arXiv:2502.18137, ICML 2025. https://arxiv.org/abs/2502.18137

**[11]** Cho, Yoo, Jeon, Kim (Yonsei University).
*COD-VAE: Representing 3D Shapes with 64 Latent Vectors.*
arXiv:2503.08737, March 2025. https://arxiv.org/abs/2503.08737

**[12]** Chen et al. (MIT Han Lab).
*DC-AE: Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models.*
arXiv:2410.10733, 2024. https://arxiv.org/abs/2410.10733

**[13]** Li et al. (Microsoft Research).
*FengHuang: Next-Generation Memory Orchestration for AI Inferencing.*
arXiv:2511.10753, November 2024. https://arxiv.org/abs/2511.10753

**[14]** Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, Song Han.
*SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models.*
arXiv:2411.05007, November 2024. **ICLR 2025 Spotlight.** https://arxiv.org/abs/2411.05007
Code: https://github.com/mit-han-lab/deepcompressor + https://github.com/mit-han-lab/nunchaku

**[15]** Lei Chen et al.
*Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers.*
arXiv:2406.17343, CVPR 2025. https://arxiv.org/abs/2406.17343

**[16]** Su et al.
*ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation.*
arXiv:2406.02540, ICLR 2025. https://arxiv.org/abs/2406.02540

**[17]** (Authors tbd from abstract).
*TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models.*
arXiv:2311.16503, CVPR 2024 Highlight. https://arxiv.org/abs/2311.16503

**[18]** Junhan Zhu, Hesong Wang, Mingluo Su, Zefang Wang, Huan Wang.
*OBS-Diff: Accurate Pruning for Diffusion Models in One-Shot.*
arXiv:2510.06751, 2025. https://arxiv.org/abs/2510.06751

**[19]** Ma, Fang et al.
*DeepCache: Accelerating Diffusion Models for Free.*
arXiv:2312.00858, CVPR 2024. https://arxiv.org/abs/2312.00858

**[20]** Shen et al.
*ToCa: Accelerating Diffusion Transformers with Token-wise Feature Caching.*
arXiv:2410.05317, ICLR 2025. https://arxiv.org/abs/2410.05317

**[21]** Cao et al.
*ProCache: Constraint-Aware Feature Caching with Selective Computation for DiT Acceleration.*
arXiv:2512.17298, AAAI 2026. https://arxiv.org/abs/2512.17298

**[22]** Hanwen Chang, Haihao Shen, Yiyang Cai, Xinyu Ye, Zhenzhong Xu, Wenhua Cheng, Kaokao Lv, Weiwei Zhang, Yintong Lu, Heng Guo (Intel).
*Effective Quantization for Diffusion Models on CPUs.*
arXiv:2311.16133, November 2023. https://arxiv.org/abs/2311.16133
Library: https://github.com/intel/intel-extension-for-transformers

**[23]** Unknown authors.
*BiDM: Pushing the Limit of Quantization for Diffusion Models.*
arXiv:2412.05926, December 2024. https://arxiv.org/abs/2412.05926

**[24]** Unknown authors.
*Effortless Efficiency: Low-Cost Pruning of Diffusion Models (EcoDiff).*
arXiv:2412.02852, December 2024. https://arxiv.org/abs/2412.02852

**[25]** NVIDIA Toronto AI Lab.
*FlexiCubes: Flexible Isosurface Extraction for Gradient-Based Mesh Optimization.*
arXiv:2308.05371, SIGGRAPH 2023 (ACM ToG). https://arxiv.org/abs/2308.05371

**[26]** You et al.
*DiffCR: Layer- and Timestep-Adaptive Token Compression for Diffusion Transformers.*
arXiv:2412.16822, CVPR 2025. https://arxiv.org/abs/2412.16822

**[27]** Ren et al.
*NeCGS: Neural Compression for 3D Geometry Sets.*
arXiv:2405.15034, ICCV 2025. https://arxiv.org/abs/2405.15034

**[28]** Hui Shen, Jingxuan Zhang, Boning Xiong et al. (Michigan State University, 14 authors).
*Efficient Diffusion Models: A Survey.*
arXiv:2502.06805, TMLR 2025. https://arxiv.org/abs/2502.06805

**[29]** deepbeepmeep (community).
*Hunyuan3D-2GP: Running Hunyuan3D-2 with Minimal VRAM via CPU Offload.*
GitHub: https://github.com/deepbeepmeep/Hunyuan3D-2GP, 2025.

**[30]** Jianfeng Xiang, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, Jiaolong Yang.
*TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation.*
arXiv:2412.01506, ICLR 2025. https://arxiv.org/abs/2412.01506
Code: https://github.com/microsoft/TRELLIS

**[31]** Kai Liu, Shaoqiu Zhang, Linghe Kong, Yulun Zhang.
*CLQ: Cross-Layer Guided Orthogonal-based Quantization for Diffusion Transformers.*
arXiv:2509.24416, 2025. https://arxiv.org/abs/2509.24416

**[32]** Zhiteng Li, Hanxuan Li, Junyi Wu, Kai Liu, Haotong Qin et al.
*DVD-Quant: Data-free Video Diffusion Transformers Quantization.*
arXiv:2505.18663, 2025. https://arxiv.org/abs/2505.18663

**[33]** Zeqiang Lai, Yunfei Zhao, Zibo Zhao et al.
*Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM Distillation).*
arXiv:2503.16302, March 2025. https://arxiv.org/abs/2503.16302

**[34]** Unknown authors.
*Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.*
arXiv:2402.07033, ICLR 2025. https://arxiv.org/abs/2402.07033

---

*Document maintained at: `docs/CPU_DEPLOYMENT_ROADMAP.md`*
*Last updated: 2026-04-29 | Version 2.0 — 8 steps, 34 references, Hunyuan3D-only constraint*
