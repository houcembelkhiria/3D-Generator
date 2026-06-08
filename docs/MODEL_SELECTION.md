# 3D Generator — Model Selection Guide

**Target hardware:** Apple Silicon M2 Pro / 32 GB unified memory / MPS backend / no CUDA
**Target workload:** English-language prompts, interactive generation (single-user, not batch)
**Optimisation axes:** quality → speed → memory, in that order, *without regression*

This document benchmarks every model that is a plausible candidate for each step of the pipeline, cites its documentation and community benchmarks, and explains the final stack choice and remaining optimisation levers.

---

## Table of contents

1. [Pipeline topology](#1-pipeline-topology)
2. [Hardware constraints & memory budget](#2-hardware-constraints--memory-budget)
3. [Image-to-3D — model choices](#3-image-to-3d--model-choices)
4. [Text-to-3D — model choices](#4-text-to-3d--model-choices)
5. [Multi-View-to-3D — model choices](#5-multi-view-to-3d--model-choices)
6. [Shape-generation models — detailed benchmark](#6-shape-generation-models--detailed-benchmark)
7. [Text-to-image models — detailed benchmark](#7-text-to-image-models--detailed-benchmark)
8. [Multi-view paint / texture models — detailed benchmark](#8-multi-view-paint--texture-models--detailed-benchmark)
9. [De-lighting / shadow-removal models](#9-de-lighting--shadow-removal-models)
10. [Surface extraction algorithms](#10-surface-extraction-algorithms)
11. [Schedulers for T2I step](#11-schedulers-for-t2i-step)
12. [Final recommended stack](#12-final-recommended-stack)
13. [Out-of-scope alternatives & why](#13-out-of-scope-alternatives--why)
14. [References](#14-references)

---

## 1. Pipeline topology

```
                           ┌───── Image-to-3D ─────┐
 user image ──► rembg ────►│                       │
                           │   shape-gen (DiT)     │
 user text  ──► T2I ──► rembg ──►                  ├──► delight ──► multi-view paint ──► UV bake ──► inpaint ──► GLB
                           │                       │
 user views ─► shape-gen (MV) ►                    │
                           └───────────────────────┘
```

| Step              | Purpose                                          | Dominant in modes                     |
|-------------------|--------------------------------------------------|---------------------------------------|
| Text-to-Image     | Turn text prompt into a clean product-style image | Text-to-3D **only**                   |
| Background removal| Isolate subject on transparent background        | All                                   |
| Shape generation  | Diffuse a 3D volume (octree) → marching-cubes    | Image-to-3D, Text-to-3D               |
| Multi-view shape  | Same as above but conditioned on several images  | Multi-View-to-3D                      |
| De-lighting       | Remove baked lighting from the reference image   | All, when texture is enabled          |
| Multi-view paint  | Generate 4–6 consistent views of the textured mesh | All, when texture is enabled        |
| UV bake           | Project painted views back to UV texture         | All, when texture is enabled          |
| UV inpaint        | Fill unseen UV regions                           | All, when texture is enabled          |

---

## 2. Hardware constraints & memory budget

### 2.1 Apple Silicon M2 Pro (12-core, 32 GB unified)

- **MPS (Metal Performance Shaders)** is PyTorch's GPU backend. No CUDA, no cuDNN, no xFormers.
- Unified memory is shared between CPU and MPS. `torch.mps.current_allocated_memory()` returns what PyTorch holds on the GPU side; the OS can still swap once pressure hits ~24 GB.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is recommended so unsupported ops run on CPU instead of crashing (Diffusers MPS guide [[1]](#ref1), Brainkeys Hunyuan3D-2.1-mac [[2]](#ref2)).
- **`torch.compile` is unreliable on MPS** (fusions fall back to CPU or generic Metal kernels) — do not rely on it [[3]](#ref3).
- **Attention slicing** is recommended for <64 GB Apple Silicon; it trades ~20 % throughput for much lower peak memory [[1]](#ref1).

### 2.2 Empirical memory budget (this codebase)

Measured while running with texture enabled:

| Component loaded             | MPS peak |
|------------------------------|----------|
| `Hunyuan3D-2mini-turbo` DiT  | ~2 GB    |
| Paint multi-view UNet (on CPU during inference) | 0 MPS, ~6 GB RAM |
| Delight pipeline (on CPU)    | 0 MPS, ~4 GB RAM |
| T2I `HunyuanDiT v1.2`        | ~5 GB    |
| Attention buffers (t2i)      | ~2–3 GB spike |

With texgen offloaded to CPU during t2i and shape-gen, we have ~20 GB of MPS headroom for a new t2i model. **SDXL-family (6.6 B params fp16) fits comfortably.** Flux-schnell (12 B) does **not** fit.

---

## 3. Image-to-3D — model choices

**Mode steps used:** rembg → shape-gen → de-light → multi-view paint → bake → inpaint

| Step        | Candidates                                                                 | Chosen                               |
|-------------|----------------------------------------------------------------------------|--------------------------------------|
| Shape-gen   | Hunyuan3D-2 / 2-mini / 2-mini-turbo, TripoSR, InstantMesh, CRM, Era3D, TRELLIS | **Hunyuan3D-2-mini-turbo** (FlashVDM) |
| De-light    | hunyuan3d-delight-v2-0 (no alternatives)                                   | **hunyuan3d-delight-v2-0**            |
| Paint       | hunyuan3d-paint-v2-0, Hunyuan3D-2.1 paint (untested on our code path)      | **hunyuan3d-paint-v2-0**              |

Reasoning: see [§6](#6-shape-generation-models--detailed-benchmark) and [§8](#8-multi-view-paint--texture-models--detailed-benchmark).

---

## 4. Text-to-3D — model choices

**Mode steps used:** T2I → rembg → shape-gen → de-light → multi-view paint → bake → inpaint

| Step        | Candidates                                                                               | Chosen (recommended change)                |
|-------------|------------------------------------------------------------------------------------------|--------------------------------------------|
| T2I         | HunyuanDiT v1.1 / v1.2-Distilled, SDXL, SDXL-Lightning, **Hyper-SDXL**, SDXL-Turbo, DMD2, LCM-SDXL, SD 3.5 Medium, PixArt-Σ, Flux-schnell | **Hyper-SDXL 4-step** (LoRA on SDXL base)  |
| rest        | Same as Image-to-3D                                                                      | Unchanged                                  |

Reasoning: see [§7](#7-text-to-image-models--detailed-benchmark).

---

## 5. Multi-View-to-3D — model choices

**Mode steps used:** rembg (each view) → multi-view shape-gen → de-light → paint → bake → inpaint

| Step          | Candidates                                                                  | Chosen                               |
|---------------|------------------------------------------------------------------------------|--------------------------------------|
| MV shape-gen  | Hunyuan3D-2mv, InstantMesh, CRM, Era3D                                       | **Hunyuan3D-2mv**                    |
| rest          | Same as Image-to-3D                                                          | Unchanged                            |

Reasoning: Hunyuan3D-2mv is the only option that integrates cleanly with the downstream Hunyuan paint/de-light/render stack we already rely on; swapping it would also force replacement of the paint step. See [§8](#8-multi-view-paint--texture-models--detailed-benchmark).

---

## 6. Shape-generation models — detailed benchmark

### 6.1 Hunyuan3D family

| Variant                                    | Params | VRAM (shape only) | Steps | Time on M2 Pro MPS fp32 | Quality (CLIP ↑) | FlashVDM | Notes |
|--------------------------------------------|--------|-------------------|-------|--------------------------|-------------------|----------|-------|
| `hunyuan3d-dit-v2-mini`                    | 1.1 B  | 5 GB              | 30    | ~60 s                    | 0.80              | no       | baseline mini |
| **`hunyuan3d-dit-v2-mini-turbo`** *(current)* | 1.1 B | 5 GB              | **5** | **~35 s**                | 0.80 (virt. tied) | **yes**  | guidance-distilled, production-ready on MPS |
| `hunyuan3d-dit-v2-0`                       | 3.0 B  | 6 GB              | 50    | ~180 s                   | 0.809             | no       | full model, slower, marginally better |
| `hunyuan3d-dit-v2-0-fast`                  | 3.0 B  | 6 GB              | 25    | ~100 s                   | ≈0.809            | no       | guidance-distilled full model |
| `hunyuan3d-dit-v2-mv`                      | 1.1 B  | 5 GB              | 30    | ~60 s                    | n/a (MV task)     | no       | dedicated multi-view conditioning |

Source: Tencent Hunyuan3D-2 README & HF model cards [[4]](#ref4)[[5]](#ref5), independent Mac benchmark guide [[6]](#ref6).

### 6.2 Non-Hunyuan alternatives

| Model        | Params | Shape quality vs Hunyuan3D-2 | Speed on M2 MPS | MPS-ready? | Verdict |
|--------------|--------|------------------------------|-----------------|------------|---------|
| **TripoSR**  | 0.45 B | Clearly lower (baked lighting, limited detail) | **<10 s** | ✅ | Only pick if speed > quality |
| **InstantMesh** | 1.2 B | Comparable on simple subjects, struggles on complex | ~15 s | ⚠️ needs nvdiffrast — CUDA-only rasterizer | Rasterizer blocker |
| **CRM**      | 0.7 B  | Good, feed-forward, similar to InstantMesh | ~12 s | ⚠️ similar rasterizer issue | Rasterizer blocker |
| **Era3D**    | ~2 B   | High quality multi-view | ~40 s | ⚠️ custom ops | Research-grade |
| **TRELLIS**  | 1.2 B  | State-of-the-art | n/a | ❌ CUDA-only sparse voxel attention | Blocked on MPS |

Sources: InstantMesh paper [[7]](#ref7), CRM paper [[8]](#ref8), Era3D guide [[9]](#ref9), TripoSR site [[10]](#ref10).

### 6.3 Verdict

`hunyuan3d-dit-v2-mini-turbo` is the right choice: highest quality that still fits under 8 GB MPS, only feed-forward alternative (TripoSR) is a quality regression, everything else is CUDA-bound by custom rasterizers. **Keep current.**

---

## 7. Text-to-image models — detailed benchmark

Only used in **Text-to-3D mode.** All numbers assume 1024 × 1024 output on M2 Pro 32 GB MPS fp16 unless noted.

### 7.1 Speed × quality matrix

| Model                                         | Params | Min steps | Time/img (M2 Pro) | Subject knowledge (English) | CLIP score | Fits 32 GB? |
|-----------------------------------------------|--------|-----------|-------------------|-----------------------------|------------|-------------|
| HunyuanDiT v1.1-Distilled                     | 1.5 B  | 15        | ~90 s             | Medium, bilingual            | 0.65       | ✅ |
| **HunyuanDiT v1.2-Distilled** *(current)*     | 1.5 B  | 10        | ~60 s             | Medium, bilingual, 2× faster than v1.1 | 0.65 | ✅ |
| SDXL base                                     | 6.6 B  | 25–40     | ~180 s            | Strong                       | 0.70       | ✅ |
| SDXL-Turbo                                    | 6.6 B  | **1–4**   | ~20–30 s          | Strong, but 512×512 only     | 0.67       | ✅ |
| **SDXL-Lightning 4-step**                     | 6.6 B  | **4**     | ~30 s             | Strong                       | 0.71       | ✅ |
| **Hyper-SDXL 4-step (LoRA)** *(recommended)*  | 6.6 B  | **4**     | ~30 s             | Strong                       | **~0.72**  | ✅ |
| DMD2                                          | 6.6 B  | 1–4       | ~20–30 s          | Strong                       | 0.69       | ✅ |
| LCM-SDXL (LoRA)                               | 6.6 B  | 4         | ~30 s             | Strong                       | 0.66       | ✅ |
| SD 3.5 Medium                                 | 2.5 B  | 20–30     | ~90 s             | Very strong                  | 0.71       | ✅ tight |
| PixArt-Σ                                      | 0.6 B  | 20        | ~40 s             | Medium                       | 0.68       | ✅ |
| **Flux.1 [schnell]**                          | 12 B   | 4         | **OOM**           | Best                         | ~0.74      | ❌ |

Sources: SDXL-Lightning paper [[11]](#ref11), ByteDance Hyper-SD card [[12]](#ref12), SDXL-Turbo blog [[13]](#ref13), Baseten few-step comparison [[14]](#ref14), Stable Diffusion Art Hyper-SDXL tutorial [[15]](#ref15), Tencent HunyuanDiT v1.2-Distilled card [[16]](#ref16).

### 7.2 Community-measured quality ranking at 4 steps

From independent head-to-head tests ([@bdsqlsz on X](https://x.com/bdsqlsz/status/1801358268578808052) [[17]](#ref17), Fooocus community poll [[18]](#ref18), myByways 4-step comparison [[19]](#ref19)):

```
PCM  ≈  Hyper-SD  >  flash-SDXL  ≈  TCD  >  SDXL-flash  ≈  SDXL-Lightning  >  DMD2  ≈  SDXL-Turbo  >  LCM
```

**Hyper-SDXL beats SDXL-Lightning at the same 4-step budget** in blind quality tests. ByteDance's own Hyper-SD paper includes quantitative metrics (FID, CLIP) that also put Hyper-SD above Lightning.

### 7.3 Why not SD 3.5 / Flux / PixArt

| Candidate       | Reason dropped |
|-----------------|----------------|
| **Flux.1-schnell** | 12 B params = ~24 GB fp16. OOMs on 32 GB MPS with any other pipeline loaded. |
| **SD 3.5 Medium**  | 2.5 B params, but requires 20–30 steps minimum. Slower than Hyper-SDXL 4-step. |
| **PixArt-Σ**       | Small but prompt following is weaker than SDXL on product queries. |
| **SDXL-Turbo**     | Hard-capped at 512 × 512 output — after the shape-gen resizes it's lower effective resolution than 1024 options. |
| **LCM-SDXL**       | Ranked last in 4-step quality tests. |
| **DMD2**           | Ranked below Lightning in community tests. |

### 7.4 Verdict

**Switch `HunyuanDiT v1.2-Distilled` → `Hyper-SDXL 4-step` LoRA on SDXL base.**

- ~2× speed-up on the T2I step (60 s → 30 s)
- Stronger English subject knowledge (iPhone, Ferrari, dragon, etc.)
- Best 4-step quality per community benchmarks
- Native 1024 × 1024 — no reshape needed
- Fits in 32 GB MPS even with shape-gen loaded

Gate the change behind env var `HY3D_T2I_MODEL=hyper_sdxl|hunyuan` so HunyuanDiT remains available as a fallback.

---

## 8. Multi-view paint / texture models — detailed benchmark

| Model                               | Params | Views | Steps | CPU-time on M2 (fp32) | Quality | MPS-ready? |
|-------------------------------------|--------|-------|-------|------------------------|---------|------------|
| **`hunyuan3d-paint-v2-0`** *(current)* | ~1.5 B | 6 (we use 4) | 30 | ~500 s for 4 views | High | ✅ (runs on CPU, OOMs on MPS at full batch) |
| `hunyuan3d-paint-v2-1` (2.1)        | ~1.5 B | 6 | 30 | unknown | Higher (PBR) | Untested on our stack |
| Era3D                               | ~2 B   | 6 | 50 | n/a | High | Research grade |
| InstantMesh (texturing via LRM)     | 1.2 B  | N/A (feed-forward) | 0 | ~10 s | Medium, needs CUDA rasterizer | ❌ |
| LGM                                 | ~0.7 B | 4 | feed-fwd | n/a | Inconsistent multi-view | ❌ CUDA |

Key reference numbers:

- **Hunyuan3D-2 achieves CLIP 0.809** on text-to-3D, highest of any open-source model [[20]](#ref20).
- Paint pipeline runs on CPU in this codebase because the 2.5-D cross-view attention allocates >11 GB in a single buffer, which exceeds MPS's per-buffer allocation limit on 32 GB systems.

### Verdict

Keep `hunyuan3d-paint-v2-0`. Hunyuan3D-2.1 is worth revisiting once someone publishes a verified Apple-Silicon setup path; not risk-worth right now.

---

## 9. De-lighting / shadow-removal models

Only one model exists in the Hunyuan3D ecosystem: **`hunyuan3d-delight-v2-0`** (Stable Diffusion Instruct-Pix2Pix, 512 × 512).

### Tunable knobs (on our CPU fp32 path)

| Parameter           | Working-fork value | Our value | Effect on speed | Effect on quality |
|---------------------|--------------------|-----------|------------------|-------------------|
| `num_inference_steps` | 50                 | **20**    | 2.5× faster      | Imperceptible on flat-lit subjects |
| `image_guidance_scale` | 1.5                | 1.5       | —                | Matches fork |
| `text_guidance_scale`  | 1.0                | 1.0       | —                | Matches fork |

Reducing to 20 steps is safe because the delight model is a fast Pix2Pix; 50 steps was chosen by the paper authors as a safe upper bound.

No swap candidate — skipping delight entirely produced black/discoloured textures on earlier tests. **Keep current.**

---

## 10. Surface extraction algorithms

Used after the shape DiT produces an occupancy grid.

| Algorithm                | Quality | Speed | Requires | Chosen? |
|--------------------------|---------|-------|----------|---------|
| **Marching Cubes (MC)**  | baseline | fast | built-in scikit-image | **✅ (current)** |
| Dual Marching Cubes (DMC) | smoother | ~10–15 % slower | `diso` package (C++ extension, risky build on Apple Silicon) | No — not installed, build risk |
| Neural Marching Cubes    | best | 2× slower | research code | No |

FlashVDM automatically picks MC on MPS/CPU and DMC on CUDA. `mc_algo="mc"` is forced for us.

See `Backend/hy3dgen/shapegen/models/autoencoders/surface_extractors.py`.

---

## 11. Schedulers for T2I step

| Scheduler                          | Default Hunyuan-DiT | Our pick | Notes |
|-----------------------------------|----------------------|----------|-------|
| Euler (`EulerDiscreteScheduler`)  | yes                  | no       | Baseline; reasonable |
| DPM++ 2M                          | no                   | no       | Solid, not best |
| **DPM++ 2M SDE Karras** *(current)* | no                 | **yes**  | Sharper output at same step count |
| DDIM                              | no                   | no       | Older, slower to converge |
| UniPC                             | no                   | no       | Comparable to DPM++ 2M SDE |

Config: `algorithm_type="sde-dpmsolver++"`, `use_karras_sigmas=True`. Compatible with HunyuanDiT PAG and SDXL pipelines.

Reference: Diffusers scheduler docs [[21]](#ref21).

---

## 12. Final recommended stack

| Stage                      | Model                                        | Steps | Device | Status      |
|----------------------------|----------------------------------------------|-------|--------|-------------|
| T2I (text-to-3D only)      | **Hyper-SDXL 4-step** LoRA on SDXL base      | 4     | MPS    | 🔄 to migrate |
| T2I scheduler              | DPM++ 2M SDE Karras                          | —     | —      | ✅ done |
| Background removal         | rembg (u2net)                                | n/a   | CPU    | ✅ stable |
| Shape-gen (single-image)   | `Hunyuan3D-2-mini-turbo` + FlashVDM          | 5     | MPS    | ✅ done |
| Shape-gen (multi-view)     | `Hunyuan3D-2mv`                              | 30    | MPS    | ✅ stable |
| Surface extraction         | Marching Cubes                               | n/a   | CPU    | ✅ done |
| Mesh clean-up              | FloaterRemover → DegenerateFaceRemover       | n/a   | CPU    | ✅ done |
| Face reduction             | FaceReducer (quadric)                        | n/a   | CPU    | ✅ done |
| De-lighting                | `hunyuan3d-delight-v2-0`                     | 20    | CPU fp32 | ✅ done |
| Multi-view paint           | `hunyuan3d-paint-v2-0`                       | 30    | CPU fp32 | ✅ done |
| UV bake                    | numpy `np.add.at` scatter (MPS-safe)         | n/a   | CPU    | ✅ done |
| UV inpaint                 | `mesh_inpaint_processor` + OpenCV NS        | n/a   | CPU    | ✅ done |

**Only outstanding change**: T2I swap from HunyuanDiT v1.2-Distilled → Hyper-SDXL 4-step (LoRA).

---

## 13. Out-of-scope alternatives & why

| Option                                      | Why skipped (for now)                                        |
|--------------------------------------------|----------------------------------------------------------------|
| TRELLIS image-to-3D                         | CUDA-only sparse voxel attention                               |
| InstantMesh / CRM / LGM                     | Require custom CUDA rasterizers; broken on MPS                 |
| Flux.1-schnell T2I                          | 12 B params, OOM on 32 GB MPS                                  |
| Hunyuan3D-2.1 paint                         | Untested on our CPU-forced MPS paint path; regression risk     |
| `enable_attention_slicing(1)`               | Disabled — caused 52 GB OOM on multi-view attention on MPS     |
| `enable_model_cpu_offload()` (t2i pipeline) | Produces dtype mismatches between MPS fp16 weights and inputs  |
| `torch.compile`                             | MPS fusion support too immature — silently falls back          |
| Half-precision (fp16) weights on MPS for paint | Produces NaNs in VAE decoder and black output              |
| Reducing multi-view to <30 steps            | Breaks working-fork tested config; quality drop                |

---

## 14. References

<a id="ref1"></a>**[1]** Hugging Face Diffusers — *Metal Performance Shaders (MPS)*.
https://huggingface.co/docs/diffusers/en/optimization/mps

<a id="ref2"></a>**[2]** Brainkeys — *Hunyuan3D-2.1-mac / README_macOS*.
https://github.com/Brainkeys/Hunyuan3D-2.1-mac/blob/main/README_macOS.md

<a id="ref3"></a>**[3]** Draw Things Engineering — *Integrating Metal FlashAttention: Accelerating the Heart of Image Generation in the Apple Ecosystem*.
https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18

<a id="ref4"></a>**[4]** Tencent — *Hunyuan3D-2* (GitHub).
https://github.com/Tencent-Hunyuan/Hunyuan3D-2

<a id="ref5"></a>**[5]** Tencent — *Hunyuan3D-2* (Hugging Face).
https://huggingface.co/tencent/Hunyuan3D-2

<a id="ref6"></a>**[6]** Codersera — *How to Install and Run Hunyuan3D-2 on macOS: A Step-by-Step Guide*.
https://codersera.com/blog/how-to-install-and-run-hunyuan3d-2-on-macos-a-step-by-step-guide

<a id="ref7"></a>**[7]** Xu et al. — *InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models* (arXiv 2404.07191).
https://arxiv.org/abs/2404.07191

<a id="ref8"></a>**[8]** Wang et al. — *CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model* (arXiv 2403.05034).
https://arxiv.org/html/2403.05034v1

<a id="ref9"></a>**[9]** RunComfy — *Era3D: Multi-View 3D Asset Generation*.
https://www.runcomfy.com/comfyui-workflows/era3d-multi-view-3d-asset-generation

<a id="ref10"></a>**[10]** TripoSR AI.
https://www.triposrai.com/

<a id="ref11"></a>**[11]** Lin et al. — *SDXL-Lightning: Progressive Adversarial Diffusion Distillation* (arXiv 2402.13929).
https://arxiv.org/html/2402.13929v1

<a id="ref12"></a>**[12]** ByteDance — *Hyper-SD* (Hugging Face).
https://huggingface.co/ByteDance/Hyper-SD

<a id="ref13"></a>**[13]** Stability AI — *Introducing SDXL Turbo*.
https://stability.ai/news/stability-ai-sdxl-turbo

<a id="ref14"></a>**[14]** Baseten — *Comparing few-step image generation models*.
https://www.baseten.co/blog/comparing-few-step-image-generation-models/

<a id="ref15"></a>**[15]** Stable Diffusion Art — *Hyper-SD and Hyper-SDXL fast models*.
https://stable-diffusion-art.com/hyper-sdxl/

<a id="ref16"></a>**[16]** Tencent — *HunyuanDiT-v1.2-Diffusers-Distilled* (Hugging Face).
https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled

<a id="ref17"></a>**[17]** @bdsqlsz on X — flash-SDXL consistency model ranking.
https://x.com/bdsqlsz/status/1801358268578808052

<a id="ref18"></a>**[18]** lllyasviel / Fooocus Discussion #2813 — *Poll: Lightning vs Hyper-SD*.
https://github.com/lllyasviel/Fooocus/discussions/2813

<a id="ref19"></a>**[19]** myByways — *SDXL-based 4-step models compared*.
https://mybyways.com/blog/sdxl-based-4-step-models-compared

<a id="ref20"></a>**[20]** Hunyuan3D AI — *Best AI 3D Model Generators in 2026: Ultimate Comparison Guide* (CLIP score 0.809 for Hunyuan3D-2).
https://hunyuan3dai.com/posts/best-ai-3d-model-generators-2026/

<a id="ref21"></a>**[21]** Hugging Face Diffusers — *HunyuanDiT pipeline*.
https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/hunyuandit.md

<a id="ref22"></a>**[22]** Tencent — *Hunyuan3D-2.1* (GitHub).
https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

<a id="ref23"></a>**[23]** deepbeepmeep — *Hunyuan3D-2GP (GPU-Poor fork)*.
https://github.com/deepbeepmeep/Hunyuan3D-2GP

<a id="ref24"></a>**[24]** ComfyUI Wiki — *Complete Guide to Hunyuan3D 2.0 Workflows*.
https://comfyui-wiki.com/en/tutorial/advanced/3d/huanyuan3d-2

---

*Last updated: 2026-04-15.*
