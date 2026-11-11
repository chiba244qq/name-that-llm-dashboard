# V53 Fix Submission & Colab Sparse-k Evaluation Architecture Report

## 1. Executive Summary

| Item | Details |
| :--- | :--- |
| **Active Submission** | **NTL 0683 V53 Apex Fusion Fix** (Ref: `58147368`) |
| **Kernel ID** | `tsuyoshichiba/ntl-0683-v53-apex-fusion` (Version 7) |
| **Status** | `SubmissionStatus.PENDING` (Estimated inference: ~6-8 hours) |
| **Key Architecture Fix** | 100% path flattening (`/kaggle/input/<slug>`), resolved 12 mount path discrepancies, commit-time assertion passed |
| **SOTA Lift Integrated** | `Native Qwen3-Embedding-8B` geometric mixture share elevated from `0.40` to **`0.50`** (matching LB `0.681` benchmark) |
| **Colab Verification Notebook** | [`colab/ntl_v54_sparse_eval_and_hybrid_optimization.ipynb`](file:///Users/echo/kaggle/name-that-llm/colab/ntl_v54_sparse_eval_and_hybrid_optimization.ipynb) |

---

## 2. Root Cause Analysis of Previous Failure & The V53 Fix

### 2.1 The Kaggle Input Mounting Trap
When notebooks attach more than 20 dataset/kernel sources on Kaggle, the mounting filesystem flattens directory roots from `/kaggle/input/datasets/<owner>/<slug>` to `/kaggle/input/<slug>`. 
Previous V53 runs encountered `FileNotFoundError` during competition rerun on 12 hardcoded dataset roots (e.g., `ntl-handwriting-blend`, `nlm-l24-adapters-v5`, `ntl-two-stage-lora`, `ultrabert-380m-ntl`).

### 2.2 Dual Verification Protocols in Version 7
1. **Dynamic Name-Based Path Resolution (`resolve_p`)**:
   Every path in `_ROOTS` and `_PATHS` checks whether a flat or nested match exists via recursive directory lookup before failing.
2. **Commit-Time Full Input Assertion**:
   All 135 model and checkpoint paths are asserted in Cell 6 prior to entering the preview bypass, ensuring that missing input errors fail at commit time rather than after hours of submission rerun.
3. **Execution Verification**:
   Version 7 completed its fast commit at 10:41:38 JST and successfully issued Kaggle submission Ref `58147368`.

---

## 3. Colab Verification Engine: Sparse-k Monte Carlo (r=0.91)

Recent community benchmarking by GoPain and Alex Chilton revealed that standard offline OOF evaluation (all 18 responses per prompt) only achieves an $r \approx 0.13$ correlation with the public leaderboard. In contrast, evaluating Balanced Accuracy on **subsampled sets of $k=2$ passages per prompt (Sparse-2 OOF)** achieves an empirical correlation of **$r = 0.91$** across 15 submissions.

The generated Colab notebook ([ntl_v54_sparse_eval_and_hybrid_optimization.ipynb](file:///Users/echo/kaggle/name-that-llm/colab/ntl_v54_sparse_eval_and_hybrid_optimization.ipynb)) provides:
1. Automated download of official train sets and SOTA prefit weights (`ntl-olmo2-native24-final-inference-assets`, `ours_mbert`, `two_stage_encoders`).
2. Isolated GPU forward passes for ModernBERT-large and DeBERTa-v3.
3. 500-iteration Monte Carlo sampling engine simulating hidden test cluster sparsity.
4. Systematic grid search across:
   - Native embedding share ($w_{native} \in [0.35, 0.60]$)
   - Extra classifier weight ($w_{cls} \in [0.00, 0.28]$)
   - Sinkhorn target marginal shrinkage ($\rho \in [0.25, 0.45]$)
