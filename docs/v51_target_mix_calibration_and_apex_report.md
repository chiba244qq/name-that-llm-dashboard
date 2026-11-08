# NTL 0681 V51 Apex Architecture & Deployment Report

**Track 1: Sovereign Apex Evolution (Step 2)**  
- **Kernel**: `tsuyoshichiba/ntl-0681-v51-apex`  
- **Target Metric**: Balanced Accuracy $\to$ 目標 `0.674` 〜 `0.675`（Silver 確固化・Gold 圏接近）  
- **Parent Baseline**: V50-Apex (`0.673`、Rank 123 / 5,112 チーム、Silver メダル圏突入）

---

## 1. 施策の背景と数理的根拠

### (1) V50（0.673）による Silver 圏到達
V50 では **ModernBERT-large（512 tokens / SDPA, $w=1.4$）** と **DeBERTa-v3（384 tokens, $w=0.8$）** のデュアル双方向エンコーダー化、および **Pre-Sinkhorn 兄弟モデル・フォレンジック残差（$max\_alpha=0.06$）** を統合し、LB `0.673`（Rank 123）を達成しました。

### (2) Step 2: Sinkhorn Target Mix 最適化（$0.35 \to 0.45$）
Sinkhorn-Knopp アルゴリズムによるマージナル平滑化において、各サンプルの予測分布を公式 train 事前分布（均等一様分布 $u = 1/18$）へブレンドする強度 $\rho$（`sinkhorn_target_mix`）を $0.35$ から $0.45$ に調整しました。

$$\mathbf{t} = (1 - \rho) \hat{\mathbf{p}}_{\text{pred}} + \rho \, \mathbf{u}_{\text{train}}$$

- **利点**:
  - テストデータ内の偏向サンプルや長文外れ値に対して、一様マージナル制約の過剰な引き締め（Over-constraining）を緩和。
  - 公開・非公開テストセット間のクラス比率変動に対する頑健性が向上。
- **リスクと制約事項**:
  - 一様分布への引き寄せを強めるため、高確信度サンプルのマージンがごく僅かに平滑化される可能性があります。

---

## 2. アーキテクチャ構成表

| コンポーネント | 設定 / 重み | 役割 / 特性 |
| :--- | :--- | :--- |
| **Encoder 1** | ModernBERT-large ($w=1.4$) | 512 tokens 長文コンテキスト、SDPA 高速推論 |
| **Encoder 2** | DeBERTa-v3-large ($w=0.8$) | 384 tokens、語彙・構文特徴分類 |
| **Dual-SVC** | Pre-fit ($w=6.0$) + Open-Pool ($w=4.0$) | 表層 N-gram & タイポグラフィ特徴（適正比率） |
| **Residual** | Pre-Sinkhorn Sibling Residual ($max\_alpha=0.06$) | 兄弟モデル間（Gemini Flash, GPT-5.6）の局所整流 |
| **ATLAS** | $w=0.20$ (Margin 0.32) | ソース条件付き対数尤度 |
| **Sinkhorn** | Strength $0.30$ / **Target Mix 0.45** | マージナル平滑化・最適輸送制約緩和 |
| **Verifier** | Pairwise Verifier ($w=0.10$) | 最終タイブレーク検証 |

---

## 3. 実行ステータス
- **Submission Ref**: `58126401`（`SubmissionStatus.PENDING`、本番推論中）
- **投入時刻**: 2026-11-08 20:50 JST
- **Kernel ID**: `tsuyoshichiba/ntl-0681-v51-apex`
- **監視プロセス**: PID `47856`（`submit_and_monitor_v51_apex.py`）
- **ログ**: `/Users/echo/kaggle/name-that-llm/submission_0681_v51_apex/monitor_v51_apex.log`
