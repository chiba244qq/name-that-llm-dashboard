# NTL 0682 V52 Sovereign Architecture & Deployment Report

**Track 1: Sovereign Apex Evolution (Step 1)**  
- **Kernel**: `tsuyoshichiba/ntl-0682-v52-sovereign` (Kaggle Kernel ID: `143433724`)  
- **Target Metric**: Balanced Accuracy $\to$ 目標 `0.680`（Rank 74 相当・Silver 上位確固化）  
- **Parent Baseline**: V50-Apex (`0.673`、Rank 151 / 5,124 チーム）

---

## 1. 施策の背景とアーキテクチャ特性

### (1) Public 0.680 ブレイクスルーの構造
CHEN Han（`concyclics/0-679`）および `lightningv08` により確立された SOTA パイプライン。
- **Native Qwen3-Embedding-8B 5-fold エキスパート（混合比率 40%）**:
  - 全 36 トランスフォーマー層を使用。
  - 8 深度正規化（Learned Normalization at 8 depths）。
  - Plain / Prompt-Centred 双頭 LoRA。
  - 512 トークン被覆ウィンドウ（Covering Windows）により長文テキストの切り捨てゼロ。
- **5-Fold Staged Parent ブランチ（混合比率 60%）**:
  - 2-stage LoRA24（Qwen3-14B, Qwen3-8B, Llama-3.1, Qwen2.5）。
  - SVC4（重み 12.0）。
  - ATLAS（10% クエリ）。
  - Gaussian Prompt Group Head。

### (2) 未知プロンプト検証データ（540件）による実証性能
`anhkhitrnnn` による 30 個の完全未知プロンプト（Unseen Prompts、各モデル30件均等）での検証結果：
- 親ブランチ単体（旧 0.666 世代）: Balanced Accuracy **`0.7870`**（425 / 540 正解）
- Native Qwen3-Embedding 40% 統合後: Balanced Accuracy **`0.8444`**（456 / 540 正解、**+5.74% 向上**）$\to$ Public LB **`0.680`**

### (3) V52 における防衛策（Dynamic Kaggle Path Resolver）
Kaggle のデータセットマウント構造がアカウントや実行環境により `/kaggle/input/ntl-olmo2-native24-final-inference-assets` または `/kaggle/input/datasets/lightningv08/...` に分岐する問題に対し、再帰的走査（`glob.glob`）と CONFIG 辞書の動的パス置換関数を注入し、`FileNotFoundError` を確実に回避する設計としました。

---

## 2. 実行ステータス
- **Kernel ID**: `tsuyoshichiba/ntl-0682-v52-sovereign`
- **監視プロセス**: PID `50516`（`submit_and_monitor_v52_sovereign.py`）
- **ログ**: `/Users/echo/kaggle/name-that-llm/submission_0682_v52_sovereign/monitor_v52_sovereign.log`
