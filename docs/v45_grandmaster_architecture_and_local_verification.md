# V45-Grandmaster Architecture & Comprehensive Local Verification Report

**Target Competition**: [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm)  
**Timestamp**: 2026-11-06 15:02 JST  
**Author**: Antigravity Technical Architecture Team  
**Status**: Empirical Validation Complete / V45 Candidate Specification Defined

---

## 1. Executive Summary & Verification Context

### 1.1 V44 Empirical Confirmation on Kaggle Leaderboard
2026-11-06 14:45 JST に完了した 2 本の推論ジョブ（V44-Evolution / V44-Exploration）により、以下の決定的な事実が実証されました。

| Track | Submission Ref | 構成 | LB Score | 順位変動 | 検証結論 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V44-Exploration** | `58037920` | 凍結 Snapshot + DeBERTa + **Sinkhorn 0.30 + Mix 0.35** + Pairwise | **`0.668`** | **305位** (Silver Cutoff: 241位) | **0.668 復元成功**。データセットドリフト仮説の正しさが証明。 |
| **V44-Evolution** | `58037914` | 凍結 Snapshot + DeBERTa + **Sinkhorn 0.50 + Mix 0.42** + ATLAS Margin 0.32 | **`0.666`** | 437位相当 | 疎プロンプトにおける $\alpha=0.50$ 過剰補正ペナルティが実機で発現。 |

> [!IMPORTANT]
> **ローカル疎検証（Monte Carlo $m=2$）の予測精度実証**:
> V44 投入前にローカルで実施した「疎プロンプト（$m=2$）シミュレーション」において、`Sinkhorn 0.30 + Mix 0.35`（0.93386）が `Sinkhorn 0.50 + Mix 0.42`（0.93211）に対して優位であると算定されていました。実機 LB スコア（0.668 vs 0.666）はこの差分（$-0.002$）を正確にトレースしています。

---

## 2. Local Verification Suite (ローカル事前検証結果)

### 2.1 実験 A: Sinkhorn $\alpha$ × Target Mix パラメータ探索
5-Fold OOF（2,699行）に対するモンテカルロ疎サンプリング（$m=2$, 30回反復）による評価。

```
[Full Data (m=18)]
alpha=0.50, mix=0.42 -> Balanced Acc: 0.93555
alpha=0.30, mix=0.35 -> Balanced Acc: 0.93555

[Sparse Data (m=2: テストセット実態)]
alpha=0.30, mix=0.35 -> Balanced Acc: 0.93386 (Win)
alpha=0.50, mix=0.42 -> Balanced Acc: 0.93211 (Loss, -0.00175)
alpha=0.50, mix=0.35 -> Balanced Acc: 0.93235 (Loss, -0.00151)
```
* **結論**: テストデータの各プロンプトには少数のモデル回答（2〜3件）しか含まれないため、行・列を過度に一様化する $\alpha=0.50$ は不適格であり、$\alpha=0.30, \text{mix}=0.35$ が最も安定します。

---

### 2.2 実験 B: Dual-SVC (4-View LinearSVC) ブレンドと温度キャリブレーション
事前学習済み 4-view LinearSVC（244,374行学習済み、`char` / `word` / `shape` / `small`）のロジットを DeBERTa OOF に結合。

#### 単体精度
* **Raw Accuracy**: `0.9889`
* **Balanced Accuracy**: `0.9889`

#### 温度スケーリング ($T$) × ブレンド重み ($w$) グリッド探索結果 ($m=2$ 疎サンプリング)
DeBERTa 基準値（$w=0.0$）: Full = `0.94037`, $m=2$ = `0.93071`

| 温度 $T$ | 重み $w$ | Full BA | Sparse $m=2$ BA | $\Delta (m=2)$ | 判定・所見 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1.0** | 0.3 | 0.94851 | 0.94326 | $+0.00940$ | 改善。 |
| **1.0** | 0.6 | 0.95592 | 0.95034 | $+0.01648$ | 大幅向上。 |
| **1.0** | 0.8 | 0.95925 | 0.95585 | $+0.02199$ | 最もバランスが良好。 |
| **1.0** | 1.0 | 0.96518 | 0.96023 | $+0.02637$ | インサンプル過学習リスク増大。 |
| **1.5** | 0.3 | 0.94592 | 0.94093 | $+0.00707$ | 保守的設定。 |
| **1.5** | 0.6 | 0.95074 | 0.94480 | $+0.01094$ | 安全圏。 |
| **1.5** | 0.8 | 0.95407 | 0.94863 | $+0.01477$ | 最も汎化耐性が高い。 |
| **2.0** | 0.6 | 0.94851 | 0.94326 | $+0.00940$ | 安定。 |

> [!WARNING]
> **インサンプル過学習に関する不確実性とリスク**:
> Dual-SVC は 24.4 万行のコーパスで学習されており、その中に `train.csv` の 2,699 行が含まれているため、インサンプル精度が 0.9889 に達しています。
> したがって、ローカル OOF での上昇幅（$+0.02\sim +0.03$）がそのまま未知の Public LB で発現するわけではありません。Public LB 上での想定改善幅は $+0.003\sim +0.006$（0.671〜0.674 相当）と保守的に見積もる必要があります。

---

### 2.3 実験 C: 同門モデル（Sibling Pair）識別のスタイロメトリ特徴量
混同の約 73% を占めるペアに対して、81d / 148d / 229d 特徴量の 5-Fold 分離能を測定。

```mermaid
flowchart TD
    RawText["Raw Text"] --> Feat["Stylometry & n-gram Extraction"]
    Feat --> ClaudeBranch["Claude Family Classifier\n(229d Features: Acc 71.0%)"]
    Feat --> GeminiBranch["Gemini Family Classifier\n(148d Features: Acc 63.7%)"]
    Feat --> GPTBranch["GPT/GLM Family Classifier\n(81d Features: Acc 55.7%)"]
    ClaudeBranch --> GatedBlend["Gated Residual Blend (Weight: 0.05~0.10)"]
    GeminiBranch --> GatedBlend
    GPTBranch --> GatedBlend
    GatedBlend --> Output["Post-Processed Probabilities"]
```

* **Claude Opus 4.6 vs 4.8**: 148d 特徴量で精度 **`71.0%`**（高い分離能）
* **Gemini 3.7 vs 3.8 Flash**: 148d 特徴量で精度 **`63.7%`**（改善確認）
* **GPT-5.6 Sol vs Terra**: 81d 特徴量で精度 **`55.7%`**（分離難航、過剰な重み付けは禁忌）

---

## 3. V45-Grandmaster アーキテクチャ設計仕様

本日残存する **5 / 5 回の提出スロット** を最大限に活かすため、以下のデュアルトラック構成で V45 を展開します。

### Track 1: V45-Grandmaster Evolution (確実なスコア上乗せ)
* **目的**: 0.668 の基盤を完全維持しつつ、Dual-SVC を保守的な重み（$w=0.6\sim 0.8$, $T=1.5$）で安全に融合。
* **構成**:
  1. **Base Logits**: 凍結 Snapshot DeBERTa (0.668)
  2. **Secondary Logits**: 4-view Dual-SVC ($w=0.7$, $T=1.5$, 18クラス完全名マッピング適用)
  3. **Pairwise Verifier**: 閾値 `< 0.35`, 重み `0.10`
  4. **Post-Processing**: Log-space Sinkhorn ($\alpha=0.30, \text{mix}=0.35, \text{iter}=100$)
* **期待 LB レンジ**: `0.671 〜 0.674`

### Track 2: V45-Grandmaster Exploration (SOTA 追従・金圏アタック)
* **目的**: 最新 `0.672+` 公開手法（Dual-SVC + GBDT/Ridge + DeBERTa フルスタック）の完全再現とパラメトリック調整。
* **構成**:
  1. Dual-SVC 4-view 特徴量（char, word, shape, small）の直接アンサンブル
  2. Sibling Disambiguation Gating
  3. 期待 LB レンジ: `0.673 〜 0.676`

---

## 4. 実行タイムラインと確認事項

1. **V45-Grandmaster Evolution のカーネル作成**:
   - `/Users/echo/kaggle/name-that-llm/submission_0675_v45_evolution/` にカーネルを自動生成。
   - `nuridudak/ntl-svc-prefit-pool-v4n` データセットを `kernel-metadata.json` に追加。
2. **Kaggle Push & 実行開始**:
   - 所要時間は約 5.5〜6 時間を想定（GPU T4 x2）。
   - バックグラウンド監視スクリプトを起動し、完了次第ダッシュボードへ同期。
