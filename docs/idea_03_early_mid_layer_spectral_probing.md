# 【実験ネタ #03】Transformer 初期〜中間層（Early-to-Mid Layers）特化プロービングと LoRA+ / 位置重み付けプーリング

**作成日時**: 2026-11-05 13:55 JST  
**カテゴリ**: Track 2（モデル表現学習・特徴抽出の高度化）  
**主要情報源**:
* **arXiv 論文**:
  * *Layered Insights: Generalizable Analysis of Authorial Style by Leveraging All Transformer Layers*
  * *TextFluoroscopy / Linear Probing Provides Robust Detection of Machine-Generated Text*
  * *LoRA+: Efficient Low Rank Adaptation of Large Models* (ICML 2024)
* **Kaggle Discussion**: [Topic 750462] *Early Qwen layers know the author better than late ones: 0.391 vs 0.311*
* **新規公開モデル資産**:
  * `qiuzilang/ntl-qwen3-8b-loraplus-r4-r8-weights` (`qwen8b_unsloth_fp16_dual_msd_position_weighted_v1`)
  * `clustifier/ntl-candidate-v12-adapters` (Open Pool v12 学習済み Llama-3.1-8B LoRA + `tokattn_head.pt`)
  * `yvliet/ntl-motif18-weights` (`motif18_projection_weights.pt`)

---

## 1. 背景と解決すべき課題（なぜ試す価値があるか）

通常のテキスト分類では Transformer の最終層（Last Hidden Layer）を用いますが、LLM 出自特定（Authorship Attribution）においては **「最終層が最も識別性能が低い」** という逆転現象が学術論文およびコンペ実験の両方で確認されています。

Discussion [Topic 750462] で共有された Qwen3-4B（全36層）の層別プロービング実験結果は以下の通りです：

| 抽出レイヤー | モデル深さ比率 | 5-Fold Grouped OOF (Balanced Acc) | 特徴の性質 |
| :--- | :---: | :---: | :--- |
| **Layer 3** | **8% (極初期層)** | **`0.391` (最高)** | **語彙選択・句読点リズム・局所構文（文体情報）** |
| **Layer 6** | 17% (初期層) | `0.379` | 表層スタイルと短距離依存関係 |
| **Layer 9** | 25% (初中期層) | `0.370` | 構文構造・接続詞の習慣 |
| **Layer 18** | 50% (中間層) | `0.327` | 意味構造への遷移領域 |
| **Layer 27** | 75% (後半層) | `0.341` | トピック・ドメイン知識の統合 |
| **Layer 36** | **100% (最終層)** | **`0.311` (最低)** | **次トークン予測用のトピック内容（未知プロンプトで過学習）** |
| **6層結合** | 複合 | **`0.414`** | 初期層の文体＋中間層の構造の統合 |

* **技術的本質**:
  テストセットのプロンプトは学習セットと一切重複しない未知トピック（Unseen Prompts）です。後半〜最終層は「何について書かれているか（Topic/Semantics）」を強くエンコードするため未知プロンプトで汎化性能が落ちるのに対し、**初期〜中間層（Layer 2〜12）は「どのように書かれているか（Style/Syntax/Rhythm）」を保持している** ため、未知プロンプトに対して高い汎化性能を発揮します。

---

## 2. 提案手法の数理・技術仕様

### アプローチ A：既存 LoRA24 メンバーの「初期層重み付けプーリング（Early-Layer Weighted Pooling）」
現在、パイプライン内の `infer.py` では、抽出層 `L = [2, 4, 6, 8, 12, 16, 20, 24]` の隠れ状態を **単純等重み平均（Equal-Weight Mean）** しています：

\[
h_{\text{pool}} = \frac{1}{|L|} \sum_{\ell \in L} \text{LayerNorm}(h^{(\ell)})
\]

これを、各層の単体識別能（初期層ほど高く後半層ほど低い）に合わせて、層別の重み係数 $w_\ell$（例: 初期層 `[2, 4, 6, 8, 12]` に重み $1.25$、後半層 `[16, 20, 24]` に重み $0.75$）を乗じるか、あるいは `infer.py` の `gauss.py` / `Ridge` ヘッドへ渡す前に初期層の寄与度を引き上げます。
※ **ただし、`infer.py` で事前学習済みの `heads.pt`（固定線形ヘッド）を使う場合は入力スケールが変わると破綻するため、再学習する Ridge / LogisticRegression ヘッド（`perpassage` 側）に対してのみ層別重み付けまたは層連結を適用します。**

### アプローチ B：新規公開アダプタ（LoRA+ / Open Pool v12 対応版）の補完スタック統合
11月4日に公開された以下の最新学習済みチェックポイントを、独立した予測メンバーとして検証します：
1. **`qiuzilang/ntl-qwen3-8b-loraplus-r4-r8-weights`**:
   * LoRA+（$\eta_B / \eta_A = 4.0$）＋ 位置重み付けプーリング（Position-Weighted Pooling）＋ Multi-Sample Dropout（5 samples）＋ 初期層重点サンプリング `[2, 4, 6, 8, 12, 24, 32, 36]` で学習された Qwen3-8B アダプタ。
2. **`clustifier/ntl-candidate-v12-adapters`**:
   * 最新の Open Pool v12 で学習された Llama-3.1-8B PEFT LoRA アダプタ ＋ トークンアテンションヘッド（`tokattn_head.pt`）。

---

## 3. 期待されるメリット

* **アンサンブル多様性（Diversity）の確保**:
  現行の全上位ノートブックは `dhruvtibarewal/nlm-l24-adapters-v5` と `nishantdahal` の同一チェックポイント群に 100% 依存しています。異なる学習レシピ（LoRA+ / Position-Weighted / v12 データ）で鍛えられた特徴表現を少量（重み 0.05〜0.10）ブレンドすることで、既存アンサンブルの死角を補完できる可能性があります。
* **計算コストの抑制**:
  アプローチ A（既存の抽出済み隠れ状態に対する Ridge 側の層別重み調整）であれば、GPU の Forward Pass 時間を **1 秒も増やさずに** 実行可能です。

---

## 4. 導入に伴うリスク・デメリット・制約事項

> [!WARNING]
> **事前学習済みヘッドとの整合性崩壊リスクとデータリーク懸念**

1. **`full_train: true` モデルにおける OOF 評価不能リスク**:
   * `qiuzilang` の `v1.bundle.json` を解析した結果、`"full_train": true, "train_rows": 2699` と記載されており、公式 `train.csv` 全件を使って学習されています。
   * つまり、手元の `train.csv` に対して推論させると In-sample（学習データそのものへの予測）となり OOF スコアが 99% に跳ね上がるため、**オフラインで真の汎化性能やブレンド重みを測定できない** という重大な制約があります。
2. **実行時間の増加**:
   * 新規の 8B アダプタを 1 つ追加で Forward Pass させると、T4 GPU 上で約 25〜35 分の推論時間が加算され、ATLAS の計算予算（`ATLAS_BUDGET`）がその分削られるトレードオフが発生します（[Topic 753177] で議論された「メンバー追加による ATLAS 時間圧迫の罠」）。

---

## 5. 検証ステップ（アクションプラン）

1. **Step 1（ゼロコスト検証）**: 手元の Phase-0 OOF 環境（または `comp_stack.py` の特徴抽出キャッシュ）において、層の重み配分を変更した際の Ridge / LogisticRegression の Grouped 5-Fold OOF を計測する。
2. **Step 2（Track 2 単体探索）**: `clustifier`（v12 Llama-3.1 アダプタ）または `qiuzilang`（LoRA+）をテストする際は、既存パイプラインにいきなり混ぜるのではなく、まず `comp_stack.py` 内の軽量差し替えとして 1 スロットで LB 反応を確認する。
