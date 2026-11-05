# 【実験ネタ #01】GLM-5.3 / Flash 実モデル重み（IQ3_XXS）による直接尤度検証（Exact Teacher Log-Likelihood）

**作成日時**: 2026-11-05 13:50 JST  
**カテゴリ**: Track 2（インテル探索 / 新規モダリティ開拓）  
**主要情報源**:
* **arXiv 論文**: *ATLAS: Source-Conditioned Likelihood* (arXiv:2610.19583) / *TRACE Fingerprinting* (2026) / *Binoculars: Zero-Shot Detection of LLM-Generated Text*
* **Kaggle 外部ミラー資産**:
  * `jamesliebel/glm53-iq3xxs-segs-a-mirror` (144GB) & `segs-b-mirror` (132GB)
  * `jamesliebel/glm53-flash-iq3xxs-segs-mirror` (117GB)
* **Kaggle Discussion**: [Topic 755314] *My classifier thinks GLM-5.3 is more Claude than GLM*

---

## 1. 背景と解決すべき課題（なぜ試す価値があるか）

当チームのオフライン 5-Fold OOF（2,699件）の深層エラー分析において、**全誤分類（176件）の 27.3%（48件）が `glm-5.3` と `glm-5.3-flash` の兄弟間混同に集中** していることが判明しています（Recall はそれぞれ 77.3% と 78.7% で全18モデル中ワースト）。

さらに本日のオフライン検証により、句読点・文長などのスタイロメトリー特徴量（148次元）や文字 n-gram（char_wb 3-5）を用いても、この 2 モデルの識別精度は **52.67%〜53.00%（ほぼコイントス）** に留まることが実証されました。

一方、学術論文（ATLAS / Binoculars / TRACE）が示す理論では、表層的な文体が同一であっても、**「生成元モデル自身の重み（Teacher Weights）にテキストを入力した際の次トークン予測対数尤度（Log-Likelihood / Cross-Entropy）は、真の生成モデルにおいて統計的に有意に高くなる（Perplexity が低くなる）」** という性質があります。

---

## 2. 提案手法の数理・技術仕様

テストセットのうち、アンサンブル予測で上位候補が `glm-5.3` または `glm-5.3-flash`（および誤認されやすい `claude` 系）に絞られた曖昧なサンプル（Disputed Rows：全体の約 10〜15%）のみを対象に、以下の対数尤度比スコアを計算します：

\[
\mathcal{L}_m(x) = -\frac{1}{T} \sum_{t=1}^{T} \log P_m(x_t \mid x_{<t}) \quad (m \in \{\text{glm-5.3}, \text{glm-5.3-flash}\})
\]

\[
\Delta_{\text{GLM}}(x) = \frac{\mathcal{L}_{\text{glm-5.3}}(x) - \mathcal{L}_{\text{glm-5.3-flash}}(x)}{\sigma_{\text{norm}}(T)}
\]

1. **対象サンプルのフィルタリング**:
   全 4,000 件のテストデータのうち、一次アンサンブル（V43/V44）で GLM 系の確率和が閾値（例: $P_{\text{GLM}} > 0.15$）を超える約 400〜600 件のみを抽出。
2. **レイヤーセグメントによる逐次計算または軽量プロービング**:
   Kaggle Dataset 上に公開されている量子化レイヤーセグメント（または GLM-4-9B 等の近似オープンウェイト）を用い、対象サンプルのトークン遷移確率・初期層アクティベーションを抽出する。

---

## 3. 期待されるメリット

* **最大のボトルネックへの直接介入**:
  他モデル（Claude や GPT）がすでに 96〜99% の Recall に達している中、77% に沈んでいる GLM 2 クラスの Recall を 10% 改善するだけで、全体の Balanced Accuracy（18クラス平均）は **$\frac{0.10 \times 2}{18} \approx +0.0111$（+1.1%：Public LB で 0.668 $\to$ 0.679 相当）** 押し上げられる計算になります。
* **表層スタイルの限界突破**:
  文字 n-gram や句読点ルールでは絶対に捉えられない「語彙選択の微小な確率分布の差（Temperature / Top-p サンプリング下のトークン順位）」を直接捉えることができます。

---

## 4. 導入に伴うリスク・デメリット・制約事項（重要）

> [!CAUTION]
> **ストレージ・I/O 帯域および 9 時間タイムアウト（VOID 崖）のリスク**

1. **巨大なモデルサイズと `/kaggle/input` の読み込み速度制限**:
   * `glm53-iq3xxs` ミラーデータセットは各 110GB〜140GB と極めて巨大です。
   * Discussion [Topic 750924] (*Extremely slow /kaggle/input reads on some GPU sessions*) で報告されている通り、Kaggle のネットワークマウントされた `/kaggle/input` から 100GB 超のファイルを読み込むと、ディスク I/O だけで **数時間を浪費し 9 時間タイムアウト（失格）に直結する懸念** が強く残ります。
2. **Notebook アタッチ容量の上限**:
   * Kaggle カーネルにアタッチできるデータセット総量の上限やメモリ制限（RAM 30GB / VRAM 16GB×2）に抵触する可能性があります。
3. **現実的な代替・緩和策**:
   * 本番推論カーネルに直接 100GB 超の重みをアタッチするのではなく、**別の独立した事前学習用 Kaggle カーネル（またはオフライン環境）で Open Pool の GLM 行と公式 Train の GLM 行から「GLM 特化の小規模蒸留ヘッド（Qwen3-4B / 8B ベースの知識蒸留アダプタ）」を学習させ、その軽量アダプタ（数十MB）だけを本番カーネルに持ち込む** 方式が安全と考えられます。

---

## 5. 検証ステップ（アクションプラン）

1. **Step 1（軽量調査）**: 独立した検証用ノートブックで `jamesliebel/glm53-...` のファイル構造（GGUF 形式か PyTorch 形式か、初期数レイヤーのみの部分ロードが可能か）を確認する。
2. **Step 2（蒸留代替の検討）**: 直接ロードの I/O コストが高い場合は、既存の `lucasboesen/name-that-llm-glm4-9b-trunc25`（すでにパイプラインにアタッチ済みの GLM4-9B 特徴抽出器）の中間層出力から、`glm-5.3` vs `glm-5.3-flash` 特化の二値分類ペアワイズヘッドをオフライン OOF で構築・検証する。
