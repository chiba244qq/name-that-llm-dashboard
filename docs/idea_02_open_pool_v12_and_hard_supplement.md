# 【実験ネタ #02】Open Pool v12（211k行）＋ Hard-Class Supplement（Vertex AI 厳密ルート）へのデータ刷新

**作成日時**: 2026-11-05 13:52 JST  
**カテゴリ**: Track 1 / Track 2 共通（データセット基盤のアップグレード）  
**主要情報源**:
* **Kaggle Discussion**: [Topic 752766] *[NOW UPDATED] Open Pool v12 is out. +20,781 passages, and the gap list is getting filled*
* **新規公開データセット**:
  * `crtproduction/name-that-llm-open-pool` (v12: **211,369 行**, 59ソース統合)
  * `aniruduh9/ntl-hard-supplement` (v2: **2,369 行**, v12未収録の難所クラス特化データ)
  * `zeminglii/ntl-own-exact-route-generations` (**Google Vertex AI 経由が証明された `gemini-3.7-flash` +263行** ＆ Z.AI 直結 GLM)
  * `kilupy/ntl-18-model-corpus` (**1,588 プロンプトすべてに Claude 全5兄弟の回答が揃った完全パネルデータ**)

---

## 1. 背景と解決すべき課題（なぜ試す価値があるか）

Discussion [Topic 754860] で複数の上位参加者が口を揃えて報告している通り、**「本コンペにおける 0.620 $\to$ 0.668+ のスコア向上の大半は、モデル構造ではなく外部データの品質とプロンプトカバレッジから生まれている」** という客観的事実があります。

現在、当チームの V40 / V43 および公開 0.668 ノートブックが参照しているコードは、旧バージョンの **Open Pool v11（190,588行）** を前提として設計されています。しかし、直近（11月3日〜4日）のアップデートにより、以下の決定的なデータ増強が行われた **Open Pool v12（211,369行）** および **Hard-Class Supplement（2,369行）** が利用可能になっています：

| 強化ポイント | Open Pool v11 (現行) | Open Pool v12 + Hard Supp (新規) | 期待される効果 |
| :--- | :---: | :---: | :--- |
| **総パッセージ数** | 190,588 行 | **213,738 行 (+23,150 行)** | n-gram SVM / TF-IDF の語彙カバレッジ拡大 |
| **全18モデル完備プロンプト数** | 592 プロンプト | **712 プロンプト (+20.3%増)** | `comp_stack.py` の Prompt-Centered 学習精度の直接向上 |
| **Claude 5兄弟完備プロンプト** | 少数 | **+1,588 プロンプト (14,148行)** | `claude-fable-5` や `opus-4-6 / 4-8` の同一プロンプト比較学習 |
| **Vertex AI 証明済 `gemini-3.7`** | 3,278 行 | **3,541 行 (+263行)** | 2026-10-08以降の API 偽装（3.7要求に3.8が返る問題）を完全排除した純粋データ |

---

## 2. 提案手法の技術仕様

パイプライン内の 3 つのデータ消費箇所に対して、v12 ＋ Hard Supplement を以下の規律で注入します：

1. **n-gram LinearSVC (`--pool`) への注入**:
   * Open Pool v12（211k行）に `aniruduh9/ntl-hard-supplement`（2,369行）を結合。
   * ただし、v12 では Claude 系の行が +14,148 行増えているため、そのまま学習するとクラス事前分布が Claude に偏る懸念があります。したがって、**クラスごとのサンプル重み正規化（またはクラス均衡サブサンプリング）** を併用します。
2. **Complementary Stack (`comp_stack.py`) の「12+ models answered prompts」拡張**:
   * 公開 0.668 が強くなった主因の一つは、Open Pool から「12個以上のモデルが回答しているプロンプト群（v11時点で 2,956 プロンプト / 43,071 行）」を抜き出して Prompt-Centered 特徴量を学習している点にあります。
   * v12 ＋ `aniruduh9` のパネル補完データを加えることで、この高密度プロンプト群がさらに拡大し、プロンプト平均差し引き（Prompt Centering）の分散推定精度が向上すると考えられます。
3. **Typography Matching (`match_weight`) の厳格適用**:
   * v12 作者（crtproduction）の検証レポートにある通り、新規追加された `kilupy` の Claude データは em-dash（`—`）の出現率が 55%（公式 Train は 83%）と表記体系がやや異なります。
   * したがって、`EXTRA` サンプリング時には無作為抽出ではなく、必ず **`match_weight` 列を用いた重み付きサンプリング** を行い、公式 Train の句読点分布と一致させる必要があります。

---

## 3. 期待されるメリット

* **再学習（GPU数時間）なしで推論パイプライン内の線形モデル（SVC / TF-IDF / Comp Stack）を即座に強化可能**:
  LoRA バックボーン自体を再学習しなくても、`infer.py` と `comp_stack.py` は推論実行時に毎回 `POOL` と `EXTRA` から線形ヘッド（LinearSVC / Ridge / LogisticRegression）をオンザフライで fit しているため、データソースの差し替えとフィルタリング調整だけで即座に恩恵を受けられます。
* **最も誤分類の多い `gemini-3.7-flash`（Recall 82.7%）と `glm-5.3` 系の純度向上**:
  `zeminglii` と `aniruduh9` によって「Vertex AI 直結ルート」「Z.AI 直結ルート」のみに厳選されたデータが追加されるため、API ルーティング汚染による決定境界のボケを軽減できます。

---

## 4. 導入に伴うリスク・デメリット・制約事項

> [!WARNING]
> **クラス不均衡バイアスと Synthetic Prompt のドメインシフト**

1. **Claude ファミリーの過剰代表（Over-representation）リスク**:
   * v12 で追加された 20,781 行のうち約 70%（14,148 行）が Claude ファミリーです。また `ntl-hard-supplement` でも 2,369 行中 1,217 行が `claude-opus-5` です。
   * 無調整で SVM や TF-IDF に流し込むと、切片（Intercept）が Claude 側にシフトし、特に文体が似ている `glm-5.3` が Claude に吸い込まれる副作用（[Topic 755314] の現象）が悪化するリスクがあります。
   * **対策**: 追加データを取り込む際は、`claude-opus-5` 等の上限数をキャップ（例: 各モデル最大 N 行まで）するか、`class_weight='balanced'` を徹底する必要があります。
2. **メモリ・前処理時間の微増**:
   * 190k 行 $\to$ 213k 行への増加に伴い、文字 n-gram TF-IDF 行列の構築時間が約 10〜12% 増加（数十秒〜1分程度）します。

---

## 5. 検証ステップ（アクションプラン）

1. **Step 1**: `ntl-hard-supplement`（手元 `/tmp/intel_datasets/hard_supp/ntl_hard_supplement.parquet` に取得済み）から、特に不足している難所クラス（`glm-5.3`, `glm-5.3-flash`, `kimi-k3`, `claude-fable-5`, `gemini-3.8-flash`）のみを抽出し、過剰な `claude-opus-5`（1,217行）を除外したクリーンサブセットを定義する。
2. **Step 2**: 手元の 5-Fold OOF 環境において、n-gram LinearSVC 単体にこのサプリメントデータを追加した際の OOF Balanced Accuracy の変化（特に GLM と Gemini の Recall 変化）を事前測定する。
