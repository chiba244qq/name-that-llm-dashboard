# Name That LLM: True 5-Fold OOF Empirical Verification & Audit Remediation Report

> [!IMPORTANT]
> **監査修正の完了**: 以前の [`ntl_v57_grandmaster_verification_lab.ipynb`](file:///Users/echo/kaggle/name-that-llm-dashboard/notebooks/ntl_v57_grandmaster_verification_lab.ipynb) Cell 4 に存在していた合成ガウス乱数（`np.random.normal`）を完全に撤廃し、`lightningv08/ntl-olmo2-native24-final-inference-assets` 内の **100% 実測の 5-Fold Held-Out OOF ロジット配列（2,699行 × 18クラス）** と本番の `topic_groups()` / `finalize()` を用いた検証パイプラインへ全面刷新・実行完了しました（GitHub `origin/main` コミット `f97b0c6` 反映済み）。

---

## 1. 取得・検証した実測 5-Fold OOF アセット一覧

[`/Users/echo/kaggle/name-that-llm/true_oof_assets/`](file:///Users/echo/kaggle/name-that-llm/true_oof_assets) にダウンロードし、公式 [`train.csv`](file:///Users/echo/kaggle/name-that-llm/train.csv)（2,699行、150プロンプト、18クラス）との ID 順序および Fold 分割（`540, 540, 540, 539, 540` 行の完全非重複分割）が一致することを検証した実測データは以下の通りです。

| コンポーネント | 実測 OOF ファイル | 配列キー・形状 | 単体 OOF BAcc (Raw $\to$ Calibrated) |
| :--- | :--- | :--- | :--- |
| **Alibaba Qwen3-Embedding-8B** | [`ntl-native-q8-multidepth-fivefold-0679/oof_reference.npz`](file:///Users/echo/kaggle/name-that-llm/true_oof_assets/ntl-native-q8-multidepth-fivefold-0679/oof_reference.npz) | `plain`, `centered`: `(2699, 18)` | `0.65164` $\to$ **`0.68090`** (TF-IDF Centering) |
| **Tencent WeMM-Embedding-4B** | [`ntl-native-wemm4-fivefold-0679/oof_reference.npz`](file:///Users/echo/kaggle/name-that-llm/true_oof_assets/ntl-native-wemm4-fivefold-0679/oof_reference.npz) | `plain`, `centered`: `(2699, 18)` | `0.62125` $\to$ **`0.64089`** (TF-IDF Centering) |
| **Dual-Embedding 単体融合** | `0.85 * Qwen3-8B + 0.15 * WeMM-4B` (親なし) | `(2699, 18)` | **`0.68758`** (TF-IDF) / **`0.68981`** (True Prompt) |
| **Sovereign Parent OOF** | [`q35_heads/oof_predictions.npz`](file:///Users/echo/kaggle/name-that-llm/true_oof_assets/q35_heads/oof_predictions.npz) | `nested_calibrated`: `(3, 2699, 18)` | Scenario 1: `0.48006` / Scen 3: `0.54380` / Scen 6: `0.57491` |

---

## 2. 本番パイプライン完全準拠・5コンテキスト実測 OOF 比較結果

CHEN Han 氏の公式キャリブレーションプロトコル（[`fold0/calibration.json`](file:///Users/echo/kaggle/name-that-llm/true_oof_assets/ntl-native-wemm4-fivefold-0679/fold0/calibration.json)）および本番の [`core_math.py`](file:///Users/echo/kaggle/name-that-llm/submission_0688_v57_apex_sovereign/ntl-0688-v57-apex-sovereign.ipynb) の `finalize()`（Sinkhorn マージナル正則化 `strength=0.30, target_mix=0.35` + 37種タイポグラフィベイズ証拠補正）を各 Outer Fold ごとに厳密適用した結果は以下の通りです。

### 2.1 主要マイルストーン構成の 5コンテキスト実測 OOF スコア

| 構成名 / ステータス | `s_native` (Qwen3-8B) | `w_wemm` (WeMM-4B) | **5-Context Composite** | Full-TFIDF (本番同一) | Sparse-6 (6応答/P) | Sparse-3 (3応答/P) | Sparse-2 ($r=0.91$) | Singleton (1応答/P) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V52 Base** (`0.680` LB) | `0.40` | `0.00` | `0.67864` | `0.69868` | `0.69756` | `0.67703` | `0.67086` | `0.64903` |
| **Public 0.681 Repro** (`dfenglei`) | `0.50` | `0.00` | `0.68222` (+0.00358) | `0.70091` | `0.70002` | `0.68194` | `0.67659` | `0.65162` |
| **V56-Fix** (`0.680` LB 完走) | `0.40` | `0.07` | `0.68316` (+0.00452) | `0.70425` | `0.70235` | `0.68256` | `0.67425` | `0.65237` |
| **V57-Apex (現在推論中 `Ref: 58222075`)** | **`0.45`** | **`0.10`** | **`0.68772` (+0.00908)** | **`0.70870`** | **`0.70644`** | **`0.68641`** | **`0.67802`** | **`0.65904`** |
| **V58-A (堅実バランス案)** | `0.50` | `0.10` | **`0.68892` (+0.01028)** | `0.70870` | `0.70675` | `0.68779` | `0.68011` | `0.66126` |
| **V58-B (Outer 5-Context 極大点)** | `0.50` | `0.15` | **`0.68977` (+0.01113)** | `0.70945` | `0.70779` | `0.68692` | `0.68119` | `0.66348` |

### 2.2 CHEN Han 氏による事前計算済み 5-Fold Inner-CV グリッドとのクロスチェック

[`ntl-native-wemm4-fivefold-0679/fold{0..4}/calibration.json`](file:///Users/echo/kaggle/name-that-llm/true_oof_assets/ntl-native-wemm4-fivefold-0679/fold0/calibration.json) 内に記録されている `base_plus_wemm4` の 5-Fold Inner-CV 平均スコアは以下の通りであり、**`w_wemm = 0.10` に明確な内部極大値（Interior Optimum）**が存在します。

- `w_wemm = 0.00`: `0.696574`
- **`w_wemm = 0.10`: `0.699630` (+0.003056, Inner-CV Peak)**
- `w_wemm = 0.20`: `0.698426`
- `w_wemm = 0.35`: `0.695093`
- `w_wemm = 0.50`: `0.686481`

---

## 3. 現在稼働中の V57 (`Ref: 58222075`) の客観的評価とリスク分析

1. **`s_native=0.45, w_wemm=0.10` の妥当性**:
   - 前回の Colab ノートブックでは合成ノイズのアーティファクトとして `(s_native=0.45, w_wemm=0.10)` が選出されましたが、**本物の 5-Fold OOF 上で検証した結果でも、実は `(s_native=0.45, w_wemm=0.10)` は V56-Fix (`0.40, 0.07`) を全 5 コンテキストで一貫して上回る（Composite `+0.00456`、V52 比 `+0.00908`）極めて有望な内部領域に位置している**ことが実証されました。
   - 特に `w_wemm = 0.10` は CHEN Han 氏の 5-Fold Inner-CV における厳密な極大点と一致しています。
2. **残存する懸念点（`w_atlas = 0.20`）**:
   - 現在走行中の V57 (`Ref: 58222075`) で唯一懸念されるのは、親スタック側の `w_atlas` が `0.20`（公開実績 `heliosli` の `0.15` より `+0.05` 高い値）に設定されている点です。
   - ただし、ATLAS は `top1 - top2 < 0.30` の曖昧サンプル（全体の約30〜40%）にのみ適用され、さらに後段で `(1 - w_wemm) * (1 - s_native) = 0.90 * 0.55 = 0.495` 倍に減衰されるため、`w_atlas` の `0.15` と `0.20` の実質的な最終確率差は対象行でも最大 `2.5%` 程度に留まります。

---

## 4. 次の一手（V58 設計方針）

- **推奨構成（V58 Candidate）**:
  - `s_native = 0.50`（`dfenglei` 等で `0.680` $\to$ `0.681` の単一要因として実証済み、かつ今回の実測 OOF でも `0.45` を全 Sparse コンテキストで上回る）
  - `w_wemm = 0.10`（CHEN Han 氏の 5-Fold Inner-CV 極大値、かつ Outer 5-Context でも `0.68892` を記録）
  - `w_atlas = 0.15`, `gsq_share = 0.06`（公開 `0.681` で実証済みの親スタック重みに戻し、`w_atlas=0.20` の過剰介入リスクを排除）
- 本日の残り提出枠は **4 / 5 回** あります。現在走行中の V57 (`Ref: 58222075`) と並行して、この実測 OOF 検証済みの **V58 (`s_native=0.50, w_wemm=0.10, w_atlas=0.15, gsq_share=0.06`)** を投入するか、あるいは V57 の完了を待つかをご判断いただけます。
