# ローカル事前検証ベンチマーク（EXP-01〜EXP-09）＆ V44 採用候補選定レポート

**作成日時**: 2026-11-05 14:30 JST  
**対象コンペティション**: Kaggle Featured Code Competition「[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)」  
**検証スクリプト**: [`run_comprehensive_local_verification.py`](file:///Users/echo/kaggle/name-that-llm/run_comprehensive_local_verification.py)  
**出力データ**: [`local_verification_results.json`](file:///Users/echo/kaggle/name-that-llm/local_verification_results.json)

---

## 1. 重大な数理発見：`oof_log_probs` の未正規化スケールと `wsum = 9.0` 較正

ローカル検証スイート（`EXP-01`〜`EXP-09`）の構築過程において、Phase-0 で抽出した 5-Fold OOF ロジット（`oof_artifacts.npz` 内の `oof_log_probs`）と本番 `infer.py` の推論パイプラインの間に、**温度スケール（Temperature / Normalization）の決定的な差異**が存在することが判明しました。

### 1.1 原因の特定
- `create_phase0_notebook.py` で 5-Fold OOF を抽出した際、`infer.py --sim {fold_idx}` は `--sinkhorn 0`（デフォルト）で実行されていました。
- `infer.py` の内部実装では、`a.sinkhorn > 0` の本番推論時のみ、9個のアンサンブルメンバー（ニューラル6種＋線形3種）のロジット総和 `tot` を有効メンバー数 `wsum = 9.0` で除算（`lpn = tot / max(1, wsum)`）してから `P = softmax(lpn)` を計算し、さらに対数空間での幾何平均補間：
  $$\text{tot}_{\text{post}} = (1 - \alpha) \log(P + 10^{-12}) + \alpha \log(Q + 10^{-12})$$
  を実行しています。
- 一方、`oof_artifacts.npz` に保存されていた `oof_log_probs` は **`wsum = 9.0` で除算される前の未正規化総和 `tot`**（最小値 `-508.8`、Top-1/Top-2 ロジット差の中央値 `47.3`）でした。これをそのまま `softmax` に通すと、98% 以上の行で最大確率が `0.9999+` に張り付き、後段の信頼度ゲート（`margin < 0.35` や `margin < 0.65`）がわずか 1.0%〜2.9% の行にしか発火しない状態となっていました。

| 正規化因子 `wsum` | ATLAS 発火率 (`margin < 0.32`) | Pairwise 発火率 (`margin < 0.35`) | Sibling 発火率 (`margin < 0.65`) | Full OOF (`sk=0.50, mix=0.42`) | 実機挙動との整合性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `wsum = 1.0` (未除算) | 0.9% (11 / 1,200行) | 1.0% (12 / 1,200行) | 2.9% (35 / 1,200行) | `0.93406` (線形補間) | ❌ 過信状態（ゲート不発） |
| **`wsum = 9.0` (`infer.py` 厳密一致)** | **8.7% (104 / 1,200行)** | **9.7% (116 / 1,200行)** | **18.9% (227 / 1,200行)** | **`0.93555` (対数空間補間)** | **✅ 実機のゲート発火率と一致** |
| `wsum = 14.0` (Comp Stack 混合相当) | 14.0% (168 / 1,200行) | 15.2% (182 / 1,200行) | 28.5% (342 / 1,200行) | `0.93481` | ⭕ 軟化アンサンブル相当 |

---

## 2. 全9実験（`EXP-01`〜`EXP-09`）のローカル定量検証結果

`infer.py` と完全同一の `wsum = 9.0` ＋ 対数空間 Sinkhorn（100反復）＋ Post-Sinkhorn ゲート順序を再現し、以下の 2 つの評価軸で全 9 実験を定量検証しました：
1. **Full OOF ($m=18$)**: 1プロンプトあたり18回答すべてが揃った 2,699 行での 5-Fold OOF Balanced Accuracy。
2. **Sparse OOF ($m=2, 3$)**: 本番テストセットの不完全なプロンプト構造（1プロンプトあたり数件のみ抽出）を模した、25回モンテカルロ疎サンプリング平均 Balanced Accuracy。

| Exp ID | 実験名 / 検証仮説 | 対応ネタ | Full OOF ($m=18$) | Sparse OOF ($m=2$) | $\Delta (m=2)$ vs V40 | Sparse ($m=3$) | LB $\text{SD}_{\text{gap}}$ | 判定 (Verdict) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | **V40 Baseline (Anchor)** | Baseline | `0.93555` | `0.93902` | `+0.00000` | `0.93514` | `±0.0000` | **Baseline** |
| **EXP-02** | **V43-Evolution (`Mix 0.42`)** | Track 1 (推論中) | `0.93555` | `0.93953` | **`+0.00051`** | `0.93444` | `±0.0008` | **Adopted (In Flight)** |
| **EXP-03** | **V40A Uniform Stylometry (`148d`)** | Idea #05 (対照) | `0.93518` | `0.93938` | `+0.00036` | `0.93462` | `±0.0010` | **Rejected (148d一括は減退)** |
| **EXP-04** | **Family-Specific Stylometry Routing** | Idea #05 | `0.93518` | **`0.93971`** | **`+0.00069`** | **`0.93469`** | `±0.0012` | **Adopted ($m=2,3$ 改善)** |
| **EXP-05** | **V34.1 Pairwise Verifier (`0.35 / 0.10`)** | Idea #04 | **`0.93555`** | **`0.93995`** | **`+0.00093`** | `0.93438` | `±0.0000` | **Top Standalone (+0.00093)** |
| **EXP-06** | **V34.1 Pairwise Verifier (`0.36 / 0.115`)** | Idea #04 | **`0.93555`** | **`0.93984`** | **`+0.00082`** | `0.93438` | `±0.0000` | **Adopted** |
| **EXP-07** | **Exact-Route Hard Supplement SVC** | Idea #02 | **`0.93555`** | `0.93940` | `+0.00038` | `0.93442` | `±0.0008` | **Adopted (SVC補強)** |
| **EXP-08** | **V44 Undamped Hybrid (全盛込み)** | Ideas #02+#04+#05 | `0.93555` | `0.93912` | `+0.00010` | `0.93435` | `±0.0008` | **Hold (過剰補正の懸念)** |
| **EXP-09** | **V44 Damped Hybrid (`Pairwise 0.10 + GPT/GLM 81d 0.05`)** | Ideas #04+#05 | **`0.93555`** | **`0.94003`** | **`+0.00101`** | **`0.93472`** | `±0.0008` | **★ Recommended V44 Core** |

---

## 3. ファミリー別スタイロメトリー検証（Idea #05 定量結果）

148次元（V40A 標準特徴量）と 81次元（Victor 氏の 100単語あたり長さ正規化特徴量 `per100w`）を主要4兄弟ペアの 5-Fold Grouped CV で直接比較した結果は以下の通りです。

| 対象兄弟ペア | V40A 標準 (`148d`) | Victor 長さ正規化 (`81d`) | 結合 (`229d`) | 選定された特徴量セット |
| :--- | :--- | :--- | :--- | :--- |
| **Claude Opus 4.6 vs 4.8** | **`0.7100` ★** | `0.6500` | `0.7033` | **148d (Char/Word/Struct)** |
| **Gemini 3.7 vs 3.8 Flash** | **`0.6367` ★** | `0.5667` | `0.6233` | **148d (Char/Word/Struct)** |
| **GPT-5.6 Sol vs Terra** | `0.5333` | **`0.5567` ★** | `0.5467` | **81d (Length-Norm `per100w`)** |
| **GLM-5.3 vs GLM-5.3-Flash** | `0.5267` | **`0.5333` ★** | `0.5200` | **81d (Length-Norm `per100w`)** |

---

## 4. リスク・デメリット・不確実性の明示と次の一手（V44 設計指針）

1. **複数後処理モジュールの単純加算による「過剰補正リスク」（`EXP-08` vs `EXP-09` の教訓）**:
   - 個別では有効な `Pairwise Verifier (0.115)`、`Family Stylometry (0.12)`、`Hard Supplement SVC (0.35)` を減衰なしで同時に重ねた `EXP-08` では、疎プロンプト $m=2$ スコアが `0.93995 -> 0.93912` へ減退しました。
   - これは同一の低マージン行（`margin < 0.35`）に対して複数の補正項が重複加算され、ベースアンサンブルのファミリー質量バランスを崩すためと考えられます。
   - したがって、次期 V44 では `EXP-09` の通り、**`V34.1 Pairwise Verifier (gate=0.35, w=0.10)` を主軸とし、スタイロメトリー補正は `81d` が明確に優位な GPT/GLM（または減衰係数 `alpha=0.05`）に限定する設計**が妥当と判断されます。
2. **Public LB における統計的揺らぎ（$\text{SD}_{\text{gap}}$）**:
   - 各施策の予測不一致率 $f$ は `0.0007〜0.0019`（2,699行中 2〜5 行のフリップ）であり、1,200行の Public LB における統計的標準偏差 $\text{SD}_{\text{gap}} = \sqrt{f / 1200}$ は **`±0.0008〜±0.0012`（約 1 ティク相当）** です。
   - 本日 19:30〜19:35 JST 頃に完了予定の **Track 1 (`V43-Evolution`: `EXP-02`)** と **Track 2 (`V43-Exploration 0.668 Repro`)** の実機スコアを確認した上で、残りの提出スロットを用いて **`EXP-09`（Damped Hybrid）** を投入する方針とします。
