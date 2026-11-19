# 戦況評価・レポート監査およびダッシュボード今後の方針（2026-11-19）

**作成日時**: 2026-11-19 09:58 JST

---

## 1. 最新の LB 構造分析（`2026-11-19 09:56 JST` 実測）

現在ダッシュボード（[`data.json`](file:///Users/echo/kaggle/name-that-llm-dashboard/data.json)）に表示されている `11/16` 時点の数値と、本日の最新 LB 実測値には大きな構造変化が生じています。

| 指標 | ダッシュボード旧値（`11/16`） | **最新実測値（`11/19`）** | 変化・戦略的意味 |
| :--- | :---: | :---: | :--- |
| **参加チーム数** | `6,008` | **`6,370`** | `+362` チーム増 |
| **現在スコア / 順位** | `0.696` / `341位` (`5.68%`) | **`0.698` / `1,009位` (`15.84%`)** | スコアは `+0.002` 更新するも順位後退 |
| **メダル圏** | `Bronze` | **`No Medal`** | 公開 `0.702` カーネル大量コピーの影響 |
| **Bronze ボーダー（Top 10%）** | `0.691`（600位） | **`0.702`（637位）** | **`+0.011` 急上昇** |
| **Silver ボーダー（Top 5%）** | `0.696`（300位） | **`0.702`（318位）** | Bronze と同点の **`0.702` に大密集** |
| **Gold ボーダー（Top 18）** | `0.743`（18位） | **`0.751`（18位）** | `+0.008` 上昇（1位は `0.776`） |

### スコア帯別のチーム分布（最重要の崖）
- `score >= 0.698`: `1,010` チーム
- `score >= 0.700`: `984` チーム
- **`score >= 0.702`**: **`707` チーム**（公開 `lightningv08 0.702` 系カーネルによる巨大な団子状態）
- **`score >= 0.705`**: **`153` チーム**（**Top 2.40% — 一気に 850 順位以上を飛び越えて安定 Silver 圏**）
- `score >= 0.710`: `116` チーム / `score >= 0.720`: `69` チーム / `score >= 0.750`: `20` チーム

> [!IMPORTANT]
> 現在のコンペ構造は **「`0.702` 公開団子（約550チームが 0.702〜0.704 に密集）」** と **「`0.705+` の独自モデル勢（153チームのみ）」** に完全に二極化しています。したがって目標ラインは「`0.700` 到達」ではなく、**まず `0.705+`（Top 153 / 安定 Silver）を超えること**に再定義する必要があります。

---

## 2. 直近レポートおよび実験群（`NTL 0700`〜`0711`）の客観的評価

### 2.1 検証により確定した成果（採用）
1. **基盤刷新の効果実証（`NTL 0700: 0.696` → [`NTL 0707: 0.698`](file:///Users/echo/kaggle/name-that-llm/submission_0707_baseline_nocomp_gsq_v1/ntl-0707-baseline-nocomp-gsq-v1.ipynb)）**
   - `comp_stack.py`（`comp_w=0.15`）を無効化し、`GSQ 27B (6.3h)`・`Native Q8 (7.2h)` の完走余地と `V1` の現実的ロード見積り（`fixed_s_flash=870 / fixed_s_glm53=1140`）を適用したことで自己ベスト `0.698` を樹立しました。
2. **`V1` カバレッジの最適点確認**
   - 同一基盤（`0707`）上の比較で、`0.15`（[`0709`](file:///Users/echo/kaggle/name-that-llm/submission_0709_baseline_v1_fixed15/ntl-0709-baseline-v1-fixed15.ipynb): `0.694`） < **`0.30`（`0707`: `0.698`）** > `0.45 + pack`（[`0711`](file:///Users/echo/kaggle/name-that-llm/submission_0711_v1_pack_cover45/ntl-0711-v1-pack-cover45.ipynb): `0.695`）を確認しました。

### 2.2 棄却確定と最も重要な方法論的教訓（OOF→LB 乖離）
1. **Surprisal 注入の 5 戦全敗（完全打ち切り判断は妥当）**
   - `0702 (0.695)`, `0703 (0.694)`, `0705 (0.690)`, `0706 (0.694)`, `0708 (0.693)` の全 5 回でベースラインを下回りました。
   - **重大な教訓**: 我々のローカル OOF 代理モデル（`qwen8 + wemm4 + q35_heads` のみ）では `P5 + Gem4-12B` が `Mean4 +0.00583` 改善したにもかかわらず、本番 `S10 + GSQ 27B + V1` 上では `-0.002 〜 -0.005` 悪化しました。**旧 3 モデル代理 OOF は本番 `S10 + V1`（特に `V1` 時間枠とのトレードオフ）を十分に再現できていない**という検証基盤側の限界が露呈しています。
2. **公開後処理スタックの全滅（ローカル OOF 監査）**
   - `Apex V4`（`Mean4 -0.00224`, `P=0.020`）、`StyleLM`（`-0.00122`）、`gm_champ V18 GRL+Ordinal+Residual`（`-0.02183`, `P=0.000`）、`Hungarian`（公開実験でも悪化）、`Open-Pool CPU Spec`（`0704: 0.684` / `0705: 0.690`）のすべてが不採用と判定されました。
3. **路線 A（Open Pool + LoRA + kNN）の評価（前ターンレポートの検証）**
   - 前ターンの評価（肯定根拠：`sadamtorres` で Open Pool 12k + 3 ファミリーが `0.493 -> 0.638` / 否定リスク：`gm_champ` で単体 `0.642` モデルを `0.651` に混ぜて `0.638` に低下＋9h 枠のゼロサム制約）は客観的かつ妥当です。
   - **追加で認識すべき現実**: `0.705+`（153チーム圏）を突破するには、もはや公開 `S10` のハイパラ調整（路線 B）だけでは期待値が低く（`±0.004〜0.007` の LB 分散内に留まる）、**「既存 `S10` の弱いメンバーを入れ替えてでも勝てる強い独自モデル」または「GLM-5.3 以外のターゲットに対する直接尤度（V1 型）」** のいずれかに踏み込む必要があります。

---

## 3. ダッシュボード（`name-that-llm-dashboard`）の現状診断と改修方針

[`/Users/echo/kaggle/name-that-llm-dashboard/`](file:///Users/echo/kaggle/name-that-llm-dashboard) のコード監査の結果、ダッシュボードが **11/16（`V58` / `0700` 時代）で停止しており、現状の意思決定支援ツールとして機能不全を起こしている 5 つのギャップ** を特定しました。

| # | 対象ファイル | 現状のギャップ（陳腐化・問題点） | 改修方針 |
| :-: | :--- | :--- | :--- |
| 1 | [`data.json`](file:///Users/echo/kaggle/name-that-llm-dashboard/data.json) / [`build_data.py`](file:///Users/echo/kaggle/name-that-llm-dashboard/build_data.py) | `11/16` 停止（`341位 / Bronze / 0.696` の誤表示）。`0.702` 団子と `0.705+` 崖が見えない | `build_data.py` を拡張し、最新順位（`1,009位 / 0.698 / No Medal`）に加え、**スコア帯別クラスタ統計（`>=0.702`, `>=0.705`, `>=0.710`, `>=0.750` のチーム数と自分の距離）** を自動算出・表示する |
| 2 | [`build_data.py`](file:///Users/echo/kaggle/name-that-llm-dashboard/build_data.py) (`timeline`, `architecture_matrix`, `active_runs`) | `NTL 0701` で停止。`0702〜0711` が未登録。`active_runs` パーサーが `V41〜0694` の旧ハードコード | `NTL 0702〜0711`（`0707: 0.698` 樹立、Surprisal 検証、V1 3点比較）を全件反映し、`active_runs` を `NTL 07xx` 以降の汎用正規表現パースへ刷新する |
| 3 | [`local_verification_results.json`](file:///Users/echo/kaggle/name-that-llm-dashboard/local_verification_results.json) | `11/11`（`V57 Currently Running`）で停止。直近の OOF 監査と OOF→LB 乖離が未記録 | `11/17〜11/19` のベンチマーク（`Apex V4`, `StyleLM`, `GRL+Ordinal`, `P5+Gem4`, `P5+Q3.5-9B`）と **OOF vs LB 乖離フラグ** を反映する |
| 4 | [`index.html`](file:///Users/echo/kaggle/name-that-llm-dashboard/index.html) | 見出しが旧名（`V38 → V58 Dual-Embedding Sovereign`）。棄却ログ（Do-Not-Repeat）と時間予算内訳がない | 見出し・ハイライトを `NTL 0707 (0.698)` 世代へ刷新し、**① 0.705+ Silver 突破トラッカー、② 棄却・失敗レジストリ（再発防止一覧）、③ 9h ランタイム配分カード** を追加する |
| 5 | [`docs/`](file:///Users/echo/kaggle/name-that-llm-dashboard/docs) & `REPORT_METADATA` | [`strategic_verification_plan_1118.md`](file:///Users/echo/.gemini/antigravity/brain/ada99bbf-90ae-436b-9d82-11c0881d9733/strategic_verification_plan_1118.md) が未同期。旧 `V52〜V57` 記事が `STEP 0` を占有 | 本レポートと `1118` 計画を `STEP 0` に昇格させ、`V57` 以前の旧設計レポートを `archived` へ整理し、[`update_and_deploy.sh`](file:///Users/echo/kaggle/name-that-llm-dashboard/update_and_deploy.sh) で GitHub Pages へ即時反映する |

---

## 4. 今後の実行ロードマップ（ダッシュボード刷新 ＋ コンペ戦略）

### フェーズ 1：ダッシュボード即時刷新（本日・提出枠消費ゼロ・所要約 10 分）
1. [`build_data.py`](file:///Users/echo/kaggle/name-that-llm-dashboard/build_data.py)・[`local_verification_results.json`](file:///Users/echo/kaggle/name-that-llm-dashboard/local_verification_results.json)・[`index.html`](file:///Users/echo/kaggle/name-that-llm-dashboard/index.html) を上記 5 ギャップ解消仕様へ改修。
2. [`update_and_deploy.sh`](file:///Users/echo/kaggle/name-that-llm-dashboard/update_and_deploy.sh) を実行し、最新 LB（`6,370チーム / 0.702 団子 / 0.705 銀崖`）・`NTL 0702〜0711` 履歴・棄却レジストリ・本レポートを GitHub Pages にデプロイ。

### フェーズ 2：コンペ本体の次期検証（目標：`0.705+` 安定 Silver 突破）
- **優先検証 1（`0.698 -> 0.702+` 差分の原因切り分け）**:
  - `NTL 0707 (0.698)` と公開 `0.702` の残り差分（`V1` の `cum<=90000` 固定 vs 動的 `870/1140`、および `light_apex` 本日版の「GLM 勝者のみ `post-glm` 確率参照＋タイポグラフィ規則」）の挙動を精査し、まずベースラインの再現上限を確定させる。
  - *リスク*: `Apex` 系後処理はローカル OOF で悪化しているため、安易な取り込みは避け、採用する場合は 1 軸単体での厳格比較に留める。
- **優先検証 2（`0.705+` 突破に向けた独自シグナル開発）**:
  - **アプローチ A（他ターゲット尤度調査）**: `V1`（`GLM-5.3 / Flash`）で `+0.012` 出た仕組みを参考に、18 クラスのうちオープンウェイトまたは近似モデルで直接対数尤度・困惑度特徴が取れる別クラス（例：`qwen3.8-max` / `kimi-k3` 周辺）がないか技術調査する。
  - **アプローチ B（路線 A 小規模パイロット）**: `NTL 0707` 内の既存メンバーの推論時間・単体強度を棚卸しした上で、Colab A100（1〜2時間）で異ファミリー（例：`Gemma-4-12B` 等）の Open Pool 拡張 LoRA パイロットを回し、**「既存メンバーと差し替えて 9h 枠内で純増するか」** を検証する。
- **優先検証 3（ローカル検証基盤の信頼性回復）**:
  - Surprisal で起きた「OOF `+0.006` → LB `-0.005`」の再発を防ぐため、ローカル評価時に `V1` 相当の GLM 尤度・`GSQ` 相当の重みを加味した評価指標、および「追加モデル 1 分あたりの `V1` カバレッジ損失コスト」をダッシュボードのゲート基準として明文化する。
