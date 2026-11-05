# Submission 37 (V43-Evolution) Deployment & Operations Report

**記録日時**: 2026-11-05 12:55 JST  
**対象コンペ**: Kaggle「Name That LLM」  
**トラックスコープ**: **Track 1: Evolution（堅実進化）**  
**カーネル**: `tsuyoshichiba/ntl-0673-v43-evolution` (Ref: 待機中)  
**コードベース**: `/Users/echo/kaggle/name-that-llm/submission_0673_v43_evolution/name-that-llm-0673-v43.ipynb`  

---

## 1. 投入背景と目的

直近の第36弾並行アブレーション（V42 A/B/C）および Kaggle 公開上位コード（0.667〜0.668）の diff 比較により、以下の事実が実証・確定しました：
1. **外部パーツのノイズ特定**: 外部の `bgunshui Dual-DeBERTa 0.75` を無調整で組み込んだことで、Public テストセットに対して -0.005 の減退が発生していた（V40: 0.666 $\to$ V42-C: 0.661）。
2. **ベースライン純度の証明**: 自チームの V40（0.666）スタック（Single DeBERTa + Clean OpenPool + Sinkhorn 0.50 + ATLAS Margin 0.32）が最も純度が高く頑健である。
3. **オフラインOOFによる裏付け**: 手元 5-Fold OOF（2,699件）によるグリッド探索において、**Target Mix 0.42** は Mix 0.35 に対し **+0.00112 の確実なアドバンテージ** を記録している。

現在、リーダーボードでは `0.666` に約 30〜40 チームが密集しており、同点タイのままタイムスタンプ順で 256 位（Top 5.47%）に位置しています。**`0.667`（+0.001）を叩き出した瞬間にこの大集団を一掃し、Top 100〜150 圏へ復帰できる** 状況です。

---

## 2. V43-Evolution の主要アーキテクチャ仕様

| 構成要素 | V40 Apex Sovereign (基準) | V43 Evolution (本モデル) | 変更理由 / 数理的根拠 |
| :--- | :--- | :--- | :--- |
| **Backbone** | Single DeBERTa + 5-LoRA | Single DeBERTa + 5-LoRA | 減退原因の外部 Dual-DeBERTa を完全破棄（Revert） |
| **Data Pool** | OpenPool v11 Clean (190k) | OpenPool v11 Clean (190k) | 汚染のないクリーンデータプールを維持 |
| **ATLAS** | Margin 0.32, w=0.20 | Margin 0.32, w=0.20 | Gemini/GPT 識別境界の鋭利化を維持 |
| **Sinkhorn強度** | $\alpha = 0.50$ | $\alpha = 0.50$ | Balanced Accuracy への強固な正則化を維持 |
| **Target Mix ($\rho$)** | **$0.35$** | **$0.42$** | **手元 5-Fold OOF 検証済み (+0.00112)。中短文の兄弟混同を平滑化** |
| **推論制限時間** | ~6.5 時間 | ~6.5 時間 | 9時間タイムアウト（VOID崖）に対して十分な安全マージン |

---

## 3. 数理仕様：Target Mix 0.42 の作用機序

Soft Sinkhorn における事後クラス分布の目標マージナル $t$ は、予測マージナル $\bar{p}$ と学習データ事前分布 $u$ の線形結合として定義されます：

\[
t_c = (1 - \rho) \bar{p}_c + \rho u_c \quad (\rho = 0.42)
\]

* **従来の $\rho = 0.35$**: 予測マージナルへの依存度が 65% と高く、テストセット内の局所的な偏り（特に GLM-5.3 や Gemini Flash 兄弟の誤予測）に引きずられやすい。
* **本モデルの $\rho = 0.42$**: 正則化事前分布へのブレンド率を 42% まで引き上げることで、50〜100 words の中短文において発生する「兄弟モデル間の過度な極端予測」を抑制し、Balanced Accuracy の期待値を押し上げる設計です。

---

## 4. デプロイおよびモニタリング計画

* **デプロイ方式**: FAST_COMMIT 経由の 2 段階実行（GPU 割当 2 枠制限を回避）。
* **監視プロセス**: バックグラウンドスクリプト [`submit_and_monitor_v43.py`](file:///Users/echo/kaggle/name-that-llm/submission_0673_v43_evolution/submit_and_monitor_v43.py) により、Kaggle API 経由でステータスおよびスコアを自動追跡。
* **ダッシュボード連携**: サブミット登録時および推論完了時に GitHub リポジトリ（`chiba244qq/name-that-llm-dashboard`）へ自動 push 同期。
