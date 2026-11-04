# Apex-II SOTA アーキテクチャ深層解剖 & 第35弾（V41）設計書

**記録日時**: 2026-11-03 16:00 JST  
**コンペティション**: Kaggle Featured Code Competition「[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)」  
**作成者**: AI Agent (Senior Technical Architect)  

---

## 1. エグゼクティブサマリー

- **現行実績と課題**:
  - 当チームは **Score `0.666`**（Rank 114 位 / 4,393 チーム / Top 2.59%）で銀メダル圏前半を維持。
  - しかし本日、**Silver Cutoff（Top 5% / 219位）が `0.666` に到達**。0.666 には 149 チームが密集しており、金メダル圏（Top 18 / `0.694`）へ抜け出すための「0.666 突破スタック」が不可欠。
- **最新インテル獲得（2026-11-03 最新ノートブック解剖）**:
  - `alexchilton/name-that-llm-0-666-all-public-what-failed`（129 votes）
  - `rahulkrttty/ntl-v16-apex-ii-public-ensemble`（本日公開された Apex-II Grandmaster SOTA）
- **核心的発見**:
  1. **プロンプト内ハンガリアン割り当ての完全失敗**:
     - 隠れテストセットには「1プロンプトあたりの文章数が少ない（few passages per prompt）」ため、オフラインで有効だった Hungarian assignment はパブリックLBで -0.010〜-0.017 の大暴落を起こす。
     - 大域的な周辺分布最適化である **Sinkhorn アルゴリズム** のみが有効。
  2. **0.666 の壁を破る「Apex-II 統合アーキテクチャ」の特定**:
     - **Dual-DeBERTa (`0.75 * deb_pool + 0.25 * deb_orig`)**: 未見プロンプトでの BA が 0.5230 → 0.5763 へ急上昇（`bgunshui/ntl-train-deberta-b` 採用）。
     - **V34.1 Evidence-Aware Pairwise A/B Verifier**: `mercurydust/ntllm-chatgpt-a-b` rev 03 による証拠スパンとプロンプト弱ラベルの照合（所要時間約16秒）。
     - **V40A Sibling-Style Residual (0.10)**: 全体質量を保持しつつ兄弟モデル間のタイ（同点）を解消。
     - **Sinkhorn パラメータの最適連動**: 兄弟・Pairwise の強いシグナルを殺さないため、`--sinkhorn 0.30`（従来 0.50 から軟化）+ `--sinkhorn_target_mix 0.35` に最適化。

---

## 2. Apex-II アーキテクチャの構成要素

```mermaid
flowchart TD
    subgraph Data Layer
        A1["Official Train/Test"]
        A2["Open Pool v11 Decontaminated (190,588 rows)"]
        A3["2x match_weight Extra (49,056 rows)"]
        A4["ChatGPT A/B Rev03 Manifest"]
    end

    subgraph Deep LM Ensemble (T4x2 DP)
        B1["Qwen3-14B Base + Adapter V5 (w=3.0, keep=27)"]
        B2["Qwen3-8B Base + Adapter V5 (w=2.0, keep=24)"]
        B3["Llama-3.1-8B Base + Adapter V5 (w=1.0, keep=24)"]
        B4["Nishant Qwen3-14B Two-Stage (w=2.5)"]
        B5["Qwen2.5-14B In-Style (w=1.0)"]
    end

    subgraph Dual-DeBERTa Stack
        C1["DeBERTa-v3-large Official (0.25)"]
        C2["DeBERTa-v3-large OpenPool Trained (0.75 / BA 0.5763)"]
    end

    subgraph Complementary & Likelihood
        D1["Complementary Stack (Electra, UltraBERT, TokAttn / w=0.15)"]
        D2["ATLAS Source-Conditioned Likelihood (Margin 0.32, w=0.20)"]
    end

    subgraph Micro-Verifiers
        E1["V34.1 Evidence-Aware Pairwise Verifier (~16s)"]
        E2["V40A Sibling-Style Residual (alpha=0.10)"]
        E3["4-view n-gram LinearSVC (w=12)"]
    end

    subgraph Global Regularization
        F["Sinkhorn-Knopp Regularization (lambda=0.30, target_mix=0.35)"]
    end

    Data Layer --> Deep LM Ensemble
    Data Layer --> Dual-DeBERTa Stack
    Data Layer --> Micro-Verifiers
    Deep LM Ensemble & Dual-DeBERTa Stack & Complementary & Likelihood & Micro-Verifiers --> F
    F --> G["Final Prediction (submission.parquet)"]
```

---

## 3. 第34弾（V40）と第35弾（V41: Apex-II）の仕様比較

| 項目 | 第34弾 (V40: 現行) | 第35弾 (V41: Apex-II) | 変更意図 / 期待ゲイン |
| :--- | :--- | :--- | :--- |
| **Public Score** | **`0.666`** (5回完全再現) | **目標: `0.670〜0.675+`** | 0.666 密集帯の脱出と金圏肉薄 |
| **DeBERTa 構成** | 単一（公式学習重みのみ） | **Dual-DeBERTa (`75% Pool + 25% Orig`)** | 未見プロンプト BA +0.053 の飛躍 |
| **A/B 検証器** | なし | **V34.1 Pairwise A/B Verifier** | ChatGPT / Claude 系の微細弁別 |
| **Sibling Residual** | なし | **V40A Sibling Residual (`0.10`)** | GLM / Gemini 兄弟の質量保存的解像度向上 |
| **ATLAS Margin** | `0.32` | `0.32` | 難関サンプルへの尤度評価集中 |
| **Sinkhorn λ** | `0.50` | **`0.30`** | Pairwise/Sibling のシャープな確率を保持 |
| **Target Mix** | `0.42` | **`0.35`** | 周辺制約の微小緩和 |
| **推論所要時間** | 約 6時間30分 | **約 6時間35分** | 9時間制限（VOID崖）に対して安全マージン |

---

## 4. リスク管理と検証計画

1. **タイムアウト・メモリ制約の厳格管理**:
   - `comp_stack.py` を別プロセスで実行して GPU メモリを解放。
   - `V34.1 Pairwise` は CPU 上で約 16 秒で完了するため、GPU 帯域および総実行時間へのインパクトは極小。
   - ATLAS の計算予算（`ATLAS_BUDGET`）を `8.3時間 - comp時間 - 600秒` で動的キャップし、いかなる場合も 9 時間タイムアウトを発生させない。
2. **Kaggle データセット・カーネル依存性の確認**:
   - 必要となる `bgunshui/ntl-train-deberta-b`、`mercurydust/ntllm-chatgpt-a-b`、`areebbb/ntl-we-back-0-666` の存在とアクセス権をローカルスクリプトで事前検証。
3. **提出パイプラインの二重防御**:
   - `wait_and_submit_v41.sh` は、CLI 出力の正規表現依存を全廃し、`kaggle competitions submissions` の最新 レコード ID・日時を直接取得して 1 回のみ提出する安全設計にする。

---

## 5. 実装完了および展開ステータス（2026-11-03 18:52 JST）

- **作業ディレクトリ**: `/Users/echo/kaggle/name-that-llm/submission_0670_v41_apex_ii/`
- **成果物構成**:
  1. `kernel-metadata.json` (ID: `tsuyoshichiba/ntl-0670-v41-apex-ii`, T4 GPU, 依存27アセット完全バインド)
  2. `name-that-llm-0670-v41.ipynb` (ハードコードパス停止脆弱性を完全排除、Dual-DeBERTa 0.75/0.25、V34.1 Pairwise A/B、V40A Sibling-Style 0.10、ATLAS Margin 0.32、Sinkhorn 0.30 統合)
  3. `submit_and_monitor_v41.py` (スロット浪費ゼロ・単発提出保証パイプライン)
- **現在の稼働状況**:
  - **Kaggle Push**: 18:49:51 JST に完了。
  - **FAST_COMMIT**: Kaggle クラウド上で現在実行中（`kernelworkerstatus.running`）。
  - **自律パイプライン**: `submit_and_monitor_v41.py`（PID 27119）が稼働中。
    1. FAST_COMMIT 完了（18:58〜19:00 JST 頃）を自動検知。
    2. 明日 2026-11-04 09:00:15 JST（UTC 00:00:15）の枠リセットまで安全にスリープ待機。
    3. 枠リセット時に **1 回のみ確実に提出** を実行（二重提出完全防止）。
    4. 提出完了後、約 6 時間 35 分の本推論を自律追跡し、スコアと順位を記録。

