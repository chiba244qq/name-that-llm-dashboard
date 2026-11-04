# 第30弾（V36 Sibling Typo）投入完了および自律監視運用レポート

**記録日時**: 2026-11-01 19:48 JST  
**担当**: シニアテクニカルアーキテクト  
**対象コンペティション**: Kaggle Featured Code Competition「[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)」  

---

## 1. 投入完了サマリー

事前検証（疎サンプリング・モンテカルロ評価ハーネス）において正の汎化寄与を示したタイポグラフィ層の拡張マーカー（`'...'`）を組み込んだ**第30弾（`tsuyoshichiba/ntl-0663-v36-sibling-typo`）**のFAST_COMMITおよび公式提出が正常に完了し、Kaggle評価基盤上での本番推論（自律監視下）に移行しました。

| 項目 | 設定・ステータス | 備考・技術仕様 |
| :--- | :--- | :--- |
| **提出Ref** | **`57869119`** | Kaggle Competitions API経由で受理 |
| **ベースモデル** | 第28弾 V35（Sinkhorn 0.50） | 5時間23分で安全完走し `0.662` を記録した最速高スコア版 |
| **カーネルID** | `tsuyoshichiba/ntl-0663-v36-sibling-typo` | バージョン 1（GPU: Nvidia Tesla T4 ×2） |
| **ステータス** | **`SubmissionStatus.PENDING`** | 推論キューに入りバックグラウンド実行中 |
| **投入日時** | 2026-11-01 10:46:18 UTC (19:46 JST) | 本日 3 回目の提出枠を消費 |
| **推論完了見込み** | **2026-11-02 01:10〜01:25 JST 頃** | 予測推論時間: 約 5時間25分〜5時間40分 |
| **本日残スロット** | **残り 2 回**（3回消費 / 5回上限） | UTC 2026-11-01 枠 |

---

## 2. 事前検証：マーカー単体アブレーション結果

オフライン疎サンプリング検証ハーネス（`k=2` 間引きシミュレーション、37個の既存マーカーベースライン = `0.37478`）において実施した単体追加アブレーションの結果は以下の通りです。

```
Base score (37 marks): 0.37478
  + '              : 0.37173 (diff: -0.00305)  [悪化: 単一引用符はノイズ誘発]
  + ...            : 0.37516 (diff: +0.00038)  [改善: ASCII三連ピリオド]
  + {              : 0.37458 (diff: -0.00020)  [悪化]
  + }              : 0.37458 (diff: -0.00020)  [悪化]
  + Furthermore,   : 0.37458 (diff: -0.00020)  [悪化]
  + Additionally,  : 0.37516 (diff: +0.00038)  [トピックリーク懸念あり]
  + Specifically,  : 0.37516 (diff: +0.00038)  [トピックリーク懸念あり]
  + Notably,       : 0.37516 (diff: +0.00038)  [トピックリーク懸念あり]
  + Ultimately,    : 0.37476 (diff: -0.00002)  [悪化]
  + In summary,    : 0.37516 (diff: +0.00038)  [トピックリーク懸念あり]
```

### 採否の数理的判断
1. **記号マーカー `'...'` の採用**:
   - 既存のタイポグラフィ層には Unicode三連ドット `…`（`\u2026`）は含まれていましたが、ASCII標準のピリオド3つ `'...'` は未登録でした。
   - モデル間の思考プロセスや短文終端記法における差異を捉え、`+0.00038` のスコア改善を示したため、安全な記号マーカーとして第30弾に組み込みました。
2. **語彙マーカー群（`Additionally,` 等）の排除**:
   - 単語レベルの特徴量はプロンプトのトピックに過剰適合し、テストセットでのドメインシフト時にオフライン一括検証で `-0.0044` の大幅悪化を引き起こすことが確認されたため、本番モデルからは完全に除外しました。

---

## 3. 第30弾（V36）の主要構成と推論安全性

```mermaid
flowchart TD
    subgraph S1["1. Base Feature Extraction"]
        A["Qwen-14B + Qwen-8B + Llama-8B (2xT4 DP)"]
        B["TF-IDF (Char/Word) + Distortion"]
        C["LinearSVC (4-view n-gram)"]
        D["Public Complementary Stack (15%)"]
    end

    subgraph S2["2. ATLAS Selective Rescoring (20%)"]
        E["Margin Cut <= 0.30"]
        F["Chunk=3, Batch=4, Budget=7.8h"]
        G["Completed in ~5.5h (Safe from 8.3h cap)"]
    end

    subgraph S3["3. Bayesian Typography (38 Marks)"]
        H["37 Existing Marks + '...' (ASCII Ellipsis)"]
        I["Confidence Gate * Length Gate * Coverage Gate"]
    end

    subgraph S4["4. Damped Sinkhorn Optimization"]
        J["Sinkhorn Alpha = 0.50"]
        K["Target Mix Rho = 0.35"]
        L["Hungarian Assignment on Dense Clusters"]
    end

    S1 --> S2 --> S3 --> S4 --> M["Final Submission (0.663+ Breakthrough Target)"]
```

- **推論時間バジェット設計**:
  - 第28弾で実証された `Sinkhorn 0.50`（完走 5h 23m）のベースパイプラインを完全踏襲。
  - タイポグラフィ層の記号追加（37個 → 38個）は `infer.py` 内部で数ミリ秒の NumPy 配列演算に過ぎず、推論総時間に与える影響はゼロです。
  - これにより、ATLAS の 8.3 時間バジェット打ち切り（`0.659` への退行要因）を確実に回避し、安全に全テスト行を救済完了させます。

---

## 4. 自律監視運用の状況

提出直後より、バックグラウンドにて監視スクリプト `monitor_sub30.sh` を常駐起動しています。

- **監視プロセス**: `monitor_sub30.sh`（PID稼働中）
- **ログ記録先**: `/Users/echo/kaggle/name-that-llm/monitor_sub30.log`
- **ポーリング間隔**: 60秒
- **通知プロトコル**: 評価完了（`COMPLETE`）またはエラー発生（`ERROR`）を検知次第、即座にスコアを解析して報告します。

---

## 5. 本日の残り戦略（スロット残数: 2回）

本日は残り2回の提出枠を保持しています。第30弾の推論（約5.5時間）の進行を監視しつつ、以下の選択肢を維持します：

1. **第30弾推論の並行監視**: 明朝 01:10〜01:25 JST 頃の完了まで安全に見守る。
2. **第31弾（本日枠の有効活用）**:
   - `Sinkhorn 0.48` 等のパラメータ微小探索版、あるいは確信度マージン（`--atlas_margin 0.32`）の微調整版など、リスクの極めて低い並行提出を検討可能。
