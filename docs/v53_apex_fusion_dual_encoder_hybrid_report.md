# NTL 0683 V53 Apex Fusion Architecture & Design Report

**Track 1: Sovereign Apex Evolution (Step 2 - V53 Apex Fusion)**  
- **Target Kernel**: `tsuyoshichiba/ntl-0683-v53-apex-fusion`  
- **Target Metric**: Balanced Accuracy $\to$ 目標 `0.682` 〜 `0.684`（Rank 50〜65 圏内、Gold メダル視野）  
- **Baselines**:
  - V50-Apex (`0.673`、Rank 151、ModernBERT 1.4 + DeBERTa 0.8)
  - V51-Apex (`PENDING`、Target Mix 0.45)
  - V52-Sovereign (`0.680` 骨格、Native Qwen3-Embedding-8B 40% + 5-Fold Staged Parent 60%)

---

## 1. 設計思想と技術的ブレイクスルー

### (1) 0.680 パイプラインの構造的盲点
公開 `0.680` パイプライン（`lightningv08` / `feiyu26`）は、**Native Qwen3-Embedding-8B（全36層、8深度正規化、5-fold、40%混合）** という極めて強力な表現学習エキスパートを備えている一方、**親ブランチ（60%）は旧 `0.666` 世代のまま凍結** されています。
具体的には、直近でスコアを `0.666` $\to$ `0.675`（+0.009）へ急騰させた以下の要素が親ブランチに未導入でした：
1. **ModernBERT-large（512 tokens / SDPA, $w=1.4$）**
2. **DeBERTa-v3 分類器（384 tokens, $w=0.8$）**

### (2) V53 Apex Fusion のハイブリッド結合方式
親ブランチの内部ロジックやバイナリキャッシュ（`ntl660_prefit` 等）を直接改造すると、厳格なアサーションチェックやファイルハッシュ検証が破綻するリスクがあります。
そこで V53 では、`ensemble.py` 内に **独立した GPU サブプロセスステージ（`extra_cls_worker.py`）** を新設しました：

```
[Core Parent Stack (60%)] + [Complementary Stack (15%)] + [ATLAS (10%)]
                        │
                        ▼
                [Initial Mixed Prior]
                        │
   ┌────────────────────┴────────────────────┐
   │                                         │
   ▼ (1 - w_cls = 0.82)                      ▼ (w_cls = 0.18)
[Parent Prior]                     [Standalone Dual-Encoder]
   │                               ├── ModernBERT-large (512 tokens, w=1.4)
   │                               └── DeBERTa-v3-large (384 tokens, w=0.8)
   │                                         │
   └────────────────────┬────────────────────┘
                        │
                        ▼
             [Fused Parent Prior] (60%)
                        │
   ┌────────────────────┴────────────────────┐
   │                                         │
   ▼ (60%)                                   ▼ (40%)
[Fused Parent]                     [Native Qwen3-Embedding-8B]
   │                               (All 36 layers, 5-fold calibrated)
   └────────────────────┬────────────────────┘
                        │
                        ▼ (Geometric Mixture)
            [Pre-Balancing Distribution]
                        │
                        ▼
   [Hierarchical Bayesian Typography + Sinkhorn-Knopp Balancing]
                        │
                        ▼
             [Final Submission Parquet]
```

### (3) 安全性と障害遮断（Graceful Fallback）
`extra_cls_worker.py` の実行およびブレンド処理は `try ... except` ブロックで厳密に保護されており、万一のメモリ不足やパス例外が発生した場合でも、自動的に純粋な V52（`0.680`）パイプラインにフォールバックして完走します。

---

## 2. 実装コンポーネント表

| コンポーネント | 役割 / 設定 | 参照リソース |
| :--- | :--- | :--- |
| **Native Embedding Expert** | 40% 幾何混合、5-fold、全36層、512 tokens | `lightningv08/ntl-olmo2-native24-final-inference-assets` |
| **Parent 2-stage LoRA24** | Qwen3-14B, Qwen3-8B, Llama-3.1, Qwen2.5 | `dhruvtibarewal/nlm-l24-adapters-v5` 等 |
| **Dual-Encoder Classifier** | ModernBERT (1.4) + DeBERTa-v3 (0.8), $w_{cls}=0.18$ | `areebbb/ntl-enc-ours-mbert`, `nuridudak/ntl-two-stage-encoders` |
| **Dynamic Path Resolver** | Kaggle 入力マウントパスの完全自動吸収 | 自作埋め込みモジュール |
| **Post-Processing** | 階層ベイズタイポグラフィ ＋ Sinkhorn マージナル平滑化 | `core_math.py` |

---

## 3. 次のステップ
- V52 の FAST_COMMIT 完了と提出（Ref 発行）を確認後、Kaggle の GPU セッション枠が空き次第、V53 を即時プッシュ・投入します。
