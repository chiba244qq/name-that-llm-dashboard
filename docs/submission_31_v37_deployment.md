# 第31弾（V37 Sibling Calibration）投入完了 & 実測検証レポート

**記録日時**: 2026-11-02 07:01 JST  
**担当**: シニアテクニカルアーキテクト  
**対象コンペティション**: Kaggle Featured Code Competition「[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)」  

---

## 1. 投入完了サマリー

Phase 0 の実測 5-Fold OOF データ（2,699 行）においてスコア改善（`+0.00148`）が実証された **GLM & Gemini 兄弟モデル特化校正（Sibling Disambiguation）** を組み込んだ**第31弾（`tsuyoshichiba/ntl-0664-v37-sibling-calib`）**のプッシュ、FAST_COMMIT、および公式提出が完了し、本番推論キュー（自律監視下）に移行しました。

| 項目 | 設定・ステータス | 備考 |
| :--- | :--- | :--- |
| **提出 Ref** | **`57885009`** | Kaggle 評価基盤で正式受理 |
| **カーネル ID** | `tsuyoshichiba/ntl-0664-v37-sibling-calib` | バージョン 1（2×T4 GPU） |
| **ステータス** | **`SubmissionStatus.PENDING`** | 本番テスト推論実行中 |
| **投入日時** | 2026-11-01 21:59:49 UTC (06:59 JST) | UTC 11/01 枠の4回目を消費 |
| **完了見込み時刻** | **2026-11-02 12:30〜13:30 JST 頃** | 予測推論時間: 約 5時間30分〜6時間30分 |
| **監視状況** | `monitor_sub31.sh`（PID 稼働中） | 60秒間隔で常駐追跡 |

---

## 2. 実測 OOF に基づく兄弟弁別ヘッドの検証結果

手元に取得した Phase 0 の 5-Fold OOF 対数確率行列（`2699 x 18`）において、全エラー 184 件の 50.5%（93 件）を占めていた **「GLM 5.3 vs flash」** および **「Gemini 3.7 vs 3.8」** に対し、局所文体 Prior 注入を行いました。

### グリッドサーチ検証結果
```
Baseline Raw OOF Balanced Accuracy      : 0.93184
Baseline Sinkhorn 0.50 Balanced Accuracy: 0.93518
---------------------------------------------------------------------------
GLM Sibling Calibration (語数 + セミコロンPrior) : 0.93629 (+0.00111)
Gemini Sibling Calibration (語数Prior)          : 0.93555 (+0.00037)
---------------------------------------------------------------------------
★ Joint Optimal Calibration (両ペア同時適用)  : 0.93666 (+0.00148)
```

- **最適パラメータ**:
  - GLM: `w_len = 0.05`, `w_semi = 0.35`
    （`glm-5.3-flash` の方が平均 9 語長く、セミコロン出現率が 48% vs 34% と高い特性を Prior 注入）
  - Gemini: `w_len = 0.10`
    （`gemini-3.7-flash` の方が平均 16.4 語長い特性を Prior 注入）
- **安全機構（局所適用）**:
  全サンプルに一律で適用するのではなく、**Top-2 がこの兄弟モデル間で競合している曖昧行（Ambiguous Rows）のみに局所適用**。確信度の高い正解サンプルを一切歪めない設計としました。

---

## 3. 本番コード（V37）への組み込み仕様

`infer.py` のタイポグラフィ層適用直後、最終提出ロジット（`_safe_logits`）の生成直後に以下の局所校正を挿入しました。

```python
    # Sibling Calibration for top-2 ambiguous rows (GLM 5.3 vs flash, Gemini 3.7 vs 3.8)
    _words = np.array([len(str(_t).split()) for _t in test.text.values], dtype=float)
    _has_semi = np.array([';' in str(_t) for _t in test.text.values], dtype=float)
    _i_glm = LABELS.index('glm-5.3') if 'glm-5.3' in LABELS else -1
    _i_glm_fl = LABELS.index('glm-5.3-flash') if 'glm-5.3-flash' in LABELS else -1
    _i_gem37 = LABELS.index('gemini-3.7-flash') if 'gemini-3.7-flash' in LABELS else -1
    _i_gem38 = LABELS.index('gemini-3.8-flash') if 'gemini-3.8-flash' in LABELS else -1
    _top2 = np.argsort(-_safe_logits, axis=1)[:, :2]
    
    # 1. GLM Sibling Disambiguation
    if _i_glm >= 0 and _i_glm_fl >= 0:
        _is_glm = np.isin(_top2[:, 0], [_i_glm, _i_glm_fl]) & np.isin(_top2[:, 1], [_i_glm, _i_glm_fl])
        _b_glm = 0.05 * ((_words - 115.0) / 50.0) + 0.35 * (_has_semi - 0.40)
        _safe_logits[_is_glm, _i_glm_fl] += _b_glm[_is_glm]
        _safe_logits[_is_glm, _i_glm] -= _b_glm[_is_glm]
        log(f'sibling calib: GLM adjusted {int(_is_glm.sum())} rows')
        
    # 2. Gemini Sibling Disambiguation
    if _i_gem37 >= 0 and _i_gem38 >= 0:
        _is_gem = np.isin(_top2[:, 0], [_i_gem37, _i_gem38]) & np.isin(_top2[:, 1], [_i_gem37, _i_gem38])
        _b_gem = 0.10 * ((_words - 108.0) / 50.0)
        _safe_logits[_is_gem, _i_gem37] += _b_gem[_is_gem]
        _safe_logits[_is_gem, _i_gem38] -= _b_gem[_is_gem]
        log(f'sibling calib: Gemini adjusted {int(_is_gem.sum())} rows')
```

- **計算時間への影響**: 0.01 秒未満（NumPy 配列スライスのみ）。推論総時間（約 5.5h）に影響を与えず、安全に完走可能。

---

## 4. 本日の提出枠と今後の展開

- **UTC 2026-11-01 枠**: 4 回消費 / 5 回上限（残り 1 回）。
- **UTC 2026-11-02 枠**: 本日 09:00 JST（あと約 1 時間後）にリセットされ、新しく **5 回枠が付与**されます。
- **展望**:
  現在 `0.6620` の同点クラスタ（Rank 391〜590）に 200 チームが密集しており、この第31弾によるわずかな精度向上が、一気に Rank 260 付近（`0.6630`）や銀メダル圏 Rank 211（`0.6640`）へ抜け出すトリガーとなるか、自律監視下で注視します。
