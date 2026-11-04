# Kaggle「Name That LLM」Live Command Center & Dashboard デプロイ完了記録

**デプロイ日時**: 2026-11-04 09:29 JST  
**公開 URL**: **[https://chiba244qq.github.io/name-that-llm-dashboard/](https://chiba244qq.github.io/name-that-llm-dashboard/)**  
**GitHub リポジトリ**: [chiba244qq/name-that-llm-dashboard](https://github.com/chiba244qq/name-that-llm-dashboard)  
**作成者**: AI Agent (Senior Technical Architect)  

---

## 1. エグゼクティブサマリー

- **ダッシュボード公開完了**:
  - Kaggle Featured Code Competition「Name That LLM」のリアルタイム進行状況を可視化するモダン・レスポンシブダッシュボードを GitHub Pages 上に構築・公開（HTTP 200 確認済み）。
  - PC・スマートフォンいずれのブラウザからもアクセス可能。
- **直近の稼働ハイライト**:
  - **第35弾（V41: Apex-II）**: 本日 09:00:17 JST に Ref `57963401` として投入成功。現在 Kaggle クラウド上で推論実行中（所要時間約6時間35分 / 15:35 JST 完了見込み）。
  - **現在のリーダーボード成績**: 4,522 チーム中 **Rank 189 位（Top 4.18% / 銀メダル安全圏内）**、最高スコア **`0.666`**。

---

## 2. ダッシュボードの主要コンポーネント

```
┌────────────────────────────────────────────────────────────────────────┐
│  K  Name That LLM | Live Command Center        ● Last Updated: 09:33   │
│     [Live Metrics] [Archives (20)]                 [Kaggle LB ↗]       │
├────────────────────────────────────────────────────────────────────────┤
│  ⚡ CURRENT RUN IN PROGRESS: V41 Apex-II (Ref: 57963401) [RUNNING]     │
│     Submitted: 09:00 JST | Target: 0.670 - 0.675+ | Est: ~15:35 JST   │
├──────────────────┬──────────────────┬──────────────────┬───────────────┤
│   Public Rank    │    Peak Score    │   Medal Tier     │ Cutoff Radar  │
│    189 / 4,522   │      0.666       │  🥈 Silver Medal │ Gold:   0.697 │
│    (Top 4.18%)   │  (Balanced Acc)  │ (Top 5% / 226位) │ Silver: 0.666 │
│                  │                  │                  │ Bronze: 0.665 │
├──────────────────┴──────────────────┴──────────────────┴───────────────┤
│  📈 Score Evolution & Breakthrough Trajectory (Interactive Chart.js)   │
│     V28 (0.662) ──> V38 (0.665) ──> V39/V40 (0.666) ──> V41 Target     │
├────────────────────────────────────────────────────────────────────────┤
│  🧩 Pipeline Evolution Matrix (V38 〜 V41 の各パラメータ・重み比較)    │
├────────────────────────────────────────────────────────────────────────┤
│  📋 Recent Submissions Log (Kaggle API 直結の公式提出履歴)             │
├────────────────────────────────────────────────────────────────────────┤
│  📚 Strategic Reports & Engineering Archives (時系列レポート集: 全20本) │
│     [すべて (20)] [提出・運用] [設計・戦略] [総合ガイド] [インテル調査] │
│     ・各カードをクリックすると、その場でフルMarkdownビューアが展開    │
│     ・見出し、テーブル、数式、コードブロックを綺麗にレンダリング      │
└────────────────────────────────────────────────────────────────────────┘
```

1. **リアルタイム実行バナー（Active Run Banner）**:
   - 現在実行中の V41（Ref: 57963401）のステータス、投入時刻、目標スコア（`0.670〜0.675+`）、推論完了予定時刻（`~15:35 JST`）をパルスアニメーション付きで表示。
2. **コア KPI グリッド**:
   - 現在順位（Rank 189 / 4,522 チーム、Top 4.18%）
   - 当チーム最高スコア（`0.666`）
   - メダル獲得ステータス（Silver Medal）
   - リアルタイムボーダーライン（Gold 0.697 / Silver 0.666 / Bronze 0.665）
3. **インタラクティブスコア推移グラフ（Chart.js）**:
   - ブロンズ密集崖（0.662）から銀メダル圏突破（0.665）、0.666完全防衛、そして最新V41の目標値までの進化の軌跡を折れ線グラフで可視化。
4. **アーキテクチャ比較マトリクス**:
   - V38、V39、V40、V41（Apex-II）のデータプール、Dual-DeBERTa比率、Pairwise A/B、Sibling Residual、Sinkhorn λ などの設計差分を表形式で整理。
5. **サブミッション履歴テーブル**:
   - Kaggle API から取得した直近の提出レコード（Ref、日時、ステータス、スコア、説明文）。
6. **時系列戦略レポート集 & インライン Markdown ビューア（新機能）**:
   - コンペ初期から作成した全 20 本の Markdown ドキュメント（中学生向け総合解説、Gold戦略書、OOF検証ログ、各回提出配備レポート等）を時系列でカード一覧化。
   - カテゴリフィルター（すべて、提出・運用、アーキテクチャ設計、総合ガイド、競合調査）完備。
   - カードをクリックすると、**`marked.js` によりその場でモーダル表示され、見出し・コード・表・数式を含んだ美しい GitHub スタイルで全文を閲覧可能**。


---

## 3. ディレクトリ構成と更新手順

### 3.1 ディレクトリ構成
```
/Users/echo/kaggle/name-that-llm-dashboard/
├── build_data.py          # Kaggle API から最新LB・提出データを取得し data.json を生成
├── data.json              # ダッシュボード用集計データ
├── index.html             # フロントエンドUI（Tailwind CSS + Chart.js）
├── update_and_deploy.sh   # データ更新 & GitHub push を行うワンクリック自動化スクリプト
└── README.md
```

### 3.2 ワンクリック更新コマンド
最新の Kaggle 順位や推論結果を反映して GitHub Pages に反映したい場合は、ターミナルから以下を実行するだけで完了します：

```bash
cd /Users/echo/kaggle/name-that-llm-dashboard
./update_and_deploy.sh
```
