#!/usr/bin/env python3
"""Build latest competition and submission metrics into data.json for dashboard."""
import os
import re
import sys
import json
import zipfile
import tempfile
import datetime
import glob
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

DOC_DATES = {
    "submission_25_sinkhorn60_deployment.md": "2026-10-31 12:00 JST",
    "dual_submission_operations_ab.md": "2026-10-31 21:00 JST",
    "ntl_master_strategy_and_operations.md": "2026-10-31 21:45 JST",
    "submission_28_sinkhorn50_deployment.md": "2026-11-01 10:27 JST",
    "offline_validation_and_sibling_mining.md": "2026-11-01 11:02 JST",
    "submission_28_29_results_and_next_steps.md": "2026-11-01 19:12 JST",
    "submission_30_v36_deployment_and_monitoring.md": "2026-11-01 19:48 JST",
    "gold_medal_master_strategy.md": "2026-11-01 23:00 JST",
    "phase0_oof_baseline_deployment.md": "2026-11-01 23:07 JST",
    "phase0_oof_results_and_breakthrough_findings.md": "2026-11-02 06:50 JST",
    "submission_31_v37_deployment.md": "2026-11-02 07:01 JST",
    "sibling_disambiguation_training_plan.md": "2026-11-02 14:34 JST",
    "submission_32_v38_deployment.md": "2026-11-02 15:11 JST",
    "v39_verification_and_tradeoff_analysis.md": "2026-11-02 21:50 JST",
    "kaggle_discussion_and_code_intel_report.md": "2026-11-02 21:51 JST",
    "submission_33_v39_deployment.md": "2026-11-02 22:33 JST",
    "submission_34_v40_deployment.md": "2026-11-03 08:10 JST",
    "apex_ii_sota_architecture_and_v41_plan.md": "2026-11-03 16:00 JST",
    "name_that_llm_comprehensive_guide.md": "2026-11-03 21:55 JST",
    "kaggle_live_dashboard_deployment.md": "2026-11-04 09:29 JST",
    "offline_oof_and_error_analysis_report.md": "2026-11-04 17:42 JST",
    "dual_track_strategy_and_operations.md": "2026-11-04 20:20 JST",
    "submission_37_v43_evolution_deployment.md": "2026-11-05 12:55 JST",
    "offline_verification_and_hungarian_trap_analysis.md": "2026-11-05 13:25 JST",
    "idea_01_glm_exact_teacher_log_likelihood.md": "2026-11-05 13:50 JST",
    "idea_02_open_pool_v12_and_hard_supplement.md": "2026-11-05 13:52 JST",
    "idea_03_early_mid_layer_spectral_probing.md": "2026-11-05 13:55 JST",
    "idea_04_atlas_fast_batching_and_focused_margin.md": "2026-11-05 13:57 JST",
    "idea_05_family_specific_stylometry_and_m2_validator.md": "2026-11-05 13:58 JST",
    "local_oof_benchmark_and_v44_selection_report.md": "2026-11-05 14:30 JST",
    "v45_grandmaster_architecture_and_local_verification.md": "2026-11-06 15:02 JST",
    "v47_clean_svc_bank_integration_and_day_strategy.md": "2026-11-07 09:51 JST",
    "v48_deberta_v3_classifier_and_dual_svc_apex_report.md": "2026-11-07 20:30 JST",
    "v49_pre_sinkhorn_sibling_forensic_residual_report.md": "2026-11-08 09:43 JST",
    "v50_dual_encoder_modernbert_sibling_forensic_apex_report.md": "2026-11-08 12:21 JST",
    "v51_target_mix_calibration_and_apex_report.md": "2026-11-08 20:51 JST",
    "v52_0680_native_embedding_lora_sovereign_report.md": "2026-11-08 22:45 JST",
    "v53_apex_fusion_dual_encoder_hybrid_report.md": "2026-11-08 22:54 JST",
    "v53_flattened_repro_and_colab_sparse_eval_plan.md": "2026-11-09 10:43 JST",
    "colab_v55_hyperparameter_optimization_report.md": "2026-11-10 10:15 JST",
    "v56_dual_embedding_dual_encoder_preliminary_design.md": "2026-11-10 13:27 JST",
    "colab_v57_sovereign_hyperparameter_verification_report.md": "2026-11-11 11:56 JST",
    "forum_review_and_verification_plan_1112.md": "2026-11-12 11:37 JST",
    "strategic_verification_plan_1113.md": "2026-11-13 10:28 JST",
    "strategic_verification_plan_1118.md": "2026-11-18 11:53 JST",
    "dashboard_and_strategy_roadmap_1119.md": "2026-11-19 09:58 JST"
}

REPORT_METADATA = {
    # --- ACTIVE: Curated Reading Order (11/19 Current Strategy) ---
    "dashboard_and_strategy_roadmap_1119.md": {
        "order": 1, "step": "STEP 0: 最新戦況・0.705+突破方針 (11/19 必読)", "status": "active",
        "note": "【最新・最重要】6,370チーム・0.702公開団子(707組) vs 0.705+(153組)の崖構造、NTL 0707(0.698)樹立、Surprisal 5連敗総括、次期方針"
    },
    "strategic_verification_plan_1118.md": {
        "order": 2, "step": "STEP 0.5: 2×2 切り分け検証計画 (11/18)", "status": "active",
        "note": "【実験設計】NTL 0706〜0708 の 2×2 要因計画と全11構成 Surprisal OOF ベンチマーク結果"
    },
    "name_that_llm_comprehensive_guide.md": {
        "order": 3, "step": "STEP 1: 超入門・全体像", "status": "active",
        "note": "【必読・全体像】コンペ概要、Balanced Accuracy、9時間VOID崖、基本解法ガイド"
    },
    "kaggle_live_dashboard_deployment.md": {
        "order": 4, "step": "STEP 2: 運用インフラ", "status": "active",
        "note": "リアルタイム監視ダッシュボード・アーカイブ統合の運用設計"
    },
    "dual_track_strategy_and_operations.md": {
        "order": 5, "step": "STEP 3: 運用体制・戦略プロトコル", "status": "active",
        "note": "Track 1 (堅実進化) ＆ Track 2 (探索) の2系統分割と1変数切り分け規律"
    },
    "offline_verification_and_hungarian_trap_analysis.md": {
        "order": 6, "step": "STEP 4: オフライン数理検証", "status": "active",
        "note": "スタイロメトリーの限界と Hungarian 割当の罠、Soft Sinkhorn の必然性"
    },
    "idea_01_glm_exact_teacher_log_likelihood.md": {
        "order": 7, "step": "STEP 5: コア技術 — 直接対数尤度 (V1)", "status": "active",
        "note": "【0.684→0.698 牽引】GLM-5.3 / Flash 実モデル重みによる直接尤度評価（他ターゲット拡張候補）"
    },
    "idea_02_open_pool_v12_and_hard_supplement.md": {
        "order": 8, "step": "STEP 5: 路線A — Open Pool 拡張学習", "status": "active",
        "note": "Open Pool 大規模データによる異ファミリー LoRA 表現学習（路線A パイロット検討資料）"
    },
    "sibling_disambiguation_training_plan.md": {
        "order": 9, "step": "STEP 6: 兄弟モデル識別設計", "status": "active",
        "note": "全エラーの大半を占める兄弟モデル（Gemini/GLM/Claude/GPT）混同の分析と対策設計"
    },
    "gold_medal_master_strategy.md": {
        "order": 10, "step": "STEP 7: 戦略ロードマップ", "status": "active",
        "note": "上位メダル獲得に向けたフェーズ計画とリソース配分方針"
    },
    # --- ARCHIVED / SUPERSEDED (過去の通過点・旧世代) ---
    "colab_v57_sovereign_hyperparameter_verification_report.md": {
        "order": 70, "step": "ARCHIVE: V57/V58 世代", "status": "archived",
        "note": "【旧世代】0.680 Dual-Embedding 時代の 5-Fold OOF 検証（0707 世代へ移行済み）"
    },
    "v56_dual_embedding_dual_encoder_preliminary_design.md": {
        "order": 71, "step": "ARCHIVE: V56 世代", "status": "archived",
        "note": "【旧世代】WeMM-4B 統合設計（0.684/0.698 世代で代替済み）"
    },
    "v52_0680_native_embedding_lora_sovereign_report.md": {
        "order": 72, "step": "ARCHIVE: 0.680 通過点", "status": "archived",
        "note": "【通過点】5-Fold Native Qwen3-Emb-8B 導入による 0.680 到達記録"
    },
    "strategic_verification_plan_1113.md": {
        "order": 73, "step": "ARCHIVE: 11/13 計画", "status": "archived",
        "note": "【完了】0.684 クリーン基準〜V1 導入前の検証計画"
    },
    "forum_review_and_verification_plan_1112.md": {
        "order": 74, "step": "ARCHIVE: 11/12 インテル", "status": "archived",
        "note": "【旧インテル】11/12 時点のフォーラム調査"
    },
    "local_oof_benchmark_and_v44_selection_report.md": {
        "order": 75, "step": "ARCHIVE: V44 世代", "status": "archived",
        "note": "【旧世代】wsum=9.0 較正発見と V44 時代のローカルベンチマーク"
    },
    "submission_37_v43_evolution_deployment.md": {
        "order": 76, "step": "ARCHIVE: V43 世代", "status": "archived",
        "note": "【旧世代】V43 (0.662) デプロイ記録"
    },
    "idea_03_early_mid_layer_spectral_probing.md": {
        "order": 77, "step": "ARCHIVE: R&D #03", "status": "archived",
        "note": "【検証済】Phase 6 初期〜中間層プロービング構想"
    },
    "idea_04_atlas_fast_batching_and_focused_margin.md": {
        "order": 78, "step": "ARCHIVE: R&D #04", "status": "archived",
        "note": "【旧世代】ATLAS バッチ推論・マージン最適化"
    },
    "idea_05_family_specific_stylometry_and_m2_validator.md": {
        "order": 79, "step": "ARCHIVE: R&D #05", "status": "archived",
        "note": "【旧世代】ファミリー別スタイロメトリー検証"
    },
    "ntl_master_strategy_and_operations.md": {
        "order": 80, "step": "ARCHIVE: 初期戦略", "status": "archived",
        "note": "【旧世代】10月末時点のマスター戦略"
    },
    "phase0_oof_results_and_breakthrough_findings.md": {
        "order": 81, "step": "ARCHIVE: Phase 0", "status": "archived",
        "note": "【旧世代】初期 OOF 基盤の分析結果"
    },
    "offline_oof_and_error_analysis_report.md": {
        "order": 82, "step": "ARCHIVE: 初期エラー分析", "status": "archived",
        "note": "【旧世代】Sinkhorn x Target Mix 初期エラー分析"
    },
    "kaggle_discussion_and_code_intel_report.md": {
        "order": 83, "step": "ARCHIVE: 11/02 インテル", "status": "archived",
        "note": "【旧インテル】0.667 時代の動向レポート"
    },
    "submission_33_v39_deployment.md": {
        "order": 84, "step": "ARCHIVE: V39 通過点", "status": "archived",
        "note": "【通過点】V39 (0.666) 記録"
    },
    "submission_34_v40_deployment.md": {
        "order": 85, "step": "ARCHIVE: V40 通過点", "status": "archived",
        "note": "【通過点】V40 (0.666) 記録"
    },
    "v39_verification_and_tradeoff_analysis.md": {
        "order": 86, "step": "ARCHIVE: V39 分析", "status": "archived",
        "note": "【旧世代】V39 トレードオフ分析"
    },
    "apex_ii_sota_architecture_and_v41_plan.md": {
        "order": 87, "step": "ARCHIVE: V41 世代", "status": "archived",
        "note": "【旧世代】Apex-II 設計解剖"
    },
    "v45_grandmaster_architecture_and_local_verification.md": {
        "order": 88, "step": "ARCHIVE: V45 世代", "status": "archived",
        "note": "【旧世代】V45 ローカル検証"
    },
    "v47_clean_svc_bank_integration_and_day_strategy.md": {
        "order": 88, "step": "ARCHIVE: V47 世代", "status": "archived",
        "note": "【旧世代】SVC バンク統合"
    },
    "v48_deberta_v3_classifier_and_dual_svc_apex_report.md": {
        "order": 88, "step": "ARCHIVE: V48 世代", "status": "archived",
        "note": "【旧世代】DeBERTa-v3 + Dual SVC"
    },
    "v49_pre_sinkhorn_sibling_forensic_residual_report.md": {
        "order": 88, "step": "ARCHIVE: V49 世代", "status": "archived",
        "note": "【旧世代】Pre-Sinkhorn Sibling Residual"
    },
    "v50_dual_encoder_modernbert_sibling_forensic_apex_report.md": {
        "order": 88, "step": "ARCHIVE: V50 世代", "status": "archived",
        "note": "【旧世代】Dual Encoder ModernBERT"
    },
    "v51_target_mix_calibration_and_apex_report.md": {
        "order": 89, "step": "ARCHIVE: V51 世代", "status": "archived",
        "note": "【旧世代】Target Mix 較正"
    },
    "v53_apex_fusion_dual_encoder_hybrid_report.md": {
        "order": 89, "step": "ARCHIVE: V53 世代", "status": "archived",
        "note": "【旧世代】V53 Apex Fusion"
    },
    "v53_flattened_repro_and_colab_sparse_eval_plan.md": {
        "order": 89, "step": "ARCHIVE: V53 評価計画", "status": "archived",
        "note": "【旧世代】V53 疎評価計画"
    },
    "colab_v55_hyperparameter_optimization_report.md": {
        "order": 90, "step": "ARCHIVE: V55 世代", "status": "archived",
        "note": "【旧世代】V55 Colab ハイパラ検証"
    },

    # --- ARCHIVED / SUPERSEDED / REJECTED (9件) ---
    "submission_32_v38_deployment.md": {
        "order": 91, "step": "ARCHIVE: 通過点", "status": "archived",
        "note": "【通過点】0.665達成記録。V39 (0.666) へ完全移行済み"
    },
    "submission_31_v37_deployment.md": {
        "order": 92, "step": "ARCHIVE: 棄却施策", "status": "archived",
        "note": "【棄却】ルールベース事前分布調整。スコア微減（0.659）のため不採用確定"
    },
    "phase0_oof_baseline_deployment.md": {
        "order": 93, "step": "ARCHIVE: 初期着手", "status": "archived",
        "note": "【初期着手】Phase 0 OOF の着手ログ。結果レポート（Step 5）に統合済み"
    },
    "submission_30_v36_deployment_and_monitoring.md": {
        "order": 94, "step": "ARCHIVE: 旧世代", "status": "archived",
        "note": "【旧世代】V36 (0.662) 銅メダル時代の提出記録。現行スタックにより代替"
    },
    "submission_28_29_results_and_next_steps.md": {
        "order": 95, "step": "ARCHIVE: 旧戦況", "status": "archived",
        "note": "【旧戦況】第28・29弾の中間スコア記録。銀メダル突破前の歴史的記録"
    },
    "offline_validation_and_sibling_mining.md": {
        "order": 96, "step": "ARCHIVE: 旧ハーネス", "status": "archived",
        "note": "【代替済み】k=2 疎サンプリング。より高精度な 5-Fold OOF（Step 5）の完成により役目を終了"
    },
    "submission_28_sinkhorn50_deployment.md": {
        "order": 97, "step": "ARCHIVE: 旧世代", "status": "archived",
        "note": "【旧世代】第28弾の初期 Sinkhorn 0.50 投入記録。V38/V39 で発展完了"
    },
    "dual_submission_operations_ab.md": {
        "order": 98, "step": "ARCHIVE: 旧並行実験", "status": "archived",
        "note": "【旧世代】第26・27弾の並行推論実験。旧モデル構成のため現在非推奨"
    },
    "submission_25_sinkhorn60_deployment.md": {
        "order": 99, "step": "ARCHIVE: 失敗検証", "status": "archived",
        "note": "【過剰補正】Sinkhorn 0.60 投入で 0.659 へ低下。Sinkhorn 0.50 回帰の教訓となった記録"
    }
}

def extract_reports():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, 'docs')
    reports = []
    
    for p in sorted(glob.glob(os.path.join(docs_dir, '*.md'))):
        fname = os.path.basename(p)
        with open(p, encoding='utf-8') as f:
            content = f.read()

        # Title
        m_title = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        raw_title = m_title.group(1).strip() if m_title else fname
        # Clean title markdown
        clean_title = re.sub(r'[\[\]\(\)\*]', '', raw_title)

        # Date
        date_str = DOC_DATES.get(fname)
        if not date_str:
            m_date = re.search(r'\*\*(?:記録日時|作成日時|更新日時|最終更新日時|デプロイ日時)\*\*:\s*([0-9\-\:\s]+(?:JST|UTC)?)', content)
            if m_date:
                date_str = m_date.group(1).strip()
            else:
                m_time = os.path.getmtime(p)
                date_str = datetime.datetime.fromtimestamp(m_time).strftime('%Y-%m-%d %H:%M JST')

        # Category
        category = '戦略・分析'
        badge_color = 'purple'
        if 'guide' in fname:
            category = '総合ガイド・入門'
            badge_color = 'cyan'
        elif fname.startswith('idea_'):
            category = '実験ネタ・R&D'
            badge_color = 'amber'
        elif 'deployment' in fname or 'submission' in fname or 'dashboard' in fname:
            category = '提出・運用レポート'
            badge_color = 'emerald'
        elif 'intel' in fname or 'discussion' in fname:
            category = '競合調査・インテル'
            badge_color = 'amber'
        elif 'plan' in fname or 'strategy' in fname:
            category = 'アーキテクチャ設計'
            badge_color = 'blue'

        # Extract Summary
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith('#') and not l.startswith('**') and not l.startswith('---') and not l.startswith('>')]
        summary = lines[0] if lines else ''
        if len(summary) > 130:
            summary = summary[:130] + '...'

        # Custom Metadata (Reading Order & Status)
        meta = REPORT_METADATA.get(fname, {
            "order": 50,
            "step": "その他",
            "status": "active",
            "note": summary
        })

        reports.append({
            "filename": fname,
            "title": clean_title,
            "date": date_str,
            "category": category,
            "badge_color": badge_color,
            "summary": summary,
            "order": meta["order"],
            "step": meta["step"],
            "status": meta["status"],
            "note": meta["note"]
        })

    # Sort primarily by recommended reading order (Active 1..12 first, then Archived 91..99)
    reports.sort(key=lambda x: x["order"])
    return reports

def main():
    api = KaggleApi()
    api.authenticate()

    # 1. Fetch submissions
    print("Fetching Kaggle submissions...")
    subs = api.competition_submissions('name-that-llm')
    sub_list = []
    running_sub = None

    for s in subs:
        score_val = None
        if s.public_score is not None and str(s.public_score).strip():
            try:
                score_val = float(s.public_score)
            except ValueError:
                pass
        
        status_str = str(s.status).replace("SubmissionStatus.", "")
        
        date_str = str(s.date)
        try:
            dt = datetime.datetime.fromisoformat(date_str)
            dt_jst = dt + datetime.timedelta(hours=9)
            date_jst = dt_jst.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            date_jst = date_str

        item = {
            "ref": s.ref,
            "date": date_jst,
            "status": status_str,
            "score": score_val,
            "description": s.description
        }
        sub_list.append(item)

    # Detect all active/running submissions (generic NTL 07xx+ parser)
    active_runs = []
    for s in sub_list:
        if "PENDING" in s["status"] or "RUNNING" in s["status"]:
            desc = s["description"] or ""
            m_ver = re.search(r'(NTL\s*\d+|V\d+(?:-[A-Z0-9]+)?)', desc, re.IGNORECASE)
            ver_name = m_ver.group(1).upper() if m_ver else f"Run #{s['ref']}"
            track_label = "Surprisal Test" if "Surprisal" in desc else ("V1 Coverage" if "max-cover" in desc or "PACK" in desc else "Baseline / Pipeline")
            active_runs.append({
                "ref": s["ref"],
                "version": ver_name,
                "track": track_label,
                "description": desc,
                "params": desc,
                "submitted_at": s["date"],
                "status": "RUNNING" if "RUNNING" in s["status"] else "PENDING",
                "target_score": "0.705+ (Silver Cliff)",
                "estimated_completion": "~8-9h from submit"
            })

    running_sub = active_runs[0] if active_runs else None

    # 2. Fetch Leaderboard metrics
    print("Fetching Kaggle leaderboard...")
    clusters = {"ge_0698": 0, "ge_0700": 0, "ge_0702": 0, "ge_0705": 0, "ge_0710": 0, "ge_0750": 0}
    with tempfile.TemporaryDirectory() as tmpdir:
        api.competition_leaderboard_download('name-that-llm', path=tmpdir)
        for f in os.listdir(tmpdir):
            if f.endswith('.zip') and 'name-that-llm' in f:
                with zipfile.ZipFile(os.path.join(tmpdir, f), 'r') as z:
                    z.extractall(tmpdir)
        csv_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.csv')]
        latest_csv = sorted(csv_files)[-1]
        df = pd.read_csv(latest_csv)
        total_teams = len(df)

        my_row = df[df['TeamMemberUserNames'].str.contains('tsuyoshichiba', na=False) | df['TeamName'].str.contains('tsuyoshichiba', na=False)]
        if not my_row.empty:
            my_rank = int(my_row.iloc[0]['Rank'])
            my_score = float(my_row.iloc[0]['Score'])
        else:
            my_rank = 100
            my_score = 0.717

        top18_cutoff = float(df.iloc[17]['Score'])
        silver_rank_cutoff = int(total_teams * 0.05)
        bronze_rank_cutoff = int(total_teams * 0.10)
        silver_cutoff = float(df.iloc[silver_rank_cutoff - 1]['Score'])
        bronze_cutoff = float(df.iloc[bronze_rank_cutoff - 1]['Score'])
        top_pct = round((my_rank / total_teams) * 100, 2)

        clusters = {
            "ge_0698": int((df['Score'] >= 0.698).sum()),
            "ge_0700": int((df['Score'] >= 0.700).sum()),
            "ge_0702": int((df['Score'] >= 0.702).sum()),
            "ge_0705": int((df['Score'] >= 0.705).sum()),
            "ge_0710": int((df['Score'] >= 0.710).sum()),
            "ge_0750": int((df['Score'] >= 0.750).sum()),
        }

    # Count today's UTC submissions for remaining quota (5/day)
    today_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    used_today = sum(
        1 for s in subs
        if str(s.date).startswith(today_utc)
    )
    quota_remaining = max(0, 5 - used_today)

    # 3. Evolution timeline
    v45_sub = next((s for s in sub_list if ("V45 Evolution" in s["description"] or "0675" in s["description"]) and "COMPLETE" in s["status"] and s.get("score") is not None), None)
    v45_score = float(v45_sub["score"]) if v45_sub else None

    v46_sub = next((s for s in sub_list if ("V46" in s["description"] or "0676" in s["description"]) and "COMPLETE" in s["status"] and s.get("score") is not None), None)
    v46_score = float(v46_sub["score"]) if v46_sub else None

    timeline = [
        {"version": "V28", "score": 0.662, "date": "10-31", "tag": "Bronze", "desc": "Sinkhorn 0.50 + ATLAS 20% (ブロンズ同点密集)"},
        {"version": "V31", "score": 0.662, "date": "11-01", "tag": "Bronze", "desc": "2xT4 DP Fast Run + ATLAS 20%"},
        {"version": "V36", "score": 0.662, "date": "11-01", "tag": "Bronze", "desc": "Typo / Ellipsis Marker"},
        {"version": "V37", "score": 0.659, "date": "11-01", "tag": "Rejected", "desc": "後処理ルールベースPrior (微減により棄却)"},
        {"version": "V38", "score": 0.665, "date": "11-02", "tag": "Silver", "desc": "OpenPool 198k + LinearSVC 12.0 (銀メダル圏突破!)"},
        {"version": "V39", "score": 0.666, "date": "11-02", "tag": "Silver Top", "desc": "OpenPool v11 Extra 49k 換装 (銀上位獲得)"},
        {"version": "V40", "score": 0.666, "date": "11-03", "tag": "Silver Defend", "desc": "Gemini ルーティング除外 + ATLAS Margin 0.32 (0.666完全再現/114位防衛)"},
        {"version": "V41", "score": 0.663, "date": "11-04", "tag": "Silver Hold", "desc": "Apex-II (Sinkhorn 0.30緩和による微減、要改善特定)"},
        {"version": "V42 A/B", "score": 0.663, "date": "11-04", "tag": "Silver", "desc": "Dual-DeBERTa+Sibling+Sinkhorn0.50 (Mix0.42: 0.663)"},
        {"version": "V42-C", "score": 0.661, "date": "11-04", "tag": "Rejected", "desc": "V40同条件アンカー (CLIデータセット自動更新による減退 -0.005)"},
        {"version": "V43 (T1)", "score": 0.662, "date": "11-05", "tag": "Dataset Drift", "desc": "Track 1 Evolution: Mix 0.42 (+0.001 vs T2, ただしOpenPool/Afify更新で0.662)"},
        {"version": "V43 (T2)", "score": 0.661, "date": "11-05", "tag": "Dataset Drift", "desc": "Track 2 Exploration: 0.668 Repro (CLI push時の外部DS強制最新化により0.661へ低下)"},
        {"version": "V44 (T1)", "score": 0.666, "date": "11-06", "tag": "Silver", "desc": "Track 1 Evolution: Sovereign-Aegis Hybrid (Sinkhorn 0.50過剰補正ペナルティ実証 -0.002)"},
        {"version": "V44 (T2)", "score": 0.668, "date": "11-06", "tag": "Silver Repro", "desc": "Track 2 Exploration: 0.668 Repro (Snapshot固定により0.668完全復元、Rank 305達成)"},
        {"version": "V45 (T1)", "score": v45_score, "date": "11-06", "tag": "Decoy Trap", "desc": f"Track 1 Evolution: Dual-SVC 1.0 + N14B 3-Adapter ({v45_score}着地、主力N14B希釈特定)"},
        {"version": "V46", "score": 0.668, "date": "11-07", "tag": "Silver", "desc": "Sovereign Refined (純N14B 2.5復元 + Mix 0.45: 0.668)"},
        {"version": "V52", "score": 0.680, "date": "11-08", "tag": "0.680 Breakthrough", "desc": "Native Qwen3-Embedding-8B 5-Fold LoRA (40%) + 5-Fold Sovereign Base (60%) で0.680大台突破"},
        {"version": "V53-Fix", "score": 0.679, "date": "11-09", "tag": "Silver", "desc": "Qwen3-Embedding 50%単一引き上げ検証 (0.679)"},
        {"version": "V54", "score": 0.671, "date": "11-10", "tag": "Diagnostic", "desc": "Standalone ModernBERT 1.4 / DeBERTa 0.8 注入 (w_cls=0.18過大により0.671へ低下・原因特定)"},
        {"version": "V55", "score": 0.674, "date": "11-10", "tag": "Diagnostic", "desc": "ModernBERT 1.6 / DeBERTa 0.4 比率調整 (+0.003改善するもextra_cls汚染を確認)"},
        {"version": "V56-Fix", "score": 0.680, "date": "11-10", "tag": "Dual-Emb Base", "desc": "extra_cls完全排除 + Qwen3-8B (40%) + Tencent WeMM-4B (7%) 並列統合で0.680完全復帰"},
        {"version": "V57", "score": None, "target": 0.680, "date": "11-11", "tag": "Timeout", "desc": "Dual-Embedding Sovereign (9時間制限超過により失格判定)"},
        {"version": "V58", "score": 0.673, "target": 0.680, "date": "11-11", "tag": "Diagnostic", "desc": "Dual-Embedding Tuned (Qwen3 50% + WeMM4 10% + ATLAS 0.15 + GSQ 0.06: LB 0.673)"},
        {"version": "V59 (0690)", "score": 0.673, "target": 0.680, "date": "11-11", "tag": "Diagnostic", "desc": "WeMM4 15% 拡張検証 (Qwen3 50% + WeMM4 15% + ATLAS 0.15 + GSQ 0.06: LB 0.673)"},
        {"version": "0691-0694", "score": 0.677, "date": "11-12", "tag": "Ablation", "desc": "V56-Fix再検証・WeMMカナリア・GSQ0.06/ATLAS0.15単体アブレーション (0.675〜0.677)"},
        {"version": "NTL 0695", "score": 0.684, "date": "11-13", "tag": "0.684 Clean Base", "desc": "All Public 0.684 Clean Benchmark (infer.py + comp_stack.py 完走)"},
        {"version": "0696 / 0697", "score": 0.679, "date": "11-13", "tag": "Diagnostic", "desc": "0.684 + WeMM 0.07 (0.678) / Sinkhorn 0.20 (0.679)"},
        {"version": "0698 / 0699", "score": None, "date": "11-15", "tag": "Error", "desc": "sanpierベース 81b1e7c4 イメージで CUDA init 失敗・リラン例外エラー"},
        {"version": "NTL 0700", "score": 0.696, "date": "11-15", "tag": "0.696 Breakthrough", "desc": "0.684 Stack + GLM-5.3/Flash V1 Graft (de153e6d GPU + 実測V1_COST + 8.75h: 0.696)"},
        {"version": "NTL 0701", "score": 0.694, "date": "11-15", "tag": "Diagnostic", "desc": "0700 + 曖昧度優先選択 (mass - 0.25*|p53 - pflash|: 0.694)"},
        {"version": "0702 / 0703", "score": 0.694, "date": "11-17", "tag": "Rejected", "desc": "Surprisal P5(0.695) / P5+Gem2(0.694): V1時間圧迫により0700比微減"},
        {"version": "0704 / 0705", "score": 0.690, "date": "11-17", "tag": "Rejected", "desc": "Open-Pool CPU Spec並列競合 + V1 fixed_s=100 超過 (0704: 0.684, 0705: 0.690)"},
        {"version": "NTL 0706", "score": 0.694, "date": "11-18", "tag": "Rejected", "desc": "Spec除去 + V1(870/1140)修正 + P5+Gem4-12B (0705比+0.004回復も0700未満)"},
        {"version": "NTL 0707", "score": 0.698, "target": 0.705, "date": "11-18", "tag": "0.698 Peak", "desc": "comp_stack無効 + GSQ 27B完走 + Native 7.2h + 安全V1(cover 0.30): 自己ベスト 0.698"},
        {"version": "NTL 0708", "score": 0.693, "date": "11-18", "tag": "Rejected", "desc": "0707基盤 + P5+Gem4-12B Surprisal (OOF最高も本番-0.005、Surprisal路線打ち切り)"},
        {"version": "0709 / 0711", "score": 0.695, "date": "11-19", "tag": "Ablation", "desc": "0707基盤 V1 cover比較: 0.15(0.694) < 0.30(0.698) > 0.45+pack(0.695)"},
        {"version": "NTL 0712", "score": 0.696, "date": "11-19", "tag": "Rejected", "desc": "0707 + Soft Typography (penalty=0.05) 単体 (0.696、自己ベスト未達)"},
        {"version": "NTL 0713", "score": 0.709, "date": "11-21", "tag": "320-Tok V1", "desc": "320-Token GLM Likelihood + Qwen Ensemble (sanpier 0.709 再現)"},
        {"version": "NTL 0714", "score": 0.710, "date": "11-21", "tag": "Fable Routing", "desc": "0713 + Fable Routing (claude-fable-5 加重キュー: +0.001)"},
        {"version": "NTL 0715", "score": 0.717, "target": 0.720, "date": "11-21", "tag": "0.717 Peak", "desc": "Prefix B (Reasoning Low + {n_words}) + Robust V1 (W=1.5): 自己ベスト 0.717"},
        {"version": "NTL 0716", "score": 0.714, "date": "11-22", "tag": "Rejected", "desc": "0715 + Fable Routing (0.717比 -0.003、Prefix B下では逆効果)"},
        {"version": "NTL 0717", "score": 0.717, "date": "11-22", "tag": "320-Tok B", "desc": "0715 + Prefix B 320-Token ll_model (OOF +0.02818、本番0.717横並び)"},
        {"version": "NTL 0718 / 0719", "score": 0.717, "date": "11-23", "tag": "0.717 Clean", "desc": "Guarded Fable Return + No comp_stack (0718/0719ともに0.717安定再現)"},
        {"version": "NTL 0720 / 0721", "score": 0.715, "date": "11-23", "tag": "Rejected", "desc": "GSQ 0.10(0.715) / Native 0.48(0.711): シェア増量は両者とも悪化・現行0.03/0.40維持"},
        {"version": "NTL 0722 v6", "score": 0.698, "date": "11-24", "tag": "Rejected", "desc": "0719 + prvsiyan Contrastive Qwen3-8B L24 (comp_w=0.12): 0.698に大幅悪化・棄却"},
        {"version": "NTL 0723", "score": 0.717, "date": "11-24", "tag": "Sibling Odds", "desc": "0719(No comp) + Sibling Odds + ozertuu V27 Stage-3: 0.717横並び"},
        {"version": "NTL 0724 / 0725", "score": 0.717, "date": "11-25", "tag": "Diff Ablation", "desc": "差分A(0724: comp_w=0.15) / 差分B(0725: argsort(-mass)純GLM順) 評価中"},
        {"version": "NTL 0726", "score": 0.722, "target": 0.725, "date": "11-25", "tag": "0.722 Peak", "desc": "LumiBee Kiro v66 PB115 Hybrid Apex (115-dim LR+LGB + Hybrid Alpha 0.25 + All-GLM-Out Guard): 自己ベスト 0.722 (+0.005 / 銀圏到達)"}
    ]

    # 4. Architecture comparison matrix
    architecture_matrix = [
        {"version": "V38", "track": "Track 1", "score": "0.665", "data_pool": "Open Pool 198k", "deberta": "Single (Official)", "pairwise_ab": "None", "sibling_style": "None", "atlas_margin": "0.0", "sinkhorn": "0.50 / Mix 0.35", "highlight": "銀メダル圏突入"},
        {"version": "V39", "track": "Track 1", "score": "0.666", "data_pool": "Open Pool v13 (190.5k)", "deberta": "Single (Official)", "pairwise_ab": "None", "sibling_style": "None", "atlas_margin": "0.0", "sinkhorn": "0.50 / Mix 0.35", "highlight": "データセット高品質化"},
        {"version": "V40", "track": "Track 1", "score": "0.666", "data_pool": "Open Pool v13 + Afify v9", "deberta": "Single (Official)", "pairwise_ab": "None", "sibling_style": "None", "atlas_margin": "0.32", "sinkhorn": "0.50 / Mix 0.35", "highlight": "3回連続0.666再現 (DS更新前)"},
        {"version": "V41", "track": "Track 2", "score": "0.663", "data_pool": "Open Pool Mutated (Nov 4)", "deberta": "Dual (0.75 Pool)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "V40A Residual (0.10)", "atlas_margin": "0.32", "sinkhorn": "0.30 / Mix 0.35", "highlight": "外部DS更新によるドリフト開始"},
        {"version": "V42", "track": "Track 1", "score": "0.663", "data_pool": "Open Pool Mutated (Nov 4)", "deberta": "Dual (0.75 Pool)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "Residual (0.10)", "atlas_margin": "0.32", "sinkhorn": "0.50 / Mix 0.42", "highlight": "Mix0.42で+0.002下支え"},
        {"version": "V42-B", "track": "Track 1", "score": "0.663", "data_pool": "Open Pool Mutated (Nov 4)", "deberta": "Dual (0.75 Pool)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "Residual (0.05)", "atlas_margin": "0.32", "sinkhorn": "0.50 / Mix 0.42", "highlight": "Sibling保守版"},
        {"version": "V42-C", "track": "Track 1", "score": "0.661", "data_pool": "Open Pool Mutated (Nov 4)", "deberta": "Dual (0.75 Pool)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "Residual (0.10)", "atlas_margin": "0.32", "sinkhorn": "0.50 / Mix 0.35", "highlight": "Mix0.35戻しで0.661"},
        {"version": "V43 (T1)", "track": "Track 1", "score": "0.662", "data_pool": "Open Pool v15 (213k) + Afify v11", "deberta": "Single (Official)", "pairwise_ab": "None", "sibling_style": "None", "atlas_margin": "0.32", "sinkhorn": "0.50 / Mix 0.42", "highlight": "Mix 0.42効果でT2に+0.001勝利 (DS汚染特定)"},
        {"version": "V43 (T2)", "track": "Track 2", "score": "0.661", "data_pool": "Open Pool v15 (213k) + Afify v11", "deberta": "Single (Official)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "Residual (0.10)", "atlas_margin": "0.32", "sinkhorn": "0.30 / Mix 0.35", "highlight": "公開0.668無改変でもDS更新により0.661へ急落"},
        {"version": "V44 (T1)", "track": "Track 1", "score": "0.666", "data_pool": "Frozen Snapshot (190.5k + 4.5k)", "deberta": "Single (Official)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "Residual (0.10)", "atlas_margin": "0.32", "sinkhorn": "0.50 / Mix 0.42", "highlight": "Sinkhorn 0.50過剰補正により0.666着地"},
        {"version": "V44 (T2)", "track": "Track 2", "score": "0.668", "data_pool": "Frozen Snapshot (190.5k + 4.5k)", "deberta": "Single (Official)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "Residual (0.10)", "atlas_margin": "0.32", "sinkhorn": "0.30 / Mix 0.35", "highlight": "0.668完全復元・銀メダルボーダー到達 (Rank 305)"},
        {"version": "V45 (T1)", "track": "Track 1", "score": str(v45_score) if v45_score else "Pending", "data_pool": "Frozen Snapshot + Dual-SVC 4-view", "deberta": "Single (Official)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "N14B 3-Adapter Blend", "atlas_margin": "0.32", "sinkhorn": "0.30 / Mix 0.35", "highlight": f"公開デコイ解除も主力N14B希釈により0.667着地"},
        {"version": "V46", "track": "Track 1", "score": "0.668", "data_pool": "Frozen Snapshot (190.5k)", "deberta": "Single (Official)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "Undiluted N14B (2.5)", "atlas_margin": "0.32", "sinkhorn": "0.30 / Mix 0.45", "highlight": "旧世代骨格での0.668到達"},
        {"version": "V52", "track": "Track 1", "score": "0.680", "data_pool": "Sovereign 5-Fold + GSQ 0.03", "deberta": "Core + Comp (.85/.15)", "pairwise_ab": "ATLAS (w=0.10)", "sibling_style": "Qwen3-Emb-8B (0.40)", "atlas_margin": "0.30", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【大台突破】Native Qwen3-8B 5-Fold LoRA融合で0.680達成"},
        {"version": "V54 / V55", "track": "Track 2", "score": "0.671 / 0.674", "data_pool": "Sovereign + Extra Cls (w=0.18)", "deberta": "ModernBERT + DeBERTa-v3", "pairwise_ab": "ATLAS (w=0.10)", "sibling_style": "Qwen3-Emb-8B (0.40)", "atlas_margin": "0.30", "sinkhorn": "0.30 / Mix 0.35", "highlight": "未較正の外部生ロジット注入による確率幾何崩壊を実証・棄却"},
        {"version": "V56-Fix", "track": "Track 1", "score": "0.680", "data_pool": "Sovereign 5-Fold + GSQ 0.03", "deberta": "Core + Comp (No ExtraCls)", "pairwise_ab": "ATLAS (w=0.10)", "sibling_style": "Qwen3 (0.40) + WeMM4 (0.07)", "atlas_margin": "0.30", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【0.680完全復帰】2xT4並列WeMM-4Bワーカー統合・5時間00分完走"},
        {"version": "V57 (Timeout)", "track": "Track 2", "score": "TIMEOUT", "data_pool": "Sovereign 5-Fold + GSQ 0.06", "deberta": "Core + Comp (.85/.15)", "pairwise_ab": "ATLAS (w=0.20)", "sibling_style": "Qwen3 (0.45) + WeMM4 (0.10)", "atlas_margin": "0.30", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【9時間超過】Ref: 58222075 (ランタイム上限超過失格)"},
        {"version": "V58 (0.673)", "track": "Track 1", "score": "0.673", "data_pool": "Sovereign 5-Fold + GSQ 0.06", "deberta": "Core + Comp (.85/.15)", "pairwise_ab": "ATLAS (w=0.15)", "sibling_style": "Qwen3 (0.50) + WeMM4 (0.10)", "atlas_margin": "0.30", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【推論完了: 0.673】Ref: 58224297 (LB 0.673 / V56比 -0.007)"},
        {"version": "V59 (0690)", "track": "Track 3", "score": "0.673", "data_pool": "Sovereign 5-Fold + GSQ 0.06", "deberta": "Core + Comp (.85/.15)", "pairwise_ab": "ATLAS (w=0.15)", "sibling_style": "Qwen3 (0.50) + WeMM4 (0.15)", "atlas_margin": "0.30", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【完了: 0.673】Ref: 58233790 (WeMM4 15% 拡張検証)"},
        {"version": "NTL 0695", "track": "Benchmark", "score": "0.684", "data_pool": "All Public 0.684 Clean", "deberta": "comp_stack.py", "pairwise_ab": "ATLAS V9", "sibling_style": "Qwen3-Emb + GSQ-27B", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【0.684再現】Ref: 58292048 (de153e6d / クリーン基準)"},
        {"version": "NTL 0700", "track": "V1 Graft", "score": "0.696", "data_pool": "0.684 Clean + V1 GLM Likelihood", "deberta": "comp_stack.py + ntl-segrun", "pairwise_ab": "ATLAS V9", "sibling_style": "Qwen3 + GSQ + GLM-5.3/Flash", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【0.696突破】Ref: 58388241 (de153e6d GPU + 実測V1_COST + 8.75h)"},
        {"version": "NTL 0706", "track": "Track 2", "score": "0.694", "data_pool": "0700 + P5+Gem4 (NO Spec)", "deberta": "comp_stack.py + ntl-segrun", "pairwise_ab": "ATLAS V9", "sibling_style": "P5+Gem4 kNN(2,0.20)", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【棄却】Ref: 58490392 (Spec競合解消もV1圧迫で0.694止まり)"},
        {"version": "NTL 0707", "track": "Track 1", "score": "0.698", "data_pool": "No comp_stack + Full GSQ 27B", "deberta": "comp_stack OFF + ntl-segrun", "pairwise_ab": "ATLAS V9", "sibling_style": "Qwen3(0.40) + GSQ(0.03) + V1(0.30)", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【現自己ベスト: 0.698】Ref: 58490991 (comp_stack廃止+GSQ完走+安全V1)"},
        {"version": "NTL 0708", "track": "Track 2", "score": "0.693", "data_pool": "0707 + P5+Gem4-12B Surprisal", "deberta": "comp_stack OFF + ntl-segrun", "pairwise_ab": "ATLAS V9", "sibling_style": "0707 + P5+Gem4 kNN(2,0.20)", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【棄却】Ref: 58494871 (Surprisal注入5戦全敗確定・打ち切り)"},
        {"version": "NTL 0709 / 0711", "track": "Track 2", "score": "0.694 / 0.695", "data_pool": "0707 + V1 Cover (0.15 / 0.45+pack)", "deberta": "comp_stack OFF + ntl-segrun", "pairwise_ab": "ATLAS V9", "sibling_style": "V1 max-cover 0.15 vs 0.45+pack", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【検証完了】0.30(0.698)が最適点と実証"},
        {"version": "NTL 0712", "track": "Track 2", "score": "0.696", "data_pool": "0707 + Soft Typo (0.05)", "deberta": "comp_stack OFF + ntl-segrun", "pairwise_ab": "ATLAS V9", "sibling_style": "Soft Typography (penalty=0.05)", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【未達】Ref: 58556097 (0707比 -0.002)"},
        {"version": "NTL 0713 / 0714", "track": "Track 1", "score": "0.709 / 0.710", "data_pool": "320-Token GLM V1", "deberta": "Prefix A (320 tok) + Apex 0.10", "pairwise_ab": "ATLAS V9", "sibling_style": "320tok V1 + Fable Routing", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【0.710突破】320tok(0.709) + Fable Routing(0.710)"},
        {"version": "NTL 0715", "track": "Track 1", "score": "0.717", "data_pool": "Prefix B + Robust V1 (W=1.5)", "deberta": "Prefix B (160 tok) + Apex 0.10", "pairwise_ab": "ATLAS V9", "sibling_style": "Reasoning Low + {n_words} + Robust V1", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【現自己ベスト: 0.717】Ref: 58665553 (Prefix B + Robust V1)"},
        {"version": "NTL 0716", "track": "Track 2", "score": "0.714", "data_pool": "0715 + Fable Routing", "deberta": "Prefix B (160 tok) + Fable", "pairwise_ab": "ATLAS V9", "sibling_style": "0715 + claude-fable-5 加重キュー", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【棄却】Ref: 58690579 (0715比 -0.003)"},
        {"version": "NTL 0717", "track": "Track 1", "score": "0.717", "data_pool": "0715 + Prefix B 320-Token", "deberta": "Prefix B (320 tok) + Robust V1", "pairwise_ab": "ATLAS V9", "sibling_style": "自前2699件320-Token再学習", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【0.717確認】Ref: 58716522 (320tok飽和確認)"},
        {"version": "NTL 0718 / 0719", "track": "Track 1", "score": "0.717", "data_pool": "Guarded Fable + No comp_stack", "deberta": "comp_stack OFF + Guarded Fable", "pairwise_ab": "ATLAS V9", "sibling_style": "Fable保護ゲート + 境界キュー", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【クリーン基盤: 0.717】Ref: 58725503 (無駄計算削減)"},
        {"version": "NTL 0720 / 0721", "track": "Track 2", "score": "0.715 / 0.711", "data_pool": "0719 + GSQ 0.10 / Native 0.48", "deberta": "comp_stack OFF + Guarded Fable", "pairwise_ab": "ATLAS V9", "sibling_style": "GSQ/Native シェア増量実験", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【棄却】Ref: 58749169/58749173 (0.03/0.40が最適)"},
        {"version": "NTL 0722 v6", "track": "Track 2", "score": "0.698", "data_pool": "0719 + prvsiyan Ctr L24", "deberta": "comp_w=0.12 (5-fold Qwen3-8B)", "pairwise_ab": "ATLAS V9", "sibling_style": "prvsiyan OOF 0.57459統合", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【棄却】Ref: 58806448 (0.698へ大幅悪化)"},
        {"version": "NTL 0723", "track": "Track 1", "score": "0.717", "data_pool": "0719 + Sibling Odds", "deberta": "comp_stack OFF + ozertuu V27", "pairwise_ab": "ATLAS V9", "sibling_style": "Sibling Odds + reconcile_typography", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【0.717維持】Ref: 58813043 (comp_w=0.0での再現)"},
        {"version": "NTL 0724 (Diff A)", "track": "Track 1", "score": "Pending", "data_pool": "0723 + comp_stack (0.15)", "deberta": "comp_stack ON + ozertuu V27", "pairwise_ab": "ATLAS V9", "sibling_style": "comp_w=0.15 + Sibling Odds", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【評価中】Ref: 58857627 (差分A検証)"},
        {"version": "NTL 0725 (Diff B)", "track": "Track 1", "score": "Pending", "data_pool": "0724 + argsort(-mass)", "deberta": "comp_stack ON + ranjeet258順", "pairwise_ab": "ATLAS V9", "sibling_style": "純GLM質量順 + Sibling Odds", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【評価中】Ref: 58857635 (差分B検証)"},
        {"version": "NTL 0726 (PB115)", "track": "Track 1", "score": "0.722", "data_pool": "No comp_stack + PB-115 (115-dim)", "deberta": "comp_stack OFF + 638.4s/1325.7s", "pairwise_ab": "ATLAS V9", "sibling_style": "PB115 LR+LGB + Hybrid 0.25 + All-GLM-Out", "atlas_margin": "0.35", "sinkhorn": "0.30 / Mix 0.35", "highlight": "【新自己ベスト: 0.722】Ref: 58859803 (銀メダル圏突入 / +0.005)"}
    ]

    # 5. Extract Markdown Reports
    reports = extract_reports()
    print(f"Extracted {len(reports)} markdown reports from docs/")

    # 6. Load Local Offline Verification Benchmark Results
    local_verif = None
    verif_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_verification_results.json'),
        '/Users/echo/kaggle/name-that-llm/local_verification_results.json',
    ]
    for vp in verif_candidates:
        if os.path.exists(vp):
            try:
                with open(vp, 'r', encoding='utf-8') as vf:
                    local_verif = json.load(vf)
                print(f"Loaded local verification results ({len(local_verif.get('experiments', []))} experiments) from {vp}")
                break
            except Exception as e:
                print(f"Warning: failed to load {vp}: {e}")

    now_jst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")
    medal_str = "Gold" if my_rank <= 18 else ("Silver" if my_rank <= silver_rank_cutoff else ("Bronze" if my_rank <= bronze_rank_cutoff else "No Medal"))
    out_data = {
        "updated_at": now_jst,
        "competition": {
            "name": "Name That LLM",
            "total_teams": total_teams,
            "my_rank": my_rank,
            "my_score": my_score,
            "medal": medal_str,
            "top_percent": top_pct,
            "gold_cutoff": top18_cutoff,
            "silver_cutoff": silver_cutoff,
            "silver_rank_cutoff": silver_rank_cutoff,
            "bronze_cutoff": bronze_cutoff,
            "bronze_rank_cutoff": bronze_rank_cutoff,
            "quota_remaining": quota_remaining,
            "clusters": clusters
        },
        "running_submission": running_sub,
        "active_runs": active_runs,
        "timeline": timeline,
        "submissions": sub_list[:20],
        "architecture_matrix": architecture_matrix,
        "local_verification": local_verif,
        "reports": reports
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"data.json successfully generated with {len(reports)} reports at: {out_path}")

if __name__ == "__main__":
    main()
