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
    "local_oof_benchmark_and_v44_selection_report.md": "2026-11-05 14:30 JST"
}

REPORT_METADATA = {
    # --- ACTIVE: Reading Order 1 to 21 ---
    "name_that_llm_comprehensive_guide.md": {
        "order": 1, "step": "STEP 1: 超入門・全体像", "status": "active",
        "note": "【必読・全体像】コンペ概要、Balanced Accuracy、9時間VOID崖、基本解法が中学生でもわかる完全ガイド"
    },
    "kaggle_live_dashboard_deployment.md": {
        "order": 2, "step": "STEP 2: 運用インフラ", "status": "active",
        "note": "リアルタイム監視ダッシュボード・自動提出・アーカイブ統合の全体像"
    },
    "dual_track_strategy_and_operations.md": {
        "order": 3, "step": "STEP 3: 運用体制・戦略プロトコル", "status": "active",
        "note": "【重要】Track 1 (堅実進化) ＆ Track 2 (インテル探索) の2系統分割とスロット配分規律"
    },
    "submission_37_v43_evolution_deployment.md": {
        "order": 4, "step": "STEP 3: Track 1 堅実進化 (最新)", "status": "active",
        "note": "【最新・本命】第37弾 V43-Evolution。V40基準スタック復帰＋OOF最適化 Target Mix 0.42"
    },
    "offline_verification_and_hungarian_trap_analysis.md": {
        "order": 5, "step": "STEP 4: オフライン数理検証", "status": "active",
        "note": "【必読・解明】スタイロメトリーの限界とHungarian割当の罠。Soft Sinkhornの必然性を数理実証"
    },
    "local_oof_benchmark_and_v44_selection_report.md": {
        "order": 6, "step": "STEP 4: ローカル実験ベンチマーク (最新)", "status": "active",
        "note": "【最新・実験総括】wsum=9.0較正の発見と全8実験 (EXP-01〜08) のFull/Sparse(m=2) OOF比較レポート"
    },
    "idea_01_glm_exact_teacher_log_likelihood.md": {
        "order": 7, "step": "STEP 5: 次世代実験ネタ #01", "status": "active",
        "note": "【実験ネタ #01】GLM-5.3 / Flash 実モデル重みによる直接尤度検証（Exact Teacher Log-Likelihood）"
    },
    "idea_02_open_pool_v12_and_hard_supplement.md": {
        "order": 8, "step": "STEP 5: 次世代実験ネタ #02", "status": "active",
        "note": "【実験ネタ #02】Open Pool v12 (211k行) ＋ Hard-Class Supplement (Vertex AI 厳密ルート) データ刷新"
    },
    "idea_03_early_mid_layer_spectral_probing.md": {
        "order": 9, "step": "STEP 5: 次世代実験ネタ #03", "status": "active",
        "note": "【実験ネタ #03】Transformer 初期〜中間層特化プロービングと LoRA+ / 位置重み付けプーリング"
    },
    "idea_04_atlas_fast_batching_and_focused_margin.md": {
        "order": 10, "step": "STEP 5: 次世代実験ネタ #04", "status": "active",
        "note": "【実験ネタ #04】ATLAS 高速バッチ推論 (BS=6) ＋ 集中 Dispute Margin (0.28) ＋ Pairwise 較正"
    },
    "idea_05_family_specific_stylometry_and_m2_validator.md": {
        "order": 11, "step": "STEP 5: 次世代実験ネタ #05", "status": "active",
        "note": "【実験ネタ #05】ファミリー別最適スタイロメトリー選択 (148d vs 81d) と疎プロンプト抽出 (m=2) 検証器"
    },
    "gold_medal_master_strategy.md": {
        "order": 12, "step": "STEP 6: 戦略ロードマップ", "status": "active",
        "note": "Top 18 金メダル獲得に向けた4段階フェーズ計画とリソース配分"
    },
    "ntl_master_strategy_and_operations.md": {
        "order": 12, "step": "STEP 6: 戦略ロードマップ", "status": "active",
        "note": "銀メダル死守から金メダル挑戦へのマスター戦略＆作戦計画"
    },
    "phase0_oof_results_and_breakthrough_findings.md": {
        "order": 13, "step": "STEP 7: 検証基盤の確立", "status": "active",
        "note": "オフライン 5-Fold OOF 基盤の確立と、Public LB との相関解明"
    },
    "offline_oof_and_error_analysis_report.md": {
        "order": 14, "step": "STEP 7: 検証基盤 (エラー深層分析)", "status": "active",
        "note": "Sinkhorn x Target Mix 2次元グリッド探索 & 176件の兄弟混同深層分析"
    },
    "sibling_disambiguation_training_plan.md": {
        "order": 15, "step": "STEP 8: コア技術・ブレークスルー", "status": "active",
        "note": "全エラーの68%を占める兄弟モデル（Gemini/GLM）特化識別器の学習仕様書"
    },
    "kaggle_discussion_and_code_intel_report.md": {
        "order": 16, "step": "STEP 8: コア技術・最新インテル", "status": "active",
        "note": "Kaggle上位陣（0.667+）の解法動向、Hungarian破綻の理由、Apex Kの分析"
    },
    "submission_33_v39_deployment.md": {
        "order": 17, "step": "STEP 9: 実戦・ブレークスルー", "status": "active",
        "note": "0.666達成！Open Pool v11 Extra 49k 換装で銀メダル上位を確定させた記録"
    },
    "submission_34_v40_deployment.md": {
        "order": 18, "step": "STEP 9: 実戦・ブレークスルー", "status": "active",
        "note": "0.666完全再現！Geminiルーティング除外 & ATLAS Margin 0.32で114位防衛"
    },
    "v39_verification_and_tradeoff_analysis.md": {
        "order": 19, "step": "STEP 9: 実戦・ブレークスルー", "status": "active",
        "note": "投入前のOOF事前検証手法とリスク・リターンのトレードオフ分析"
    },
    "apex_ii_sota_architecture_and_v41_plan.md": {
        "order": 20, "step": "STEP 10: 旧SOTAアーキテクチャ解剖", "status": "active",
        "note": "Apex-II 構成（Dual-DeBERTa + Pairwise + Sibling + ATLAS）の設計解剖"
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

    # Detect all active/running submissions
    active_runs = []
    for s in sub_list:
        if "PENDING" in s["status"] or "RUNNING" in s["status"]:
            desc = s["description"]
            ver_name = "In Progress"
            params_summary = ""
            track_label = "Track 1 (Evolution)"
            est_comp = "~15:30 JST"
            if "V45 Evolution" in desc or "V45" in desc or "0675" in desc:
                ver_name = "V45 (Disarmed Grandmaster)"
                params_summary = "Snapshot + Dual-SVC 1.0 + Calibrated ATLAS 0.32/0.20 + Sinkhorn 0.30"
                track_label = "Track 1 (Evolution)"
                est_comp = "~20:45 JST"
            elif "V44 Evolution" in desc or "V44-Evolution" in desc:
                ver_name = "V44 (Sovereign-Aegis Hybrid)"
                params_summary = "Frozen Snapshot | Sinkhorn 0.50 | Mix 0.42 | Pairwise Verifier | Single-DeBERTa"
                track_label = "Track 1 (Evolution)"
                est_comp = "Completed (0.666)"
            elif "V44 Exploration" in desc or "V44-Exploration" in desc:
                ver_name = "V44 (0.668 Frozen Repro)"
                params_summary = "Frozen Snapshot | Sinkhorn 0.30 | Mix 0.35 | Pairwise Verifier | areebbb 1x"
                track_label = "Track 2 (Exploration)"
                est_comp = "Completed (0.668)"
            elif "V43-Evolution" in desc:
                ver_name = "V43 (Evolution Anchor+Mix)"
                params_summary = "Sinkhorn 0.50 | Mix 0.42 | Single-DeBERTa | ATLAS 0.32"
                track_label = "Track 1 (Evolution)"
                est_comp = "~19:30 JST"
            elif "V43-Exploration" in desc:
                ver_name = "V43 (0.668 Repro Verifier)"
                params_summary = "Sinkhorn 0.30 | Mix 0.35 | Pairwise A/B | Sibling 0.10 | areebbb 1x"
                track_label = "Track 2 (Exploration)"
                est_comp = "~19:35 JST"
            elif "V42c" in desc or "V42-C" in desc:
                ver_name = "V42-C (Apex V40Anchor)"
                params_summary = "Sinkhorn 0.50 | Mix 0.35 | Sibling 0.10 | Dual-DeBERTa 0.75"
                track_label = "Track 1 (Evolution Anchor)"
            elif "V42-B" in desc:
                ver_name = "V42-B (Apex Conservative)"
                params_summary = "Sinkhorn 0.50 | Mix 0.42 | Sibling 0.05 | Dual-DeBERTa 0.75"
                track_label = "Track 1 (Evolution Conservative)"
            elif "V42 " in desc or "V42 Apex" in desc:
                ver_name = "V42 (Apex Re-Anchored)"
                params_summary = "Sinkhorn 0.50 | Mix 0.42 | Sibling 0.10 | Dual-DeBERTa 0.75"
                track_label = "Track 1 (Evolution Aggressive)"
            elif "V41" in desc:
                ver_name = "V41 (Apex-II)"
                params_summary = "Sinkhorn 0.30 | Mix 0.35 | Sibling 0.10 | Dual-DeBERTa 0.75"
                track_label = "Track 2 (Exploration Prototype)"

            active_runs.append({
                "ref": s["ref"],
                "version": ver_name,
                "track": track_label,
                "description": desc,
                "params": params_summary,
                "submitted_at": s["date"],
                "status": "RUNNING" if "RUNNING" in s["status"] else "PENDING",
                "target_score": "0.667 - 0.668+",
                "estimated_completion": est_comp
            })

    running_sub = active_runs[0] if active_runs else None

    # 2. Fetch Leaderboard metrics
    print("Fetching Kaggle leaderboard...")
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
            my_rank = 189
            my_score = 0.666

        top18_cutoff = float(df.iloc[17]['Score'])
        silver_rank_cutoff = int(total_teams * 0.05)
        bronze_rank_cutoff = int(total_teams * 0.10)
        silver_cutoff = float(df.iloc[silver_rank_cutoff - 1]['Score'])
        bronze_cutoff = float(df.iloc[bronze_rank_cutoff - 1]['Score'])
        top_pct = round((my_rank / total_teams) * 100, 2)

    # 3. Evolution timeline
    v45_sub = next((s for s in sub_list if ("V45 Evolution" in s["description"] or "0675" in s["description"]) and "COMPLETE" in s["status"] and s.get("public_score")), None)
    v45_score = float(v45_sub["public_score"]) if (v45_sub and str(v45_sub.get("public_score", "")).strip() not in ["None", ""]) else None

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
        {"version": "V45 (T1)", "score": v45_score, "target": 0.672, "date": "11-06", "tag": "Silver SOTA" if v45_score and v45_score >= 0.670 else "Pending", "desc": f"Track 1 Evolution: Disarmed Grandmaster (Snapshot + Dual-SVC 1.0 + Calibrated ATLAS 0.32/0.20 + Sinkhorn 0.30: {v45_score if v45_score else '推論中'})"}
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
        {"version": "V45 (T1)", "track": "Track 1", "score": str(v45_score) if v45_score else "Pending", "data_pool": "Frozen Snapshot + Dual-SVC 4-view", "deberta": "Single (Official)", "pairwise_ab": "V34.1 A/B Verifier", "sibling_style": "N14B 3-Adapter Blend", "atlas_margin": "0.32", "sinkhorn": "0.30 / Mix 0.35", "highlight": f"【本命】公開デコイ解除 + Dual-SVC 1.0 ({'着地: ' + str(v45_score) if v45_score else '推論中'})"}
    ]

    # 5. Extract Markdown Reports
    reports = extract_reports()
    print(f"Extracted {len(reports)} markdown reports from docs/")

    # 6. Load Local Offline Verification Benchmark Results
    local_verif = None
    verif_candidates = [
        '/Users/echo/kaggle/name-that-llm/local_verification_results.json',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_verification_results.json')
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
            "bronze_rank_cutoff": bronze_rank_cutoff
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
