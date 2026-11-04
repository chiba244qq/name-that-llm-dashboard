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
    "kaggle_live_dashboard_deployment.md": "2026-11-04 09:29 JST"
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

        reports.append({
            "filename": fname,
            "title": clean_title,
            "date": date_str,
            "category": category,
            "badge_color": badge_color,
            "summary": summary
        })

    # Sort descending by date
    reports.sort(key=lambda x: x["date"], reverse=True)
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

        if running_sub is None and ("PENDING" in status_str or "RUNNING" in status_str):
            running_sub = {
                "ref": s.ref,
                "version": "V41 (Apex-II)" if "V41" in s.description else "In Progress",
                "description": s.description,
                "submitted_at": date_jst,
                "status": "RUNNING" if "RUNNING" in status_str else "PENDING",
                "target_score": "0.670 - 0.675+",
                "estimated_completion": "15:35 JST"
            }

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
    timeline = [
        {"version": "V28", "score": 0.662, "date": "10-31", "tag": "Bronze", "desc": "Sinkhorn 0.50 + ATLAS 20% (ブロンズ同点密集)"},
        {"version": "V31", "score": 0.662, "date": "11-01", "tag": "Bronze", "desc": "2xT4 DP Fast Run + ATLAS 20%"},
        {"version": "V36", "score": 0.662, "date": "11-01", "tag": "Bronze", "desc": "Typo / Ellipsis Marker"},
        {"version": "V37", "score": 0.659, "date": "11-01", "tag": "Rejected", "desc": "後処理ルールベースPrior (微減により棄却)"},
        {"version": "V38", "score": 0.665, "date": "11-02", "tag": "Silver", "desc": "OpenPool 198k + LinearSVC 12.0 (銀メダル圏突破!)"},
        {"version": "V39", "score": 0.666, "date": "11-02", "tag": "Silver Top", "desc": "OpenPool v11 Extra 49k 換装 (銀上位獲得)"},
        {"version": "V40", "score": 0.666, "date": "11-03", "tag": "Silver Defend", "desc": "Gemini ルーティング除外 + ATLAS Margin 0.32 (0.666完全再現/114位防衛)"},
        {"version": "V41", "score": None, "target": 0.672, "date": "11-04", "tag": "Running", "desc": "Apex-II (Dual-DeBERTa + Pairwise A/B + Sibling Residual 0.10 + Sinkhorn 0.30)"}
    ]

    # 4. Architecture comparison matrix
    architecture_matrix = [
        {"version": "V38", "score": "0.665", "data_pool": "Open Pool 198k", "deberta": "Single (Official)", "pairwise_ab": "None", "sibling_style": "None", "atlas_margin": "0.0", "sinkhorn": "0.50", "highlight": "銀メダル圏突入"},
        {"version": "V39", "score": "0.666", "data_pool": "Open Pool v11 (Extra 49k)", "deberta": "Single (Official)", "pairwise_ab": "None", "sibling_style": "None", "atlas_margin": "0.0", "sinkhorn": "0.50", "highlight": "データセット高品質化"},
        {"version": "V40", "score": "0.666", "data_pool": "Open Pool v11 Clean", "deberta": "Single (Official)", "pairwise_ab": "None", "sibling_style": "None", "atlas_margin": "0.32", "sinkhorn": "0.50", "highlight": "ルーティング汚染行排除"},
        {"version": "V41 (Apex-II)", "score": "0.670+ (目標)", "data_pool": "Open Pool v11 190k + 49k", "deberta": "Dual (0.75 Pool + 0.25 Orig)", "pairwise_ab": "V34.1 A/B Verifier (~16s)", "sibling_style": "V40A Residual (0.10)", "atlas_margin": "0.32", "sinkhorn": "0.30", "highlight": "3大SOTA統合スタック"}
    ]

    # 5. Extract Markdown Reports
    reports = extract_reports()
    print(f"Extracted {len(reports)} markdown reports from docs/")

    now_jst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")
    out_data = {
        "updated_at": now_jst,
        "competition": {
            "name": "Name That LLM",
            "total_teams": total_teams,
            "my_rank": my_rank,
            "my_score": my_score,
            "medal": "Silver",
            "top_percent": top_pct,
            "gold_cutoff": top18_cutoff,
            "silver_cutoff": silver_cutoff,
            "silver_rank_cutoff": silver_rank_cutoff,
            "bronze_cutoff": bronze_cutoff,
            "bronze_rank_cutoff": bronze_rank_cutoff
        },
        "running_submission": running_sub,
        "timeline": timeline,
        "submissions": sub_list[:15],
        "architecture_matrix": architecture_matrix,
        "reports": reports
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"data.json successfully generated with {len(reports)} reports at: {out_path}")

if __name__ == "__main__":
    main()
