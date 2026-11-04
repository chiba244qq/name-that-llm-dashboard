#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating Kaggle metrics from API..."
python3 build_data.py

if [ -d ".git" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Committing and pushing to GitHub Pages..."
    git add data.json index.html
    git commit -m "Auto-update Kaggle metrics: $(date '+%Y-%m-%d %H:%M:%S JST')" || echo "No changes to commit"
    git push origin main || echo "Git push skipped or failed"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dashboard updated successfully!"
else
    echo "Git repository not initialized yet."
fi
