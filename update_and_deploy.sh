#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing latest markdown artifacts from brain..."
mkdir -p "$DIR/docs"
cp /Users/echo/.gemini/antigravity/brain/ada99bbf-90ae-436b-9d82-11c0881d9733/*.md "$DIR/docs/" 2>/dev/null || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating Kaggle metrics and reports metadata..."
python3 build_data.py

if [ -d ".git" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Committing and pushing to GitHub Pages..."
    git add data.json index.html docs/ build_data.py README.md update_and_deploy.sh
    git commit -m "Auto-update Kaggle metrics & archives: $(date '+%Y-%m-%d %H:%M:%S JST')" || echo "No changes to commit"
    git push origin main || echo "Git push skipped or failed"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dashboard and Archives updated successfully!"
else
    echo "Git repository not initialized yet."
fi
