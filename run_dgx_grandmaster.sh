#!/usr/bin/env bash
set -e

echo "================================================================="
echo "🚀 [STEP 1/4] Pulling latest code from GitHub main..."
echo "================================================================="
git fetch origin main
git checkout main
git pull origin main

echo ""
echo "================================================================="
echo "🔍 [STEP 2/4] Verifying GPU and ML libraries..."
echo "================================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

echo ""
echo "================================================================="
echo "🧪 [STEP 3/4] Running Offline F0.5 Metric Simulator..."
echo "================================================================="
python3 simulate_f05_score.py

echo ""
echo "================================================================="
echo "⚡ [STEP 4/4] Executing Grandmaster Full Pipeline..."
echo "================================================================="
python3 run_grandmaster_pipeline.py

echo ""
echo "================================================================="
echo "✅ [VALIDATION] Verifying Official Submission TSV..."
echo "================================================================="
if [ -f "output_grandmaster/matching_results.tsv" ]; then
    python3 validate_dgx_tsv.py output_grandmaster/matching_results.tsv
    echo ""
    echo "🎉 READY FOR SUBMISSION: output_grandmaster/matching_results.tsv"
    ls -lh output_grandmaster/matching_results.tsv
fi
