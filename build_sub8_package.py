import os
import shutil
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(REPO, 'sub8_staging')
ZIP_NAME = 'sub8_team_submission'

print("[1/5] Cleaning up staging...")
if os.path.exists(STAGING):
    shutil.rmtree(STAGING)

print("[2/5] Copying output TSV files from output_final/...")
out_dir = os.path.join(STAGING, 'output')
os.makedirs(out_dir, exist_ok=True)
shutil.copy2(os.path.join(REPO, 'output_final', 'matching_results.tsv'), out_dir)
shutil.copy2(os.path.join(REPO, 'output_final', 'candidate_pairs.tsv'), out_dir)

print("[3/5] Packaging reproduction code...")
code_dir = os.path.join(STAGING, 'code', 'business_entity_resolution')
src_dir = os.path.join(code_dir, 'src')
os.makedirs(src_dir, exist_ok=True)

for f in os.listdir(os.path.join(REPO, 'src')):
    if f.endswith('.py'):
        shutil.copy2(os.path.join(REPO, 'src', f), src_dir)

if os.path.exists(os.path.join(REPO, 'xgb_reranker_v2.json')):
    shutil.copy2(os.path.join(REPO, 'xgb_reranker_v2.json'), code_dir)

req_path = os.path.join(REPO, 'requirements.txt')
if os.path.exists(req_path):
    shutil.copy2(req_path, code_dir)

with open(os.path.join(code_dir, 'README.md'), 'w', encoding='utf-8') as fh:
    fh.write('''# Business Entity Resolution — Submission 8 Reproduction Guide
Amazon ML Challenge 2026

## Architecture:
1. Stage 1: Tight Candidate Blocker (Multi-key inverted index + Geographic anchors)
2. Stage 2: XGBoost v2 (10 features: string similarities + US state conflict + address Jaccard)
3. Stage 3: Sentence-Transformer Cross-Script Multilingual Matching for Indic records

## Execution:
```bash
python run_final_pipeline.py
```
Output files in `output/`:
- matching_results.tsv
- candidate_pairs.tsv
''')

print("[4/5] Copying methodology documentation...")
doc_path = os.path.join(REPO, 'student_resource', 'Documentation_template.md')
shutil.copy2(doc_path, STAGING)

print("[5/5] Creating zip archive...")
shutil.make_archive(os.path.join(REPO, ZIP_NAME), 'zip', STAGING)
shutil.rmtree(STAGING)

final_zip = os.path.join(REPO, ZIP_NAME + '.zip')
size_mb = os.path.getsize(final_zip) / (1024 * 1024)
print(f"🎉 SUCCESS: Created {final_zip} ({size_mb:.2f} MB)")
