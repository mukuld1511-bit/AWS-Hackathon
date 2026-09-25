import os
import shutil
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(REPO, 'submission_staging')
ZIP_NAME = 'final_submission.zip'

print("[1/5] Cleaning up staging directory...")
if os.path.exists(STAGING):
    shutil.rmtree(STAGING)

# 1. output/ — both TSV files
print("[2/5] Copying TSV output files...")
out_dir = os.path.join(STAGING, 'output')
os.makedirs(out_dir, exist_ok=True)
shutil.copy2(os.path.join(REPO, 'output', 'matching_results.tsv'), out_dir)
shutil.copy2(os.path.join(REPO, 'output', 'candidate_pairs.tsv'), out_dir)

# 2. code/business_entity_resolution/src/
print("[3/5] Packaging reproduction code...")
code_dir = os.path.join(STAGING, 'code', 'business_entity_resolution')
src_dir = os.path.join(code_dir, 'src')
os.makedirs(src_dir, exist_ok=True)

for f in os.listdir(os.path.join(REPO, 'src')):
    if f.endswith('.py'):
        shutil.copy2(os.path.join(REPO, 'src', f), src_dir)

if os.path.exists(os.path.join(REPO, 'xgb_reranker.json')):
    shutil.copy2(os.path.join(REPO, 'xgb_reranker.json'), code_dir)

with open(os.path.join(code_dir, 'README.md'), 'w', encoding='utf-8') as fh:
    fh.write('''# Business Entity Resolution — Solution Reproduction Guide
Amazon ML Challenge 2026

## Environment
- Python 3.10+
- Dependencies: pip install -r requirements.txt

## Pipeline Architecture
1. Candidate Generation & Blocking:
   - Multi-key inverted index (Country, Exact Name, Postal Code, Geo-Anchors)
2. Matching & Re-ranking:
   - Gradient Boosted Decision Tree (XGBoost) with calibrated threshold (0.99)
3. Cross-Script Multilingual Matching:
   - Sentence-Transformers MiniLM for Indic languages (Tamil, Hindi) with address verification

## Reproduction Steps
Run the end-to-end pipeline:
```bash
python src/baseline.py
```
Output files generated in `output/`:
- `matching_results.tsv` (Final entity matches)
- `candidate_pairs.tsv` (Candidate pairs from blocking)
''')

req_path = os.path.join(REPO, 'requirements.txt')
if os.path.exists(req_path):
    shutil.copy2(req_path, code_dir)

# 3. Documentation template
print("[4/5] Copying completed documentation template...")
doc_path = os.path.join(REPO, 'student_resource', 'Documentation_template.md')
shutil.copy2(doc_path, STAGING)

# 4. Create ZIP
print("[5/5] Creating final_submission.zip archive...")
zip_path = os.path.join(REPO, 'final_submission')
shutil.make_archive(zip_path, 'zip', STAGING)
shutil.rmtree(STAGING)

final_zip = os.path.join(REPO, ZIP_NAME)
size_mb = os.path.getsize(final_zip) / (1024 * 1024)
print(f"🎉 SUCCESS: {final_zip} created ({size_mb:.2f} MB)")
