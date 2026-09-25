# Build final submission zip for Amazon ML Challenge 2026
import os
import shutil

REPO = os.getcwd()
STAGING = os.path.join(REPO, "submission_staging")
ZIP_NAME = "team_submission"

# Clean staging
if os.path.exists(STAGING):
    shutil.rmtree(STAGING)

# 1. output/ — both TSV files
out_dir = os.path.join(STAGING, "output")
os.makedirs(out_dir, exist_ok=True)
shutil.copy2(os.path.join(REPO, "output", "matching_results.tsv"), out_dir)
shutil.copy2(os.path.join(REPO, "output", "candidate_pairs.tsv"), out_dir)

# 2. code/business_entity_resolution/src/ — all source code
code_dir = os.path.join(STAGING, "code", "business_entity_resolution")
src_dir = os.path.join(code_dir, "src")
os.makedirs(src_dir, exist_ok=True)
for f in os.listdir(os.path.join(REPO, "src")):
    if f.endswith(".py"):
        shutil.copy2(os.path.join(REPO, "src", f), src_dir)

# code README
with open(os.path.join(code_dir, "README.md"), "w", encoding="utf-8") as fh:
    fh.write("""# Business Entity Resolution — Reproduction Guide

## Prerequisites
- Python 3.8+
- pip install -r requirements.txt

## Steps to Reproduce
1. Download and extract dataset into `student_resource/` folder
2. Run the pipeline:
   ```bash
   python src/baseline.py
   ```
3. Output files are generated in `output/`:
   - `matching_results.tsv` — final entity matches
   - `candidate_pairs.tsv` — candidate blocking pairs
""")

# requirements.txt
shutil.copy2(os.path.join(REPO, "requirements.txt"), code_dir)

# 3. Documentation template
doc_src = os.path.join(REPO, "student_resource", "Documentation_template.md")
if os.path.exists(doc_src):
    shutil.copy2(doc_src, STAGING)

# Print structure
print("=== Submission staging contents ===")
for root, dirs, files in os.walk(STAGING):
    level = root.replace(STAGING, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        fp = os.path.join(root, f)
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"{indent}  {f} ({size_mb:.2f} MB)")

# 4. Create ZIP
zip_path = os.path.join(REPO, ZIP_NAME)
print(f"\nCreating zip: {zip_path}.zip ...")
shutil.make_archive(zip_path, 'zip', STAGING)

final_zip = zip_path + ".zip"
final_size = os.path.getsize(final_zip) / (1024 * 1024)
print(f"\n✅ DONE! Final zip: {final_zip}")
print(f"   Size: {final_size:.2f} MB")

# Cleanup staging
shutil.rmtree(STAGING)
print("   Staging cleaned up.")
