import os
import shutil

REPO = os.getcwd()
STAGING = os.path.join(REPO, "sub3_staging")
ZIP_NAME = "sub3_team_submission"

if os.path.exists(STAGING):
    shutil.rmtree(STAGING)

# 1. output/ — Unstop specifically looks for this folder inside the zip!
out_dir = os.path.join(STAGING, "output")
os.makedirs(out_dir, exist_ok=True)
shutil.copy2(os.path.join(REPO, "sub3_output", "matching_results.tsv"), out_dir)
# (candidate_pairs is optional for leaderboard, but we can include it if it exists)
if os.path.exists(os.path.join(REPO, "sub3_output", "candidate_pairs.tsv")):
    shutil.copy2(os.path.join(REPO, "sub3_output", "candidate_pairs.tsv"), out_dir)

# 2. code/
code_dir = os.path.join(STAGING, "code", "business_entity_resolution")
src_dir = os.path.join(code_dir, "src")
os.makedirs(src_dir, exist_ok=True)
for f in os.listdir(os.path.join(REPO, "src")):
    if f.endswith(".py"):
        shutil.copy2(os.path.join(REPO, "src", f), src_dir)

# Requirements and model
if os.path.exists("requirements.txt"):
    shutil.copy2("requirements.txt", code_dir)
if os.path.exists("xgb_reranker.json"):
    shutil.copy2("xgb_reranker.json", code_dir)

# Create ZIP
zip_path = os.path.join(REPO, ZIP_NAME)
print(f"Creating zip: {zip_path}.zip ...")
shutil.make_archive(zip_path, 'zip', STAGING)

final_zip = zip_path + ".zip"
print(f"✅ DONE! Final zip ready for Unstop: {final_zip}")
shutil.rmtree(STAGING)
