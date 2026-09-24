"""
Optimized TSV Submission Checker for Amazon ML Challenge 2026.
Strictly enforces all competition rules:
1. Tab-separated format (.tsv).
2. Exactly 1,732,544 rows matching test_source1.tsv.
3. Proper singleton representation (empty string, no NaN, no null).
4. S2/S3 ID validity and no S1 self-matches.
5. Matching results must be a strict subset of candidate pairs.
"""
import os
import sys
import subprocess

def check_submission(
    matching_path: str = "output/matching_results.tsv",
    candidate_path: str = "output/candidate_pairs.tsv",
    test_dir: str = "student_resource/dataset/test"
):
    print("=" * 70)
    print("🔍 RUNNING STRICT SUBMISSION INTEGRITY CHECK (TSV OPTIMIZED)")
    print("=" * 70)

    # 1. Existence check
    for p, name in [(matching_path, "Matching Results"), (candidate_path, "Candidate Pairs")]:
        if not os.path.exists(p):
            print(f"❌ [FAIL] Missing file: {name} at {p}")
            return False
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"✅ Found {name}: {size_mb:.2f} MB")

    # 2. Run official student_resource validator script
    validator_script = os.path.join("student_resource", "utils", "validate_submission.py")
    if not os.path.exists(validator_script):
        print(f"❌ [FAIL] Official validator script not found at {validator_script}")
        return False

    print("\n[*] Invoking official competition validator...")
    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    if proc.returncode == 0:
        print("=" * 70)
        print("🎉 [PERFECT PASS] ALL RULES SATISFIED! SAFE TO SUBMIT TO UNSTOP.")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print("❌ [VALIDATION FAILED] DO NOT SUBMIT. Fix the errors listed above.")
        print("=" * 70)
        return False

if __name__ == "__main__":
    m_path = sys.argv[1] if len(sys.argv) > 1 else "output/matching_results.tsv"
    c_path = sys.argv[2] if len(sys.argv) > 2 else "output/candidate_pairs.tsv"
    t_dir = sys.argv[3] if len(sys.argv) > 3 else "student_resource/dataset/test"
    check_submission(m_path, c_path, t_dir)
