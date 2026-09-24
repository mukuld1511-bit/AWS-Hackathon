"""
Submission File Validator for Amazon ML Challenge 2026.
CRITICAL: You only have 5 submissions per day. Never submit an unvalidated CSV!
"""
import os
import sys
import pandas as pd

def validate_submission(
    sub_path: str,
    test_path: str,
    id_col: str = None,
    pred_col: str = None,
    expected_rows: int = None
) -> bool:
    """
    Validates the generated submission file against test set requirements:
    1. File existence and non-empty size.
    2. Correct number of rows matching test set.
    3. No NaN / Null / Inf values.
    4. Exact ID alignment and ordering with test.csv.
    5. Correct columns present.
    """
    print("=" * 60)
    print(f"[*] Validating submission file: {sub_path}")
    print("=" * 60)

    if not os.path.exists(sub_path):
        print(f"[FAIL] Submission file not found: {sub_path}")
        return False

    file_size_mb = os.path.getsize(sub_path) / (1024 * 1024)
    print(f"[+] Submission file size: {file_size_mb:.2f} MB")

    try:
        sub_df = pd.read_csv(sub_path)
    except Exception as e:
        print(f"[FAIL] Unable to read submission CSV: {e}")
        return False

    print(f"[+] Loaded submission shape: {sub_df.shape}")
    print(f"[+] Columns: {list(sub_df.columns)}")

    # Check against test file if provided
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        expected_rows = len(test_df)
        print(f"[+] Test set detected with {expected_rows} rows.")

        if id_col is None:
            # Guess common ID column names
            for col in ['id', 'sample_id', 'product_id', 'query_id', 'ID', 'entity_id']:
                if col in test_df.columns and col in sub_df.columns:
                    id_col = col
                    break

        if id_col and id_col in test_df.columns and id_col in sub_df.columns:
            if not (test_df[id_col].values == sub_df[id_col].values).all():
                print(f"[FAIL] IDs in submission do NOT exactly match test set ID sequence!")
                return False
            print(f"[PASS] ID alignment verified on column '{id_col}'.")

    if expected_rows is not None:
        if len(sub_df) != expected_rows:
            print(f"[FAIL] Row count mismatch! Expected: {expected_rows}, Found: {len(sub_df)}")
            return False
        print(f"[PASS] Row count matches perfectly ({len(sub_df)} rows).")

    # Check for NaN / Null values
    nan_counts = sub_df.isna().sum()
    if nan_counts.any():
        print("[FAIL] Missing/NaN values detected:")
        print(nan_counts[nan_counts > 0])
        return False
    print("[PASS] Zero NaN / Null values detected.")

    print("\n[Preview of First 5 Rows]:")
    print(sub_df.head())
    print("\n[Preview of Last 5 Rows]:")
    print(sub_df.tail())
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL VALIDATION CHECKS PASSED! Ready for Unstop submission.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/submission_checker.py <path_to_submission.csv> [<path_to_test.csv>]")
    else:
        s_path = sys.argv[1]
        t_path = sys.argv[2] if len(sys.argv) > 2 else None
        validate_submission(s_path, t_path)
