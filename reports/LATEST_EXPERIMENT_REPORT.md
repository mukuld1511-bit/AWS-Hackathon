# 📊 DGX Experiment Log & Status

| Field | Details |
|---|---|
| **Last Updated** | Initial Sync Setup |
| **Pipeline Stage** | Baseline Complete (Score: 0.385) ➔ Moving to Stage 2 |
| **Target Score** | 0.90+ |
| **Next Action on DGX** | Run `python3 run_dgx_pipeline.py` & log metrics here |

---

## 📝 Recent Runs & Metrics

### Run 1: Baseline Exact Match
- **Score (Unstop LB):** 0.385
- **Observations:** High false-positive penalty on common names. Precision-capping needed.

---

### Run 2: Precision Capping Threshold 85.0
- **Status:** Pipeline executing locally with stricter acceptance criteria (85.0 threshold).
- **Goal:** Reduce false-positives significantly to boost F_0.5 score.
- **Score (Unstop LB):** TBD (Pending evaluation)

---

*(DGX updates will be pushed here and synced automatically to Local PC)*
