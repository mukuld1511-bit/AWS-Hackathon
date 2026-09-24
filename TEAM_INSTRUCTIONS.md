# 🚨 TEAM ROLES & WORKFLOW PROTOCOL (Amazon ML Challenge 2026)

> **Motto:** Mukul owns the core engine and official submissions. Teammates run parallel experiments to boost our ensemble and manage documentation.

---

## 👑 1. Submission Authority & Core Pipeline
* **Portal Submissions:** **ONLY Mukul** will submit files to Unstop.
* **Core Engine:** Mukul manages the primary training pipeline, local validation (CV), and benchmark models.
* **Rule:** Daily limit is strictly **5 submissions per day**.

---

## 👥 2. Strategic Team Allocation

### 🎯 Mukul (Lead / Core Engine)
- **Focus:** Primary end-to-end ML pipeline, leak-free 5-Fold Cross-Validation, core GBDT & Transformer models.
- **Responsibility:** Final model selection, ensemble integration, and official Unstop submissions.

### 🃏 Harsh & Prateek (Parallel Model Explorers / Alpha Boosters)
- **Role:** Independent high-upside model experiments.
- **Harsh:** Explores alternative model architectures (e.g., secondary transformer backbones, distinct embedding models).
- **Prateek:** Focuses on dedicated modular experiments (e.g., specialized candidate retrieval, unique feature engineering sets).
- **Strategy:** If your experiment outperforms or provides diverse predictions, we blend it with Mukul's core model for an instant leaderboard boost. If not, Mukul's core model keeps our rank secure.

### 📝 Ayush (Documentation, EDA & Analysis)
- **Role:** Competition documentation, EDA visualizations, and experiment tracking.
- **Responsibility:** Continuously draft the **mandatory 1-2 page Approach Document** (Problem formulation, Blocking strategy, Architecture) required by Amazon for Top 100 shortlisting.

---

## 🌿 3. Branching & Git Guidelines
- Mukul: `main`
- Ayush: `ayush`
- Harsh: `harsh`
- Prateek: `prateek`

**Workflow:**
1. Work inside your branch.
2. Push your models and notebooks to your branch so Mukul can pull and inspect them.
3. If you generate a candidate prediction CSV, validate it using:
   ```bash
   python src/submission_checker.py <your_file.csv> <test.csv>
   ```
   and pass it to Mukul for local CV evaluation and potential ensemble.
