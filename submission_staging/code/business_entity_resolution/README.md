# Business Entity Resolution — Reproduction Guide

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
