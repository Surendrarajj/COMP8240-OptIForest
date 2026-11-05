# COMP8240 final project — OptIForest reproduction and extension

Surendra Raj Pandey, student ID 60985194, Macquarie University.

This project reruns the original detector on AD, evaluates five additional ADBench datasets and a constructed synthetic dataset, compares conventional Isolation Forest, and records an exploratory LLM explanation audit. The detector comes from Xiang et al. (2023), https://www.ijcai.org/proceedings/2023/264 . Original code source https://github.com/xiagll/OptIForest .

## Setup

Use Python 3.12 on Linux or macOS for the environment used in the final rerun. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

NumPy must remain at 1.26.4 because the original detector uses `np.mat`. The requirements also pin the plotting dependency that could otherwise upgrade NumPy. The earlier Mac progress experiment used Python 3.9.6. Its summarized results and timing are distinct from the final Linux rerun.

## Check the data and code

```bash
python scripts/prepare_data.py
python -m unittest discover -s test -v
```

All 14 upstream tests pass after correcting relative paths in the test harness and removing an extra sample-index column from a HierHash fixture. No detector source changes were made. `results/provenance.json` records hashes and the upstream tree checked.

`python scripts/prepare_data.py --convert` regenerates CSVs from supplied NPZs. The default command only checks them. The synthetic generator writes relative to the working directory. Run `python scripts/generate_synthetic.py` from this project root to reproduce its CSVs. A separate regeneration check was completed and recorded under `results/`.

## Rerun the experiments

Raw AD reproduces the original input protocol:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_final.py --dataset ad --runs 5
```

The final report gives the actual completed AD count, which may exceed this minimum. Its saved consecutive seeds start at 42.

Standardise all extension feature matrices with one common protocol:

```bash
python scripts/run_final.py --dataset annthyroid --standardize --runs 3
python scripts/run_final.py --dataset pageblocks --standardize --runs 3
python scripts/run_final.py --dataset breastw --standardize --runs 15
python scripts/run_final.py --dataset pima --standardize --runs 15
python scripts/run_final.py --dataset yeast --standardize --runs 15
python scripts/run_final.py --dataset synthetic --standardize --runs 15
```

Each seed evaluates OptIForest and conventional Isolation Forest, both with 100 trees and 512 sampled records per tree. Fit and scoring use the same full feature matrix. Labels are used only for metrics. Full-matrix scaling is part of this transductive protocol. No held-out deployment performance or cross-validation is claimed. All duplicates remain. Both Python and NumPy generators are seeded.

The runner saves one CSV record and a score NPZ after each completed model run. It resumes matching records and refuses metadata mismatches. It locks each dataset output on Linux/macOS to prevent overlapping writers. Raw pilot records live under `results/`; final extensions live under `results_scaled/`. The `STOP_RAW` marker prevents accidental continuation of the incomplete raw extension pilot. To investigate that pilot intentionally, first read `results/raw_pilot_notes.json` and explicitly remove the marker. Raw Pima/PageBlocks were computationally impractical under the observed budget.

Runtime warnings from an equal-child-size local statistic are counted, retained in the results and discussed in the report. All saved scores are checked for finiteness. Execution-time records came from shared-host concurrent runs and should be treated as operational measurements.

## Build the figures and report

```bash
python scripts/build_results.py
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The results builder checks at least 15 paired runs for each smaller extension, at least three for Annthyroid/PageBlocks and at least five for AD, with consecutive seed sequences. It generates the tables and scientific figures from the saved results. The report gives the exact number used for every condition. Use the final PDF for iLearn.

For Overleaf, upload the contents of `report/`, set `main.tex` as the main document and compile with pdfLaTeX. The official ACL style and bibliography style are bundled.

## LLM-as-judge disclosure

`results/judge/` contains 12 explanations, their exact evidence and rubric, scores, reasons and an execution disclosure. The judge was the ChatGPT/Codex assistant in the preparation conversation. It also helped construct the templates. This is an unblinded exploratory self-audit, with no independent human raters. Exact deployed model ID, temperature and snapshot were unavailable. No API run, independent validation, human agreement or automatic reproduction of the judgments is claimed. `scripts/prepare_judge.py` recreates the evidence and template cases. Reading the stored judgments is sufficient to reproduce their summary statistics.

## Submission and attribution

The assessment requires a publicly accessible GitHub repository with its URL in a footnote of the report. `report/repository.tex` must contain that actual URL before the final submission. Preserve upstream attribution. This project does not assign a blanket licence to third-party code or historical data. See `DATA_SOURCES.md` for source-specific notes and unresolved provenance.

AI assisted code preparation, experimental orchestration, analysis, drafting and the disclosed explanation audit. The student author should verify the work and read the cited sources before submitting, as required by the assessment brief.
