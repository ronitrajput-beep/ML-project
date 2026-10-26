# Mask augmentation under channel loss

A completed, reproducible course research study of **fault classification when input channels disappear**. The manuscript uses the IEEE two-column class and contains Abstract, Introduction, Literature Review, Experimental Setup, Proposed Methodology, Results and Analysis, Conclusion, Discussion, and References.

**Read the paper:** [PDF](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Full-text literature notes](literature/REVIEW_NOTES.md)

## What was measured

Two datasets, three split seeds, three principal model families, and an extra corruption control:

1. Balanced logistic regression with training-only preprocessing.
2. Native-missingness histogram gradient boosting (HGB).
3. Group-outage-augmented HGB (Group-HGB).
4. Random-cell-augmented HGB (Cell-HGB), a matched selected-cell-budget control.

All 24 fits in the fixed principal protocol were executed. Six additional, explicitly post-hoc clean-repeat fits test the representation-count caveat. The fixed-protocol archive contains 576 condition/model/threshold evaluations, all confusion counts, raw probabilities, split indices, 24 saved models, fit timings, protocol hashes, a fresh-fit reproduction check, and an audit that recomputes every result row.

| Positive-class F1 | APS, real truck data | AI4I, synthetic |
|---|---:|---:|
| HGB, clean + three anticipated outages | 0.8359 | 0.5610 |
| Group-HGB, same conditions | 0.8435 | 0.6006 |
| HGB, two untrained group outages | 0.839 | 0.488 |
| Group-HGB, two untrained group outages | 0.843 | 0.404 |
| Cell-HGB, two untrained group outages | 0.841 | 0.582 |

Values are three-split means; full precision and sample SD are in `results/primary_summary.csv` and `results/summary.csv`. The primary APS gain has a conditional paired-bootstrap 95% interval spanning zero. On the untrained AI4I rotational-speed outage, Group-HGB F1 is 0.223 versus HGB's 0.441. No test scores were used to select thresholds.

**The finding is a transfer boundary, not a universal improvement.** Targeted augmentation can improve the anticipated-outage average yet underperform a cheap random-cell control on different outage identities. An exploratory clean-repeat control also reaches 0.614 on AI4I without any masking, so a mask-specific benefit is not established. This is an original empirical case study; global method novelty and publication acceptance have not been established. The paper does not claim a new learning algorithm or state-of-the-art scores.

## Reproduce

Python 3.12 was used. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m unittest discover -s tests -v
python scripts/run_experiments.py
python scripts/analyze_results.py
python scripts/audit_data.py
python scripts/exploratory_redundancy.py
python scripts/exploratory_clean_repeat.py
python scripts/verify_results.py
python scripts/reproduce_one.py
```

On Windows activate `.venv\Scripts\activate` instead. CPU-only; four numerical-library threads. Measured fit time across the original 24 fits was 88.7 seconds on this cloud instance; acquisition, evaluation, bootstrap, reproduction, and paper production take additional time. Laptop/Colab timings will differ. The full run rewrites generated results; preserve the committed outputs before conducting changed-protocol experiments. `reproduce_one.py` uses temporary outputs and preserves the original artifacts.

No API credentials, paid service, GPU, or experiment-tracking account are needed for model training. Data acquisition uses pinned public mirrors and committed checksums. Read [data provenance and terms](data/README.md); raw tables are not redistributed.

## Apply a saved model

```bash
python scripts/infer.py \
  --artifact models/aps_11_group_hgb.joblib \
  --csv data/aps_test.csv \
  --policy environment \
  --output predictions.csv
```

The CSV output contains a score, the fixed threshold and the predicted label. Balanced-loss scores are not guaranteed calibrated fault probabilities. Missing input columns are treated as unavailable channels. AI4I raw product type is one-hot encoded automatically. These are research artifacts, not validated operational maintenance systems.

## Paper and literature

The ten-paper targeted review covers 2020–2025. Publisher-owned GitHub mirrors allowed full-text acquisition despite blocked scholarly hosts. It is not an exhaustive review of all 2026 research. Main-text methods, evaluations and limitations were reviewed; proofs and every appendix were not exhaustively audited. Source URLs, pins, checksums, page-grounded notes, and rejected broad novelty claims are in `literature/`.

Optional literature/PDF tools:

```bash
python -m pip install -r requirements-literature.txt
python scripts/retrieve_literature.py
python scripts/build_tables.py
python scripts/draw_architecture.py
```

A TeX installation containing `pdflatex`, BibTeX and the included `IEEEtran` bibliography style builds the paper. The IEEEtran class and bibliography style are included with its original licence notices.

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The PDF's generic course-project author line makes no claim about author identity or affiliation. Replace it with the actual authors' details before submitting the manuscript, and review the technical claims with your supervisor.

## Audit trail

- [Fixed protocol](docs/PROTOCOL.md), saved before fitting; internally frozen, not externally registered.
- [Research decisions and limitations](docs/DECISIONS.md).
- [Exploratory sensitivity plan](docs/EXPLORATORY_PLAN.md), separate from confirmatory experiments.
- `results/reproduction_check.json`: a fresh APS seed-11 fit reproduces all 34 arrays exactly in this environment.
- `results/verification.json`: all 576 metric rows recomputed and all 24 saved models reloaded and prediction-checked.
- `results/bootstrap.json`: conditional paired intervals and their scope.
- [Model inventory](models/README.md).

AI tools assisted code implementation, analysis and manuscript drafting. The measured results are generated by the released scripts; stylistic editing is not evidence of scientific novelty or a guarantee about AI-detector outputs. Third-party source documents, datasets, software and IEEE class retain their original terms.
