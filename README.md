# Italian PoS tagging with BERT

Fine-tuning [`dbmdz/bert-base-italian-cased`](https://huggingface.co/dbmdz/bert-base-italian-cased) for UPOS tagging and morphological feature prediction on [UD Italian-ISDT 2.15](https://github.com/UniversalDependencies/UD_Italian-ISDT/tree/r2.15), with gold word and sentence boundaries.

The project compares frozen BERT, fine-tuned BERT, and BERT with a UPOS CRF against UDPipe 2. Training has two stages: UPOS first, then UPOS + FEATS from the selected UPOS checkpoint. Each stage runs for 10 epochs with development-set checkpoint selection, using three seeds (`42`, `123`, `456`).

## Results

Test scores (%), averaged over three seeds for BERT. Full results and standard deviations are in the [report](report/report.pdf).

| Model | UPOS accuracy | FEATS exact accuracy | FEATS macro-F1 |
|---|---:|---:|---:|
| UDPipe 2 | 98.75 | 98.29 | 90.04 |
| BERT, UPOS only | 98.74 | — | — |
| BERT, UPOS + FEATS | 98.86 | 98.45 | 90.23 |

FEATS macro-F1 considers attribute-value pairs present in the test set.

## Contents

- [`BERT_Italian_PoS_UD_2_15.ipynb`](BERT_Italian_PoS_UD_2_15.ipynb): data preparation, training, and evaluation.
- [`scripts/analysis/`](scripts/analysis/): metric recalculation, paired bootstrap, error examples, and corpus comparison.
- [`scripts/figures/`](scripts/figures/): diagrams and result plots for the report.
- [`report/`](report/): LaTeX source, bibliography, figures, and compiled PDF.

Local datasets and experiment outputs live in `data/` and `outputs/`, which are excluded from Git. UPOS runs are stored in `outputs/bert_upos_then_feats_v1/`; joint-stage runs and combined summaries are in `outputs/bert_upos_then_feats_v2/`.

## Usage

Open the notebook from this directory and run its cells in order. It includes dependency installation and dataset download. For reproducibility, use the configuration and package versions recorded in `experiment.json`; use new output directories for new experiment settings.

To recalculate analyses from saved predictions (Python 3.11+, NumPy):

```console
python -B scripts/analysis/evaluate_udpipe_morphology.py
python -B scripts/analysis/evaluate_upos_uncertainty.py
python -B scripts/analysis/extract_error_examples.py
```

These commands require the local corpus, saved predictions, and cached UDPipe responses in `outputs/references/udpipe_2_15/`. They do not rerun training or inference. Figure scripts additionally require Matplotlib and write to `report/figures/`.

To compile the report with LaTeX and `latexmk`:

```console
latexmk -cd -pdf report/report.tex
```
