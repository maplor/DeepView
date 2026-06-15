# Reproduction: `active_supContrast_iter20_visUnlabelData.ipynb`

Self-contained, modular re-implementation of the active-learning + supervised
contrastive notebook (`deepview/calculate_results/pape_results/`).

## Modules

| File | Notebook block it reproduces |
|------|------------------------------|
| `data_preprocessing.py` | 数据读取 — load `all_norm_data.pkl`, majority-vote labels, drop unknown(-2), 80/20 stratified split, 1% labelled / 99% unlabelled split. Also holds `majority_value`, the 2-arg `data_loader_umineko`, and the label dictionaries/colour maps. |
| `model.py` | `SimpleNN` (dual conv-AE encoders → 32-d embedding → projector / classifier) and `SupContrastiveLoss`. |
| `train_eval.py` | `train_model`, freeze/unfreeze strategies, `evaluate_model`, `evaluate_supContrast_model` (confidence threshold 0.5), `AE_eval_time_series`, entropy `uncertainty_sampling`. |
| `visualize.py` | UMAP projection, `LabelSpreading` propagation, plotly scatter HTMLs, matplotlib time-series / raw-accel PDFs. |
| `run_reproduce.py` | End-to-end runner: the 20-iteration active-learning loop + saving + figures. |
| `model_func.py` | Verbatim copy of the encoder/decoder definitions (`Autoencoder3d4`, `Autoencoder1d`, …) so the package needs no external repo modules. |

## Data requirements

All **code** in this folder is self-contained (no imports from the `deepview`
package — `model_func.py` carries verbatim copies of every encoder/decoder).
The only external dependency is the **data files**, which the scripts load via
relative paths pointing one level *above* `reproduce/`. They must exist at the
repo root for the scripts/notebooks to run:

| Entrypoint(s) | Required data file(s) |
|---------------|------------------------|
| `run_reproduce.py`, `reproduce_walkthrough.ipynb` (umineko) | `../analysis/pape_results/all_norm_data.pkl` |
| `reproduce_walkthrough_bear.ipynb` (`bear_data.py`) | `../data/bear/{data,label}.npy` |
| `reproduce_walkthrough_turtle.ipynb` (`turtle_data.py`) | `../data/turtle/{data,label}.npy` |
| `run_turtle_fixed.py` | `../data/turtle/{data_w40,label_w40}.npy` |
| `reproduce_walkthrough_omizu.ipynb`, `omizu_stats.py` (`omizu_data.py`) | `../data/omizunagidori/{data,label}.npy` |

These `.npy`/`.pkl` files are `.gitignore`d (large), so they are **not** copied
into `reproduce/` — running depends on them being present at the repo root.
The `build_notebook_*.py` scripts regenerate the corresponding `.ipynb`
walkthroughs from the modules.

## Run

```bash
# from repo root, using the project venv (Python 3.9, torch 2.4)
.venv/bin/python reproduce/run_reproduce.py            # metrics + all figures
.venv/bin/python reproduce/run_reproduce.py --no-viz   # metrics only (faster)
```

Data path defaults to `../analysis/pape_results/all_norm_data.pkl`
(override with `--pkl /path/to/all_norm_data.pkl`).

## Outputs (written to `reproduce/results/`)

- `metrics.csv` — per-iteration labelled-set size, train/test Accuracy, Macro-F1, Micro-F1.
- `results.pkl` — full run: metrics, per-iteration model weights, selected labels, test predictions.
- `final_model.pth` — final model `state_dict`.
- `label_propagation_*_iter20_warmup20.html`, `true_label_all_iter20_warmup20.html` — plotly UMAP scatters.
- `propagationlabel_20_warmup20.pdf`, `modelpred_accel_20_warmup20.pdf`, `propagation_accel_20.pdf` — matplotlib time-series figures.

## Notes on exact reproduction

Seeds match the notebook (`seed=2025`, splits `random_state=42`), and the
training order / hyper-parameters are copied verbatim. Absolute metric values
may differ by a few hundredths from the notebook's printed table because the
original ran on a different OS / torch build (BLAS and RNG kernels differ
across platforms even on CPU). The **trend** — test accuracy rising from
~0.79 to ~0.92 and Macro-F1 from ~0.44 to ~0.84 over 20 iterations — is what
is reproduced.
