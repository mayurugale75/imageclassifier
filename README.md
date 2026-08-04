# Satellite Land-Use Classifier

A computer-vision project that classifies Sentinel-2 satellite image tiles into EuroSAT land-use categories and compares two observations using ResNet-18 feature embeddings and cosine similarity.

## Features

- Transfer learning with ImageNet-pretrained ResNet-18
- Reproducible train/validation split and checkpoint metadata
- Macro F1 score, classification report, confusion matrix, and one-vs-rest ROC curves
- Streamlit dashboard for single-image classification
- Temporal change analysis with global cosine similarity and patch-level embedding heatmap

## Dataset

The project uses [EuroSAT](https://github.com/phelber/EuroSAT), a 10-class Sentinel-2 land-use dataset. Dataset files and trained models are excluded from Git. Download the dataset locally:

```powershell
python download_dataset.py
```

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --epochs 10
python evaluate.py --checkpoint checkpoints/best_model.pt
streamlit run streamlit_app.py
```

Evaluation writes `metrics.txt`, `confusion_matrix.png`, and `roc_curves.png` to `artifacts/`. The dashboard's second tab expects two co-registered images of the same scene; bright areas in the map identify high embedding change.

## Structure

```text
models/              ResNet-18 classifier and embedding encoder
utils/               data and temporal-comparison utilities
train.py             training entry point
evaluate.py          F1, confusion matrix, and ROC generation
streamlit_app.py     interactive dashboard
```

## Scope and limitation

The temporal component is unsupervised embedding comparison, so it indicates visual/semantic change rather than a labelled change type. For reliable use, input images should show the same aligned location with similar illumination. A production extension should validate the similarity threshold using labelled paired time-series data and incorporate geospatial metadata.

## Baseline results

The existing legacy ResNet-18 checkpoint was evaluated on a fixed 256-image slice of the reproducible validation partition. It achieved **0.9117 macro F1** and **0.91 accuracy**. This is a demonstration benchmark, not the final full-partition result, because the original checkpoint was trained before the split seed was recorded. Retraining with `train.py` and running `evaluate.py` without `--max-samples` will produce the final reproducible result.

- [Metrics report](docs/results/metrics.txt)
- [Confusion matrix](docs/results/confusion_matrix.png)
- [ROC curves](docs/results/roc_curves.png)
