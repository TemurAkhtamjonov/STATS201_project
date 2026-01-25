# Visual Framing of Political Protest Images

## Project Overview
This project studies how political protests are visually framed in news images. Using computer vision and machine learning, we test whether image-derived features can help identify visual framing patterns in political images and test whether simple visual proxies (e.g., crowd presence) can be predicted from image-derived signals.

## Research Question
**What visual patterns characterize protest images, and can these patterns be used to classify images as protest-related?**

## Machine Learning Task (Week 3 baseline)
For Week 3, we build **baseline classifiers** for a binary label:
- **Target (y):** `iscrowd` (proxy outcome; class 1 = crowd present)
- **Baseline feature (X):** `predicted_labels` (cluster label per image from `clusters_preds_caravan_newsapi.csv`)

Because the label is imbalanced, we report metrics beyond accuracy (F1, balanced accuracy, ROC-AUC).

## Data Source
We use replication data from:
Torres (2024). *A Framework for the Unsupervised and Semi-Supervised Analysis of Visual Frames*.

The replication package includes:
- News images
- Image-level metadata
- Pre-extracted visual features for copyrighted images

Key files used so far:
- `metadata_caravan_media_short.csv` (image metadata, includes `iscrowd`)
- `clusters_preds_caravan_newsapi.csv` (cluster label per image filename)

## Repository Structure
- `notebooks/` — Colab notebooks (Week 2/3)
- `figures/` — saved plots (e.g., label distribution)
- `data/` — (optional) small derived data files only (no large/copyrighted raw media)
- `reports/` — written report drafts and final submission
- `src/` — helper scripts (if added later)

## Week 2 — Feasibility + Data Understanding
### What we did
1. Loaded metadata and inspected columns
2. Examined label distribution for `iscrowd`
3. Confirmed class imbalance and dataset size

### Outputs
**Label distribution (`iscrowd`)**
![Label distribution of iscrowd](figures/label_distribution.png)

## Week 3 — Baseline Models & Evaluation

### Data merge
We merged metadata with cluster predictions on `imageid`:
- Metadata rows: **688**
- Matched rows after merge: **517**

This means Week 3 baselines are trained/evaluated on **517** images that appear in both files.

### Train/Test split
We used a **stratified 70/30 split** to preserve class proportions:
- `test_size=0.30`, `random_state=42`, `stratify=y`

### Baseline models implemented
1. **Majority class baseline**
   - Predicts the most common label for every example.
2. **Cluster-rate baseline**
   - Estimates `P(y=1 | cluster)` on train, predicts 1 if probability ≥ 0.5.
3. **Logistic regression baseline**
   - One-hot encodes `predicted_labels`
   - Logistic regression with `class_weight="balanced"`

### Evaluation metrics
We report:
- Accuracy
- Confusion matrix
- **F1 score (class 1)**
- **Balanced accuracy**
- **ROC-AUC** (when probabilities available)
- 5-fold CV F1 (StratifiedKFold)

### Current results (Week 3)

Latest run (one-hot + logistic regression with `class_weight="balanced"`; stratified 70/30 split, `random_state=42`).

| Model | Accuracy | F1 (class 1) | Balanced Acc | ROC-AUC | Notes |
|---|---:|---:|---:|---:|---|
| Majority baseline | 0.788 | 0.000 | 0.500 | N/A | Predicts all 0 (no crowds) |
| Cluster-rate baseline | 0.872 | 0.722 | 0.841 | N/A | Uses train cluster positive rates |
| Logistic (one-hot clusters, balanced) | 0.872 | 0.722 | 0.841 | 0.895 | Uses `predicted_labels` as categorical |

**Confusion matrix (logistic):**  
TN=110, FP=13, FN=7, TP=26

**5-fold CV (StratifiedKFold) F1 mean:** 0.717 (std: 0.067)

## How to Reproduce (Colab)

### Requirements
- Google Colab (Python 3)
- Data folder (Torres replication) stored in Google Drive

### Steps
1. Open the notebook:
   - `notebooks/week3_baseline_models.ipynb`

2. Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

### Links 
- Colab notebook: https://colab.research.google.com/drive/1n7k6R5bAs274ZNCcv-ckoQO0DQK9VgDW?usp=sharing
- Data folder in Drive: https://drive.google.com/drive/folders/1_EA2hMWBHYXhxKx_hGfaqKbFpd3fvr-I?usp=drive_link
