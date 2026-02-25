# Predicting Crowd Presence in News Images

## Project Overview
This project predicts **crowd presence** in news images.

The target variable (`iscrowd`) indicates whether an image contains a visible crowd. Importantly, this is **not a protest classifier**. The model predicts crowd presence as a visual proxy and does not distinguish protest crowds from other types of gatherings.

We evaluate whether **face-derived visual features** add predictive signal beyond an **unsupervised cluster-based image representation**.

## Research Question
**Substantive:** How can we measure **crowd presence** as a component of visual framing in a large corpus of news images?

**Operational:** Can image-derived representations predict whether an image contains a crowd (`iscrowd`)?

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
- `notebooks/` — Colab notebooks (Weeks 2–6 + final_model)
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

**Confusion matrix (logistic regression baseline):**  
![Week 3 confusion matrix](figures/confusion_matrix_logistic.png)

**5-fold CV (StratifiedKFold) F1 mean:** 0.717 (std: 0.067)

## Week 4 Progress (Feature Construction & Model Comparison)

**Goal.** Compare multiple representations/models for predicting `iscrowd` using the Torres replication data.

### Data + Features
- **Base representation:** `predicted_labels` (cluster ID from `clusters_preds_caravan_newsapi.csv`)
- **Extended representation:** `predicted_labels + newspaperid + ideol_allsides + topicality`
  - `topicality` had missing values, so we compared:
    - **Dropna** (remove rows with missing topicality)
    - **Median imputation** (fill missing topicality with the median to keep all rows)

### Models Compared (70/30 split, stratified, random_state=42)
- Majority-class baseline
- Logistic Regression (cluster-only)
- Logistic Regression (cluster + metadata, dropna)
- Logistic Regression (cluster + metadata, median-imputed)
- Decision Tree (median-imputed)
- Random Forest (median-imputed)
- Logistic Regression with **threshold tuning** (optimize F1 for class 1)

### Key Results (class 1 = `iscrowd`)
| Model | Acc | Prec_1 | Rec_1 | F1_1 |
|---|---:|---:|---:|---:|
| RandomForest (imputed) | 0.910 | 0.771 | 0.818 | 0.794 |
| LogReg (imputed) tuned thr=0.65 | 0.897 | 0.730 | 0.818 | 0.771 |
| LogReg cluster+meta (median-impute) | 0.872 | 0.651 | 0.848 | 0.737 |
| LogReg cluster-only | 0.872 | 0.667 | 0.788 | 0.722 |
| DecisionTree (imputed) | 0.865 | 0.636 | 0.848 | 0.727 |
| LogReg cluster+meta (dropna) | 0.862 | 0.647 | 0.759 | 0.698 |
| Majority baseline | 0.788 | 0.000 | 0.000 | 0.000 |

### Model Comparison Figure
![Week 4 model comparison](figures/week4_model_comparison.png)

### Code
Week 4 notebook:
- `notebooks/week4.ipynb`

## Week 5 – Diagnostics, Errors, and Robustness (Image-Based Analysis)

### Goal
The goal of Week 5 was to **diagnose model behavior and failure modes** by introducing an explicit image-based representation and evaluating its robustness. Unlike earlier weeks, which relied on precomputed visual clusters and metadata, this week directly processes raw images to extract an interpretable signal: **the number of detected faces in each image**.

This enables targeted error analysis and sensitivity checks using a simple, human-interpretable visual heuristic.

### Image-Based Feature: Face Count
We applied a pretrained **MTCNN face detection model** to all available images (N = 517).  
For each image, we extracted the following features:

- `face_count`: total number of detected faces  
- `face_count_hi`: number of high-confidence detections  
- `max_face_prob`: maximum detection confidence in the image  

This representation directly reflects the intuition that crowd images contain more visible faces, while remaining independent of text or metadata cues.

### Diagnostic Analysis: Distribution by Label
We first examined how detected face counts differ between crowd and non-crowd images.

![Face count distribution by crowd label](figures/facecount_by_label_boxplot.png)

**Observation:**
- Images labeled `iscrowd = 1` exhibit substantially higher face counts on average.
- The distribution is highly skewed, with extreme outliers (up to 400+ faces).
- A log scale is required to visualize variability, indicating rare but very large gatherings.

This confirms that face count is a meaningful but noisy visual signal.

### Robustness Check: Threshold Sensitivity
We evaluated a simple rule-based classifier:

> Predict `iscrowd = 1` if `face_count ≥ K`

We swept thresholds K in [0, 50] and evaluated F1 score on both training and test splits.

![Sensitivity of face-count threshold](figures/k_sweep_f1_train_test.png)

**Observation:**
- Performance is highly sensitive to the choice of threshold \( K \).
- The optimal threshold on the training set is **K = 6**.
- Performance degrades sharply for larger thresholds, revealing brittleness.

This highlights the instability of hard decision rules and motivates more flexible modeling approaches.

### Error Analysis: Confusion Matrix (K = 6)
Using the best-performing threshold \( K = 6 \), we evaluated test-set errors.

![Confusion matrix for K=6](figures/confusion_matrix_k6_test.png)

**Findings:**
- False positives often correspond to images with many visible faces but no labeled crowd (e.g., collages, repeated faces, studio audiences).
- False negatives include crowd scenes where faces are small, occluded, or viewed from a distance.

This demonstrates that face count alone cannot reliably capture crowd semantics.

### Failure Modes and Bias
Key failure modes identified:
- **Overcounting:** repeated faces, posters, or screens inflate face counts.
- **Undercounting:** distant or occluded crowds yield few detections.
- **Context blindness:** face count ignores spatial layout and interaction between people.

These issues highlight how naive visual heuristics can encode systematic bias.

### Implications and Next Steps
Week 5 diagnostics show that:
- Image-based features provide strong, interpretable signals.
- Simple rules are brittle and sensitive to parameter choice.
- Error analysis reveals where visual heuristics fail semantically.

## Week 6 – Synthesis & Final Model (updated)

### Goal
Synthesize prior stages into a final, interpretable model for predicting `iscrowd` (crowd presence).

**Note on metadata models (Week 4):** 
We explored metadata-augmented models that achieved higher predictive performance, but we exclude metadata from the final model to keep the task visually grounded and aligned with `iscrowd` as a crowd-presence label.

### Final Model Specification
The final model is a **logistic regression classifier** combining:

### Visual Representation
- `predicted_labels` (image cluster ID)

### Image-Based Features
- `face_count`
- `face_count_hi`
- `max_face_prob`

Processing steps:
- `predicted_labels` → one-hot encoded
- Numeric face features → passed through directly (missing filled with 0)
- 70/30 stratified train–test split (`random_state=42`)
- `class_weight="balanced"` to handle label imbalance

### Final Results
Baseline vs final (held-out test set):

| Model | Accuracy | Prec_1 | Rec_1 | F1_1 |
|---|---:|---:|---:|---:|
| Baseline (clusters only) | 0.8718 | 0.6667 | 0.7879 | 0.7222 |
| Final (clusters + faces) | 0.8590 | 0.6170 | 0.8788 | 0.7250 |

Interpretation:
- Face features **increase recall** (more crowd images detected) and slightly increase F1.
- Precision and overall accuracy decrease slightly, indicating a recall–precision trade-off.

**Confusion matrix (Final: clusters + faces):**  
![Confusion matrix (clusters + faces)](figures/confusion_matrix_clusters_faces_model.png)

### Substantive Takeaway
The results support a cautious claim: crowd presence can be predicted from image-derived representations, and face-derived cues increase sensitivity to crowd images, but the improvement is modest and introduces more false positives.

# Model Evolution

| Week | Representation | Model | Goal |
|------|---------------|-------|------|
| Week 3 | Clusters only | Logistic regression | Baseline |
| Week 4 | Clusters + metadata | Multiple models | Representation exploration |
| Week 5 | Face-based features | Threshold rule + diagnostics | Error analysis |
| Week 6 (updated) | Clusters + face features | Logistic regression | Final comparison (visual-only) |

This progression demonstrates how representation changes model behavior.

# Why We Stop Here

We stop development at this point for three reasons:

1. Performance improvements plateau.
2. The final model balances interpretability and predictive power.
3. The project goal was diagnostic and comparative — not state-of-the-art optimization.

Further deep learning experimentation (e.g., CNN fine-tuning) would increase complexity without clear theoretical justification given the dataset size (N = 517).

# Substantive Interpretation

The results suggest:

- Unsupervised **visual cluster labels** capture meaningful visual regularities associated with crowd presence in this image corpus.
- Adding **face-derived features** shifts the model toward **higher sensitivity** (higher recall for `iscrowd=1`) with a small trade-off in precision/accuracy.
- Crowd presence is partially measurable from image-derived representations, but the improvement from face features is **modest**.

What a reader should conclude:
- The pipeline can support **large-scale measurement of crowd presence** (`iscrowd`) in the dataset.

What a reader should NOT conclude:
- This is not a system for identifying protests or interpreting political meaning from images.

# Scope

This model is intended for:

- Media research on visual framing
- Archival filtering of **crowd vs non-crowd images** in the corpus
- Exploratory social science measurement workflows

It is **not** designed for:
- Real-time surveillance
- General-purpose crowd detection outside this dataset
- High-stakes automated deployment

# Limitations

- Face detection fails in distant or occluded crowds.
- Cluster labels are unsupervised and may encode noise.
- The dataset size limits more complex supervised vision models.
- Results may not generalize beyond this corpus and label definition (`iscrowd`).

## How to Reproduce (Colab)

### Requirements
- Install dependencies: `pip install -r requirements.txt`
- Torres replication data available in Drive (see link below)

### Run order
1. Main notebook (current): `notebooks/final_model.ipynb`
2. Weekly notebooks (archive / earlier iterations): see `notebooks/`

### Links 
- Colab notebook: Find All Google Colab Notebooks in the "notebooks" folder
- Data folder in Drive: https://drive.google.com/drive/folders/1_EA2hMWBHYXhxKx_hGfaqKbFpd3fvr-I?usp=drive_link
