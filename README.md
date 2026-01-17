# Visual Framing of Political Protest Images

## Project Overview
This project studies how political protests are visually framed in news images.
Using computer vision and machine learning, we analyze whether interpretable
visual features extracted from images can distinguish protest-related images
from non-protest political images.

## Research Question
What visual features characterize protest images, and can these features be
used to classify images as protest-related?

## Machine Learning Task
We frame this as a supervised image classification task:
- **Input:** visual features extracted from images
- **Output:** protest vs non-protest label (or crowd presence proxy)

## Data Source
We use replication data from:

Torres (2024). *A Framework for the Unsupervised and Semi-Supervised Analysis of
Visual Frames*.

The replication package includes:
- News images
- Image-level metadata
- Pre-extracted visual features for copyrighted images

## Planned Methods
- Image preprocessing (resize, normalization)
- Visual feature extraction (edge density, brightness, color saturation,
  person-density proxies)
- Baseline models:
  - Logistic regression
  - Random forest
- Model evaluation using a train/test split

## Project Status
Week 2: Problem formulation, dataset feasibility, and repository setup.
