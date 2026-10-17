# Flood Detection in Agricultural Fields Using High-Resolution SAR Imagery and Deep Learning

<p style="font-size: 1.4em;">DATA*6700 Master of Data Science Project | University of Guelph</p>

<br>

## Project Overview

Project Overview

This project investigates agricultural flood detection using multi-temporal, high-resolution ICEYE Synthetic Aperture Radar (SAR) imagery over the Elora Research Station in Ontario, Canada. A Siamese Network with a shared ResNet-18 backbone is developed to analyze changes between SAR images acquired at different times and identify potential flooding.

The project also explores model explainability using Grad-CAM to investigate which spatial features contribute to model predictions. Because detailed pixel-level ground-truth flood masks are unavailable, a controlled-patch evaluation approach is developed to quantitatively assess the spatial relevance of Grad-CAM highlights using field-verified flood locations.

<p align="center">
  <img src="report/figures/fig1.png" width="600">
</p>
<p align="center">
  <em>Figure 1. Examples of Waterlogging in the Study Area. Photographs provided by Dr. Berg and Abbey Papadimitriou.</em>
</p>

<br><br>

<p align="center">
  <img src="report/figures/fig2.png" width="300">
</p>
<p align="center">
  <em>Figure 2. Study Area and Test Patch Locations. The left map shows the Elora Research Station, including four field-verified areas outlined by orange polygons and the extent of one SAR image. The right map shows the locations of the 30 test patches.</em>
</p>

<br><br>

<p align="center">
  <img src="report/figures/fig3.png" width="600">
</p>
<p align="center">
  <em>Figure 3. Model Architecture.</em>
</p>

<br><br>

<p align="center">
  <img src="report/figures/fig4.png" width="600">
</p>
<p align="center">
  <em>Figure 4. Depression Map and Sample NDVI Image.</em>
</p>

<br><br>

<p align="center">
  <img src="report/figures/fig5.png" width="300">
</p>
<p align="center">
  <em>Figure 5. Training Patch Locations.</em>
</p>

<br><br>

<p align="center">
  <img src="report/figures/fig6.png" width="300">
</p>
<p align="center">
  <em>Figure 6. Training and Validation Curves for the Classification and Regression Models.</em>
</p>

<br><br>

<p align="center"><strong>Table 1. Precipitation Prediction Scores (Normal)</strong></p>

<div align="center">

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Classification | 86.7 | 86.7 | 86.7 | 86.7 |
| Regression | 86.7 | 86.7 | 86.7 | 86.7 |
</div>

<br><br>

<p align="center"><strong>Table 2. Precipitation Prediction Scores (Controlled)</strong></p>

<div align="center">

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Classification | 73.3 | 84.0 | 70.0 | 76.4 |
| Regression | 86.7 | 86.7 | 86.7 | 86.7 |
</div>

<br><br>

<p align="center">
  <img src="report/figures/fig7.png" width="300">
</p>
<p align="center">
  <em>Figure 7. Scatter Plots between Classification Probablilities and Regression Predictions.</em>
</p>

<br><br>

<p align="center"><strong>Table 3. Flood Detection Scores</strong></p>

<div align="center">

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Classification | 70.0 | 80.0 | 53.3 | 64.0 |
| Regression | 75.0 | 82.6 | 63.3 | 71.7 |
</div>

<br><br>

<p align="center">
  <img src="report/figures/fig8.png" width="300">
</p>
<p align="center">
  <em>Figure 8. Example Patch Groups.</em>
</p>

<br><br>

<p align="center"><strong>Table 4. Ablation study - Precipitation Prediction (Normal)</strong></p>

<div align="center">

| Input Combination | Model | TP | FN | FP | TN | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAR + NDVI + dep + meta | Classification | 26 | 4 | 4 | 26 | 0.87 | 0.87 | 0.87 | 0.87 |
| SAR + NDVI + dep + meta | Regression | 28 | 2 | 4 | 26 | 0.90 | 0.88 | 0.93 | 0.90 |
| SAR + dep + meta | Classification | 28 | 2 | 9 | 21 | 0.82 | 0.76 | 0.93 | 0.84 |
| SAR + dep + meta | Regression | 27 | 3 | 5 | 25 | 0.87 | 0.84 | 0.90 | 0.87 |
| SAR + NDVI + meta | Classification | 25 | 5 | 5 | 25 | 0.83 | 0.83 | 0.83 | 0.83 |
| SAR + NDVI + meta | Regression | 22 | 8 | 6 | 24 | 0.77 | 0.79 | 0.73 | 0.76 |
| SAR + NDVI + dep | Classification | 26 | 4 | 8 | 22 | 0.80 | 0.76 | 0.87 | 0.81 |
| SAR + NDVI + dep | Regression | 26 | 4 | 15 | 15 | 0.68 | 0.63 | 0.87 | 0.73 |
| SAR + meta | Classification | 25 | 5 | 2 | 28 | 0.88 | 0.93 | 0.83 | 0.88 |
| SAR + meta | Regression | 23 | 7 | 6 | 24 | 0.78 | 0.79 | 0.77 | 0.78 |
| SAR-only | Classification | 27 | 3 | 7 | 23 | 0.83 | 0.79 | 0.90 | 0.84 |
| SAR-only | Regression | 26 | 4 | 10 | 20 | 0.77 | 0.72 | 0.87 | 0.79 |
</div>

<br><br>

<p align="center"><strong>Table 5. Ablation study - Precipitation Prediction (Controlled)</strong></p>

<div align="center">

| Input Combination | Model | TP | FN | FP | TN | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAR + NDVI + dep + meta | Classification | 22 | 8 | 4 | 26 | 0.80 | 0.85 | 0.73 | 0.79 |
| SAR + NDVI + dep + meta | Regression | 26 | 4 | 5 | 25 | 0.85 | 0.84 | 0.87 | 0.85 |
| SAR + dep + meta | Classification | 25 | 5 | 9 | 21 | 0.77 | 0.74 | 0.83 | 0.78 |
| SAR + dep + meta | Regression | 24 | 6 | 5 | 25 | 0.82 | 0.83 | 0.80 | 0.81 |
| SAR + NDVI + meta | Classification | 27 | 3 | 5 | 25 | 0.87 | 0.84 | 0.90 | 0.87 |
| SAR + NDVI + meta | Regression | 21 | 9 | 6 | 24 | 0.75 | 0.78 | 0.70 | 0.74 |
| SAR + NDVI + dep | Classification | 22 | 8 | 8 | 22 | 0.73 | 0.73 | 0.73 | 0.73 |
| SAR + NDVI + dep | Regression | 22 | 8 | 15 | 15 | 0.62 | 0.59 | 0.73 | 0.66 |
| SAR + meta | Classification | 21 | 9 | 2 | 28 | 0.82 | 0.91 | 0.70 | 0.79 |
| SAR + meta | Regression | 20 | 10 | 6 | 24 | 0.73 | 0.77 | 0.67 | 0.71 |
| SAR-only | Classification | 20 | 10 | 7 | 23 | 0.72 | 0.74 | 0.67 | 0.70 |
| SAR-only | Regression | 25 | 5 | 10 | 20 | 0.75 | 0.71 | 0.83 | 0.77 |
</div>

<br><br>

<p align="center"><strong>Table 6. Ablation study - Flood Detection</strong></p>

<div align="center">

| Input Combination | Model | TP | FN | FP | TN | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAR + NDVI + dep + meta | Classification | 17 | 13 | 2 | 28 | 0.75 | 0.89 | 0.57 | 0.69 |
| SAR + NDVI + dep + meta | Regression | 20 | 10 | 5 | 25 | 0.75 | 0.80 | 0.67 | 0.73 |
| SAR + dep + meta | Classification | 22 | 8 | 8 | 22 | 0.73 | 0.73 | 0.73 | 0.73 |
| SAR + dep + meta | Regression | 20 | 10 | 5 | 25 | 0.75 | 0.80 | 0.67 | 0.73 |
| SAR + NDVI + meta | Classification | 22 | 8 | 4 | 26 | 0.80 | 0.85 | 0.73 | 0.79 |
| SAR + NDVI + meta | Regression | 17 | 13 | 6 | 24 | 0.68 | 0.74 | 0.57 | 0.64 |
| SAR + NDVI + dep | Classification | 17 | 13 | 5 | 25 | 0.70 | 0.77 | 0.57 | 0.65 |
| SAR + NDVI + dep | Regression | 16 | 14 | 15 | 15 | 0.52 | 0.52 | 0.53 | 0.52 |
| SAR + meta | Classification | 19 | 11 | 2 | 28 | 0.78 | 0.90 | 0.63 | 0.75 |
| SAR + meta | Regression | 17 | 13 | 6 | 24 | 0.68 | 0.74 | 0.57 | 0.64 |
| SAR-only | Classification | 15 | 15 | 5 | 25 | 0.67 | 0.75 | 0.50 | 0.60 |
| SAR-only | Regression | 18 | 12 | 10 | 20 | 0.63 | 0.64 | 0.60 | 0.62 |
</div>

<br><br>

<p align="center">
  <img src="report/figures/fig9.png" width="600">
</p>
<p align="center">
  <em>Figure 9. Correlation Boxplots between Grad-CAM highlight and Several Images.</em>
</p>

For detailed methodology, experiments, and analysis:

- [Wetland Segmentation Report](report/iceye-flood-detection-report.md)

<br>

## Repository Structure

```text
iceye-flood-detection/
├── data/
│   ├── 0_metadata/          # Metadata CSV files
│   ├── 1_org/               # Acquired/original data
│   ├── 1-5_intermediate/    # Intermediate data generated during preprocessing
│   ├── 2_processed/         # Processed data ready for modelling
│   ├── 3_results/           # Model outputs and prediction results
│   └── 4_evaluation/        # Evaluation results
│
├── data_process/            # Notebooks for data preprocessing
├── depression/              # Tools for generating depression maps
├── evaluation/              # Notebooks for model evaluation
└── modelling/               # Notebooks for model training and inference