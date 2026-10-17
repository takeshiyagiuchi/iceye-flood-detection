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

<div align="center">
**Table 1. Precipitation Prediction Scores (Normal)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Classification | 86.7 | 86.7 | 86.7 | 86.7 |
| Regression | 86.7 | 86.7 | 86.7 | 86.7 |
</div>

<br><br>

<div align="center">
**Table 2. Precipitation Prediction Scores (Controlled)**

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
**Table 3. Flood Detection Scores**

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