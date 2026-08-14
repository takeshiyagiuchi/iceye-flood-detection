# ICEYE Flood Detection

## Abstract

Waterlogging in agricultural fields can negatively affect soil conditions and agricultural operations, but detecting small-scale waterlogging using satellite imagery remains challenging. In particular, the limited availability of field-verified flooding evidence makes it difficult to train and evaluate conventional image segmentation models. This study proposes a deep learning approach that combines a Siamese Network and Grad-CAM to detect potential waterlogged areas from high-resolution ICEYE Synthetic Aperture Radar (SAR) imagery without requiring pixel-level segmentation labels.

The proposed model takes a pair of SAR images acquired at different times and predicts precipitation conditions between the acquisitions. ResNet-18 is used as the shared backbone of the Siamese Network to extract spatial features from the two images. Terrain information, vegetation information, and SAR acquisition metadata are also incorporated as additional inputs. Grad-CAM is applied to the final convolutional layer of ResNet-18 to identify spatial regions contributing to the precipitation prediction. The hypothesis is that spatial features associated with waterlogging may contribute to precipitation prediction and consequently be highlighted by Grad-CAM.

A dataset consisting of 960 image pairs was constructed from ICEYE imagery acquired over the Elora Research Station in Ontario, Canada, during 2024 and 2025. The dataset was divided into 720 training, 180 validation, and 60 test pairs. Due to the limited field evidence, synthetic test patches were created to enable quantitative evaluation of the spatial overlap between Grad-CAM highlights and field-verified flooded areas.

The results show that both classification and regression models can predict precipitation events with reasonable performance, while Grad-CAM produces spatial highlights in most test patches. However, the highlights do not consistently correspond to waterlogged areas and can also react to other land cover types and unchanged dry areas. Therefore, although the proposed approach demonstrates the potential for flood detection without pixel-level annotations, further field-verified data and improved evaluation methods are required to establish its reliability.


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