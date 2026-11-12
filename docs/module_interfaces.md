# Module Interfaces

This document defines the data contracts between project modules.

## Pipeline

MVTec AD -> data split -> preprocessing -> feature extraction -> anomaly detector
-> patch score aggregation -> heatmap / mask -> evaluation

## Module responsibilities

### data.py
Input: MVTec AD root and category name.

Output:
- image path
- category
- split
- image-level label
- ground-truth mask path when available

The official train/test split must be preserved.

### preprocess.py
Input: raw image.

Output: preprocessed image with a fixed, documented representation.

Preprocessing parameters must be fitted or selected without using test anomaly labels.

### features.py
Input: image or image patch.

Output: one-dimensional numeric feature vector.

Initial feature families:
- LBP
- HOG
- grayscale statistics
- gradient statistics

### patches.py
Input:
- image
- patch size
- stride

Output:
- patch image
- patch coordinates
- patch size

Coordinates must be sufficient to map patch scores back to the original image.

### models.py
Input:
- normal training feature matrix
- test feature matrix

Output:
- anomaly scores

Initial detectors:
- Isolation Forest
- One-Class SVM
- KNN

The detector layer must not know how patches are visualized or evaluated.

### heatmap.py
Input:
- patch anomaly scores
- patch coordinates
- original image shape

Output:
- spatial anomaly score map

### postprocess.py
Input:
- anomaly score map
- threshold determined from normal validation data

Output:
- binary anomaly mask

Test anomaly labels must not be used to select the threshold.

### metrics.py
Input:
- image-level scores and labels
- pixel-level score maps and ground-truth masks

Output:
- image AUROC
- image AP
- pixel AUROC
- pixel AP

## Cross-module contract

A result for one test image must remain traceable through:

image_id -> patch coordinates -> patch score -> heatmap
-> image score / mask -> ground truth -> evaluation metrics
