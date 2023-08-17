# CRCHistoPhenotypes Image Classification

This repository contains a machine learning system capable of classifying 
histopathology images of cancerous cells based on the modified 
["CRCHistoPhenotypes" dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe ).

## Project Overview

Histopathology refers to the examination of a biopsy or surgical specimen 
by a pathologist, after the specimen has been processed and histological 
sections have been placed onto glass slides. In this project, we 
specifically focused on the classification of cancerous cells based on RGB 
images.

### Dataset

The dataset used in this project consists of:
- 27x27 RGB images of cells from 99 different patients.
- For the first 60 patients, labels `isCancerous` and `cell-type` have 
been provided by medical experts.
- For the remaining 39 patients, only the `isCancerous` label has been 
provided.


### Tasks

1. **Binary Classification**: Classify images according to whether a given 
cell image represents cancerous cells or not (`isCancerous`).
2. **Multi-Class Classification**: Classify images according to cell-type, 
namely: 
    - Fibroblast
    - Inflammatory
    - Epithelial 
    - Others

#### Results

The achieved metrics for the multi-class classification model are:
- **Accuracy**: 81.77%
- **Precision**: 84.23%
- **Recall**: 79.29%
- **F1 Score**: 81.82%

## HD Extension

To explore the potential of the additional data, we investigated its use 
for improving the cell-type classification model. We explored a 
semi-supervised learning method, [**FixMatch**](https://github.com/google-research/fixmatch). This approach makes use of 
labeled and unlabeled data to enhance the classification performance.
