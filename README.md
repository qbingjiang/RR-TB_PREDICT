# Tri Model: Deep Learning for Rifampicin-Resistant Tuberculosis (RR-TB) Detection  

## Overview  
Rifampicin-resistant tuberculosis (RR-TB) is a critical barrier to global TB control, undermining progress toward the WHO End TB Strategy. Early and accurate detection of RR-TB is essential to guide timely treatment.  

The **Tri model** is a deep learning system designed to address this challenge. It automatically segments the lung region from CT scans and directly predicts RR-TB vs. drug-sensitive TB (DS-TB) using the segmented whole-lung volume—*without requiring manual annotation of lesions*.  

This repository provides resources related to the Tri model, including its architecture, performance metrics, and interpretability tools.  


## Key Features  
- **End-to-end automation**: Integrates lung segmentation and RR-TB prediction in a single pipeline, eliminating manual lesion annotation.  
- **Dual contextual modeling**: Combines local and global CT feature learning with expert-guided encoding to capture TB-specific patterns.  
- **Strong generalizability**: Validated across 3 independent clinical centers, demonstrating robust performance across diverse cohorts.  
- **Interpretability**: Generates class activation maps (CAMs) to visually explain model predictions, enhancing clinical trust.  


## Performance  
The Tri model achieved consistent performance across all cohorts, with AUC (Area Under the ROC Curve) values ranging from **0.809 to 0.906**.  

It outperformed comparative methods, including:  
- Clinical risk factors  
- Radiomics features  
- Other deep learning baselines  


## Interpretability  
To enhance transparency, the model generates class activation maps (CAMs) that highlight regions in the lung CT volume most influential to its RR-TB/DS-TB predictions. These maps help clinicians correlate model decisions with anatomical features.  


## Applications  
The Tri model offers a rapid, objective tool to support clinical decision-making for RR-TB, enabling:  
- Timely identification of drug resistance  
- Individualized treatment planning  
- Streamlined TB care workflows  



