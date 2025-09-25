# Machine Learning & Biomedical Signal Processing Portfolio
This repository contains end-to-end projects implemented in **Python (Google Colab)** and **MATLAB**, focused on computer vision and physiological signal analysis.

## 🧭 Contents
- Jupyter notebooks (`.ipynb`) for data preprocessing, modeling, and evaluation
- MATLAB scripts for signal processing pipelines
- Clear metrics, datasets, and reproducibility notes

## 📂 Projects

- **Skin Lesion Segmentation (ISIC)**  
  Unsupervised K-Means + post-processing, compared against ground-truth masks.  
  **Key metrics:** F1 ≈ 0.75, Acc ≈ 0.82 (sample split)

- **HEp-2 Radiomics Classification**  
  First-order + texture + shape features → **Random Forest** classifier.  
  **Best Accuracy:** ~0.94–0.95 (holdout)

- **ECG Apnoea Detection (RR/HRV)**  
  RR-interval features per minute → **Logistic Regression (balanced)** with LOPOCV.  
  **Avg Test:** Acc ≈ 0.70, Sens ≈ 0.76, Spec ≈ 0.47

- **SpO₂ + QRS Apnoea Detection (MATLAB)**  
  Per-second SpO₂ + HR/HRV features → **logistic model** with smoothing & min-run post-processing.  
  **Hold-out (per record):** Sens ≈ 0.81, PPV ≈ 0.79, **F1 ≈ 0.84**

## File Guide
- `project*_*.ipynb` — Google Colab notebooks (Python)
- `matlab/project6_spo2_qrs_apnoea_detection.m` — MATLAB end-to-end pipeline

## Tech Stack
**Python:** numpy, pandas, scikit-learn, scipy, scikit-image, matplotlib, h5py  
**MATLAB:** Signal Processing, Statistics and Machine Learning
