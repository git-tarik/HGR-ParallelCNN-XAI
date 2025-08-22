# Explainable Hand Gesture Recognition using SHAP-Enhanced ParallelCNN

> MANIT Bhopal – Summer Internship Project · LeapGestRecog · PyTorch · XAI (SHAP)

This project implements a lightweight **ParallelCNN** for static hand-gesture recognition on **LeapGestRecog**, and pairs it with **SHAP** to generate pixel-level attribution maps.  
On our setup, the model reached **~99.5% validation** and **>99% test** accuracy, while explanations consistently highlighted finger contours and palm regions that drive predictions.  
The repo contains the fully computed notebook, exported figures, and minimal code so others can view results quickly and re-run locally.
<!-- Badges -->
<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

**Quick links:**  
- 📓 Notebook: [`notebooks/Parallel_CNN.ipynb`](notebooks/Parallel_CNN.ipynb)  
- 🧩 Confusion Matrix: [`results/confusion_matrix.png`](results/confusion_matrix.png)  
- 📈 Training Curve: [`results/train_accuracy.png`](results/train_accuracy.png)  
- ✅ Validation Curve: [`results/val_accuracy.png`](results/val_accuracy.png)  
- 🔬 Saliency Grid: [`results/shap_examples/shap_grid_01.png`](results/shap_examples/shap_grid_01.png)  
- 🏗️ Architecture: [`assets/fig_architecture.png`](assets/fig_architecture.png)
les/shap_grid_01.png" alt="SHAP/IG Saliency Grid" width="420">
</p>

## 🔧 Workflow

<img src="assets/fig_workflow.png" alt="Pipeline: dataset → preprocessing → ParallelCNN training → SHAP/IG explanations" width="720">

*Figure 1 — End-to-end pipeline. Depth frames from LeapGestRecog are preprocessed, fed to the ParallelCNN for training/evaluation, and then explained with SHAP/IG to highlight pixels that drive each prediction.*

---

## 🧱 Architecture

<img src="assets/fig_architecture.png" alt="ParallelCNN with two convolutional branches (3×3 and 5×5) fused before classifier" width="720">

*Figure 2 — ParallelCNN. Two convolutional branches (3×3 and 5×5) extract complementary features. Feature maps are concatenated and passed to a compact classifier head.*

---

## 🖼️ Dataset Samples

<img src="assets/fig_dataset_samples.png" alt="Example depth images across 10 static hand gestures" width="720">

*Figure 3 — LeapGestRecog samples. Example depth frames across the 10 gesture classes used in this project.*

---

## 📊 Results

**Numbers:** see [`results/metrics.json`](results/metrics.json)

### 1) Confusion Matrix

<img src="results/confusion_matrix.png" alt="Confusion matrix heatmap across 10 classes" width="720">

*Figure 4 — Nearly all mass is on the diagonal, indicating very high per-class accuracy with only a few minor off-diagonal mistakes.*

### 2) Training Accuracy

<img src="results/train_accuracy.png" alt="Training accuracy vs epochs" width="720">

*Figure 5 — Training accuracy climbs rapidly and plateaus around ~99–100%.*

### 3) Validation Accuracy

<img src="results/val_accuracy.png" alt="Validation accuracy vs epochs" width="720">

*Figure 6 — Validation accuracy remains ~99% with small fluctuations, suggesting good generalization for this dataset.*

### 4) Saliency / SHAP Grid

<img src="results/shap_examples/shap_grid_01.png" alt="Saliency/SHAP heatmaps highlighting discriminative regions" width="720">

*Figure 7 — Attribution heatmaps consistently highlight finger tips and palm edges that are discriminative for the predicted gesture.*
