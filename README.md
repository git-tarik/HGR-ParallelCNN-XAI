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
