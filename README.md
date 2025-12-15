# Bayesian-Optimized Hybrid Ensemble for Cybersecurity Anomaly Detection

This repository contains the complete implementation and experimental pipeline for a **hybrid machine learning framework** designed for **cybersecurity anomaly detection**. The work focuses on evaluating classical machine learning models, deep learning architectures, and a proposed **hybrid ensemble** to detect network intrusions in a multi-class setting.

The project accompanies an IEEE-style research paper and emphasizes **feature engineering, robustness, interpretability, and performance under class imbalance**.

---

## 📌 Key Contributions

- Comprehensive evaluation of **classical ML, deep learning, and hybrid models**
- Multi-stage **feature engineering pipeline** for high-dimensional network data
- **Bayesian Optimization** for systematic hyperparameter tuning
- **Hybrid ensemble model** combining the strengths of multiple learners
- Use of **SHAP** for model interpretability
- Robustness analysis to assess model stability and generalization

---

## 🧠 Models Implemented

### Classical Machine Learning
- Logistic Regression  
- Decision Tree  
- Random Forest  
- HistGradient Boosting  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  

### Deep Learning
- Artificial Neural Network (ANN)  
- Long Short-Term Memory (LSTM)  

### Proposed Model
- **Bayesian-Optimized Hybrid Ensemble**

---

## 🧪 Dataset

- **Network Intrusion Dataset** (Kaggle)  
  https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

**Characteristics:**
- Multi-class intrusion detection (12 attack categories)
- High-dimensional feature space
- Strong class imbalance (realistic cybersecurity setting)

---

## ⚙️ Feature Engineering Pipeline

The raw dataset is transformed using a multi-stage process:

1. **Statistical Feature Extraction**
   - Entropy-based features to capture traffic randomness

2. **Dimensionality Reduction**
   - Principal Component Analysis (PCA) to retain global structure

3. **Feature Selection**
   - Minimum Redundancy Maximum Relevance (mRMR)

This pipeline significantly improves classical model performance and stability.

---

## 🔍 Hyperparameter Optimization

- **Bayesian Optimization** is used instead of grid/random search
- Efficient exploration of hyperparameter space
- Improves convergence speed and final model performance
- Applied to tree-based models and ensemble components

---

## 📊 Evaluation Metrics

Given severe class imbalance, performance is evaluated using:

- Accuracy  
- **Macro F1-score** (primary metric)  
- ROC-AUC  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- Training Time  

> Note: Accuracy alone is misleading due to dominant majority classes.

---

## 🧩 Interpretability & Robustness

### SHAP (SHapley Additive exPlanations)
- Identifies feature contributions to predictions
- Improves transparency of the hybrid ensemble
- Helps validate learned patterns against domain knowledge

### Robustness Analysis
- Tests model stability across dataset splits
- Evaluates sensitivity to feature perturbations
- Ensures reliable deployment in real-world environments

---

## 📈 Results Summary

- Tree-based models outperform deep learning models on static intrusion data
- Feature engineering yields **significant gains** across all models
- The **hybrid ensemble** achieves the best balance of:
  - Accuracy
  - Macro F1-score
  - Robustness
  - Interpretability
