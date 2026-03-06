# Interpretable Gradient Boosting Under Distribution Shift

This repository contains the implementation and experimental framework for evaluating interpretable gradient boosting models (XGBoost, LightGBM, Gradient Boosting)
in the presence of data distribution shifts across multiple demographic sub-categories represented in Telco Customer Churn dataset. 

## 🚀 Overview
The project focuses on:
- **Quantifying the impact** of Distribution and Susbpopulation shift on model performance and interpretability.
- **Benchmarking** black-box vs. interpretable boosting methods vs. other classification benchmarks (i.e., Logistic Regression, Random Forest).
- **Analyzing** SHAP-stability and Fairness-Bias across presented models when tested under the distribution shifted data.

## Requirements 

To ensure reproducibility and avoid dependency conflicts (especially with the specific **XGBoost 1.6.2** requirement), it is recommended to use a dedicated Conda environment.

### setup instructions:

1. **Create the environment**:

   ```bash
   conda create -n interpret_gb python=3.10.5
    ```
2. **Activate the environment:**

   ```bash
   conda activate interpret_gb
   ```

Reproducing the experiments requires ```bash pip-```installing the following library packages:

```
```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
scipy>=1.10.0
shap>=0.42.0
interpret>=0.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost==1.6.2
lightgbm>=3.3.5
```

## Main Experiment

All core experiments, including the simulation of distribution shifts and the subsequent interpretability analysis, are contained within the primary notebook. 
After installing all required packages and 
