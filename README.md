# Survival Prediction Challenge

Machine learning models for Myeloid Leukemia patient survival prediction competition by QRT.

## Project Structure

```
SurvivalPrediction/
├── notebooks/              # Jupyter notebooks (01-07)
│   ├── 01_data_processing_feature_engineering.ipynb
│   ├── 02_cv_strategies.ipynb
│   ├── 03_cox_ph_regression.ipynb
│   ├── 04_xgboost_aft.ipynb
│   ├── 05_deepsurv_cox.ipynb
│   ├── 06_two_model_approach.ipynb
│   ├── 07_ensembling_strategies.ipynb
│   └── outputs/submissions
├── data/                   # Processed feature data
│   ├── X_train_*.csv
│   ├── X_test_*.csv
│   └── target_train_clean_aligned.csv
```

## Evaluation Metric

Perform 5-fold stratified cross validation to compute risk scores for all training samples, then compute global out-of-fold ipcw C-index (C_overall), C_test_like and C_high_risk (to be explained). Evaluation uses a weighted combination of C_overall, C_test_like, and C_high_risk. The weighted C-index is given by:

```python
Weighted C-index = 0.3 × C_overall + 0.4 × C_test_like + 0.3 × C_high_risk
```
We define 5 risk groups based on:
- **Risk groups**: Defined by BM_BLAST > 10, TP53 mutations, HB < 10, PLT < 50, high cytogenetic risk

C_test_like is ipcw C-index computed on training data with risk group at least 1, since the test data consists of sicker patients. C_high_risk is ipcw C-index computed on training data with risk group at least 2.

## Models

### 1. XGBoost AFT (Accelerated Failure Time)
- **Features**: 83 unfixed (no NaN imputation, uses native XGBoost handling)
- **Distribution**: Normal
- **CV Score**: 0.6964 weighted C-index

### 2. Cox Proportional Hazards (Elastic Net)
- **Features**: 128 fixed (NaN imputed, scaled)
- **Package**: `CoxnetSurvivalAnalysis` from scikit-survival
- **Feature selection**: L1 zeroed out 67/128 features, keeping 61 active
- **CV Score**: 0.6907 weighted C-index

### 3. DeepSurv (Neural Network Cox Model)
- **Features**: 83 fixed scaled
- **Architecture**: [64, 64, 64] with SELU activation
- **Dropout**: 0.66
- **CV Score**: 0.6902 weighted C-index

### 4. CatBoost CLF + LightGBM REG (Two-Model Approach)

Simple version of the 4th place solution (which the author generously made public) in Kaggle competition CIBMTR
- **Features**: 128 fixed scaled
- **CatBoost classifier**: Predicts death probability with sample weights
  - Events: weight = 1.0; Censored: weight = F(t) / F_max (cumulative density)
- **LightGBM regressor**: Predicts survival time (trained on events only)
- **Merge**: risk = pred_event × (1 + odds(avg_pred_event) × (1 - pred_time_norm))
- **CV Score**: 0.6912 weighted C-index

## Ensemble Strategy

Grid search over 1,771 weight combinations (step=0.05) found optimal 4-model ensemble:

| Model | Weight | Individual Score |
|-------|--------|------------------|
| **XGBoost AFT 83 unfixed** | **50%** | 0.6964 |
| **CoxPH 128 fixed (elastic net)** | **5%** | 0.6907 |
| **DeepSurv 83 fixed** | **15%** | 0.6902 |
| **CatBoost CLF + LGB REG 128 fixed** | **30%** | 0.6912 |

**Ensemble CV Score: 0.6998** (+0.0034 over best single model)

## Cross-Validation

- **Global OOF CV**: Single C-index computed on all 3,120 out-of-fold predictions
- **Stratification**: By event status and TP53 mutation
- **5-fold**: Stratified K-Fold with shuffle

```python
# Global OOF procedure
oof_preds = np.zeros(n_samples)
for fold in folds:
    model.fit(X_train[fold], y_train[fold])
    oof_preds[val_idx] = model.predict(X_val[fold])

# Single global evaluation
oof_normalized = zscore(oof_preds)
score = weighted_cindex_ipcw(oof_normalized, y_surv, risk_groups)
```

## Key Results

| Model | Overall | Test-like | High-risk | Weighted |
|-------|---------|-----------|-----------|----------|
| XGBoost AFT 83 unfixed | 0.7214 | 0.6967 | 0.6709 | 0.6964 |
| CatBoost CLF + LGB REG 128 fixed | 0.7184 | 0.6937 | 0.6608 | 0.6912 |
| CoxPH 128 fixed (elastic net) | 0.7175 | 0.6931 | 0.6607 | 0.6907 |
| DeepSurv 83 fixed | 0.7169 | 0.6918 | 0.6614 | 0.6902 |
| **Ensemble (50/5/15/30)** | **0.7253** | **0.7008** | **0.6733** | **0.6998** |

## Requirements

```
numpy
pandas
scikit-learn
scikit-survival
xgboost
torch
catboost
lightgbm
lifelines
optuna
```

## Usage

**Run notebooks in order** (01-07) for full pipeline


## License

MIT License
