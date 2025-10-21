# Death Prediction Model

A machine learning project for predicting in-hospital mortality in ICU patients using clinical data and ECG features.

## 📋 Project Overview

This project develops predictive models to assess mortality risk in Intensive Care Unit (ICU) patients. The models utilize a combination of:
- Clinical parameters (vital signs, laboratory results, SOFA scores)
- ECG-derived features (heart rate variability, rhythm characteristics)
- Patient demographics and medical history

## 🎯 Model Performance

### Latest Results (v5)

| Model | Accuracy | Best Parameters |
|-------|----------|----------------|
| **Voting Ensemble** | **0.76** | Hard/soft voting across all three models |
| **Random Forest** | 0.74 | n_estimators: 218, max_depth: 3, min_samples_split: 6 |
| **XGBoost** | 0.73 | n_estimators: 332, max_depth: 5, learning_rate: 0.015 |
| **SVM** | 0.71 | C: 38.57, gamma: 0.0012, kernel: sigmoid |

### Version History

#### v5 (Current) - Ensemble Voting System
- **Focus**: Voting ensemble combining multiple optimized models
- **Models**: Random Forest, XGBoost, SVM (voting classifier)
- **Best Result**: **0.76 accuracy** with voting ensemble (0.74 RF, 0.73 XGBoost, 0.71 SVM individually)
- **Key Innovation**: Ensemble voting achieves +2% improvement over best individual model
- **Features**: 31 features including age squared (WIEK2) and full HRV metrics
- **Architecture**: Hard/soft voting across all three model types for robust predictions

#### v4 - Multi-Model Voting Ensemble
- **Focus**: Voting classifier combining RF, XGBoost, and SVM predictions
- **Accuracy**: 0.68-0.70 (ensemble), individual models tested separately
- **Key Innovation**: First implementation of ensemble voting; separate models for clinical-only vs. clinical+ECG data
- **Models Saved**: `rf_model.joblib`, `xgb_model.json`, individual and ensemble versions
- **Finding**: ECG features provided marginal improvement (~2%), ensemble reduced variance

#### v3 - Optuna Hyperparameter Tuning
- **Focus**: XGBoost optimization using Optuna framework
- **Accuracy**: 0.72-0.74
- **Key Innovation**: Automated hyperparameter search (50 trials)
- **Features**: 21 clinical + ECG features
- **Notable**: First version to consistently break 0.72 accuracy barrier
- **Optimization**: ROC-AUC scoring with 3-fold CV

#### v2 - ECG Feature Integration
- **Focus**: Adding ECG-derived features to clinical data
- **Accuracy**: 0.70-0.74
- **Key Innovation**: Heart rate variability (HRV) metrics and ECG signal features
- **Features**: 35 features (clinical + 15 ECG metrics)
- **ECG Processing**: NeuroKit2 library for signal analysis
- **Finding**: RR interval std and QT interval emerged as important predictors

#### v1 - Baseline Clinical Model
- **Focus**: Pure clinical data with XGBoost
- **Accuracy**: 0.65-0.72
- **Features**: 21 clinical features (no ECG)
- **Key Predictors**: Sepsis, Age, Lactate, Sodium-chloride difference
- **Models**: Multiple XGBoost configurations saved
- **Finding**: Established baseline performance and identified core clinical predictors

## 📊 Dataset

**Total Patients**: ~200+ cases  
**Test Set**: 50 patients (24% mortality rate)  
**Training Set**: ~150 patients

### Key Features

#### Clinical Parameters (Top Importance)
1. **Lactate (Lac)** - First 24h measurement
2. **Sodium-Chloride Difference** - TISS 2 & 3
3. **Base Excess (BE)** - First 24h
4. **Age (WIEK)**
5. **SOFA Score** - Sequential Organ Failure Assessment
6. **PaO2/FiO2 Ratio** - Oxygenation index

#### ECG-Derived Features
- Heart rate (HR) and variability (SDNN, RMSSD, pNN50)
- RR intervals (mean, std)
- QRS duration and QT interval
- ECG signal statistics (mean, std, skewness, kurtosis)
- Entropy measures (approximate, sample)

#### Laboratory Values
- Interleukin-6, Procalcitonin
- Glucose, Creatinine
- Temperature, MAP (Mean Arterial Pressure)

## 🗂️ Project Structure

```
Death Prediction Model/
├── data/
│   ├── labels.txt                    # Feature descriptions
│   └── digitized_json_files/         # Raw ECG data (200+ files)
├── ecg_processing/
│   ├── data.py                       # ECG data extraction
│   ├── ecg_info.py                   # ECG feature engineering
│   └── ecg_features.csv              # Processed ECG features
├── v1/ - v5/                         # Model iterations
│   ├── data.py                       # Data preprocessing
│   ├── data_ecg.py                   # ECG feature extraction
│   ├── data_join.py                  # Merge clinical + ECG data
│   ├── train.py / train_*.py         # Model training scripts
│   ├── predict.py                    # Inference script
│   ├── train.csv / test.csv          # Split datasets
│   └── *.joblib / *.json             # Saved models
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost optuna joblib matplotlib seaborn
```

### Training a Model

```bash
# Navigate to the latest version
cd v5

# Train Random Forest (best performer)
python train_rf.py

# Train XGBoost
python train_xgb.py

# Train SVM
python train_svm.py
```

### Making Predictions

```bash
python predict.py
```

## 🔬 Methodology

### Data Processing Pipeline

1. **Data Collection**
   - Clinical data from ICU admissions
   - ECG recordings digitized to JSON format
   - Laboratory results from first 72 hours

2. **Feature Engineering**
   - ECG signal processing (NeuroKit2)
   - Sodium-chloride difference calculation
   - Heart rate variability metrics
   - Temporal aggregation (TISS 1-3 measurements)

3. **Data Splitting**
   - Random shuffle with fixed seed (random_state=111)
   - 50 test samples, ~150 training samples
   - Stratification considered for class balance

4. **Model Training**
   - Hyperparameter optimization using Optuna
   - Cross-validation (3-5 folds)
   - ROC-AUC as primary metric
   - Regularization to prevent overfitting

5. **Evaluation**
   - Accuracy, Confusion Matrix
   - Feature importance analysis
   - Per-patient error analysis

### Handling Overfitting

The models implement several strategies to prevent overfitting:
- L1/L2 regularization (`reg_alpha`, `reg_lambda`)
- Tree depth constraints (`max_depth: 3-5`)
- Minimum samples per leaf (`min_samples_leaf`)
- Cross-validation during hyperparameter tuning
- Removal of zero-importance features

### Missing Data

Missing values are handled by:
- Dropping rows with NaN values (`dropna()`)
- Feature selection based on availability

## 📈 Feature Importance

### Top 10 Features (v5 Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Lac (1st 24h) | 0.127 |
| 2 | Na-Cl difference (TISS 3) | 0.113 |
| 3 | BE (1st 24h) | 0.080 |
| 4 | Na-Cl difference (TISS 2) | 0.067 |
| 5 | Age | 0.065 |
| 6 | SOFA Score | 0.063 |
| 7 | QT Interval | 0.059 |
| 8 | Lactic Acid | 0.045 |
| 9 | ECG Kurtosis | 0.043 |
| 10 | QRS Duration | 0.039 |

## 🔍 Model Details

### Random Forest (v5 - Best)
- **Accuracy**: 0.74 (test), 0.80 (train)
- **Configuration**: 218 estimators, max_depth=3, bootstrap=True
- **Strengths**: Robust, interpretable, handles non-linear relationships

### XGBoost
- **Accuracy**: 0.71-0.73
- **Configuration**: Gradient boosting with L1/L2 regularization
- **Strengths**: Fast training, good performance on structured data

### SVM
- **Accuracy**: 0.68-0.71
- **Configuration**: RBF/Sigmoid kernel with scaling
- **Strengths**: Works well with high-dimensional data

## 📝 Notes

- **Class Imbalance**: Approximately 48% mortality rate in test set (24/50)
- **Data Size**: Limited dataset (~200 patients) - results should be validated on larger cohorts
- **Overfitting Risk**: Training accuracy often exceeds test accuracy by 10-15%
- **Feature Redundancy**: Some features show zero importance and can be removed

## 🎓 Clinical Significance

The models identify key predictors of ICU mortality:
- **Metabolic dysfunction**: Elevated lactate and base deficit
- **Electrolyte imbalance**: Sodium-chloride difference
- **Organ failure**: High SOFA scores
- **Cardiac dysfunction**: ECG abnormalities and heart rate variability

## 🔮 Future Improvements

- [ ] Increase dataset size for better generalization
- [ ] Implement SMOTE for handling class imbalance
- [ ] Add temporal features (trend analysis over 72h)
- [ ] Ensemble methods combining multiple models
- [ ] External validation on independent cohort
- [ ] Feature selection using recursive elimination
- [ ] Deep learning approaches (LSTM for time-series ECG)

## 📄 License

This is a research project for educational purposes.

## 🤝 Contributors

Medical University Project - ICU Mortality Prediction

---

**Last Updated**: October 2025  
**Current Version**: v5  
**Best Model**: Voting Ensemble (0.76 accuracy)
