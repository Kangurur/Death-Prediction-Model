import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- Wczytanie danych ---
data = pd.read_csv('data/train.csv')
X = data.drop(columns=['zgon', 'KG'])
y = data['zgon']

test = pd.read_csv('data/test.csv')
X_test = test.drop(columns=['zgon', 'KG'])
y_test = test['zgon']
idx = test['KG']

# --- Funkcja celu dla Optuna ---
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': 'logloss',
        #'gamma': trial.suggest_float('gamma', 0, 5),
        #'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        #'use_label_encoder': False,
        'random_state': 2137
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='recall', n_jobs=-1).mean()
    return score

# --- Optymalizacja ---
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Najlepsze parametry:", study.best_params)

# --- Trenowanie najlepszego modelu ---
best_model = XGBClassifier(**study.best_params)
best_model.fit(X, y)

# --- Predykcja i ewaluacja ---
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print(confusion_matrix(y_test, y_pred))

# --- Ważność cech ---
importances = best_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

print("\nTop cechy wg ważności:")
print(feat_imp_df)

for i in range(test.shape[0]):
    if y_pred[i] != y_test.iloc[i]:
        print(f"Niepoprawna predykcja dla KG {idx[i]}: {y_test.iloc[i]} vs {y_pred[i]}")

if accuracy >= 0.65:
    best_model.save_model(f'models/v1({accuracy:.2f}).json')
