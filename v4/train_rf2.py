import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('v4/train_ecg.csv')
X = data.drop(columns=['zgon', 'KG'])
y = data['zgon']

test = pd.read_csv('v4/test_ecg.csv')
X_test = test.drop(columns=['zgon', 'KG'])
y_test = test['zgon']
idx = test['KG']

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }

    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
    return score

# Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Najlepsze parametry:", study.best_params)

best_model = RandomForestClassifier(**study.best_params)
best_model.fit(X, y)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print(confusion_matrix(y_test, y_pred))

importances = best_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

print("\nTop cechy wg ważności:")
print(feat_imp_df)

# Błędy klasyfikacji
#for i in range(test.shape[0]):
#    if y_pred[i] != y_test.iloc[i]:
#        print(f"Niepoprawna predykcja dla KG {idx[i]}: {y_test.iloc[i]} vs {y_pred[i]}")

print(f"Accuracy train: {accuracy_score(y, best_model.predict(X)):.2f}")

import joblib
if accuracy >= 0.65:
    joblib.dump(best_model, f'v4/rf_model2_ecg.joblib')
