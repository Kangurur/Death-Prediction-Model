from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import optuna
import pandas as pd

data = pd.read_csv('v2/train.csv')

test = pd.read_csv('v2/test.csv')

data=data.dropna()
test=test.dropna()

X = data.drop(columns=['zgon', 'KG'])
y = data['zgon']
X_test = test.drop(columns=['zgon', 'KG'])
y_test = test['zgon']
idx = test['KG']

def objective(trial):
    params = {
        'C': trial.suggest_float('C', 0.01, 100, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        # 'degree': trial.suggest_int('degree', 2, 5)  # tylko dla 'poly'
    }
    
    model = make_pipeline(StandardScaler(), SVC(**params, probability=True))
    score = cross_val_score(model, X, y, cv=3, scoring='recall', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

print("Najlepsze parametry:", study.best_params)

# Trening końcowego modelu
best_model = make_pipeline(StandardScaler(), SVC(**study.best_params, probability=True))
best_model.fit(X, y)

# Predykcja
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print(confusion_matrix(y_test, y_pred))


# Błędy klasyfikacji
#for i in range(test.shape[0]):
#    if y_pred[i] != y_test.iloc[i]:
#        print(f"Niepoprawna predykcja dla KG {idx[i]}: {y_test.iloc[i]} vs {y_pred[i]}")

print(f"Accuracy train: {accuracy_score(y, best_model.predict(X)):.2f}")

import joblib
if accuracy >= 0.67:
    joblib.dump(best_model, 'v4/svm_model.pkl')
    