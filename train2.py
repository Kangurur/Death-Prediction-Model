import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Wczytanie danych
data = pd.read_csv('data/train.csv')
X = data.drop(columns=['zgon', 'KG'])  # Załóżmy, że 'KG' to cecha, którą pomijasz
y = data['zgon']

# Testowy zbiór
test = pd.read_csv('data/test.csv')
X_test = test.drop(columns=['zgon', 'KG'])
y_test = test['zgon']

# Model XGBoost
model = XGBClassifier(eval_metric='logloss', random_state=2137)

# Parametry do strojenia
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1, 0.5],
    'colsample_bytree': [0.8, 1, 0.5],
    #'use_label_encoder': [False],
}

# Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=4, scoring='accuracy', n_jobs=-1, verbose=1)

grid_search.fit(X, y)

# Predykcja na danych testowych
y_pred = grid_search.best_estimator_.predict(X_test)

# Ewaluacja
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(grid_search.best_estimator_, X_test, y_test)
print("Best parameters found: ", grid_search.best_params_)



# --- Ważność cech ---
best_model = grid_search.best_estimator_
importances = best_model.feature_importances_
feature_names = X.columns

# Stworzenie DataFrame
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feat_imp_df)