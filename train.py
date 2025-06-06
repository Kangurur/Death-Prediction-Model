import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data=pd.read_csv('mpsi/0Projekt/data/train.csv')
X=data.drop(columns=['zgon','KG'])
y=data['zgon']


rf = RandomForestClassifier(n_estimators=20, random_state=1)
rf.fit(X, y)

model=RandomForestClassifier(random_state=42)
grid_params = {
    'n_estimators': [100, 50, 10, 20],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}


#grid_search = GridSearchCV(estimator=model, param_grid=grid_params, cv=10, scoring='accuracy')
#grid_search.fit(X, y)
#print("Best parameters found: ", grid_search.best_params_)

test=pd.read_csv('mpsi/0Projekt/data/test.csv')
X_test=test.drop(columns=['zgon','KG'])
y_test=test['zgon']


#y_pred = grid_search.best_estimator_.predict(X_test)
y_pred = rf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
print(confusion_matrix(y_test, y_pred))