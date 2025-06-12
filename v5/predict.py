import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

test = pd.read_csv('v3/test.csv')
test = test.dropna()
X = test.drop(columns=['zgon', 'KG'])
y = test['zgon']
idx = test['KG']

svm= joblib.load('v5/svm_model2.pkl')
xgb = XGBClassifier()
xgb.load_model('v5/xgb_model.json')
rf = joblib.load('v5/rf_model.joblib')

y_svm = svm.predict(X)
y_xgb = xgb.predict(X)
y_rf = rf.predict(X)
pred = []

for i in range(len(y_svm)):
    #pred.append([int(y_svm[i]), int(y_xgb[i]), int(y_rf[i])])
    #continue
    ans = int(y_svm[i]) + int(y_xgb[i]) + int(y_rf[i])
    if ans >= 2:
        pred.append(1)
    else:
        pred.append(0)
    

print("accuracy:", accuracy_score(y, pred))
#print(y)
#for i in range(len(pred)):
#    print(pred[i], y.iloc[i])