import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

test = pd.read_csv('v4/test.csv')
test_ecg = pd.read_csv('v4/test_ecg.csv')
test = test.dropna()
test_ecg = test_ecg.dropna()
common_kg = set(test_ecg['KG'])
test = test[test['KG'].isin(common_kg)].reset_index(drop=True)
X_test = test.drop(columns=['zgon', 'KG'])
y_test = test['zgon']
idx = test['KG']
X_test_ecg = test_ecg.drop(columns=['KG', 'zgon'])

xgb= XGBClassifier()
xgb.load_model('v4/xgb_model.json')
xgb_ecg = XGBClassifier()
xgb_ecg.load_model('v4/xgb_model2_ecg.json')

svm=joblib.load('v4/svm_model.pkl')
svm_ecg=joblib.load('v4/svm_model_ecg.pkl')

rf = joblib.load('v4/rf_model.joblib')
rf_ecg = joblib.load('v4/rf_model_ecg.joblib')


#y_pred = svm.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(accuracy)

y_svm = svm.predict(X_test)
y_svm_ecg = svm_ecg.predict(X_test_ecg)
y_xgb = xgb.predict(X_test)
y_xgb_ecg = xgb_ecg.predict(X_test_ecg)
y_rf = rf.predict(X_test)
y_rf_ecg = rf_ecg.predict(X_test_ecg)

pred=[]
#print(len(y_svm), len(y_svm_ecg), )

for i in range(len(y_svm)):
    res = y_svm[i] + y_xgb[i] + y_rf[i]
    ans = 0
    if res >= 2:
        ans = 1
        
    res_ecg = y_svm_ecg[i] + y_xgb_ecg[i] + y_rf_ecg[i]
    ans_ecg = 0
    if res_ecg >= 2:
        ans_ecg = 1
        
    #pred.append([int(y_svm[i]), int(y_xgb[i]), int(y_rf[i]), int(y_svm_ecg[i]), int(y_xgb_ecg[i]), int(y_rf_ecg[i]), ans, ans_ecg])
    pred.append(int(ans or ans_ecg))
    
    
print(f"Accuracy: {accuracy_score(y_test, pred):.2f}")
#print("Recall:", recall_score(y_test, pred))
#for i in range(len(y_test)):
#    print(pred[i],y_test[i])