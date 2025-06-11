import pandas as pd
import numpy as np

data=pd.read_csv('v4/data.csv')
ecg=pd.read_csv('v4/data_ecg.csv')

for i in data['KG']:
    if i not in ecg['id'].values:
        print(i)

merged = pd.merge(data, ecg, left_on='KG', right_on='id', how='inner')
merged = merged.drop(columns=['id'])
merged = merged.drop_duplicates(subset=['KG'], keep='first')

#merged['lead1_arrhythmia']= merged['lead1_arrhythmia'].astype(bool)
#merged['lead2_arrhythmia']= merged['lead2_arrhythmia'].astype(bool)
#merged['lead3_arrhythmia']= merged['lead3_arrhythmia'].astype(bool)
#merged['arrhythmia'] = merged['arrhythmia'].astype(bool)

print(merged.shape)
print(merged.columns)


merged=merged.sample(frac=1, random_state=111)
train=merged[50:]
test=merged[:50]
print(test['zgon'].sum()) #24

ecg_col = [
    'KG', 'ecg_mean', 'ecg_std', 'ecg_min', 'ecg_max', 
    'ecg_skew', 'ecg_kurtosis', 'entropy_approx', 
    'entropy_sample', 'qrs_duration_s', 'qt_interval_s', 
    'hr', 'mean_rr_ms', 'rr_std', 'hr_sdnn', 
    'hr_rmssd', 'hr_pnn50',
    'WIEK', 'zgon', 'akcja serca przyjęcie (TISS nr 1)',
    'Lac (1. gaz. 1sza doba)',
    'BE (1. gaz. 1sza doba)',
    "sodium chloride difference tiss 3",
]

col=['KG',
     'WIEK',
     #'follow up 30 dni',
     'zgon',
     #'Operowany przed przyjęciem (0/1)',
     'Interleukina 6',
     'Prokalcytonina',
     'MAP 1sza doba',
     'pao2/fio2 1sza doba',
     #'BMI',
     'Glukoza (1. gaz. 1sza doba)',
     'Lac (1. gaz. 1sza doba)',
     'BE (1. gaz. 1sza doba)',
     'LA (1. gaz. 1 sza doba)',
     'SOFA - punktacja',
     'Sepsa (0/1)',
     "temperatura ciała przy przyjęciu (TISS nr 1)",
     "sodium chloride difference tiss 1",
    "sodium chloride difference tiss 2",
    "sodium chloride difference tiss 3",
    "akcja serca przyjęcie (TISS nr 1)",
    "Wentylacja mechaniczna TAK=1, NIE =0 (TISS nr 1)",
    'hr'
]

train1=train[col]
test1=test[col]
train_ecg=train[ecg_col]
test_ecg=test[ecg_col]

train1.to_csv('v4/train.csv', index=False)
test1.to_csv('v4/test.csv', index=False)
train_ecg.to_csv('v4/train_ecg.csv', index=False)
test_ecg.to_csv('v4/test_ecg.csv', index=False)