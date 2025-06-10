import pandas as pd

data=pd.read_csv('v2/data.csv')
ecg=pd.read_csv('v2/ecg_features.csv')

for i in data['KG']:
    if i not in ecg['id'].values:
        print(i)

merged = pd.merge(data, ecg, left_on='KG', right_on='id', how='left')
merged = merged.drop(columns=['id'])
merged = merged.drop_duplicates(subset=['KG'], keep='first')

#merged['lead1_arrhythmia']= merged['lead1_arrhythmia'].astype(bool)
#merged['lead2_arrhythmia']= merged['lead2_arrhythmia'].astype(bool)
#merged['lead3_arrhythmia']= merged['lead3_arrhythmia'].astype(bool)
merged['arrhythmia'] = merged['arrhythmia'].astype(bool)

print(merged.shape)
print(merged.columns)

train=merged[:400]
test=merged[400:]

train.to_csv('v2/train.csv', index=False)
test.to_csv('v2/test.csv', index=False)