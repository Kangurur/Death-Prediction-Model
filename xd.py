import pandas as pd

data=pd.read_csv('data/test.csv')

train=pd.read_csv('data/train.csv')
print(train[train['zgon'] == 1].count())

print(data[data['zgon'] == 1].count())
print(data[data['zgon'] == 0].count())