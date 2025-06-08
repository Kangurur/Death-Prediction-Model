import pandas as pd

data=pd.read_excel('data/annotations_ecg.xlsx')
xd = data.columns.tolist()
with open('data/labels.txt', 'w') as f:
    for label in xd:
        f.write(f"{label}\n")