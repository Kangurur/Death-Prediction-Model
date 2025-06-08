import pandas as pd

data=pd.read_excel('data/annotations_ecg.xlsx')
col=['KG',
     'WIEK',
     #'follow up 30 dni',
     'zgon w ciągu pierwszych 30 dni od przyjęcia do OIT',
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
    
]
data=data[col]
data=data.drop_duplicates(subset=['KG'], keep='first')


data['KG'] = data['KG'].astype(str)
data=data[data["Interleukina 6"]!="Nie znaleziono"]
data["Interleukina 6"] = data["Interleukina 6"].astype(float)
data["Prokalcytonina"] = data["Prokalcytonina"].astype(float)
data["Sepsa (0/1)"] = data["Sepsa (0/1)"].astype(bool)
#data["Operowany przed przyjęciem (0/1)"] = data["Operowany przed przyjęciem (0/1)"].astype(bool)
data.rename(columns={'zgon w ciągu pierwszych 30 dni od przyjęcia do OIT': 'zgon'}, inplace=True)
#data['zgon'] = data['zgon'].astype(bool)
#print(data.count()) #454

#data['WIEK'] = data['WIEK'].apply(
#    lambda x: 0 if x<=14 else (1 if 15 <= x <= 30 else (2 if 31 <= x <= 60 else 3)))

data['WIEK2'] = data['WIEK'].apply(lambda x: x//10)
data["infection"] = data['Interleukina 6']* data['Prokalcytonina']
data["organs"] = data["Lac (1. gaz. 1sza doba)"] * data["pao2/fio2 1sza doba"] 

data['LA+'] = data['LA (1. gaz. 1 sza doba)'].apply(
    lambda x: 0 if 8 <= x <= 14 else (x - 14 if x > 14 else 8 - x)
)



#data=data.sample(frac=1, random_state=1)
data=data.sample(frac=1, random_state=2137)
data.set_index('KG', inplace=True, drop=False)
#data=data.reset_index(drop=True)

train=data[:400]
test=data[400:]
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)