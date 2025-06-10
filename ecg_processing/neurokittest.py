import neurokit2 as nk
import json
import numpy as np
import pandas as pd

with open('data/digitized_json_files/19_2020.json', 'r') as f:
    raw = json.load(f)
lead= raw['leads'][0]['signal']  # Wybierz pierwszy lead

#print(len(lead),type(lead))
import numpy as np
lead = np.array(lead, dtype=np.float32)  
lead[0] = lead[1]  

#ecg = nk.ecg_simulate(duration=3, sampling_rate=320, noise=0.2)
ecg = lead

#print(len(ecg),type(ecg))

ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=320)

signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=320)
print(info)
print(signals.columns)

hrv = nk.hrv(info["ECG_R_Peaks"], sampling_rate=320, show=False)
hrv_nonlinear = nk.hrv_nonlinear(info["ECG_R_Peaks"], sampling_rate=320)

print(hrv.columns)

print("delineate")

delineate = nk.ecg_delineate(ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=320, method="dwt")[1]
print(delineate)


qrs_dur = np.nanmean(np.array(delineate["ECG_S_Peaks"]) - np.array(delineate["ECG_Q_Peaks"])) / 320
qt_dur = np.nanmean(np.array(delineate["ECG_T_Offsets"]) - np.array(delineate["ECG_Q_Peaks"])) / 320
#pr_int = np.nanmean(delineate["ECG_R_Peaks"] - delineate["ECG_P_Peaks"]) / 320

print("morpho")
print(qrs_dur, qt_dur)

qrs2 = np.nanmean(signals["ECG_S_Peaks"] - signals["ECG_Q_Peaks"]) / 320
qt2 = np.nanmean(signals["ECG_T_Offsets"] - signals["ECG_Q_Peaks"]) / 320
print("morpho2")
print(qrs2, qt2)


print(np.where(signals["ECG_R_Peaks"]==1))

print(hrv.columns)
print(signals.columns)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(ecg, label='Cleaned ECG Signal')
plt.show()