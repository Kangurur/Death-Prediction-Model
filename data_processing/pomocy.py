import numpy as np
import pandas as pd
import neurokit2 as nk
import json

# --- 1. Przetwarzanie sygnału EKG ---
with open('data/digitized_json_files/19_2020.json', 'r') as f:
    raw = json.load(f)
lead= raw['leads'][0]['signal']  # Wybierz pierwszy lead
lead[0]=lead[1]

ekg_signal = np.array(lead, dtype=np.float32)  # <-- Twoje dane
sampling_rate = 320  # Hz

#signals, info = nk.ecg_process(ekg_signal, sampling_rate=sampling_rate)

#cleaned_signal = nk.ecg_clean(ekg_signal, sampling_rate=sampling_rate, method="neurokit")
#rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=sampling_rate)

#quality = nk.ecg_quality(cleaned_signal, sampling_rate=sampling_rate)

signals, info = nk.ecg_process(ekg_signal, sampling_rate=sampling_rate)
#analysis = nk.ecg_analyze(signals, sampling_rate=sampling_rate, method="interval-related")

#analysis.to_csv('data_processing/analysis_test.csv')