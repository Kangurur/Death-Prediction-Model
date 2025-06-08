import numpy as np
import pandas as pd
import neurokit2 as nk
import json

# --- 1. Przetwarzanie sygnału EKG ---
with open('data/digitized_json_files/19_2020.json', 'r') as f:
    raw = json.load(f)
lead= raw['leads'][0]['signal'][1:]  # Wybierz pierwszy lead

ekg_signal = np.array(lead, dtype=np.float32)  # <-- Twoje dane
sampling_rate = 320  # Hz

#signals, info = nk.ecg_process(ekg_signal, sampling_rate=sampling_rate)

cleaned_signal = nk.ecg_clean(ekg_signal, sampling_rate=sampling_rate, method="neurokit")
rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=sampling_rate)

quality = nk.ecg_quality(cleaned_signal, rpeaks=rpeaks, sampling_rate=sampling_rate)