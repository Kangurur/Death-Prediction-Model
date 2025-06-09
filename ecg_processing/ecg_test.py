import numpy as np
import pandas as pd
import neurokit2 as nk
import json

# --- 1. Przetwarzanie sygnału EKG ---
with open('data/digitized_json_files/19_2020.json', 'r') as f:
    raw = json.load(f)
lead= raw['leads'][0]['signal']  # Wybierz pierwszy lead

ekg_signal = np.array(lead, dtype=np.float32)  # <-- Twoje dane
sampling_rate = 320  # Hz

signals, info = nk.ecg_process(ekg_signal, sampling_rate=sampling_rate)

# --- 2. HRV: cechy czasowe, widmowe, geometryczne ---
hrv = nk.hrv(info["ECG_R_Peaks"], sampling_rate=sampling_rate, show=False)
hrv_nonlinear = nk.hrv_nonlinear(info["ECG_R_Peaks"], sampling_rate=sampling_rate)
hrv_features = pd.concat([hrv, hrv_nonlinear], axis=1)

# --- 3. Cechy statystyczne surowego sygnału ---
stats = {
    "ekg_mean": np.mean(ekg_signal),
    "ekg_std": np.std(ekg_signal),
    "ekg_min": np.min(ekg_signal),
    "ekg_max": np.max(ekg_signal),
    "ekg_skew": pd.Series(ekg_signal).skew(),
    "ekg_kurtosis": pd.Series(ekg_signal).kurt()
}

# --- 4. Entropia sygnału ---
entropy_features = {
    "entropy_approx": nk.entropy_approximate(ekg_signal),
    "entropy_sample": nk.entropy_sample(ekg_signal)
}

# --- 5. Delineacja: wykrycie P, Q, R, S, T ---
delineate = nk.ecg_delineate(ekg_signal, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate, method="dwt")

# --- 6. Cechy morfologiczne (średnie wartości) ---
qrs_dur = np.nanmean(delineate["ECG_S_Peaks"] - delineate["ECG_Q_Peaks"]) / sampling_rate
qt_dur = np.nanmean(delineate["ECG_T_Offsets"] - delineate["ECG_Q_Peaks"]) / sampling_rate
pr_int = np.nanmean(delineate["ECG_R_Peaks"] - delineate["ECG_P_Peaks"]) / sampling_rate

morpho = {
    "qrs_duration_s": qrs_dur,
    "qt_interval_s": qt_dur,
    "pr_interval_s": pr_int
}

# --- 7. Typ rytmu: HR, tachykardia, bradykardia ---
hr_mean = hrv_features["HRV_MeanNN"].values[0]
rhythm = "normal"
if hr_mean < 60:
    rhythm = "bradycardia"
elif hr_mean > 100:
    rhythm = "tachycardia"

rhythm_features = {
    "mean_rr_ms": hr_mean,
    "rhythm_type": rhythm
}

# --- 8. Wykrycie arytmii (prosto): nieregularność RR ---
rr = np.diff(info["ECG_R_Peaks"]) / sampling_rate
rr_std = np.std(rr)
is_irregular = rr_std > 0.12  # wartość progowa przykładowa (~120 ms)

arrhythmia = {
    "rr_std_sec": rr_std,
    "possible_arrhythmia": is_irregular
}

# --- 9. Scalenie wszystkich cech ---
features = hrv_features.iloc[0].to_dict()
features.update(stats)
features.update(entropy_features)
features.update(morpho)
features.update(rhythm_features)
features.update(arrhythmia)
