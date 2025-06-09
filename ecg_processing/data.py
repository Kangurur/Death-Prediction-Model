import numpy as np
import pandas as pd
import neurokit2 as nk
import json

def process_ecg_signal(lead, sampling_rate=320):
    ecg_signal = np.array(lead, dtype=np.float32)
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    
    hrv = nk.hrv(info["ECG_R_Peaks"], sampling_rate=sampling_rate, show=False)
    hrv_nonlinear = nk.hrv_nonlinear(info["ECG_R_Peaks"], sampling_rate=sampling_rate)
    hrv_features = pd.concat([hrv, hrv_nonlinear], axis=1)
    stats = {
        "ecg_mean": np.mean(cleaned),
        "ecg_std": np.std(cleaned),
        "ecg_min": np.min(cleaned),
        "ecg_max": np.max(cleaned),
        "ecg_skew": pd.Series(cleaned).skew(),
        "ecg_kurtosis": pd.Series(cleaned).kurt()
    }
    
    entropy_features = {
        "entropy_approx": nk.entropy_approximate(ecg_signal)[0],
        "entropy_sample": nk.entropy_sample(ecg_signal)[0]
    }
    
    delineate = nk.ecg_delineate(cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate, method="dwt")[1]
    qrs_dur = np.nanmean(np.array(delineate["ECG_S_Peaks"]) - np.array(delineate["ECG_Q_Peaks"])) / sampling_rate
    qt_dur = np.nanmean(np.array(delineate["ECG_T_Offsets"]) - np.array(delineate["ECG_Q_Peaks"])) / sampling_rate
    
    morpho = {
        "qrs_duration_s": qrs_dur,
        "qt_interval_s": qt_dur,
    }
    
    hr_mean = hrv_features["HRV_MeanNN"].values[0]
    rr = np.diff(info["ECG_R_Peaks"]) / sampling_rate * 1000
    rr_std = np.std(rr)
    is_irregular = rr_std > 120 #ms
    hr = 60000 / hr_mean if hr_mean > 0 else 0
    
    rhythm_features = {
        "hr" : hr,
        "mean_rr_ms": hr_mean,
        "rr_std": rr_std,
        "arrhythmia": is_irregular,
        "hr_sdnn": hrv_features.iloc[0].to_dict()["HRV_SDNN"],
        "hr_rmssd": hrv_features.iloc[0].to_dict()["HRV_RMSSD"],
        "hr_pnn50": hrv_features.iloc[0].to_dict()["HRV_pNN50"],
    }
    #features = hrv_features.iloc[0].to_dict()
    features = stats
    features.update(entropy_features)
    features.update(morpho)
    features.update(rhythm_features)
    
    return features


with open('data/digitized_json_files/19_2020.json', 'r') as f:
    raw = json.load(f)
lead= raw['leads'][0]['signal']+raw['leads'][6]['signal']  
lead[0] = lead[1]

print(process_ecg_signal(lead))



    