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
    #pq
    #p len
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


folder = 'data/digitized_json_files'

#xd=["19_2020.json","44_2020.json","51_2020.json"]
import warnings
warnings.filterwarnings("ignore")

import os
results = []

for filename in os.listdir(folder):
    if filename.endswith('.json'):
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as f:
            raw = json.load(f)
        leads = raw['leads']
        sampling_rate = leads[0]['sampling_frequency']
        lead1=None
        lead2=None
        lead3=None
        if(len(leads[0]['signal']) < 1000):
            lead1 = leads[0]['signal']+leads[3]['signal']+leads[6]['signal']+leads[9]['signal']
            lead2= leads[1]['signal']+leads[4]['signal']+leads[7]['signal']+leads[10]['signal']
            lead3= leads[2]['signal']+leads[5]['signal']+leads[8]['signal']+leads[11]['signal']
        elif(len(leads[0]['signal']) < 2000):
            lead1 = leads[0]['signal']+leads[6]['signal']
            lead2= leads[1]['signal']+leads[7]['signal']
            lead3= leads[2]['signal']+leads[8]['signal']
        else:
            lead1 = leads[0]['signal']
            lead2 = leads[1]['signal']
            lead3 = leads[2]['signal']
            
        lead1 = [x if x is not None else 0 for x in lead1]
        lead2 = [x if x is not None else 0 for x in lead2]
        lead3 = [x if x is not None else 0 for x in lead3]
        lead1 = np.array(lead1, dtype=np.float32)
        lead2 = np.array(lead2, dtype=np.float32)
        lead3 = np.array(lead3, dtype=np.float32)
        lead1 = np.nan_to_num(lead1, nan=0.0)
        lead2 = np.nan_to_num(lead2, nan=0.0)
        lead3 = np.nan_to_num(lead3, nan=0.0)

        #row = {"id": filename.split('_')[0]}
        try:
            features1 = process_ecg_signal(lead1, sampling_rate=sampling_rate)
            #row.update({f"lead1_{k}": v for k, v in features1.items()})
        except Exception as e:
            print("Error processing lead 1 in file", filename, ":", e)
        try:
            features2 = process_ecg_signal(lead2, sampling_rate=sampling_rate)
            #row.update({f"lead2_{k}": v for k, v in features2.items()})
        except Exception as e:
            #row.update({f"lead2_{k}": v for k, v in features1.items()})
            features2 = features1.copy()
            print("Error processing lead 2 in file", filename, ":", e)
        try:
            features3 = process_ecg_signal(lead3, sampling_rate=sampling_rate)
            #row.update({f"lead3_{k}": v for k, v in features3.items()})
        except Exception as e:
            #row.update({f"lead3_{k}": v for k, v in features1.items()})
            features3 = features1.copy()
            print("Error processing lead 3 in file", filename, ":", e)

        features = {
            "id": filename.split('_')[0],
            'ecg_mean': (features1['ecg_mean'] + features2['ecg_mean'] + features3['ecg_mean']) / 3,
            'ecg_std': (features1['ecg_std'] + features2['ecg_std'] + features3['ecg_std']) / 3,
            'ecg_min': min(features1['ecg_min'], features2['ecg_min'], features3['ecg_min']),
            'ecg_max': max(features1['ecg_max'], features2['ecg_max'], features3['ecg_max']),
            'ecg_skew': (features1['ecg_skew'] + features2['ecg_skew'] + features3['ecg_skew']) / 3,
            'ecg_kurtosis': (features1['ecg_kurtosis'] + features2['ecg_kurtosis'] + features3['ecg_kurtosis']) / 3,
            'entropy_approx': (features1['entropy_approx'] + features2['entropy_approx'] + features3['entropy_approx']) / 3,
            'entropy_sample': (features1['entropy_sample'] + features2['entropy_sample'] + features3['entropy_sample']) / 3,
            'qrs_duration_s': (features1['qrs_duration_s'] + features2['qrs_duration_s'] + features3['qrs_duration_s']) / 3,
            'qt_interval_s': (features1['qt_interval_s'] + features2['qt_interval_s'] + features3['qt_interval_s']) / 3,
            'hr': (features1['hr'] + features2['hr'] + features3['hr']) / 3,
            'mean_rr_ms': (features1['mean_rr_ms'] + features2['mean_rr_ms'] + features3['mean_rr_ms']) / 3,
            'rr_std': (features1['rr_std'] + features2['rr_std'] + features3['rr_std']) / 3,
            #'arrhythmia': features1['arrhythmia'] or features2['arrhythmia'] or features3['arrhythmia'],
            'hr_sdnn': (features1['hr_sdnn'] + features2['hr_sdnn'] + features3['hr_sdnn']) / 3,
            'hr_rmssd': (features1['hr_rmssd'] + features2['hr_rmssd'] + features3['hr_rmssd']) / 3,
            'hr_pnn50': (features1['hr_pnn50'] + features2['hr_pnn50'] + features3['hr_pnn50']) / 3,
        }
        
        results.append(features)

df = pd.DataFrame(results)
df.to_csv("v3/ecg_features.csv", index=False)





