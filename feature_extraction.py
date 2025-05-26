
import numpy as np
import pandas as pd
from PyEMD import EMD  # pip install EMD-signal
import antropy as ant  # pip install antropy

def extract_emd_features(channel_data):
    """
    Apply EMD to one EEG channel, extract basic stats and entropy for each IMF.
    """
    emd = EMD()
    IMFs = emd(channel_data)
    features = []
    for imf in IMFs:
        mean_val = np.mean(imf)
        var_val = np.var(imf)
        energy_val = np.sum(imf ** 2)
        apen = ant.app_entropy(imf)
        sampen = ant.sample_entropy(imf)
        features.extend([mean_val, var_val, energy_val, apen, sampen])
    return features

def extract_features_multichannel(eeg_data):
    """
    Extract features from all channels in eeg_data (shape: channels x samples)
    Returns a 1D array of all features concatenated.
    """
    all_features = []
    for channel_idx in range(eeg_data.shape[0]):
        channel_features = extract_emd_features(eeg_data[channel_idx, :])
        all_features.extend(channel_features)
    return np.array(all_features)

def process_raw_to_features(raw):
    """
    Takes an MNE Raw object, returns extracted features for all channels.
    """
    data = raw.get_data()  # shape: (channels, samples)
    features = extract_features_multichannel(data)
    return features

def batch_extract_features(list_of_raws):
    """
    Takes a list of MNE Raw objects, returns a DataFrame of features for all.
    """
    features_list = []
    for raw in list_of_raws:
        features = process_raw_to_features(raw)
        features_list.append(features)
    return pd.DataFrame(features_list)

# ------------- Example usage (for main.py or notebook) ------------- #
if __name__ == "__main__":
    # Example: given an MNE Raw object 'raw'
    # features = process_raw_to_features(raw)
    pass
