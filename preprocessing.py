
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mne  # MNE for EEG signal processing

# --------------- For raw EEG data (e.g., .edf, .set, .bdf files) ---------------- #

def load_eeg_raw(file_path, file_format='edf'):
    """
    Loads raw EEG file using MNE.
    Supported formats: 'edf', 'bdf', 'set', etc.
    """
    if file_format == 'edf':
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_format == 'bdf':
        raw = mne.io.read_raw_bdf(file_path, preload=True)
    elif file_format == 'set':
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    else:
        raise ValueError("Unsupported file format.")
    return raw

def bandpass_filter(raw, l_freq=0.5, h_freq=50.0):
    """
    Band-pass filter the EEG data (default 0.5–50 Hz).
    """
    return raw.filter(l_freq, h_freq, fir_design='firwin')

def run_ica(raw, n_components=15, random_state=42):
    """
    Run ICA for artifact rejection (e.g., eye blinks, ECG).
    Returns ICA-cleaned Raw object.
    """
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter='auto')
    ica.fit(raw)
    # Usually, manual or automated artifact component detection/removal here.
    # For full automation, try e.g. ica.exclude = [0, 1]
    raw_clean = ica.apply(raw)
    return raw_clean

def baseline_correction(raw):
    """
    Perform baseline correction (remove mean per channel).
    """
    data = raw.get_data()
    data -= np.mean(data, axis=1, keepdims=True)
    raw._data = data
    return raw

def normalize_channels(raw):
    """
    Normalize channels (z-score per channel).
    """
    data = raw.get_data()
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data.T).T  # Transpose for sklearn, then back
    raw._data = data_norm
    return raw

def preprocess_raw_eeg(file_path, file_format='edf'):
    """
    Complete pipeline from file to preprocessed EEG data.
    """
    raw = load_eeg_raw(file_path, file_format)
    raw = bandpass_filter(raw)
    raw = run_ica(raw)
    raw = baseline_correction(raw)
    raw = normalize_channels(raw)
    return raw

# ----------- For feature CSVs (as in your dataset: features_raw.csv) ------------ #

def load_feature_csv(feature_path):
    """
    Loads features from CSV and drops empty/NaN columns.
    """
    df = pd.read_csv(feature_path)
    # Drop columns with all NaN or unnecessary columns
    df = df.dropna(axis=1, how='all')
    # Optional: drop label columns here if included
    return df

def normalize_features(features_df):
    """
    Normalize (z-score) all feature columns.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)
    features_df_scaled = pd.DataFrame(features_scaled, columns=features_df.columns)
    return features_df_scaled

def preprocess_features(feature_path):
    """
    Complete preprocessing pipeline for feature CSVs.
    Returns preprocessed DataFrame.
    """
    df = load_feature_csv(feature_path)
    df_norm = normalize_features(df)
    return df_norm

# ----------- Example usage (for main.py or notebook) ------------ #

if __name__ == "__main__":
    # Example for raw EEG file:
    # preprocessed_raw = preprocess_raw_eeg('subject1.edf')

    # Example for feature CSV:
    features_preprocessed = preprocess_features('features_raw.csv')
    print(features_preprocessed.head())
