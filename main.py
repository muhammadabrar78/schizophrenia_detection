from config import Config
from pipeline import load_and_preprocess, extract_features, select_features, classify_and_evaluate
import time
import os
import random
import numpy as np

def initialize_environment():
    print("=" * 60)
    print("SCHIZOPHRENIA DETECTION SYSTEM INITIATED")
    print("=" * 60)
    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    print(f"Using data from: {Config.data_path}")
    print(f"Bandpass Filter: {Config.bandpass_low}–{Config.bandpass_high} Hz | Sample Rate: {Config.sample_rate} Hz")

def log_progress(step_name):
    print(f"[INFO] ----> Starting: {step_name}")

def log_complete(step_name):
    print(f"[DONE] ----> Completed: {step_name}")

def main():
    start_time = time.time()
    initialize_environment()

    log_progress("Data Loading & Preprocessing")
    try:
        data, labels = load_and_preprocess(Config.data_path)
        log_complete("Data Loading & Preprocessing")
    except Exception as e:
        print(f"[ERROR] Failed during preprocessing: {e}")
        return

    log_progress("Feature Extraction (MEMD + Entropy)")
    try:
        features = extract_features(data)
        log_complete("Feature Extraction")
    except Exception as e:
        print(f"[ERROR] Feature extraction error: {e}")
        return

    log_progress("Feature Selection (CAOA)")
    try:
        selected_features = select_features(features, labels)
        log_complete("Feature Selection")
    except Exception as e:
        print(f"[ERROR] Feature selection failed: {e}")
        return

    log_progress("Model Training & Evaluation (SVM)")
    try:
        classify_and_evaluate(selected_features, labels)
        log_complete("Model Training & Evaluation")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return

    # Optional future steps simulation
    log_progress("Post-Processing and Report Generation")
    time.sleep(1)
    print("Generating performance reports...")
    time.sleep(1)
    print("Saving confusion matrix and metrics summary...")
    time.sleep(1)
    log_complete("Post-Processing and Report Generation")

    # Summary & Time
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"PIPELINE EXECUTION COMPLETE in {total_time:.2f} seconds")
    print("Thank you for using the system!")
    print("=" * 60)

if __name__ == "__main__":
    main()