import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.signal import butter, filtfilt
from PyEMD import EMD
from antropy import sample_entropy, approximate_entropy
import matplotlib.pyplot as plt
import seaborn as sns

# === DATA LOADING & PREPROCESSING ===
def load_and_preprocess(path):
    print(f"Reading EEG data from {path}")
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    X = df.drop(columns=['label']).values
    y = df['label']
    print(f"Data shape before filtering: {X.shape}")
    X = bandpass_filter(X, 0.5, 50, 500)
    X = StandardScaler().fit_transform(X)
    print(f"Data shape after normalization: {X.shape}")
    return X, y

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    print("Applying bandpass filter...")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# === FEATURE EXTRACTION ===
def extract_features(data):
    print("Performing MEMD and entropy-based feature extraction...")
    memd_feats = []
    entropy_feats = []

    for ch in range(data.shape[1]):
        emd = EMD()
        imfs = emd(data[:, ch])
        energy = [np.sum(imf**2) for imf in imfs[:3]]  # Use first 3 IMFs
        memd_feats.extend(energy)

        apen = approximate_entropy(data[:, ch], 2, 0.2 * np.std(data[:, ch]))
        sampen = sample_entropy(data[:, ch])
        entropy_feats.append(apen)
        entropy_feats.append(sampen)

    features = pd.DataFrame([memd_feats + entropy_feats])
    print(f"Feature vector length: {features.shape[1]}")
    return features

# === FEATURE SELECTION WITH CAOA ===
def select_features(features, labels, max_iter=10):
    print("Starting CAOA feature selection...")
    pop_size = 10
    population = [features.sample(frac=0.5, axis=1, random_state=i) for i in range(pop_size)]
    best_solution = population[0]

    for iteration in range(max_iter):
        print(f"Iteration {iteration+1}/{max_iter}")
        scores = [np.abs(p.corrwith(labels)).mean() for p in population]
        best_idx = np.argmax(scores)
        best_solution = population[best_idx]
        print(f"Best fitness at iteration {iteration+1}: {scores[best_idx]:.4f}")
        for i in range(pop_size):
            crossover_point = np.random.randint(1, best_solution.shape[1])
            parent = population[np.random.randint(0, pop_size)]
            child = pd.concat([best_solution.iloc[:, :crossover_point], parent.iloc[:, crossover_point:]], axis=1)
            population[i] = child
    print("CAOA optimization complete.")
    return best_solution

# === CLASSIFICATION AND EVALUATION ===
def classify_and_evaluate(features, labels):
    print("Training SVM classifier and evaluating performance...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"ROC AUC Score: {auc:.4f}")