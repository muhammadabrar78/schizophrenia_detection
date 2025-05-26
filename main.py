
import pandas as pd
import numpy as np
from preprocessing import preprocess_features
from feature_selection import combined_feature_selection
from model import train_svm_classifier, predict, svm_grid_search
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

features = preprocess_features('features_raw.csv')
labels = pd.read_csv('labels.csv').values.flatten()

X_train, X_test, y_train, y_test = train_test_split(features.values, labels, test_size=0.2, random_state=42)

X_train_sel, selected_names = combined_feature_selection(X_train, y_train, n_features=10, feature_names=features.columns)
X_test_sel = X_test[:, [features.columns.get_loc(name) for name in selected_names]]

svm = train_svm_classifier(X_train_sel, y_train)

y_pred = predict(svm, X_test_sel)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
