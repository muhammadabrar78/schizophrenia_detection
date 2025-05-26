
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

def train_svm_classifier(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42):
    svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=random_state)
    svm.fit(X_train, y_train)
    return svm

def svm_grid_search(X, y, param_grid=None, cv=5, scoring='accuracy'):
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=cv, scoring=scoring)
    grid.fit(X, y)
    print(f"Best SVM Params: {grid.best_params_}")
    return grid.best_estimator_

def predict(model, X_test):
    return model.predict(X_test)

def predict_proba(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    else:
        raise ValueError("Model does not support probability prediction.")

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = train_svm_classifier(X_train, y_train)
    preds = predict(svm, X_test)
    print("Example predictions:", preds[:5])
    save_model(svm, "svm_model.joblib")
    loaded_svm = load_model("svm_model.joblib")
    preds2 = predict(loaded_svm, X_test)
    print("Predictions from loaded model:", preds2[:5])
