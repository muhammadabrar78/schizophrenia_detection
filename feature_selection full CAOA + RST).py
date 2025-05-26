
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------- CAOA Implementation ------------------- #

class CAOAFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Full CAOA for feature selection.
    - Population: binary vectors (feature subsets)
    - Fitness: SVM accuracy on validation set
    """
    def __init__(self, n_features=10, pop_size=20, max_iter=30, crossover_rate=0.5, random_state=42, test_size=0.3):
        self.n_features = n_features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        self.test_size = test_size
        self.selected_indices = None

    def fitness(self, X, y, feature_mask):
        """Fitness = SVM accuracy using the selected features."""
        idx = np.where(feature_mask == 1)[0]
        if len(idx) == 0:
            return 0  # Avoid empty feature sets
        X_sub = X[:, idx]
        X_train, X_val, y_train, y_val = train_test_split(X_sub, y, test_size=self.test_size, random_state=self.random_state)
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        return accuracy_score(y_val, y_pred)

    def initialize_population(self, n_features, pop_size):
        # Random binary matrix, each row = a feature subset
        pop = np.zeros((pop_size, n_features), dtype=int)
        for i in range(pop_size):
            indices = np.random.choice(n_features, self.n_features, replace=False)
            pop[i, indices] = 1
        return pop

    def run(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        pop = self.initialize_population(n_features, self.pop_size)
        # Archimedes variables: volume, density, acceleration for each solution
        volume = np.random.rand(self.pop_size)
        density = np.random.rand(self.pop_size)
        acceleration = np.random.rand(self.pop_size)

        best_idx = 0
        best_fitness = 0
        best_solution = None

        for it in range(self.max_iter):
            fitness_vals = np.array([self.fitness(X, y, sol) for sol in pop])
            # Update best
            max_fit_idx = np.argmax(fitness_vals)
            if fitness_vals[max_fit_idx] > best_fitness:
                best_fitness = fitness_vals[max_fit_idx]
                best_solution = pop[max_fit_idx].copy()
                best_idx = max_fit_idx

            # Volume and density update (Eq. 10-11)
            lr1, lr2 = 0.5, 0.5
            volume = lr1 * volume + (1 - lr1) * volume[best_idx]
            density = lr2 * density + (1 - lr2) * density[best_idx]
            # Acceleration update (Eq. 12)
            acceleration = density / (volume + 1e-6)  # Add small epsilon to avoid zero division

            # Solution update
            for i in range(self.pop_size):
                # Velocity update (Eq. 14)
                velocity = acceleration[i] * (np.random.rand(n_features) - 0.5)
                # Position update (Eq. 13)
                prob = 1 / (1 + np.exp(-velocity))
                pop[i] = (prob > 0.5).astype(int)
                # Keep n_features features
                if np.sum(pop[i]) > self.n_features:
                    ones = np.where(pop[i] == 1)[0]
                    pop[i][np.random.choice(ones, size=np.sum(pop[i]) - self.n_features, replace=False)] = 0
                elif np.sum(pop[i]) < self.n_features:
                    zeros = np.where(pop[i] == 0)[0]
                    pop[i][np.random.choice(zeros, size=self.n_features - np.sum(pop[i]), replace=False)] = 1

            # Crossover operation (Eq. 15)
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size and np.random.rand() < self.crossover_rate:
                    mask = np.random.rand(n_features) < 0.5
                    child1 = np.where(mask, pop[i], pop[i+1])
                    child2 = np.where(mask, pop[i+1], pop[i])
                    pop[i], pop[i+1] = child1, child2

        self.selected_indices = np.where(best_solution == 1)[0]
        return self

    def fit(self, X, y):
        return self.run(X, y)

    def transform(self, X):
        if self.selected_indices is None:
            raise RuntimeError("You must fit the selector first.")
        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# ---------------- Rough Set Theory (RST) Implementation ------------------- #

def dependency_degree(X, y, attr_indices):
    """
    Calculate the dependency degree of a subset of attributes (features) with respect to the decision attribute y.
    """
    # Combine selected features and label into tuples for equivalence classes
    combined = np.hstack([X[:, attr_indices], y.reshape(-1, 1)])
    # Count the number of unique rows (equivalence classes)
    total_classes = len(np.unique(combined, axis=0))
    feature_classes = len(np.unique(X[:, attr_indices], axis=0))
    # Dependency degree = #consistent rows / total samples (simplified)
    return feature_classes / total_classes

def rough_set_reduction(X, y, feature_names=None, min_dependency=0.7):
    """
    Simple RST reduction: iteratively drop features that reduce dependency degree least.
    Keeps dropping until dependency degree < min_dependency.
    """
    attrs = list(range(X.shape[1]))
    curr_dep = dependency_degree(X, y, attrs)
    while len(attrs) > 1 and curr_dep >= min_dependency:
        best_dep = 0
        drop_idx = None
        for i in attrs:
            temp_attrs = [j for j in attrs if j != i]
            dep = dependency_degree(X, y, temp_attrs)
            if dep > best_dep:
                best_dep = dep
                drop_idx = i
        if drop_idx is not None and best_dep >= min_dependency:
            attrs.remove(drop_idx)
            curr_dep = best_dep
        else:
            break
    X_rst = X[:, attrs]
    if feature_names is not None:
        selected_names = [feature_names[i] for i in attrs]
    else:
        selected_names = None
    return X_rst, selected_names

# ----------------- Combined Pipeline -------------------- #

def combined_feature_selection(X, y, n_features=10, feature_names=None, min_dependency=0.7):
    """
    CAOA for initial selection, then RST for further reduction.
    """
    caoafs = CAOAFeatureSelector(n_features=n_features)
    X_caoa = caoafs.fit_transform(X, y)
    selected_indices = caoafs.selected_indices
    if feature_names is not None:
        selected_names = [feature_names[i] for i in selected_indices]
    else:
        selected_names = None

    X_rst, final_names = rough_set_reduction(X_caoa, y, selected_names, min_dependency=min_dependency)
    return X_rst, final_names

# ----------------- Example usage ------------------- #
if __name__ == "__main__":
    # Example dummy data
    X = np.random.rand(120, 32)
    y = np.random.randint(0, 2, 120)
    feature_names = [f"Ch_{i}" for i in range(X.shape[1])]

    X_selected, selected_names = combined_feature_selection(X, y, n_features=8, feature_names=feature_names)
    print("Final selected feature names:", selected_names)
    print("Shape after selection:", X_selected.shape)
