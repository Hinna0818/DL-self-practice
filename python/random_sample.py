## a function for making data subsets randomly
import numpy as np
def get_random_subsets(X, y, n_subsets, size = 2, replacements=True, seed=42):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    np.random.seed(seed)
    n_samples = X.shape[0]
    subsets = []

    for _ in range(n_subsets):
        indices = np.random.choice(n_samples, size = size, replace = replacements)
        X_subset = X[indices].tolist()
        y_subset = y[indices].tolist()
        subsets.append((X_subset, y_subset))
    
    return subsets

## example
np.random.seed(1234)  
X = np.random.rand(10, 3)  
y = np.random.randint(0, 2, 10)  

subsets = get_random_subsets(X, y, n_subsets = 3, size = 4, replacements=True)
print(subsets)