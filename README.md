## PCA from Scratch (NumPy)

A clean, modular implementation of Principal Component Analysis (PCA) using only NumPy. Includes:
- Centering the data
- Computing the covariance matrix
- Eigen decomposition
- Sorting eigenpairs by eigenvalue magnitude
- Projecting onto the top-k principal components

### Files
- `pca_from_scratch.py`: PCA implementation and a runnable demo.

### Quick Start
Create and activate a virtual environment, then run the demo:

```bash
python3 -m venv /Users/mananawasthi/Desktop/Pca_scratch/.venv
source /Users/mananawasthi/Desktop/Pca_scratch/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
python /Users/mananawasthi/Desktop/Pca_scratch/pca_from_scratch.py
```

Expected output includes eigenvalues, explained variance ratios, principal components, and a sample of projected points.

### Usage in Your Code
Import and call the `pca` function. It returns the projected data, components, eigenvalues, and mean vector.

```python
import numpy as np
from pca_from_scratch import pca, explained_variance_ratio

X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0]])

X_proj, components, eigenvalues, mean_vec = pca(X, k=2)
ratios_k, cum_k = explained_variance_ratio(eigenvalues, k=2)
```

### API Overview
- `center_data(X) -> (X_centered, mean_vector)`
- `compute_covariance_matrix(X_centered, bias=False) -> cov`
- `eigen_decomposition(cov) -> (eigenvalues, eigenvectors)`
- `sort_eigens(eigenvalues, eigenvectors) -> (eigenvalues_sorted, eigenvectors_sorted)`
- `project_data(X_centered, components, k) -> X_projected`
- `pca(X, k, bias=False) -> (X_projected, components_sorted, eigenvalues_sorted, mean_vector)`
- `explained_variance_ratio(eigenvalues_sorted, k) -> (ratios_k, cumulative_k)`

Notes:
- `components` are returned as columns; the first `k` columns are the top-k principal directions.
- Input `X` must be shaped `(n_samples, n_features)`.

### Reproducible Demo
The built-in demo (`_demo()`) constructs a 2D correlated dataset, runs PCA for `k=1` and `k=2`, and prints results. Run via:

```bash
python /Users/mananawasthi/Desktop/Pca_scratch/pca_from_scratch.py
```

### License
MIT


