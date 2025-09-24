"""
Principal Component Analysis (PCA) from scratch using NumPy only.

Steps implemented:
1) Center the data (subtract the mean)
2) Compute the covariance matrix
3) Find eigenvalues and eigenvectors
4) Sort by eigenvalue magnitude (descending)
5) Project the data onto the top k principal components

Includes a simple demo at the bottom.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def center_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center the data matrix X by subtracting the mean of each feature (column).

    Args:
        X: Data matrix of shape (n_samples, n_features).

    Returns:
        X_centered: Centered data matrix of shape (n_samples, n_features).
        mean_vector: Mean of each feature, shape (n_features,).
    """
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array of shape (n_samples, n_features)")

    mean_vector = np.mean(X, axis=0)
    X_centered = X - mean_vector
    return X_centered, mean_vector


def compute_covariance_matrix(X_centered: np.ndarray, bias: bool = False) -> np.ndarray:
    """
    Compute the covariance matrix of centered data.

    Args:
        X_centered: Centered data matrix of shape (n_samples, n_features).
        bias: If True, use biased estimator (divide by n). If False, divide by (n-1).

    Returns:
        Covariance matrix of shape (n_features, n_features).
    """
    if X_centered.ndim != 2:
        raise ValueError("X_centered must be a 2D array")

    n_samples = X_centered.shape[0]
    if n_samples < 2 and not bias:
        raise ValueError("At least 2 samples required for unbiased covariance (n-1 in denominator)")

    denom = n_samples if bias else (n_samples - 1)
    cov = (X_centered.T @ X_centered) / denom
    return cov


def eigen_decomposition(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a symmetric covariance matrix.

    Args:
        cov: Covariance matrix of shape (n_features, n_features), symmetric positive semi-definite.

    Returns:
        eigenvalues: Array of shape (n_features,) containing eigenvalues.
        eigenvectors: Matrix of shape (n_features, n_features), columns are eigenvectors.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square 2D array")

    # Use eigh for symmetric matrices for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvalues, eigenvectors


def sort_eigens(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort eigenvalues and eigenvectors by eigenvalue magnitude in descending order.

    Args:
        eigenvalues: Array of shape (n_features,).
        eigenvectors: Matrix of shape (n_features, n_features), columns are eigenvectors.

    Returns:
        sorted_eigenvalues, sorted_eigenvectors
    """
    if eigenvectors.shape[0] != eigenvectors.shape[1] or eigenvectors.shape[0] != eigenvalues.shape[0]:
        raise ValueError("Shapes of eigenvalues and eigenvectors are inconsistent")

    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def project_data(X_centered: np.ndarray, components: np.ndarray, k: int) -> np.ndarray:
    """
    Project centered data onto the top-k principal components.

    Args:
        X_centered: Centered data of shape (n_samples, n_features).
        components: Principal axes (eigenvectors as columns), shape (n_features, n_features).
        k: Number of principal components to keep (1 <= k <= n_features).

    Returns:
        X_projected: Transformed data in k-dimensional space of shape (n_samples, k).
    """
    n_features = X_centered.shape[1]
    if not (1 <= k <= n_features):
        raise ValueError(f"k must be between 1 and {n_features}, got {k}")

    top_components = components[:, :k]
    X_projected = X_centered @ top_components
    return X_projected


def pca(X: np.ndarray, k: int, bias: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full PCA pipeline.

    Args:
        X: Data matrix (n_samples, n_features).
        k: Number of principal components to retain.
        bias: If True, use biased covariance estimator (divide by n) instead of (n-1).

    Returns:
        X_projected: Data projected onto top-k components (n_samples, k)
        components: Principal axes (eigenvectors as columns), sorted (n_features, n_features)
        eigenvalues: Sorted eigenvalues (n_features,)
        mean_vector: Mean used for centering (n_features,)
    """
    X_centered, mean_vector = center_data(X)
    cov = compute_covariance_matrix(X_centered, bias=bias)
    eigenvalues, eigenvectors = eigen_decomposition(cov)
    eigenvalues_sorted, eigenvectors_sorted = sort_eigens(eigenvalues, eigenvectors)
    X_projected = project_data(X_centered, eigenvectors_sorted, k)
    return X_projected, eigenvectors_sorted, eigenvalues_sorted, mean_vector


def explained_variance_ratio(eigenvalues: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """
    Compute explained variance ratio for top-k components and cumulative ratio.

    Args:
        eigenvalues: Sorted eigenvalues (descending) of shape (n_features,).
        k: Number of components to consider.

    Returns:
        ratios_k: Array of shape (k,) with per-component explained variance ratios.
        cumulative_k: Cumulative explained variance ratio up to k.
    """
    if k < 1 or k > eigenvalues.shape[0]:
        raise ValueError("Invalid k for explained variance ratio")
    total = np.sum(eigenvalues)
    if total <= 0:
        raise ValueError("Sum of eigenvalues must be positive to compute ratios")
    ratios = eigenvalues / total
    ratios_k = ratios[:k]
    cumulative_k = float(np.sum(ratios_k))
    return ratios_k, cumulative_k


def _demo() -> None:
    """
    Demonstration using a simple 2D synthetic dataset with correlation.
    """
    rng = np.random.default_rng(42)

    # Create an elongated 2D Gaussian blob:
    # First generate uncorrelated data, then apply a linear transform to induce correlation
    n_samples = 200
    base = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
    transform = np.array([[2.5, 1.0], [0.0, 0.5]])  # shape (2, 2)
    X = base @ transform.T + np.array([5.0, -3.0])  # add mean shift

    # Run PCA for k=1 and k=2
    for k in (1, 2):
        X_proj, components, eigenvalues, mean_vec = pca(X, k=k)
        ratios_k, cum_k = explained_variance_ratio(eigenvalues, k)

        print("\n================ PCA Demo ================")
        print(f"n_samples={n_samples}, n_features=2, k={k}")
        print("Mean vector:", np.round(mean_vec, 3))
        print("Eigenvalues (sorted):", np.round(eigenvalues, 5))
        print("Explained variance ratio (top-k):", np.round(ratios_k, 5))
        print(f"Cumulative explained variance (k={k}): {cum_k:.5f}")
        print("Principal components (columns):\n", np.round(components[:, :k], 5))
        print("Projected shape:", X_proj.shape)

    # Show a small sample of projected points for k=1
    X_proj_1, comps, evals, mean_vec = pca(X, k=1)
    print("\nSample of first 5 projected points onto PC1:")
    print(np.round(X_proj_1[:5, 0], 5))


if __name__ == "__main__":
    _demo()


