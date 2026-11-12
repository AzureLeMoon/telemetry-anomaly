"""From-scratch Linear Algebra and Probability/Statistics primitives.

This module implements core mathematical routines
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def vector_inner_product(u: NDArray[np.floating[Any]], v: NDArray[np.floating[Any]]) -> float:
    r"""Compute the Euclidean inner product (dot product) of two 1D vectors.

    Mathematical Formulation:
        $$\langle u, v \rangle = u^T v = \sum_{i=1}^D u_i v_i$$

    Args:
        u: First 1D vector of shape `(D,)`.
        v: Second 1D vector of shape `(D,)`.

    Returns:
        Scalar inner product value as a float.

    Raises:
        ValueError: If input arrays are not 1D or do not share the same length.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.ndim != 1 or v_arr.ndim != 1:
        raise ValueError("Both inputs to vector_inner_product must be 1D arrays.")
    if u_arr.shape[0] != v_arr.shape[0]:
        raise ValueError(f"Dimension mismatch: {u_arr.shape[0]} != {v_arr.shape[0]}.")
    return float(u_arr @ v_arr)


def l2_norm(v: NDArray[np.floating[Any]]) -> float:
    r"""Compute the Euclidean L2 norm of a vector via inner product.

    Mathematical Formulation:
        $$\|v\|_2 = \sqrt{\langle v, v \rangle} = \sqrt{v^T v} = \sqrt{\sum_{i=1}^D v_i^2}$$

    Args:
        v: 1D vector of shape `(D,)`.

    Returns:
        Euclidean L2 norm as a non-negative float.
    """
    v_arr = np.asarray(v, dtype=np.float64).ravel()
    return float(np.sqrt(vector_inner_product(v_arr, v_arr)))


def compute_mean(X: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    r"""Compute column-wise sample mean

    Mathematical Formulation:
        $$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$

    Args:
        X: Input data matrix of shape `(N, D)` or 1D array of shape `(N,)`.

    Returns:
        Sample mean vector of shape `(D,)` (or scalar-shaped 1D array if 1D input).

    Raises:
        ValueError: If `X` is empty.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.size == 0:
        raise ValueError("Input array X must not be empty.")
    n_samples = X_arr.shape[0]
    ones = np.ones(n_samples, dtype=np.float64)
    return (ones @ X_arr) / float(n_samples)


def compute_variance(X: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    r"""Compute column-wise unbiased sample variance using Bessel's correction.

    Mathematical Formulation:
        $$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \mu)^2$$

    Args:
        X: Input data matrix of shape `(N, D)` or 1D array of shape `(N,)`.

    Returns:
        Unbiased sample variance vector of shape `(D,)`.

    Raises:
        ValueError: If `X` has fewer than 2 samples ($N < 2$).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n_samples = X_arr.shape[0]
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to compute unbiased sample variance.")
    mu = compute_mean(X_arr)
    centered = X_arr - mu
    squared_deviations = centered * centered
    ones = np.ones(n_samples, dtype=np.float64)
    return (ones @ squared_deviations) / float(n_samples - 1)


def z_score_normalize(
    X: NDArray[np.floating[Any]],
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Standardize a feature matrix to zero mean and unit variance per feature.

    Mathematical Formulation:
        $$Z = \frac{X - \mu}{\sigma}, \quad \sigma = \sqrt{s^2}$$

    Args:
        X: Input feature matrix of shape `(N, D)`.
        eps: Small constant to prevent division by zero for constant features.

    Returns:
        A 3-tuple `(Z, mu, sigma)` containing:
            - `Z`: Standardized data matrix of shape `(N, D)`.
            - `mu`: Column-wise mean vector of shape `(D,)`.
            - `sigma`: Column-wise sample standard deviation vector of shape `(D,)`.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    mu = compute_mean(X_arr)
    var = compute_variance(X_arr)
    sigma = np.sqrt(var)
    safe_sigma = np.where(sigma < eps, 1.0, sigma)
    Z = (X_arr - mu) / safe_sigma
    return Z, mu, sigma


def compute_covariance_matrix(Z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    r"""Compute the unbiased sample covariance matrix via explicit matrix multiplication.

    Mathematical Formulation:
        $$\Sigma = \frac{1}{N - 1} (Z - \mu_Z)^T (Z - \mu_Z)$$
        When $Z$ is already zero-centered ($\mu_Z = \mathbf{0}$), this simplifies to:
        $$\Sigma = \frac{1}{N - 1} Z^T Z$$

    Args:
        Z: Input 2D matrix of shape `(N, D)`.

    Returns:
        Symmetric sample covariance matrix $\Sigma$ of shape `(D, D)`.

    Raises:
        ValueError: If `Z` is not 2D or has fewer than 2 rows.
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Z_arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix for covariance calculation, got {Z_arr.ndim}D.")
    n_samples = Z_arr.shape[0]
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to compute the covariance matrix.")
    mu_z = compute_mean(Z_arr)
    Z_centered = Z_arr - mu_z
    cov = (Z_centered.T @ Z_centered) / float(n_samples - 1)
    return cov


def compute_correlation_matrix(cov: NDArray[np.floating[Any]], eps: float = 1e-12) -> NDArray[np.float64]:
    r"""Compute the Pearson correlation matrix from a covariance matrix.

    Mathematical Formulation:
        $$\sigma = \sqrt{\text{diag}(\Sigma)}, \quad R_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}$$

    Args:
        cov: Square covariance matrix $\Sigma$ of shape `(D, D)`.
        eps: Numerical floor for standard deviations.

    Returns:
        Pearson correlation matrix $R$ of shape `(D, D)` with unit diagonal.

    Raises:
        ValueError: If `cov` is not a 2D square matrix.
    """
    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError("Covariance matrix must be 2D and square.")
    variances = np.diag(cov_arr)
    if np.any(variances < 0.0):
        raise ValueError("Covariance matrix diagonal contains negative variances.")
    std_devs = np.sqrt(np.maximum(variances, eps))
    denom = np.outer(std_devs, std_devs)
    corr = cov_arr / denom
    return np.clip(corr, -1.0, 1.0)


def matrix_diagnostics(
    M: NDArray[np.floating[Any]],
    reg_epsilon: float = 1e-6,
    cond_threshold: float = 1e12,
) -> dict[str, Any]:
    r"""Compute key linear algebra diagnostics and regularized inverse of a square matrix.

    Mathematical Formulation:
        - Trace: $\text{Tr}(M) = \sum_{i=1}^D M_{ii}$
        - Determinant: $\det(M)$
        - Frobenius Norm: $\|M\|_F = \sqrt{\text{Tr}(M^T M)}$
        - Inverse: $M^{-1}$, or $(M + \epsilon I)^{-1}$ if $M$ is singular/ill-conditioned.

    Args:
        M: Square matrix of shape `(D, D)`.
        reg_epsilon: Tikhonov regularization strength $\epsilon$ added to the diagonal
            if the matrix is nearly singular.
        cond_threshold: Condition number threshold triggering $\Sigma + \epsilon I$ regularization.

    Returns:
        Dictionary with keys:
            - `"trace"`: Float matrix trace.
            - `"determinant"`: Float matrix determinant.
            - `"frobenius_norm"`: Float Frobenius norm computed via $\sqrt{\text{Tr}(M^T M)}$.
            - `"inverse"`: 2D NDArray representing $M^{-1}$ (or regularized inverse).
            - `"regularized"`: Boolean indicating whether Tikhonov regularization was applied.
            - `"condition_number"`: Float 2-norm condition number of $M$.

    Raises:
        ValueError: If `M` is not a 2D square matrix.
    """
    M_arr = np.asarray(M, dtype=np.float64)
    if M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1]:
        raise ValueError("Input matrix M must be 2D and square.")

    dim = M_arr.shape[0]
    diag_elements = np.diag(M_arr)
    trace_val = float(np.sum(diag_elements))

    det_val = float(np.linalg.det(M_arr))

    gram_matrix = M_arr.T @ M_arr
    gram_trace = float(np.sum(np.diag(gram_matrix)))
    frobenius_val = float(np.sqrt(max(gram_trace, 0.0)))

    cond_num = float(np.linalg.cond(M_arr))
    needs_regularization = (abs(det_val) < 1e-12) or (not np.isfinite(cond_num)) or (cond_num > cond_threshold)

    if needs_regularization:
        M_reg = M_arr + reg_epsilon * np.eye(dim, dtype=np.float64)
        inv_matrix = np.linalg.inv(M_reg)
        regularized = True
    else:
        try:
            inv_matrix = np.linalg.inv(M_arr)
            regularized = False
        except np.linalg.LinAlgError:
            M_reg = M_arr + reg_epsilon * np.eye(dim, dtype=np.float64)
            inv_matrix = np.linalg.inv(M_reg)
            regularized = True

    return {
        "trace": trace_val,
        "determinant": det_val,
        "frobenius_norm": frobenius_val,
        "inverse": inv_matrix,
        "regularized": regularized,
        "condition_number": cond_num,
    }


def gaussian_pdf(
    x: NDArray[np.floating[Any]],
    mu: float,
    var: float,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    r"""Evaluate the univariate (1D) Gaussian probability density function.

    Mathematical Formulation:
        $$f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

    Args:
        x: Array of evaluation points of arbitrary shape.
        mu: Mean $\mu$ of the Gaussian distribution.
        var: Variance $\sigma^2 > 0$ of the Gaussian distribution.
        eps: Minimum variance floor for numerical stability.

    Returns:
        Array of probability density values with the same shape as `x`.

    Raises:
        ValueError: If `var <= 0`.
    """
    if var <= 0.0:
        raise ValueError(f"Variance must be strictly positive, got {var}.")
    x_arr = np.asarray(x, dtype=np.float64)
    safe_var = max(float(var), eps)
    coeff = 1.0 / np.sqrt(2.0 * np.pi * safe_var)
    exponent = -((x_arr - float(mu)) ** 2) / (2.0 * safe_var)
    return coeff * np.exp(exponent)


def multivariate_gaussian_pdf(
    X: NDArray[np.floating[Any]],
    mu: NDArray[np.floating[Any]],
    cov: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    r"""Evaluate the multivariate Gaussian probability density function for each row in X.

    Mathematical Formulation:
        $$f(x) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}}
        \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

    Args:
        X: Data matrix of shape `(N, D)` or single vector of shape `(D,)`.
        mu: Mean vector $\mu$ of shape `(D,)`.
        cov: Covariance matrix $\Sigma$ of shape `(D, D)`.

    Returns:
        1D array of density evaluations of shape `(N,)`.
    """
    X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
    mu_arr = np.asarray(mu, dtype=np.float64).ravel()
    cov_arr = np.asarray(cov, dtype=np.float64)

    dim = X_arr.shape[1]
    diag_info = matrix_diagnostics(cov_arr)
    cov_inv = diag_info["inverse"]
    det_cov = abs(diag_info["determinant"])
    if det_cov < 1e-12:
        det_cov = float( abs(np.linalg.det(cov_arr + 1e-6 * np.eye(dim))) )

    diff = X_arr - mu_arr
    # Quadratic form (x_i - mu)^T Sigma^-1 (x_i - mu) for each sample i
    quad_form = np.sum((diff @ cov_inv) * diff, axis=1)
    norm_const = 1.0 / (np.power(2.0 * np.pi, 0.5 * dim) * np.sqrt(det_cov))
    return norm_const * np.exp(-0.5 * quad_form)
