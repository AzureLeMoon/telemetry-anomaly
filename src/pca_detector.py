"""PCA & Mahalanobis Distance Multivariate Anomaly Detection Engine.

Implements spectral decomposition of the sample covariance matrix, principal
subspace projection, reconstruction residual scoring, and Mahalanobis distance
evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.linalg_stats import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_mean,
    l2_norm,
    matrix_diagnostics,
    vector_inner_product,
    z_score_normalize,
)


class MultivariateAnomalyDetector:
    r"""Unsupervised multivariate anomaly detector using PCA and Mahalanobis distance.

    Mathematical Overview:
        1. Standardization: $Z = (X - \mu) \oslash \sigma$
        2. Sample Covariance: $\Sigma = \frac{1}{N-1} Z^T Z$
        3. Spectral Decomposition: $\Sigma V = V \Lambda$, where $V^T V = I$ and
           $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$.
        4. Trace Identity Check: $\sum_{i=1}^D \lambda_i = \text{Tr}(\Sigma)$.
        5. Subspace Projection & Reconstruction:
           $$P = Z V_k, \quad \hat{Z} = P V_k^T = Z V_k V_k^T$$
        6. Reconstruction Error:
           $$e_i = z_i - \hat{z}_i, \quad \|e_i\|_2 = \sqrt{e_i^T e_i}$$
        7. Mahalanobis Distance:
           $$D_M(z_i) = \sqrt{(z_i - \mu_z)^T \Sigma^{-1} (z_i - \mu_z)}$$

    Attributes:
        n_components: Default number of principal components $k$ retained for reconstruction.
        mu_: Fitted feature mean vector $\mu \in \mathbb{R}^D$.
        sigma_: Fitted feature standard deviation vector $\sigma \in \mathbb{R}^D$.
        mu_z_: Mean of standardized training matrix $\mu_z \in \mathbb{R}^D$.
        cov_matrix_: Sample covariance matrix $\Sigma \in \mathbb{R}^{D \times D}$.
        corr_matrix_: Pearson correlation matrix $R \in \mathbb{R}^{D \times D}$.
        eigenvalues_: Descending eigenvalues $\lambda \in \mathbb{R}^D$.
        eigenvectors_: Orthonormal eigenvectors $V \in \mathbb{R}^{D \times D}$ (columns).
        explained_variance_ratio_: Fraction of total variance per component.
        diagnostics_: Dictionary of matrix diagnostics (trace, determinant, inverse, Frobenius norm).
    """

    def __init__(self, n_components: int = 2) -> None:
        """Initialize the PCA & Mahalanobis anomaly detector.

        Args:
            n_components: Number of top principal components $k$ to retain during projection.

        Raises:
            ValueError: If `n_components < 1`.
        """
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        self.n_components = int(n_components)

        self.mu_: NDArray[np.float64] | None = None
        self.sigma_: NDArray[np.float64] | None = None
        self.mu_z_: NDArray[np.float64] | None = None
        self.cov_matrix_: NDArray[np.float64] | None = None
        self.corr_matrix_: NDArray[np.float64] | None = None
        self.eigenvalues_: NDArray[np.float64] | None = None
        self.eigenvectors_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.diagnostics_: dict[str, Any] | None = None

    def _check_is_fitted(self) -> None:
        """Verify that the detector has been fitted prior to inference."""
        if (
            self.mu_ is None
            or self.sigma_ is None
            or self.mu_z_ is None
            or self.cov_matrix_ is None
            or self.eigenvalues_ is None
            or self.eigenvectors_ is None
            or self.diagnostics_ is None
        ):
            raise RuntimeError("MultivariateAnomalyDetector must be fitted via .fit(X) before use.")

    def _transform_to_z(self, X: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Standardize input matrix X using fitted training mean and standard deviation."""
        self._check_is_fitted()
        assert self.mu_ is not None and self.sigma_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        safe_sigma = np.where(self.sigma_ < 1e-12, 1.0, self.sigma_)
        return (X_arr - self.mu_) / safe_sigma

    def fit(self, X: NDArray[np.floating[Any]]) -> MultivariateAnomalyDetector:
        r"""Fit the PCA and Mahalanobis statistical model on telemetry matrix X.

        Steps:
            1. Standardize $X \to Z$.
            2. Compute sample covariance matrix $\Sigma = \frac{1}{N-1} Z^T Z$.
            3. Perform symmetric eigendecomposition (`np.linalg.eigh`) to obtain $\lambda, V$.
            4. Sort eigenvalues and eigenvectors in descending order.
            5. Assert the spectral trace identity $\sum_{i=1}^D \lambda_i = \text{Tr}(\Sigma)$.
            6. Compute explained variance ratio $\lambda_i / \sum_{j=1}^D \lambda_j$.

        Args:
            X: Input telemetry matrix of shape `(N, D)`.

        Returns:
            Self (`MultivariateAnomalyDetector`) with fitted spectral attributes.

        Raises:
            ValueError: If `X` is not 2D or `n_components > D`.
            AssertionError: If $\sum \lambda_i \neq \text{Tr}(\Sigma)$ beyond numerical tolerance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array X of shape (N, D), got {X_arr.ndim}D.")
        _, n_features = X_arr.shape
        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed feature dimension D ({n_features})."
            )

        Z, mu, sigma = z_score_normalize(X_arr)
        self.mu_ = mu
        self.sigma_ = sigma
        self.mu_z_ = compute_mean(Z)

        cov = compute_covariance_matrix(Z)
        self.cov_matrix_ = cov
        self.corr_matrix_ = compute_correlation_matrix(cov)
        self.diagnostics_ = matrix_diagnostics(cov)

        # Symmetric eigendecomposition: returns eigenvalues in ascending order
        eigvals_asc, eigvecs_asc = np.linalg.eigh(cov)

        # Sort in descending order
        sort_idx = np.argsort(eigvals_asc)[::-1]
        self.eigenvalues_ = np.maximum(eigvals_asc[sort_idx], 0.0)
        self.eigenvectors_ = eigvecs_asc[:, sort_idx]

        # Spectral Trace Identity assertion: sum(lambda_i) == Tr(Sigma)
        eigenvalue_sum = float(np.sum(self.eigenvalues_))
        trace_sigma = float(self.diagnostics_["trace"])
        assert np.isclose(eigenvalue_sum, trace_sigma, rtol=1e-7, atol=1e-9), (
            f"Spectral trace identity violated: sum(lambda)={eigenvalue_sum:.8f} != Tr(Sigma)={trace_sigma:.8f}"
        )

        self.explained_variance_ratio_ = self.eigenvalues_ / eigenvalue_sum
        return self

    def transform_pc_scores(
        self,
        X: NDArray[np.floating[Any]],
        k: int | None = None,
    ) -> NDArray[np.float64]:
        r"""Project standardized data onto the top k principal eigenvectors.

        Mathematical Formulation:
            $$P = Z V_k \in \mathbb{R}^{N \times k}$$

        Args:
            X: Input telemetry matrix of shape `(N, D)`.
            k: Number of principal components to project onto (defaults to `self.n_components`).

        Returns:
            Latent principal component score matrix $P$ of shape `(N, k)`.
        """
        self._check_is_fitted()
        assert self.eigenvectors_ is not None
        num_k = self.n_components if k is None else int(k)
        if num_k < 1 or num_k > self.eigenvectors_.shape[1]:
            raise ValueError(f"k must be between 1 and {self.eigenvectors_.shape[1]}, got {num_k}.")
        Z = self._transform_to_z(X)
        V_k = self.eigenvectors_[:, :num_k]
        return Z @ V_k

    def project(
        self,
        X: NDArray[np.floating[Any]],
        k: int | None = None,
    ) -> NDArray[np.float64]:
        r"""Project standardized data onto top k eigenvectors and reconstruct in Z-space.

        Mathematical Formulation:
            $$P = Z V_k, \quad \hat{Z} = P V_k^T$$

        Args:
            X: Input telemetry matrix of shape `(N, D)`.
            k: Number of principal components to use (defaults to `self.n_components`).

        Returns:
            Reconstructed standardized telemetry matrix $\hat{Z}$ of shape `(N, D)`.
        """
        self._check_is_fitted()
        assert self.eigenvectors_ is not None
        num_k = self.n_components if k is None else int(k)
        P = self.transform_pc_scores(X, k=num_k)
        V_k = self.eigenvectors_[:, :num_k]
        Z_hat = P @ V_k.T
        return Z_hat

    def compute_reconstruction_error(
        self,
        X: NDArray[np.floating[Any]],
        k: int | None = None,
    ) -> NDArray[np.float64]:
        r"""Compute row-wise L2 reconstruction error between Z and reconstructed Z_hat.

        Mathematical Formulation:
            $$e_i = z_i - \hat{z}_i, \quad \|e_i\|_2 = \sqrt{e_i^T e_i}$$

        Args:
            X: Input telemetry matrix of shape `(N, D)`.
            k: Number of principal components retained (defaults to `self.n_components`).

        Returns:
            1D array of Euclidean reconstruction errors of shape `(N,)`.
        """
        Z = self._transform_to_z(X)
        Z_hat = self.project(X, k=k)
        residuals = Z - Z_hat
        n_samples = residuals.shape[0]
        errors = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            errors[i] = l2_norm(residuals[i])
        return errors

    def compute_mahalanobis_distance(
        self,
        X: NDArray[np.floating[Any]],
    ) -> NDArray[np.float64]:
        r"""Compute the Mahalanobis distance for every sample in standardized space.

        Mathematical Formulation:
            $$D_M(z_i) = \sqrt{(z_i - \mu_z)^T \Sigma^{-1} (z_i - \mu_z)}$$

        Args:
            X: Input telemetry matrix of shape `(N, D)`.

        Returns:
            1D array of Mahalanobis distances $D_M(z_i)$ of shape `(N,)`.
        """
        self._check_is_fitted()
        assert self.mu_z_ is not None and self.diagnostics_ is not None
        Z = self._transform_to_z(X)
        cov_inv: NDArray[np.float64] = self.diagnostics_["inverse"]

        n_samples = Z.shape[0]
        distances = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            diff = Z[i] - self.mu_z_
            transformed = cov_inv @ diff
            quad_form = vector_inner_product(diff, transformed)
            distances[i] = np.sqrt(max(quad_form, 0.0))
        return distances

    def detect(
        self,
        X: NDArray[np.floating[Any]],
        percentile_threshold: float = 97.5,
        k: int | None = None,
    ) -> NDArray[np.int64]:
        r"""Flag multivariate anomalies exceeding empirical/Gaussian score thresholds.

        Combines Mahalanobis distance $D_M(z_i)$ and PCA subspace reconstruction
        error $\|z_i - \hat{z}_i\|_2$. A sample is flagged as anomalous (`1`) if
        its composite or thresholded anomaly score exceeds the specified percentile.

        Args:
            X: Input telemetry matrix of shape `(N, D)`.
            percentile_threshold: Percentile cutoff in `(0, 100)` (e.g., `97.5` for 2.5% tail).
            k: Number of principal components for reconstruction error (defaults to `self.n_components`).

        Returns:
            Binary prediction array `y_pred` of shape `(N,)` (`1` = anomaly, `0` = normal).

        Raises:
            ValueError: If `percentile_threshold` is outside `(0, 100)`.
        """
        if not (0.0 < percentile_threshold < 100.0):
            raise ValueError("percentile_threshold must lie strictly between 0 and 100.")
        mahal_dist = self.compute_mahalanobis_distance(X)
        recon_err = self.compute_reconstruction_error(X, k=k)

        # Combine standardized Mahalanobis and PCA residual scores for robust detection
        mahal_norm = mahal_dist / (float(np.sqrt(compute_mean(mahal_dist ** 2))) + 1e-12)
        recon_norm = recon_err / (float(np.sqrt(compute_mean(recon_err ** 2))) + 1e-12)
        composite_score = 0.5 * mahal_norm + 0.5 * recon_norm

        cutoff = float(np.percentile(composite_score, percentile_threshold))
        return (composite_score >= cutoff).astype(np.int64)
