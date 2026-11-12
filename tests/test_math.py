"""Unit tests verifying from-scratch Linear Algebra, Statistics, PCA, and Calculus routines.

All custom mathematical primitives are validated against NumPy reference builtins
and analytical mathematical identities.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import TelemetryGenerator
from src.linalg_stats import (
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_mean,
    compute_variance,
    gaussian_pdf,
    l2_norm,
    matrix_diagnostics,
    multivariate_gaussian_pdf,
    vector_inner_product,
    z_score_normalize,
)
from src.optimizer import PolynomialTrendOptimizer
from src.pca_detector import MultivariateAnomalyDetector


def test_vector_inner_product_and_l2_norm() -> None:
    """Verify inner product and Euclidean L2 norm against np.dot and np.linalg.norm."""
    rng = np.random.default_rng(101)
    u = rng.normal(size=16)
    v = rng.normal(size=16)

    assert np.isclose(vector_inner_product(u, v), float(np.dot(u, v)), rtol=1e-12)
    assert np.isclose(l2_norm(u), float(np.linalg.norm(u, ord=2)), rtol=1e-12)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        vector_inner_product(u, v[:5])


def test_mean_variance_and_z_score_against_numpy() -> None:
    """Verify sample mean, unbiased sample variance (ddof=1), and Z-score normalization."""
    rng = np.random.default_rng(202)
    X = rng.normal(loc=[10.0, -5.0, 25.0, 0.0], scale=[2.0, 0.5, 4.0, 1.5], size=(250, 4))

    mu = compute_mean(X)
    var = compute_variance(X)
    Z, mu_z, sigma_z = z_score_normalize(X)

    np.testing.assert_allclose(mu, np.mean(X, axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(var, np.var(X, axis=0, ddof=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mu_z, mu, rtol=1e-12)
    np.testing.assert_allclose(sigma_z, np.std(X, axis=0, ddof=1), rtol=1e-12)

    # Standardized columns must have zero mean and unit unbiased variance
    np.testing.assert_allclose(compute_mean(Z), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(compute_variance(Z), np.ones(4), rtol=1e-12)


def test_covariance_and_correlation_matrices_against_numpy() -> None:
    """Verify covariance and Pearson correlation matrices against np.cov and np.corrcoef."""
    rng = np.random.default_rng(303)
    base = rng.normal(size=(300, 1))
    X = np.hstack([
        base,
        2.5 * base + rng.normal(scale=0.3, size=(300, 1)),
        -1.2 * base + rng.normal(scale=0.8, size=(300, 1)),
        rng.normal(scale=1.5, size=(300, 1)),
    ])

    Z, _, _ = z_score_normalize(X)
    cov_z = compute_covariance_matrix(Z)
    expected_cov_z = np.cov(Z, rowvar=False, ddof=1)
    np.testing.assert_allclose(cov_z, expected_cov_z, rtol=1e-12, atol=1e-12)

    cov_x = compute_covariance_matrix(X)
    expected_cov_x = np.cov(X, rowvar=False, ddof=1)
    np.testing.assert_allclose(cov_x, expected_cov_x, rtol=1e-12, atol=1e-12)

    corr_x = compute_correlation_matrix(cov_x)
    expected_corr_x = np.corrcoef(X, rowvar=False)
    np.testing.assert_allclose(corr_x, expected_corr_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.diag(corr_x), np.ones(4), atol=1e-12)


def test_matrix_diagnostics_and_singular_regularization() -> None:
    """Verify trace, determinant, Frobenius norm, matrix inverse, and Tikhonov regularization."""
    M = np.array(
        [
            [4.0, 1.2, -0.5],
            [1.2, 3.0, 0.8],
            [-0.5, 0.8, 2.5],
        ],
        dtype=np.float64,
    )
    diag = matrix_diagnostics(M)

    assert np.isclose(diag["trace"], float(np.trace(M)), rtol=1e-12)
    assert np.isclose(diag["determinant"], float(np.linalg.det(M)), rtol=1e-12)
    assert np.isclose(diag["frobenius_norm"], float(np.linalg.norm(M, ord="fro")), rtol=1e-12)
    assert diag["regularized"] is False
    np.testing.assert_allclose(diag["inverse"] @ M, np.eye(3), atol=1e-12)

    # Rank-deficient singular matrix should trigger regularization (Sigma + eps * I)
    singular_M = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    singular_diag = matrix_diagnostics(singular_M, reg_epsilon=1e-4)
    assert singular_diag["regularized"] is True
    assert np.all(np.isfinite(singular_diag["inverse"]))


def test_gaussian_pdf_univariate_and_multivariate() -> None:
    """Verify univariate and multivariate Gaussian PDF normalization and numerical accuracy."""
    mu = 2.5
    var = 1.75
    x = np.linspace(mu - 8.0 * np.sqrt(var), mu + 8.0 * np.sqrt(var), 5001)
    pdf = gaussian_pdf(x, mu=mu, var=var)

    # Numerical Riemann/trapezoidal integral over [-8 sigma, +8 sigma] must equal 1.0
    integral = float(np.trapezoid(pdf, x) if hasattr(np, "trapezoid") else np.trapz(pdf, x))
    assert np.isclose(integral, 1.0, atol=1e-4)

    # Peak density at x == mu must equal 1 / sqrt(2 * pi * var)
    peak = float(gaussian_pdf(np.array([mu]), mu=mu, var=var)[0])
    assert np.isclose(peak, 1.0 / np.sqrt(2.0 * np.pi * var), rtol=1e-12)

    # Multivariate Gaussian PDF at mean x == mu must equal 1 / ((2*pi)^(D/2) * sqrt(det(Sigma)))
    mu_vec = np.array([1.0, -2.0], dtype=np.float64)
    cov_mat = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=np.float64)
    mv_peak = float(multivariate_gaussian_pdf(mu_vec, mu_vec, cov_mat)[0])
    expected_mv_peak = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov_mat)))
    assert np.isclose(mv_peak, expected_mv_peak, rtol=1e-12)


def test_pca_detector_spectral_identities_and_detection() -> None:
    """Verify PCA eigendecomposition, trace identity, reconstruction error, and anomaly detection."""
    gen = TelemetryGenerator(n_samples=1000, n_anomalies=25, random_seed=42)
    X_raw, _, y_true = gen.generate()

    detector = MultivariateAnomalyDetector(n_components=2)
    detector.fit(X_raw)

    assert detector.eigenvalues_ is not None
    assert detector.eigenvectors_ is not None
    assert detector.diagnostics_ is not None
    assert detector.explained_variance_ratio_ is not None

    # 1. Eigenvalues must be sorted in descending order and sum to Trace(Sigma)
    assert np.all(np.diff(detector.eigenvalues_) <= 0.0)
    assert np.isclose(np.sum(detector.eigenvalues_), detector.diagnostics_["trace"], rtol=1e-10)
    assert np.isclose(np.sum(detector.explained_variance_ratio_), 1.0, rtol=1e-12)

    # 2. Eigenvectors must be orthonormal: V^T V = I
    np.testing.assert_allclose(
        detector.eigenvectors_.T @ detector.eigenvectors_,
        np.eye(4),
        atol=1e-12,
    )

    # 3. Full rank k=D reconstruction must yield zero reconstruction error
    full_recon_err = detector.compute_reconstruction_error(X_raw, k=4)
    np.testing.assert_allclose(full_recon_err, np.zeros(1000), atol=1e-11)

    # 4. Injected multivariate anomalies must have markedly higher Mahalanobis distance than normal points
    mahal = detector.compute_mahalanobis_distance(X_raw)
    mean_anom_mahal = float(np.mean(mahal[y_true == 1]))
    mean_norm_mahal = float(np.mean(mahal[y_true == 0]))
    assert mean_anom_mahal > 3.5 * mean_norm_mahal

    y_pred = detector.detect(X_raw, percentile_threshold=97.5, k=2)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    recall = tp / 25.0
    assert recall >= 0.85


def test_optimizer_analytical_gradients_and_convex_convergence() -> None:
    """Verify chain-rule analytical gradients against finite differences and check convex convergence."""
    t = np.linspace(0.0, 1.0, 200, dtype=np.float64)
    y = 12.0 * (t ** 2) + 8.0 * t + 45.0

    opt = PolynomialTrendOptimizer(learning_rate=0.3, epochs=500, alpha=1e-3)
    test_w = np.array([3.5, -1.2], dtype=np.float64)
    test_b = 10.0

    grad_w, grad_b = opt.compute_gradients(t, y, weights=test_w, bias=test_b)

    # Central finite difference verification of partial derivatives
    eps = 1e-6
    num_grad_w = np.zeros_like(test_w)
    for j in range(2):
        w_plus = test_w.copy()
        w_minus = test_w.copy()
        w_plus[j] += eps
        w_minus[j] -= eps
        loss_plus = opt.compute_loss(t, y, weights=w_plus, bias=test_b)
        loss_minus = opt.compute_loss(t, y, weights=w_minus, bias=test_b)
        num_grad_w[j] = (loss_plus - loss_minus) / (2.0 * eps)

    loss_b_plus = opt.compute_loss(t, y, weights=test_w, bias=test_b + eps)
    loss_b_minus = opt.compute_loss(t, y, weights=test_w, bias=test_b - eps)
    num_grad_b = (loss_b_plus - loss_b_minus) / (2.0 * eps)

    np.testing.assert_allclose(grad_w, num_grad_w, rtol=1e-6, atol=1e-6)
    assert np.isclose(grad_b, num_grad_b, rtol=1e-6, atol=1e-6)

    # Fit for 500 epochs and verify monotonic convex loss reduction
    opt.fit(t, y)
    assert len(opt.loss_history_) == 500
    assert opt.loss_history_[-1] < 1e-2 * opt.loss_history_[0]
    losses = np.asarray(opt.loss_history_)
    assert np.all(np.diff(losses) <= 1e-12)
