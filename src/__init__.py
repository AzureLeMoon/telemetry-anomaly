"""Multivariate Time-Series Anomaly Detection Engine using PCA, Mahalanobis Distance, and Calculus."""

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
from src.visualizer import TelemetryVisualizer

__all__ = [
    "TelemetryGenerator",
    "MultivariateAnomalyDetector",
    "PolynomialTrendOptimizer",
    "TelemetryVisualizer",
    "compute_mean",
    "compute_variance",
    "z_score_normalize",
    "compute_covariance_matrix",
    "compute_correlation_matrix",
    "matrix_diagnostics",
    "gaussian_pdf",
    "multivariate_gaussian_pdf",
    "vector_inner_product",
    "l2_norm",
]
