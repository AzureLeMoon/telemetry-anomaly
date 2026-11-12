"""CLI Entrypoint for the Multivariate Time-Series Anomaly Detection Engine.

Executes synthetic telemetry generation, Gradient Descent baseline trend calibration,
PCA spectral decomposition, Mahalanobis distance scoring, evaluation metrics,
and scientific visualization generation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.generator import TelemetryGenerator
from src.linalg_stats import compute_mean, multivariate_gaussian_pdf
from src.optimizer import PolynomialTrendOptimizer
from src.pca_detector import MultivariateAnomalyDetector
from src.visualizer import TelemetryVisualizer


def configure_logging(debug: bool = False) -> logging.Logger:
    """Configure structured CLI logging.

    Args:
        debug: If True, sets logging level to `DEBUG`; otherwise `INFO`.

    Returns:
        Configured root module logger instance.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logging.getLogger("telemetry_anomaly_pca")


def compute_classification_metrics(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.integer[Any]],
) -> dict[str, float]:
    r"""Compute binary anomaly detection Precision, Recall, F1-Score, and Accuracy.

    Mathematical Formulation:
        $$\text{Precision} = \frac{TP}{TP + FP}, \quad
        \text{Recall} = \frac{TP}{TP + FN}, \quad
        F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

    Args:
        y_true: Ground-truth binary array of shape `(N,)`.
        y_pred: Predicted binary array of shape `(N,)`.

    Returns:
        Dictionary containing `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, `f1`, `accuracy`.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    tp = float(np.sum((yt == 1) & (yp == 1)))
    fp = float(np.sum((yt == 0) & (yp == 1)))
    tn = float(np.sum((yt == 0) & (yp == 0)))
    fn = float(np.sum((yt == 1) & (yp == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    accuracy = (tp + tn) / float(yt.shape[0])

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the anomaly detection pipeline.

    Args:
        argv: Optional list of CLI argument strings.

    Returns:
        Parsed `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Multivariate Time-Series Anomaly Detection Engine using PCA, "
            "Mahalanobis Distance, and Numerical Gradient Descent."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of synthetic telemetry timestamps N to generate (default: 1000).",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=2,
        help="Number of principal components k to retain for reconstruction (default: 2).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.25,
        help="Learning rate eta for Gradient Descent trend calibration (default: 0.25).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of Gradient Descent optimization epochs (default: 500).",
    )
    parser.add_argument(
        "--anomalies",
        type=int,
        default=25,
        help="Number of multivariate anomalies to inject (default: 25).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated matplotlib figures (default: output).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose DEBUG logging and interactive inspection hooks.",
    )
    return parser.parse_args(argv)


def print_summary_table(
    diagnostics: dict[str, Any],
    eigenvalues: NDArray[np.float64],
    explained_ratio: NDArray[np.float64],
    optimizer: PolynomialTrendOptimizer,
    metrics: dict[str, float],
    mean_mv_density: float,
    saved_plots: list[Path],
) -> None:
    """Print a structured mathematical and statistical summary table to stdout."""
    border = "=" * 78
    divider = "-" * 78
    eig_str = ", ".join(f"{val:.4f}" for val in eigenvalues)
    var_str = ", ".join(f"{val * 100.0:.2f}%" for val in explained_ratio)

    print(f"\n{border}")
    print("  MULTIVARIATE TELEMETRY ANOMALY DETECTION - MATHEMATICAL SUMMARY TABLE")
    print(border)
    print(f"  {'Metric / Property':<40} | {'Value':<32}")
    print(divider)
    print(f"  {'Covariance Trace Tr(Sigma)':<40} | {diagnostics['trace']:.6f}")
    print(f"  {'Sum of Eigenvalues sum(lambda_i)':<40} | {float(np.sum(eigenvalues)):.6f}")
    print(f"  {'Covariance Determinant det(Sigma)':<40} | {diagnostics['determinant']:.6e}")
    print(f"  {'Covariance Frobenius Norm ||Sigma||_F':<40} | {diagnostics['frobenius_norm']:.6f}")
    print(f"  {'Covariance Condition Number kappa(Sigma)':<40} | {diagnostics['condition_number']:.4f}")
    print(f"  {'Eigenvalues [lambda_1 .. lambda_D]':<40} | [{eig_str}]")
    print(f"  {'Explained Variance Ratio (%)':<40} | [{var_str}]")
    print(f"  {'Mean Multivariate Gaussian PDF f(z)':<40} | {mean_mv_density:.6e}")
    print(divider)
    print(
        f"  {'GD Trend Parameters [w2, w1, b]':<40} | "
        f"[{optimizer.weights_[0]:.4f}, {optimizer.weights_[1]:.4f}, {optimizer.bias_:.4f}]"
    )
    print(
        f"  {'GD Convex Loss (Initial -> Final)':<40} | "
        f"{optimizer.loss_history_[0]:.4f} -> {optimizer.loss_history_[-1]:.4f}"
    )
    print(divider)
    print(
        f"  {'Confusion Matrix (TP / FP / FN / TN)':<40} | "
        f"{int(metrics['tp'])} / {int(metrics['fp'])} / {int(metrics['fn'])} / {int(metrics['tn'])}"
    )
    print(f"  {'Anomaly Detection Precision':<40} | {metrics['precision'] * 100.0:.2f}%")
    print(f"  {'Anomaly Detection Recall':<40} | {metrics['recall'] * 100.0:.2f}%")
    print(f"  {'Anomaly Detection F1-Score':<40} | {metrics['f1'] * 100.0:.2f}%")
    print(f"  {'Overall Classification Accuracy':<40} | {metrics['accuracy'] * 100.0:.2f}%")
    print(divider)
    for plot_path in saved_plots:
        print(f"  {'Saved Diagnostic Figure':<40} | {str(plot_path)}")
    print(f"{border}\n")


def main(argv: list[str] | None = None) -> int:
    """Run the complete multivariate anomaly detection and calculus calibration pipeline.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code (`0` on success).
    """
    args = parse_args(argv)
    logger = configure_logging(debug=args.debug)

    logger.info(
        "Starting Multivariate Telemetry Anomaly Engine (N=%d, k=%d, lr=%.4f, epochs=%d)",
        args.samples,
        args.components,
        args.learning_rate,
        args.epochs,
    )

    # 1. Synthetic Multivariate Telemetry Generation
    generator = TelemetryGenerator(
        n_samples=args.samples,
        n_anomalies=args.anomalies,
        random_seed=42,
    )
    X_raw, timestamps, y_true = generator.generate()
    logger.debug("Generated telemetry matrix X_raw with shape %s", X_raw.shape)

    # 2. Calculus & Gradient Descent Baseline Trend Calibration
    optimizer = PolynomialTrendOptimizer(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        alpha=1e-4,
    )
    optimizer.fit(timestamps, X_raw[:, 0])
    trend_curve = optimizer.predict(timestamps)
    X_detrended = optimizer.detrend(timestamps, X_raw, primary_col=0)
    logger.info(
        "Gradient Descent converged: initial_loss=%.4f -> final_loss=%.4f",
        optimizer.loss_history_[0],
        optimizer.loss_history_[-1],
    )

    # 3. PCA & Mahalanobis Distance Anomaly Detection
    detector = MultivariateAnomalyDetector(n_components=args.components)
    detector.fit(X_detrended)

    assert detector.diagnostics_ is not None
    assert detector.eigenvalues_ is not None
    assert detector.explained_variance_ratio_ is not None
    assert detector.corr_matrix_ is not None
    assert detector.cov_matrix_ is not None
    assert detector.mu_z_ is not None

    # Interactive debugging readiness hook (uncomment breakpoint() when inspecting tensors interactively):
    if args.debug:
        logger.debug("Debug flag active: spectral decomposition and covariance tensors ready.")
        # breakpoint()  # Or: import pdb; pdb.set_trace()

    mahal_distances = detector.compute_mahalanobis_distance(X_detrended)
    recon_errors = detector.compute_reconstruction_error(X_detrended, k=args.components)
    pc_scores = detector.transform_pc_scores(X_detrended, k=X_raw.shape[1])

    # Calibrate anomaly percentile cutoff to match expected anomaly rate
    anomaly_fraction = float(args.anomalies) / float(args.samples)
    percentile_cutoff = 100.0 * (1.0 - anomaly_fraction)
    y_pred = detector.detect(X_detrended, percentile_threshold=percentile_cutoff, k=args.components)

    # Compute Multivariate Gaussian PDF densities in standardized space
    Z_std = (X_detrended - detector.mu_) / detector.sigma_
    mv_densities = multivariate_gaussian_pdf(Z_std, detector.mu_z_, detector.cov_matrix_)
    mean_mv_density = float(compute_mean(mv_densities))

    logger.debug(
        "Mean Mahalanobis distance=%.4f, Mean reconstruction error=%.4f",
        float(compute_mean(mahal_distances)),
        float(compute_mean(recon_errors)),
    )

    # 4. Evaluate Classification Metrics against Ground Truth
    metrics = compute_classification_metrics(y_true, y_pred)
    logger.info(
        "Anomaly Detection Performance: Precision=%.2f%%, Recall=%.2f%%, F1=%.2f%%",
        metrics["precision"] * 100.0,
        metrics["recall"] * 100.0,
        metrics["f1"] * 100.0,
    )

    # 5. Generate Publication-Quality Matplotlib Diagnostic Figures
    visualizer = TelemetryVisualizer(output_dir=args.output_dir)
    mahal_threshold = float(np.percentile(mahal_distances, percentile_cutoff))

    plot_overview = visualizer.plot_telemetry_overview(
        timestamps=timestamps,
        X_raw=X_raw,
        y_pred=y_pred,
        feature_names=generator.feature_names,
        trend_curve=trend_curve,
        y_true=y_true,
    )
    plot_pca = visualizer.plot_pca_diagnostics(
        corr_matrix=detector.corr_matrix_,
        eigenvalues=detector.eigenvalues_,
        explained_variance_ratio=detector.explained_variance_ratio_,
        pc_scores=pc_scores,
        y_pred=y_pred,
        feature_names=generator.feature_names,
    )
    plot_calculus = visualizer.plot_calculus_convergence(
        loss_history=optimizer.loss_history_,
        anomaly_scores=mahal_distances,
        threshold_value=mahal_threshold,
    )

    saved_plots = [plot_overview, plot_pca, plot_calculus]
    print_summary_table(
        diagnostics=detector.diagnostics_,
        eigenvalues=detector.eigenvalues_,
        explained_ratio=detector.explained_variance_ratio_,
        optimizer=optimizer,
        metrics=metrics,
        mean_mv_density=mean_mv_density,
        saved_plots=saved_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
