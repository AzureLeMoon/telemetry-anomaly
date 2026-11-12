"""Scientific Plotting Module using Matplotlib.

Generates publication-quality multi-panel diagnostic figures saved to `output/`:
    1. `telemetry_overview.png`: 4-subplot time-series with detected anomalies and fitted trend.
    2. `pca_diagnostics.png`: Correlation matrix heatmap, eigenvalue scree plot, and 2D PCA
       projection with Mahalanobis confidence ellipse.
    3. `calculus_convergence.png`: Gradient descent convex loss trajectory and univariate
       Gaussian PDF distribution fit over anomaly scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from numpy.typing import NDArray

from src.linalg_stats import compute_covariance_matrix, compute_mean, compute_variance, gaussian_pdf


class TelemetryVisualizer:
    """Publication-quality scientific visualizer for multivariate anomaly detection.

    Attributes:
        output_dir: Directory path where generated PNG figures are saved.
        dpi: Resolution (dots per inch) for saved raster figures.
    """

    def __init__(self, output_dir: str | Path = "output", dpi: int = 180) -> None:
        """Initialize the visualizer and ensure the output directory exists.

        Args:
            output_dir: Target directory for output plots.
            dpi: Figure resolution in dots per inch.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)

    def plot_telemetry_overview(
        self,
        timestamps: NDArray[np.floating[Any]],
        X_raw: NDArray[np.floating[Any]],
        y_pred: NDArray[np.integer[Any]],
        feature_names: Sequence[str],
        trend_curve: NDArray[np.floating[Any]] | None = None,
        y_true: NDArray[np.integer[Any]] | None = None,
    ) -> Path:
        r"""Generate a 4-subplot time-series figure with highlighted multivariate anomalies.

        Args:
            timestamps: 1D array of timestamps of shape `(N,)`.
            X_raw: Raw telemetry matrix of shape `(N, 4)`.
            y_pred: Binary anomaly flags of shape `(N,)` (`1` = detected anomaly).
            feature_names: Sequence of 4 metric names.
            trend_curve: Optional fitted quadratic trend $\hat{y}(t)$ for the primary metric.
            y_true: Optional ground-truth binary anomaly labels of shape `(N,)`.

        Returns:
            Path to the saved `telemetry_overview.png` image.
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True)
        fig.suptitle(
            "Multivariate Server Telemetry Streams & PCA/Mahalanobis Anomaly Detection",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        units = ["Load (%)", "Usage (GB)", "Wait (ms)", "Throughput (Mbps)"]
        anom_mask = np.asarray(y_pred, dtype=bool)

        for idx, ax in enumerate(axes):
            series = X_raw[:, idx]
            ax.plot(
                timestamps,
                series,
                color=colors[idx],
                linewidth=1.1,
                alpha=0.85,
                label=f"{feature_names[idx]}",
            )
            if idx == 0 and trend_curve is not None:
                ax.plot(
                    timestamps,
                    trend_curve,
                    color="#111111",
                    linestyle="--",
                    linewidth=1.8,
                    label=r"Gradient Descent Trend $\hat{y}(t)=w_2 t^2 + w_1 t + b$",
                )
            if y_true is not None:
                true_mask = np.asarray(y_true, dtype=bool)
                ax.scatter(
                    timestamps[true_mask],
                    series[true_mask],
                    s=58,
                    facecolors="none",
                    edgecolors="#555555",
                    linewidths=1.2,
                    zorder=4,
                    label="Injected Ground Truth",
                )
            ax.scatter(
                timestamps[anom_mask],
                series[anom_mask],
                color="#d62728",
                s=28,
                zorder=5,
                label="Detected Multivariate Anomaly",
            )
            ax.set_ylabel(f"{feature_names[idx]}\n[{units[idx]}]", fontsize=9.5)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        axes[-1].set_xlabel(r"Normalized Time $t \in [0, 1]$", fontsize=10.5)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        out_path = self.output_dir / "telemetry_overview.png"
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def plot_pca_diagnostics(
        self,
        corr_matrix: NDArray[np.floating[Any]],
        eigenvalues: NDArray[np.floating[Any]],
        explained_variance_ratio: NDArray[np.floating[Any]],
        pc_scores: NDArray[np.floating[Any]],
        y_pred: NDArray[np.integer[Any]],
        feature_names: Sequence[str],
    ) -> Path:
        r"""Generate PCA diagnostic figure with correlation heatmap, scree plot, and 2D subspace.

        Layout:
            - Left Top: Pearson Correlation Matrix Heatmap $R$.
            - Left Bottom: Eigenvalue Scree Plot & Cumulative Explained Variance.
            - Right: 2D PCA subspace projection ($PC_1$ vs $PC_2$) with a Mahalanobis
              confidence ellipse separating normal traffic from anomalies.

        Args:
            corr_matrix: Pearson correlation matrix of shape `(D, D)`.
            eigenvalues: Sorted eigenvalues $\lambda$ of shape `(D,)`.
            explained_variance_ratio: Variance ratios of shape `(D,)`.
            pc_scores: Projected principal component coordinates of shape `(N, >=2)`.
            y_pred: Binary anomaly flags of shape `(N,)`.
            feature_names: Sequence of metric names.

        Returns:
            Path to the saved `pca_diagnostics.png` image.
        """
        fig = plt.figure(figsize=(14, 7.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.1, 0.9], hspace=0.38, wspace=0.28)

        ax_corr = fig.add_subplot(gs[0, 0])
        ax_scree = fig.add_subplot(gs[1, 0])
        ax_pca = fig.add_subplot(gs[:, 1])

        # Panel 1: Correlation Matrix Heatmap
        im = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax_corr.set_title("Pearson Correlation Matrix $R$", fontsize=11, fontweight="bold")
        n_features = len(feature_names)
        ax_corr.set_xticks(range(n_features))
        ax_corr.set_yticks(range(n_features))
        ax_corr.set_xticklabels(feature_names, rotation=25, ha="right", fontsize=8.5)
        ax_corr.set_yticklabels(feature_names, fontsize=8.5)
        for i in range(n_features):
            for j in range(n_features):
                val = corr_matrix[i, j]
                text_color = "white" if abs(val) > 0.65 else "black"
                ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5)
        fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

        # Panel 2: Eigenvalue Scree Plot & Explained Variance
        pcs = np.arange(1, len(eigenvalues) + 1)
        cum_var = np.cumsum(explained_variance_ratio) * 100.0
        pct_var = explained_variance_ratio * 100.0
        ax_scree.bar(pcs, pct_var, color="#1f77b4", alpha=0.8, width=0.55, label="Individual Var (%)")
        ax_scree.plot(pcs, cum_var, color="#d62728", marker="o", linewidth=1.8, label="Cumulative Var (%)")
        for idx, (ev, pv) in enumerate(zip(eigenvalues, pct_var)):
            ax_scree.text(pcs[idx], pv + 2.5, f"$\\lambda_{idx+1}$={ev:.2f}", ha="center", fontsize=8)
        ax_scree.set_xticks(pcs)
        ax_scree.set_xticklabels([f"PC{i}" for i in pcs])
        ax_scree.set_ylim(0, 115)
        ax_scree.set_ylabel("Explained Variance (%)", fontsize=9.5)
        ax_scree.set_title("Eigenvalue Scree Plot ($\\sum \\lambda_i = \\mathrm{Tr}(\\Sigma)$)", fontsize=11, fontweight="bold")
        ax_scree.grid(True, linestyle=":", alpha=0.5)
        ax_scree.legend(loc="center right", fontsize=8)

        # Panel 3: 2D PCA Subspace Projection (PC1 vs PC2) + Mahalanobis Confidence Ellipse
        anom_mask = np.asarray(y_pred, dtype=bool)
        normal_mask = ~anom_mask
        pc12 = pc_scores[:, :2]

        ax_pca.scatter(
            pc12[normal_mask, 0],
            pc12[normal_mask, 1],
            c="#1f77b4",
            s=18,
            alpha=0.55,
            label=f"Normal Telemetry (n={int(np.sum(normal_mask))})",
        )
        ax_pca.scatter(
            pc12[anom_mask, 0],
            pc12[anom_mask, 1],
            c="#d62728",
            s=48,
            marker="x",
            linewidths=1.8,
            label=f"Detected Anomalies (n={int(np.sum(anom_mask))})",
        )

        # Compute 2D Mahalanobis confidence ellipse from normal PC scores
        normal_pc12 = pc12[normal_mask] if np.any(normal_mask) else pc12
        center_2d = compute_mean(normal_pc12)
        cov_2d = compute_covariance_matrix(normal_pc12)
        eigvals_2d, eigvecs_2d = np.linalg.eigh(cov_2d)
        order_2d = np.argsort(eigvals_2d)[::-1]
        eigvals_2d = np.maximum(eigvals_2d[order_2d], 1e-12)
        eigvecs_2d = eigvecs_2d[:, order_2d]
        angle_deg = float(np.degrees(np.arctan2(eigvecs_2d[1, 0], eigvecs_2d[0, 0])))

        # 3-sigma (~99% 2D Gaussian contour radius = 3.035) Mahalanobis boundary
        chi2_radius = 3.035
        width = 2.0 * chi2_radius * float(np.sqrt(eigvals_2d[0]))
        height = 2.0 * chi2_radius * float(np.sqrt(eigvals_2d[1]))
        ellipse = Ellipse(
            xy=(float(center_2d[0]), float(center_2d[1])),
            width=width,
            height=height,
            angle=angle_deg,
            edgecolor="#111111",
            facecolor="#2ca02c",
            alpha=0.12,
            linestyle="--",
            linewidth=2.0,
            label="99% Mahalanobis Confidence Ellipse",
        )
        ax_pca.add_patch(ellipse)
        # Also draw crisp ellipse border
        ellipse_border = Ellipse(
            xy=(float(center_2d[0]), float(center_2d[1])),
            width=width,
            height=height,
            angle=angle_deg,
            edgecolor="#111111",
            facecolor="none",
            linestyle="--",
            linewidth=1.8,
        )
        ax_pca.add_patch(ellipse_border)

        ax_pca.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax_pca.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax_pca.set_xlabel(f"Principal Component 1 ({explained_variance_ratio[0]*100:.1f}% Variance)", fontsize=10.5)
        ax_pca.set_ylabel(f"Principal Component 2 ({explained_variance_ratio[1]*100:.1f}% Variance)", fontsize=10.5)
        ax_pca.set_title("2D PCA Subspace Projection ($PC_1$ vs $PC_2$) & Mahalanobis Boundary", fontsize=11.5, fontweight="bold")
        ax_pca.grid(True, linestyle=":", alpha=0.5)
        ax_pca.legend(loc="upper right", fontsize=9)

        fig.suptitle("Spectral PCA & Covariance Diagnostics", fontsize=13.5, fontweight="bold", y=0.98)
        out_path = self.output_dir / "pca_diagnostics.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_calculus_convergence(
        self,
        loss_history: Sequence[float],
        anomaly_scores: NDArray[np.floating[Any]],
        threshold_value: float,
    ) -> Path:
        r"""Generate a 2-panel plot of Gradient Descent convergence and Gaussian score density.

        Args:
            loss_history: Sequence of convex loss values $\mathcal{L}(w, b)$ over epochs.
            anomaly_scores: 1D array of Mahalanobis anomaly scores $D_M(z_i)$.
            threshold_value: Decision threshold separating normal vs anomalous telemetry.

        Returns:
            Path to the saved `calculus_convergence.png` image.
        """
        fig, (ax_loss, ax_pdf) = plt.subplots(1, 2, figsize=(13, 5.2))

        # Left Panel: Convex Loss Trajectory over Gradient Descent Epochs
        epochs = np.arange(1, len(loss_history) + 1)
        ax_loss.plot(epochs, loss_history, color="#1f77b4", linewidth=2.2, label=r"$\mathcal{L}(w, b) = \mathrm{MSE} + \alpha \|w\|_2^2$")
        ax_loss.scatter([1, len(loss_history)], [loss_history[0], loss_history[-1]], color="#d62728", zorder=5)
        ax_loss.annotate(
            f"Initial: {loss_history[0]:.2f}",
            xy=(1, loss_history[0]),
            xytext=(max(len(loss_history) * 0.12, 5), loss_history[0] * 0.88),
            arrowprops={"arrowstyle": "->", "color": "#444444"},
            fontsize=9,
        )
        ax_loss.annotate(
            f"Converged: {loss_history[-1]:.2f}",
            xy=(len(loss_history), loss_history[-1]),
            xytext=(len(loss_history) * 0.58, loss_history[0] * 0.28),
            arrowprops={"arrowstyle": "->", "color": "#444444"},
            fontsize=9,
        )
        ax_loss.set_title("Gradient Descent Convex Loss Convergence", fontsize=11.5, fontweight="bold")
        ax_loss.set_xlabel("Optimization Epoch", fontsize=10)
        ax_loss.set_ylabel(r"Regularized Loss $\mathcal{L}(w, b)$", fontsize=10)
        ax_loss.grid(True, linestyle=":", alpha=0.5)
        ax_loss.legend(loc="upper right", fontsize=9.5)

        # Right Panel: Empirical Histogram & Fitted Univariate Gaussian PDF over Anomaly Scores
        scores_arr = np.asarray(anomaly_scores, dtype=np.float64)
        inlier_scores = scores_arr[scores_arr <= threshold_value]
        if inlier_scores.size < 2:
            inlier_scores = scores_arr

        mu_score = float(compute_mean(inlier_scores))
        var_score = float(compute_variance(inlier_scores))

        x_grid = np.linspace(0.0, float(np.max(scores_arr)) * 1.08, 300)
        pdf_vals = gaussian_pdf(x_grid, mu=mu_score, var=var_score)

        ax_pdf.hist(
            scores_arr,
            bins=38,
            density=True,
            color="#1f77b4",
            alpha=0.55,
            edgecolor="white",
            label="Empirical Score Density",
        )
        ax_pdf.plot(
            x_grid,
            pdf_vals,
            color="#2ca02c",
            linewidth=2.2,
            label=rf"Gaussian PDF Fit ($\mu={mu_score:.2f}, \sigma^2={var_score:.2f}$)",
        )
        ax_pdf.axvline(
            threshold_value,
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label=f"Anomaly Threshold ({threshold_value:.2f})",
        )
        ax_pdf.set_title("Mahalanobis Score Distribution & Gaussian PDF Fit", fontsize=11.5, fontweight="bold")
        ax_pdf.set_xlabel("Mahalanobis Distance $D_M(z_i)$", fontsize=10)
        ax_pdf.set_ylabel("Probability Density $f(x)$", fontsize=10)
        ax_pdf.grid(True, linestyle=":", alpha=0.5)
        ax_pdf.legend(loc="upper right", fontsize=9)

        fig.suptitle("Numerical Optimization & Probabilistic Threshold Calibration", fontsize=13, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        out_path = self.output_dir / "calculus_convergence.png"
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path
