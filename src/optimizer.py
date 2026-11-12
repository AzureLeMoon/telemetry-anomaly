r"""Calculus & Numerical Gradient Descent Optimization Module.

Demonstrates explicit differential calculus, partial derivatives, the chain rule,
and convex optimization on an L2-regularized Mean Squared Error (MSE) objective
to fit a quadratic baseline trend curve $\hat{y}(t) = w_2 t^2 + w_1 t + b$.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.linalg_stats import vector_inner_product


class PolynomialTrendOptimizer:
    r"""Fits a quadratic baseline trend $\hat{y}(t) = w_2 t^2 + w_1 t + b$ via Gradient Descent.

    Mathematical Formulation:
        Let $X_t = [t^2, t] \in \mathbb{R}^{N \times 2}$ be the polynomial design matrix,
        $w = [w_2, w_1]^T \in \mathbb{R}^2$ be the weight vector, and $b \in \mathbb{R}$
        be the scalar intercept.

        Forward Model:
            $$\hat{y} = X_t w + b \mathbf{1}_N$$

        Convex Objective Function (MSE with L2 Regularization):
            $$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 + \alpha \|w\|_2^2$$

        Analytical Gradients via the Chain Rule:
            Let residual $r = \hat{y} - y \in \mathbb{R}^N$.
            By the chain rule $\frac{\partial \mathcal{L}}{\partial w} =
            \frac{\partial \mathcal{L}}{\partial r} \frac{\partial r}{\partial \hat{y}}
            \frac{\partial \hat{y}}{\partial w} + \frac{\partial}{\partial w}(\alpha w^T w)$:
            $$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} X_t^T (\hat{y} - y) + 2\alpha w$$
            $$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

    Attributes:
        learning_rate: Step size $\eta > 0$ for gradient descent updates.
        epochs: Number of iterative optimization steps (default 500).
        alpha: L2 regularization coefficient $\alpha \ge 0$.
        weights_: Learned polynomial weights $[w_2, w_1]^T$ of shape `(2,)`.
        bias_: Learned scalar bias $b$.
        loss_history_: List of objective values $\mathcal{L}(w^{(k)}, b^{(k)})$ across epochs.
        grad_norm_history_: List of Euclidean gradient norms across epochs.
    """

    def __init__(
        self,
        learning_rate: float = 0.25,
        epochs: int = 500,
        alpha: float = 1e-4,
    ) -> None:
        r"""Initialize the polynomial trend optimizer.

        Args:
            learning_rate: Gradient descent learning rate $\eta$.
            epochs: Number of optimization iterations (default 500).
            alpha: L2 regularization hyperparameter $\alpha \ge 0$.

        Raises:
            ValueError: If `learning_rate <= 0`, `epochs < 1`, or `alpha < 0`.
        """
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.alpha = float(alpha)

        self.weights_: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []
        self.grad_norm_history_: list[float] = []

    @staticmethod
    def _build_design_matrix(t: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        r"""Construct the quadratic polynomial design matrix $X_t = [t^2, t]$.

        Args:
            t: 1D array of timestamps of shape `(N,)`.

        Returns:
            2D feature matrix of shape `(N, 2)` where column 0 is $t^2$ and column 1 is $t$.
        """
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        return np.column_stack([t_arr ** 2, t_arr])

    def predict(self, t: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        r"""Evaluate the quadratic polynomial trend $\hat{y}(t) = w_2 t^2 + w_1 t + b$.

        Args:
            t: 1D array of timestamps of shape `(N,)`.

        Returns:
            1D array of predicted baseline values $\hat{y}$ of shape `(N,)`.
        """
        X_t = self._build_design_matrix(t)
        return (X_t @ self.weights_) + self.bias_

    def compute_loss(
        self,
        t: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]] | None = None,
        bias: float | None = None,
    ) -> float:
        r"""Compute the convex L2-regularized Mean Squared Error loss.

        Mathematical Formulation:
            $$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 + \alpha \|w\|_2^2$$

        Args:
            t: 1D array of timestamps of shape `(N,)`.
            y: 1D array of target metric values of shape `(N,)`.
            weights: Optional weight vector $[w_2, w_1]^T$ (defaults to `self.weights_`).
            bias: Optional intercept scalar $b$ (defaults to `self.bias_`).

        Returns:
            Scalar loss value $\mathcal{L}(w, b)$.
        """
        X_t = self._build_design_matrix(t)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        w = self.weights_ if weights is None else np.asarray(weights, dtype=np.float64).ravel()
        b = self.bias_ if bias is None else float(bias)

        y_hat = (X_t @ w) + b
        residual = y_hat - y_arr
        n_samples = float(y_arr.shape[0])

        mse_term = vector_inner_product(residual, residual) / n_samples
        l2_penalty = self.alpha * vector_inner_product(w, w)
        return float(mse_term + l2_penalty)

    def compute_gradients(
        self,
        t: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]] | None = None,
        bias: float | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        r"""Compute exact analytical partial derivatives via the Chain Rule.

        Mathematical Formulation:
            $$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} X_t^T (\hat{y} - y) + 2\alpha w$$
            $$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \mathbf{1}_N^T (\hat{y} - y)$$

        Args:
            t: 1D array of timestamps of shape `(N,)`.
            y: 1D array of target metric values of shape `(N,)`.
            weights: Optional weight vector $[w_2, w_1]^T$ (defaults to `self.weights_`).
            bias: Optional scalar bias $b$ (defaults to `self.bias_`).

        Returns:
            Tuple `(grad_w, grad_b)` where `grad_w` has shape `(2,)` and `grad_b` is a float.
        """
        X_t = self._build_design_matrix(t)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        w = self.weights_ if weights is None else np.asarray(weights, dtype=np.float64).ravel()
        b = self.bias_ if bias is None else float(bias)

        n_samples = float(y_arr.shape[0])
        y_hat = (X_t @ w) + b
        residual = y_hat - y_arr

        grad_w = (2.0 / n_samples) * (X_t.T @ residual) + (2.0 * self.alpha * w)
        ones = np.ones(y_arr.shape[0], dtype=np.float64)
        grad_b = float((2.0 / n_samples) * (ones @ residual))
        return grad_w, grad_b

    def fit(
        self,
        t: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
    ) -> PolynomialTrendOptimizer:
        r"""Run iterative Gradient Descent to minimize the convex regularized MSE loss.

        Update Rule at epoch $k$:
            $$w^{(k+1)} = w^{(k)} - \eta \frac{\partial \mathcal{L}}{\partial w}$$
            $$b^{(k+1)} = b^{(k)} - \eta \frac{\partial \mathcal{L}}{\partial b}$$

        Args:
            t: 1D array of timestamps of shape `(N,)`.
            y: 1D array of target primary metric observations of shape `(N,)`.

        Returns:
            Self (`PolynomialTrendOptimizer`) with fitted `weights_`, `bias_`, and `loss_history_`.
        """
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if t_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Timestamps t and target y must have the same length.")

        self.weights_ = np.zeros(2, dtype=np.float64)
        self.bias_ = 0.0
        self.loss_history_ = []
        self.grad_norm_history_ = []

        for _ in range(self.epochs):
            loss_val = self.compute_loss(t_arr, y_arr)
            self.loss_history_.append(loss_val)

            grad_w, grad_b = self.compute_gradients(t_arr, y_arr)
            grad_vec = np.array([grad_w[0], grad_w[1], grad_b], dtype=np.float64)
            self.grad_norm_history_.append(float(np.sqrt(vector_inner_product(grad_vec, grad_vec))))

            self.weights_ = self.weights_ - self.learning_rate * grad_w
            self.bias_ = self.bias_ - self.learning_rate * grad_b

        return self

    def detrend(
        self,
        t: NDArray[np.floating[Any]],
        X: NDArray[np.floating[Any]],
        primary_col: int = 0,
    ) -> NDArray[np.float64]:
        """Subtract the fitted quadratic baseline trend from the primary telemetry column.

        Args:
            t: 1D array of timestamps of shape `(N,)`.
            X: Telemetry matrix of shape `(N, D)`.
            primary_col: Column index of the primary metric to detrend (default `0`).

        Returns:
            Copy of `X` with the primary column detrended around its mean level.
        """
        X_copy = np.array(X, dtype=np.float64, copy=True)
        trend = self.predict(t)
        mean_trend = float(np.sum(trend) / trend.shape[0])
        X_copy[:, primary_col] = X_copy[:, primary_col] - (trend - mean_trend)
        return X_copy
