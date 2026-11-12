# Multivariate Time-Series Anomaly Detection Engine

A Python engine for **Multivariate Anomaly Detection on System Telemetry** using **Principal Component Analysis (PCA)**, **Mahalanobis Distance**, and **Numerical Gradient Descent**.

---

## 1. Mathematical & Architectural Overview

Modern distributed systems emit correlated multivariate telemetry streams (`cpu_load`, `memory_usage`, `io_wait`, `network_throughput`). Univariate static thresholds frequently fail to detect structural anomalies in which individual metrics remain within nominal ranges while their **cross-correlation structure** breaks down (e.g., surging `io_wait` during low `cpu_load`).


### 1.1 Linear Algebra & Statistical Primitives (`src/linalg_stats.py`)
- **Vector Inner Product & Euclidean $L_2$ Norm:**
  $$\langle u, v \rangle = u^T v = \sum_{i=1}^D u_i v_i, \qquad \|v\|_2 = \sqrt{v^T v}$$
- **Sample Mean & Unbiased Sample Variance (Bessel's Correction):**
  $$\mu = \frac{1}{N} \sum_{i=1}^N x_i, \qquad s^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \mu)^2$$
- **Z-Score Standardization & Sample Covariance Matrix:**
  $$Z = \frac{X - \mu}{\sigma}, \qquad \Sigma = \frac{1}{N - 1} Z^T Z$$
- **Pearson Correlation Matrix:**
  $$R_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii} \Sigma_{jj}}}$$
- **Matrix Diagnostics & Regularized Inversion:**
  Computes $\text{Tr}(\Sigma) = \sum_{i=1}^D \Sigma_{ii}$, $\det(\Sigma)$, Frobenius norm $\|\Sigma\|_F = \sqrt{\text{Tr}(\Sigma^T \Sigma)}$, and Tikhonov-regularized inverse $(\Sigma + \epsilon I)^{-1}$.
- **Univariate & Multivariate Gaussian Probability Density Functions (PDF):**
  $$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \qquad f(\mathbf{x}) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

### 1.2 Calculus & Convex Gradient Descent (`src/optimizer.py`)
Before spectral scoring, `PolynomialTrendOptimizer` calibrates a quadratic baseline drift curve $\hat{y}(t) = w_2 t^2 + w_1 t + b$ by minimizing the convex $L_2$-regularized Mean Squared Error (MSE) objective:
$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 + \alpha \|w\|_2^2$$
Using the **chain rule**, the analytical partial derivatives are:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} X_t^T (\hat{y} - y) + 2\alpha w, \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
where $X_t = [t^2, t] \in \mathbb{R}^{N \times 2}$.

### 1.3 Spectral PCA & Mahalanobis Distance Engine (`src/pca_detector.py`)
1. **Symmetric Eigendecomposition:** Solves $\Sigma V = V \Lambda$ with eigenvalues sorted $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$ and verifies the spectral trace identity:
   $$\sum_{i=1}^D \lambda_i = \text{Tr}(\Sigma)$$
2. **Principal Subspace Projection & Reconstruction Residual:**
   $$P = Z V_k, \qquad \hat{Z} = P V_k^T, \qquad \|e_i\|_2 = \sqrt{(z_i - \hat{z}_i)^T (z_i - \hat{z}_i)}$$
3. **Mahalanobis Distance:**
   $$D_M(z_i) = \sqrt{(z_i - \mu_z)^T \Sigma^{-1} (z_i - \mu_z)}$$

---

## 2. Repository Structure

```text
telemetry-anomaly-pca/
├── environment.yml          # Conda environment specification
├── requirements.txt         # PIP requirements (numpy, matplotlib, pytest)
├── README.md                # Overview of the project
├── main.py                  # Entrypoint with argparse and logging debugger
├── src/
│   ├── __init__.py
│   ├── generator.py         # Synthetic multivariate telemetry & anomaly injection
│   ├── linalg_stats.py      # Linear Algebra & Statistics primitives
│   ├── pca_detector.py      # PCA & Mahalanobis Distance anomaly detection model
│   ├── optimizer.py         # Gradient Descent threshold/trend calibrator (Calculus)
│   └── visualizer.py        # Matplotlib multi-panel scientific plotting
└── tests/
    └── test_math.py         # Unit tests verifying custom math against numpy builtins
```

---

## 3. Installation & Usage

### Environment Setup

```bash
# Option A: Using pip / virtualenv
pip install -r requirements.txt

# Option B: Using Conda
conda env create -f environment.yml
conda activate telemetry-anomaly-pca
```

### Running the Anomaly Detection Pipeline

```bash
python main.py --samples 1000 --components 2 --learning-rate 0.25 --debug
```

Generated scientific figures are saved to `output/`:
- `output/telemetry_overview.png`: 4-subplot time-series with fitted Gradient Descent baseline trend and detected multivariate anomalies.
- `output/pca_diagnostics.png`: Pearson correlation heatmap, eigenvalue scree plot, and 2D PCA subspace projection ($PC_1$ vs $PC_2$) with 99% Mahalanobis confidence ellipse.
- `output/calculus_convergence.png`: Gradient Descent convex loss convergence curve over 500 epochs and univariate Gaussian PDF fit over Mahalanobis anomaly scores.

### Running Unit Tests

```bash
pytest -v
```
