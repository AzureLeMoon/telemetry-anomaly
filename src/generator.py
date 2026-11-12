"""Synthetic Multivariate Server Telemetry & Anomaly Injection Generator.

Simulates N correlated server telemetry streams (`cpu_load`, `memory_usage`,
`io_wait`, `network_throughput`) and injects realistic multivariate structural
anomalies that break inter-feature correlations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class TelemetryGenerator:
    r"""Generates synthetic multivariate server telemetry with injected anomalies.

    The generator synthesizes $D=4$ coupled metrics over $N$ timestamps:
        1. `cpu_load`: Latent sinusoidal diurnal workload + quadratic thermal/load drift + noise.
        2. `memory_usage`: Linearly coupled with `cpu_load` plus cumulative thermal drift.
        3. `io_wait`: Non-linearly coupled with `cpu_load` and disk pressure.
        4. `network_throughput`: Strongly linearly correlated with `cpu_load`.

    Attributes:
        n_samples: Number of timestamps $N$ to simulate (default 1000).
        n_anomalies: Number of multivariate anomalies to inject (default 25).
        random_seed: Seed for deterministic pseudo-random generation.
        feature_names: Tuple of metric names corresponding to columns $0, \dots, 3$.
    """

    feature_names: tuple[str, str, str, str] = (
        "cpu_load",
        "memory_usage",
        "io_wait",
        "network_throughput",
    )

    def __init__(
        self,
        n_samples: int = 1000,
        n_anomalies: int = 25,
        random_seed: int = 42,
    ) -> None:
        """Initialize the synthetic telemetry generator.

        Args:
            n_samples: Total number of time steps $N$.
            n_anomalies: Total number of multivariate anomaly timestamps to inject.
            random_seed: Integer seed for reproducible random sampling.

        Raises:
            ValueError: If `n_samples <= n_anomalies` or `n_samples < 10`.
        """
        if n_samples < 10:
            raise ValueError("n_samples must be at least 10.")
        if n_anomalies < 0 or n_anomalies >= n_samples:
            raise ValueError("n_anomalies must satisfy 0 <= n_anomalies < n_samples.")
        self.n_samples = int(n_samples)
        self.n_anomalies = int(n_anomalies)
        self.random_seed = int(random_seed)

    def generate(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        r"""Generate the multivariate telemetry matrix, timestamps, and anomaly labels.

        Mathematical Model:
            Let $t \in [0, 1]$ be normalized time.
            - Base trend: $\tau(t) = 12.0 t^2 + 8.0 t + 45.0$
            - Diurnal cycle: $s(t) = 15.0 \sin(4\pi t) + 5.0 \cos(8\pi t)$
            - `cpu_load`: $x_1(t) = \tau(t) + s(t) + \epsilon_1$
            - `memory_usage`: $x_2(t) = 0.65 x_1(t) + 18.0 t + 25.0 + \epsilon_2$
            - `io_wait`: $x_3(t) = 0.12 x_1(t) + 0.0018 x_1(t)^2 + 1.5 \sin(4\pi t) + \epsilon_3$
            - `network_throughput`: $x_4(t) = 4.2 x_1(t) + 30.0 + \epsilon_4$

            Injected multivariate anomalies break the cross-correlation structure
            (e.g., spiking `io_wait` while suppressing `cpu_load`, or dropping
            `network_throughput` during high `cpu_load`), making them stand out strongly
            in the minor principal components and Mahalanobis distance.

        Returns:
            A 3-tuple `(X_raw, timestamps, y_true)` where:
                - `X_raw`: Telemetry matrix of shape `(N, 4)`.
                - `timestamps`: Normalized time vector of shape `(N,)` in $[0, 1]$.
                - `y_true`: Binary ground-truth label vector of shape `(N,)` (`1` = anomaly, `0` = normal).
        """
        rng = np.random.default_rng(self.random_seed)
        timestamps = np.linspace(0.0, 1.0, self.n_samples, dtype=np.float64)

        # Quadratic baseline drift + harmonic diurnal workload
        baseline_drift = 12.0 * (timestamps ** 2) + 8.0 * timestamps + 45.0
        harmonic_wave = 15.0 * np.sin(4.0 * np.pi * timestamps) + 5.0 * np.cos(8.0 * np.pi * timestamps)

        cpu_load = baseline_drift + harmonic_wave + rng.normal(0.0, 1.8, size=self.n_samples)

        # Memory usage: strongly correlated with CPU load + gradual thermal/cache drift
        memory_usage = (
            0.65 * cpu_load
            + 18.0 * timestamps
            + 25.0
            + rng.normal(0.0, 0.9, size=self.n_samples)
        )

        # I/O wait: non-linearly coupled with CPU and disk pressure
        io_wait = (
            0.12 * cpu_load
            + 0.0018 * (cpu_load ** 2)
            + 1.5 * np.sin(4.0 * np.pi * timestamps)
            + rng.normal(0.0, 0.35, size=self.n_samples)
        )

        # Network throughput (Mbps): linearly correlated with CPU load
        network_throughput = (
            4.2 * cpu_load
            + 30.0
            + rng.normal(0.0, 4.0, size=self.n_samples)
        )

        X_raw = np.column_stack([cpu_load, memory_usage, io_wait, network_throughput])
        y_true = np.zeros(self.n_samples, dtype=np.int64)

        if self.n_anomalies > 0:
            candidate_indices = np.arange(20, self.n_samples - 20)
            anomaly_indices = np.sort(
                rng.choice(candidate_indices, size=self.n_anomalies, replace=False)
            )
            y_true[anomaly_indices] = 1

            for idx_pos, sample_idx in enumerate(anomaly_indices):
                mode = idx_pos % 3
                if mode == 0:
                    # Broken correlation Mode A: I/O deadlock (low CPU, high I/O wait, dropped network)
                    X_raw[sample_idx, 0] -= 18.0
                    X_raw[sample_idx, 2] += 14.5
                    X_raw[sample_idx, 3] -= 85.0
                elif mode == 1:
                    # Broken correlation Mode B: Memory leak / swap thrash (normal CPU, spiked memory & I/O)
                    X_raw[sample_idx, 1] += 26.0
                    X_raw[sample_idx, 2] += 12.0
                    X_raw[sample_idx, 3] -= 65.0
                else:
                    # Broken correlation Mode C: Network packet storm / NIC saturation with stalled I/O
                    X_raw[sample_idx, 0] -= 14.0
                    X_raw[sample_idx, 1] += 22.0
                    X_raw[sample_idx, 2] += 13.5
                    X_raw[sample_idx, 3] += 135.0

        return X_raw, timestamps, y_true
