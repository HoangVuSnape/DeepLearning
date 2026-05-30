"""Linear Regression with Gradient Descent from scratch."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        # === Bước 1: Khởi tạo thông số (Initialization) ===
        self.lr = learning_rate
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def _mse_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """MSE = (1/N) * sum((y - y_hat)^2)"""
        return float(np.mean((y - y_hat) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # === Bước 2: Lan truyền tiến (Forward Pass) ===
            # y_hat = X . w + b
            y_hat = X @ self.w + self.b

            # === Bước 3: Tính sai số (Compute Loss) ===
            loss = self._mse_loss(y, y_hat)
            self.loss_history.append(loss)

            # === Bước 4: Tính Gradient (Backward Pass) ===
            # dL/dw = (1/N) * X^T . (y_hat - y)      (đạo hàm MSE theo w)
            # dL/db = (1/N) * sum(y_hat - y)           (đạo hàm MSE theo b)
            error = y_hat - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # === Bước 5: Cập nhật trọng số (Update Weights) ===
            # w_j^{t+1} = w_j^t - mu * dL/dw_j
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


def main() -> None:
    np.random.seed(42)

    X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegressionGD(learning_rate=0.05, epochs=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_mse = float(np.mean((y_test - y_pred) ** 2))
    print(f"[Linear Regression] Final Train Loss (MSE): {model.loss_history[-1]:.4f}")
    print(f"[Linear Regression] Test MSE: {test_mse:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Loss curve ---
    axes[0].plot(model.loss_history, color="steelblue", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Linear Regression — Loss Curve")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Regression line vs data ---
    axes[1].scatter(X_test, y_test, color="steelblue", alpha=0.7, label="Actual")
    sort_idx = X_test[:, 0].argsort()
    axes[1].plot(X_test[sort_idx], y_pred[sort_idx], color="tomato", linewidth=2, label="Predicted")
    axes[1].set_xlabel("X (scaled)")
    axes[1].set_ylabel("y")
    axes[1].set_title("Linear Regression — Predictions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("linear_regression_results.png", dpi=150)
    plt.show()
    print("Saved: linear_regression_results.png")


if __name__ == "__main__":
    main()
