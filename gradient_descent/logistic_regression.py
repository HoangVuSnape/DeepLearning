"""Logistic Regression with Gradient Descent from scratch."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        # === Bước 1: Khởi tạo thông số (Initialization) ===
        self.lr = learning_rate
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid: sigma(z) = 1 / (1 + e^{-z})"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
        """BCE = -(1/N) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))"""
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # === Bước 2: Lan truyền tiến (Forward Pass) ===
            # y_hat = Sigmoid(X . w + b)
            z = X @ self.w + self.b
            y_hat = self._sigmoid(z)

            # === Bước 3: Tính sai số (Compute Loss) ===
            loss = self._binary_cross_entropy(y, y_hat)
            self.loss_history.append(loss)

            # === Bước 4: Tính Gradient (Backward Pass) ===
            # Gradient của BCE sau khi rút gọn cũng có dạng:
            # dL/dw = (1/N) * X^T . (y_hat - y)
            # dL/db = (1/N) * sum(y_hat - y)
            error = y_hat - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # === Bước 5: Cập nhật trọng số (Update Weights) ===
            # w_j^{t+1} = w_j^t - mu * dL/dw_j
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def main() -> None:
    np.random.seed(42)

    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegressionGD(learning_rate=0.1, epochs=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    print(f"[Logistic Regression] Final Train Loss (BCE): {model.loss_history[-1]:.4f}")
    print(f"[Logistic Regression] Test Accuracy: {accuracy:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Loss curve ---
    axes[0].plot(model.loss_history, color="darkorange", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title("Logistic Regression — Loss Curve")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Decision boundary ---
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid).reshape(xx.shape)

    axes[1].contourf(xx, yy, probs, levels=50, cmap="RdYlBu_r", alpha=0.6)
    scatter = axes[1].scatter(
        X_test[:, 0], X_test[:, 1], c=y_test,
        cmap="RdYlBu_r", edgecolors="black", linewidth=0.5, s=50,
    )
    axes[1].set_xlabel("Feature 1 (scaled)")
    axes[1].set_ylabel("Feature 2 (scaled)")
    axes[1].set_title("Logistic Regression — Decision Boundary")
    plt.colorbar(scatter, ax=axes[1], label="Class probability")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logistic_regression_results.png", dpi=150)
    plt.show()
    print("Saved: logistic_regression_results.png")


if __name__ == "__main__":
    main()
