"""MultiLayer Perceptron with different Optimizers (SGD, Momentum, RMSprop, Adam) implemented from scratch."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLPClassifierWithOptimizers:
    """MultiLayer Perceptron Classifier with customizable optimizers (SGD, Momentum, RMSprop, Adam)."""

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        optimizer: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer.lower()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize weights and biases
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(1, len(layer_sizes)):
            # He initialization
            limit = np.sqrt(2.0 / layer_sizes[i - 1])
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * limit)
            self.b.append(np.zeros((layer_sizes[i], 1)))

        # Initialize optimizer states
        self.v_W: list[np.ndarray] = [np.zeros_like(w) for w in self.W]
        self.v_b: list[np.ndarray] = [np.zeros_like(bias) for bias in self.b]
        self.m_W: list[np.ndarray] = [np.zeros_like(w) for w in self.W]
        self.m_b: list[np.ndarray] = [np.zeros_like(bias) for bias in self.b]

        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        a_values = [X]
        z_values = []
        a = X
        for W_l, b_l in zip(self.W, self.b):
            z = W_l @ a + b_l
            a = self._sigmoid(z)
            z_values.append(z)
            a_values.append(a)
        return a_values, z_values

    def _backward(self, a_values: list[np.ndarray], y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n_samples = y.shape[1]
        dW = []
        db = []

        # Output layer error
        a_out = a_values[-1]
        delta = (a_out - y) * self._sigmoid_derivative(a_out)

        num_layers = len(self.W)
        for l in reversed(range(num_layers)):
            a_prev = a_values[l]
            dW_l = (1.0 / n_samples) * (delta @ a_prev.T)
            db_l = (1.0 / n_samples) * np.sum(delta, axis=1, keepdims=True)

            dW.insert(0, dW_l)
            db.insert(0, db_l)

            if l > 0:
                delta = (self.W[l].T @ delta) * self._sigmoid_derivative(a_prev)

        return dW, db

    def _update_parameters(self, dW: list[np.ndarray], db: list[np.ndarray], t: int) -> None:
        """Update weights and biases using the selected optimizer."""
        for l in range(len(self.W)):
            if self.optimizer == "sgd":
                # SGD: W = W - lr * dW
                self.W[l] -= self.lr * dW[l]
                self.b[l] -= self.lr * db[l]

            elif self.optimizer == "momentum":
                # Momentum: v = beta1 * v + (1 - beta1) * dW
                #           W = W - lr * v
                self.v_W[l] = self.beta1 * self.v_W[l] + (1 - self.beta1) * dW[l]
                self.v_b[l] = self.beta1 * self.v_b[l] + (1 - self.beta1) * db[l]
                self.W[l] -= self.lr * self.v_W[l]
                self.b[l] -= self.lr * self.v_b[l]

            elif self.optimizer == "rmsprop":
                # RMSprop: s = beta2 * s + (1 - beta2) * (dW^2)
                #          W = W - lr * dW / (sqrt(s) + eps)
                self.v_W[l] = self.beta2 * self.v_W[l] + (1 - self.beta2) * (dW[l] ** 2)
                self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (db[l] ** 2)
                self.W[l] -= self.lr * dW[l] / (np.sqrt(self.v_W[l]) + self.epsilon)
                self.b[l] -= self.lr * db[l] / (np.sqrt(self.v_b[l]) + self.epsilon)

            elif self.optimizer == "adam":
                # Adam: m = beta1 * m + (1 - beta1) * dW (Moment 1)
                #       v = beta2 * v + (1 - beta2) * (dW^2) (Moment 2)
                #       m_hat = m / (1 - beta1^t)
                #       v_hat = v / (1 - beta2^t)
                #       W = W - lr * m_hat / (sqrt(v_hat) + eps)
                self.m_W[l] = self.beta1 * self.m_W[l] + (1 - self.beta1) * dW[l]
                self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * db[l]

                self.v_W[l] = self.beta2 * self.v_W[l] + (1 - self.beta2) * (dW[l] ** 2)
                self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (db[l] ** 2)

                # Bias correction
                m_W_hat = self.m_W[l] / (1.0 - self.beta1 ** t)
                m_b_hat = self.m_b[l] / (1.0 - self.beta1 ** t)
                v_W_hat = self.v_W[l] / (1.0 - self.beta2 ** t)
                v_b_hat = self.v_b[l] / (1.0 - self.beta2 ** t)

                # Update weights and biases
                self.W[l] -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                self.b[l] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifierWithOptimizers":
        X_input = X.T
        if len(y.shape) == 1:
            y_input = y.reshape(1, -1)
        else:
            y_input = y.T

        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(1, self.epochs + 1):
            # Forward Pass
            a_values, _ = self._forward(X_input)
            y_hat = a_values[-1]

            # Compute Loss (MSE)
            loss = 0.5 * np.mean(np.sum((y_hat - y_input) ** 2, axis=0))
            self.loss_history.append(loss)

            # Compute Accuracy
            preds = (y_hat >= 0.5).astype(int)
            acc = float(np.mean(preds == y_input))
            self.accuracy_history.append(acc)

            # Backward Pass
            dW, db = self._backward(a_values, y_input)

            # Update parameters (t = epoch, 1-based index)
            self._update_parameters(dW, db, epoch)

        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        X_input = X.T
        a_values, _ = self._forward(X_input)
        probs = a_values[-1].T
        return (probs >= threshold).astype(int).ravel()


def main() -> None:
    # Set seed for reproducibility
    np.random.seed(42)

    # 1. Create dataset (make_moons)
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Network topology: 2 inputs -> 10 hidden -> 5 hidden -> 1 output
    layer_sizes = [2, 10, 5, 1]
    epochs = 1500

    # Dictionary to hold the trained classifiers and their configurations
    # We choose appropriate learning rates for each optimizer to make the comparison fair.
    # Adaptive methods like RMSprop/Adam can use a smaller learning rate and converge fast.
    optimizers_config = {
        "SGD": {"optimizer": "sgd", "lr": 0.1},
        "Momentum": {"optimizer": "momentum", "lr": 0.1, "beta1": 0.9},
        "RMSprop": {"optimizer": "rmsprop", "lr": 0.01, "beta2": 0.99},
        "Adam": {"optimizer": "adam", "lr": 0.01, "beta1": 0.9, "beta2": 0.999},
    }

    results = {}

    print("=" * 60)
    print("  So sánh các bộ tối ưu hóa (Optimizers) trên tập Moons")
    print("=" * 60)

    for name, config in optimizers_config.items():
        print(f"\nĐang huấn luyện với bộ tối ưu: {name} (learning rate = {config['lr']})...")
        mlp = MLPClassifierWithOptimizers(
            layer_sizes=layer_sizes,
            learning_rate=config["lr"],
            epochs=epochs,
            optimizer=config["optimizer"],
            beta1=config.get("beta1", 0.9),
            beta2=config.get("beta2", 0.999),
        )
        mlp.fit(X_train, y_train)

        # Evaluate
        test_preds = mlp.predict(X_test)
        test_acc = float(np.mean(test_preds == y_test))
        print(f"--> Hoàn thành! Test Accuracy: {test_acc:.2%} | Final Loss: {mlp.loss_history[-1]:.6f}")

        results[name] = {
            "loss_history": mlp.loss_history,
            "accuracy_history": mlp.accuracy_history,
            "test_accuracy": test_acc,
        }

    # Plot Comparison Curves
    plt.figure(figsize=(14, 6))

    # Plot 1: Loss History
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data["loss_history"], label=f"{name} (Test Acc: {data['test_accuracy']:.2%})", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Lịch sử Loss của các bộ tối ưu hóa")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Accuracy History
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data["accuracy_history"], label=name, linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Lịch sử Accuracy trong quá trình huấn luyện")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_path = "gradient_descent/optimizers_loss_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[Thành công] Đã lưu đồ thị so sánh các bộ tối ưu vào: {plot_path}")


if __name__ == "__main__":
    main()
